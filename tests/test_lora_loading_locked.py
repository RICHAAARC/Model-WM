"""Locked CPU interface regression; synthetic modules only, no checkpoint or image.

Run separately from the lightweight controls as documented in LOCAL_VALIDATION.md.
Version mismatches fail explicitly; they must not silently skip this release check.
"""
import importlib.metadata
import platform
import unittest
from unittest.mock import patch

import torch
from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from transformers import CLIPTextConfig, CLIPTextModel

from qualification.run import convert_message


LOCKED = {
    "torch": "2.0.1+cpu", "torchvision": "0.15.2+cpu",
    "diffusers": "0.24.0", "transformers": "4.26.1", "accelerate": "0.28.0",
    "huggingface-hub": "0.21.4", "safetensors": "0.4.2", "numpy": "1.24.3",
}


class SyntheticMapper(torch.nn.Module):
    def forward(self, bits):
        # Fixed inputs, no random draw and no real mapper weights.
        return (1. + bits[:, :1]) * torch.linspace(.5, 1.5, 320)[None, :]


class LockedLoraLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if platform.python_version() != "3.10.14":
            raise RuntimeError("This regression requires Python 3.10.14")
        actual = {name: importlib.metadata.version(name) for name in LOCKED}
        if actual != LOCKED:
            raise RuntimeError(f"Locked CPU versions required: {actual}")
        torch.set_num_threads(1)

    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        # A regression must never download weights or enter a denoising forward.
        for target in ["diffusers.StableDiffusionPipeline.from_pretrained",
                       "diffusers.UNet2DConditionModel.from_pretrained",
                       "diffusers.UNet2DConditionModel.forward",
                       "torch.load", "torch.cuda._lazy_init"]:
            guard = patch(target, side_effect=AssertionError("No real weights, forward or CUDA in interface check"))
            guard.start()
            self.addCleanup(guard.stop)
        self.mapper = SyntheticMapper()
        prefix = "unet.down_blocks.0.attentions.0."
        self.modules = {
            prefix + "transformer_blocks.0.attn1.to_q": (32, 32, False),
            prefix + "transformer_blocks.0.ff.net.2": (128, 32, False),
            prefix + "proj_in": (32, 32, True),
            prefix + "proj_out": (32, 32, True),
        }
        self.source = {}
        for name, (width, height, conv) in self.modules.items():
            down = torch.arange(320 * width, dtype=torch.float32).reshape(320, width) / (320 * width)
            up = torch.full((height, 320), .001, dtype=torch.float32)
            if conv:
                down, up = down[:, :, None, None], up[:, :, None, None]
            self.source[name + ".lora.down.weight"] = down
            self.source[name + ".lora.up.weight"] = up
        self.source["text_encoder.ignored.lora.down.weight"] = torch.ones(1)
        self.source_before = {k: v.clone() for k, v in self.source.items()}

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())
        for key, value in self.source.items():
            self.assertTrue(torch.equal(value, self.source_before[key]), key)

    def pipeline(self):
        unet = UNet2DConditionModel(
            sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(32, 32),
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=32, attention_head_dim=4)
        text = CLIPTextModel(CLIPTextConfig(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=4))
        vae = AutoencoderKL(block_out_channels=(32, 32),
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"), latent_channels=4)
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=None,
            unet=unet, scheduler=DPMSolverMultistepScheduler(steps_offset=1),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        pipe.unet.requires_grad_(False)
        return pipe

    def test_conversion_preserves_values_for_all_official_branches(self):
        normal = convert_message(self.mapper, self.source, [0, 1] * 24)
        # Recreate the previous inference-tensor context without editing production code.
        with torch.inference_mode():
            legacy = convert_message(self.mapper, self.source, [0, 1] * 24)
        self.assertEqual(len(normal), 8)
        self.assertEqual(normal.keys(), legacy.keys())
        for key, value in normal.items():
            self.assertTrue(torch.equal(value, legacy[key]), key)
            self.assertFalse(value.is_inference(), key)
            self.assertFalse(value.requires_grad, key)

    def test_legacy_context_reproduces_the_loader_error(self):
        with torch.inference_mode():
            legacy = convert_message(self.mapper, self.source, [0] * 48)
        with self.assertRaisesRegex(RuntimeError, "inference tensor outside InferenceMode"):
            self.pipeline().load_lora_weights(legacy)

    def test_linear_and_conv_load_fuse_unload(self):
        pipe = self.pipeline()
        before = {k: v.clone() for k, v in pipe.unet.state_dict().items()}
        pipe.load_lora_weights(convert_message(self.mapper, self.source, [0] * 48))
        pipe.fuse_lora(lora_scale=1.0)
        pipe.unload_lora_weights()
        after = pipe.unet.state_dict()
        self.assertEqual(before.keys(), after.keys())
        for name in self.modules:
            key = name.removeprefix("unet.") + ".weight"
            self.assertFalse(torch.equal(before[key], after[key]), key)
        self.assertTrue(all(getattr(m, "lora_layer", None) is None for m in pipe.unet.modules()))

    def test_switching_messages_restores_base_without_accumulation(self):
        pipe = self.pipeline()
        clean = {k: v.clone() for k, v in pipe.unet.state_dict().items()}
        key = next(iter(self.modules)).removeprefix("unet.") + ".weight"
        observed = []
        for bit in [0, 1, 0]:
            pipe.unload_lora_weights()
            pipe.unet.load_state_dict(clean, strict=True)
            pipe.load_lora_weights(convert_message(self.mapper, self.source, [bit] * 48))
            pipe.fuse_lora(lora_scale=1.0)
            pipe.unload_lora_weights()
            observed.append(pipe.unet.state_dict()[key].clone())
        self.assertFalse(torch.equal(observed[0], observed[1]))
        self.assertTrue(torch.equal(observed[0], observed[2]))
        pipe.unet.load_state_dict(clean, strict=True)
        for name, value in pipe.unet.state_dict().items():
            self.assertTrue(torch.equal(value, clean[name]), name)


if __name__ == "__main__":
    unittest.main()
