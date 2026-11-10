"""Pinned official calls. Imports and real weights occur only in the user-run worker."""
import os
import sys
from pathlib import Path


def load_wam(source, checkpoint, device):
    import torch
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise RuntimeError("Official WAM checkpoint absent; never use a random fallback")
    sys.path[:0] = [str(source), str(source/"notebooks")]
    from inference_utils import load_model_from_checkpoint
    cwd = Path.cwd()
    try:
        os.chdir(source)
        model = load_model_from_checkpoint("checkpoints/params.json", str(checkpoint))
    finally:
        os.chdir(cwd)
    if model.attenuation is None:
        raise RuntimeError("Official JND attenuation failed to construct")
    if model.scaling_w != 2.0 or model.scaling_i != 1.0:
        raise RuntimeError("Unexpected official WAM scaling")
    return model.eval().to(device)


def features_and_native(preds):
    import torch
    import torch.nn.functional as F
    from watermark_anything.data.metrics import msg_predict_inference
    if tuple(preds.shape) != (1, 33, 256, 256) or not torch.isfinite(preds).all():
        raise RuntimeError("Unreadable/nonfinite continuous evidence")
    raw = preds.detach().float().cpu()
    pooled = F.adaptive_avg_pool2d(raw, (4, 4))
    masks = raw[:, :1].sigmoid()
    occupied = int((masks > .5).sum())
    native = msg_predict_inference(raw[:, 1:], masks, method="semihard")
    pooled_masks = pooled[:, :1].sigmoid()
    pooled_occupied = int((pooled_masks > .5).sum())
    pooled_native = msg_predict_inference(pooled[:, 1:], pooled_masks, method="semihard")
    # An empty mask makes official means NaN before its Boolean comparison.
    # Preserve it as unresolved, never count the resulting all-zero vector as correct.
    return raw[0].numpy(), pooled[0].numpy(), dict(
        status="VALID" if occupied else "UNRESOLVED_EMPTY_MASK", mask_pixels=occupied,
        bits=native[0].int().tolist() if occupied else None,
        pooled_semihard_bits=pooled_native[0].int().tolist() if pooled_occupied else None,
        pooled_mask_cells=pooled_occupied,
        native_pool_disagreements=int((native!=pooled_native).sum()) if occupied and pooled_occupied else None)


class Backend:
    def __init__(self, manifest, source, checkpoint, base):
        import torch
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        if not torch.cuda.is_available():
            raise RuntimeError("User must select a Colab GPU runtime")
        torch.set_num_threads(1)
        torch.manual_seed(manifest["master_seed"])
        self.torch, self.manifest = torch, manifest
        self.wam = load_wam(source, checkpoint, "cuda")
        self.pipe = StableDiffusionPipeline.from_pretrained(str(base), local_files_only=True,
            torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        from watermark_anything.data.transforms import default_transform, unnormalize_img
        self.transform, self.unnormalize = default_transform, unnormalize_img

    def synchronize(self):
        self.torch.cuda.synchronize()

    def generate(self, work):
        import numpy as np
        g = self.manifest["generation"]
        with self.torch.inference_mode():
            image = self.pipe(work["prompt"], negative_prompt=None, height=g["height"], width=g["width"],
                num_inference_steps=g["steps"], guidance_scale=g["cfg"],
                generator=self.torch.Generator("cuda").manual_seed(work["noise_seed"])).images[0]
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def write(self, rgb, bits):
        from PIL import Image
        torch = self.torch
        with torch.inference_mode():
            x = self.transform(Image.fromarray(rgb)).unsqueeze(0).to("cuda")
            msg = torch.tensor([bits], dtype=torch.float32, device="cuda")
            output = self.unnormalize(self.wam.embed(x, msgs=msg)["imgs_w"])
            if not torch.isfinite(output).all():
                raise RuntimeError("Nonfinite WAM image")
            # torchvision.save_image's published PNG quantization, not mixture rint.
            return output[0].mul(255).add(.5).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    def extract(self, rgb):
        from PIL import Image
        with self.torch.inference_mode():
            x = self.transform(Image.fromarray(rgb)).unsqueeze(0).to("cuda")
            return features_and_native(self.wam.detect(x)["preds"])
