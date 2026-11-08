"""Real single-attempt GPU runner. Invoked only by the Colab supervisor."""
import ast
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models.efficientnet as efficientnet
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file

from .metrics import read_rgb, mix_rgb, quality, native_gate, interaction
from .records import Ledger, write_json, append_json, total_bytes


def official_mapper(source):
    """Use the upstream class verbatim without importing its training-only modules."""
    tree = ast.parse(Path(source).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MapperNet"]
    if len(nodes) != 1:
        raise RuntimeError("Expected one upstream MapperNet definition")
    ns = dict(torch=torch, nn=nn, init=init)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns["MapperNet"](input_size=48, output_size=320)


class SecretDecoder(nn.Module):
    """Upstream architecture. All weights are replaced by strict checkpoint load.

    weights=None avoids downloading unused ImageNet initialization. The architecture,
    bilinear resize and [B,48,2] logits are unchanged from upstream utils/models.py.
    """
    def __init__(self):
        super().__init__()
        self.model = efficientnet.efficientnet_b1(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 96, bias=True)

    def forward(self, x):
        x = F.interpolate(x, size=(512, 512), mode="bilinear")
        return self.model(x).view(-1, 48, 2)


def convert_message(mapper, source, bits):
    # Same branches and algebra as upstream scripts/create_wm_lora.py; cached CPU
    # source weights instead of rereading 0.54 GB for every message. No training.
    # The loader wraps these tensors as new Parameters outside this context.
    # no_grad keeps ordinary tensors; inference tensors would reject that load.
    with torch.no_grad():
        mapped = mapper(torch.tensor([bits], dtype=torch.float32))
        result = {}
        for key, value in source.items():
            if "unet" in key:
                if "attn" in key or "ff" in key:
                    if "up.weight" in key:
                        result[key] = value
                    elif "down.weight" in key:
                        result[key] = torch.diag_embed(mapped)[0] @ value * 1.03
                if "proj_in" in key or "proj_out" in key:
                    if "up.weight" in key:
                        result[key] = value
                    elif "down.weight" in key:
                        result[key] = value * mapped[0][:, None, None, None] * 1.03
            elif "text_encoder" in key:
                pass  # Upstream conversion intentionally omits text-encoder weights.
            else:
                raise ValueError(f"Unknown LoRA key {key}")
        if not result or not all(torch.isfinite(x).all().item() for x in result.values()):
            raise ValueError("Empty or non-finite message LoRA")
        return result


def decode(decoder, image):
    image = image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    array = (np.asarray(image, dtype=np.uint8)/127.5-1.).astype(np.float32)
    x = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.inference_mode():
        z = decoder(x).float().cpu().numpy()[0]
    if z.shape != (48, 2) or not np.isfinite(z).all():
        raise ValueError("Invalid native [48,2] logits")
    d = z[:, 1]-z[:, 0]
    return dict(logits=z.tolist(), difference=d.tolist(), soft=np.tanh(d/2).tolist(),
                hard=(2*(d > 0).astype(int)-1).tolist(), decoded_bits=(d > 0).astype(int).tolist())


class Runner:
    def __init__(self, root, assets):
        self.root, self.assets = Path(root), Path(assets)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        self.ledger = Ledger(root, self.manifest)
        self.media = self.root/"media"
        self.media.mkdir()
        (self.root/"visual_review").mkdir()
        self.stage_costs, self.native_rows, self.pair_rows, self.decoded = [], [], [], {}
        self.budget = self.manifest["budget"]
        self.started = time.monotonic()
        self.work_costs, self.probe_costs = [], []

    def stage(self, name, function, **context):
        wall, cpu = time.monotonic(), time.process_time()
        try:
            return function()
        finally:
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            row = dict(stage=name, wall_seconds=time.monotonic()-wall,
                       cpu_seconds=time.process_time()-cpu, **context)
            self.stage_costs.append(row)
            append_json(self.root/"stage_costs.jsonl", row)
            self.check()

    def check(self):
        if (self.root/"STOP_REQUESTED.json").exists():
            raise RuntimeError("Supervisor requested a budget stop")
        if total_bytes(self.root) > self.budget["result_bytes"]:
            raise RuntimeError("Result storage budget exceeded")
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved() > self.budget["gpu_bytes"]:
            raise RuntimeError("GPU memory budget exceeded")

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU inference fallback")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(2026110902)
        self.mapper = official_mapper(self.assets/"official/utils/models.py").eval()
        self.mapper.load_state_dict(torch.load(self.assets/"aqualora/mapper.pt", map_location="cpu", weights_only=True), strict=True)
        self.source_lora = load_file(str(self.assets/"aqualora/pytorch_lora_weights.safetensors"), device="cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            str(self.assets/"base"), torch_dtype=torch.float32, safety_checker=None,
            feature_extractor=None, local_files_only=True, use_safetensors=True)
        self.scheduler_config = dict(self.pipe.scheduler.config)
        self.clean_unet = {k: v.detach().cpu().clone() for k, v in self.pipe.unet.state_dict().items()}
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.decoder = SecretDecoder()
        self.decoder.load_state_dict(torch.load(self.assets/"aqualora/msgdecoder.pt", map_location="cpu", weights_only=True), strict=True)
        self.decoder.eval().cuda()
        for module in [self.pipe.unet, self.pipe.vae, self.pipe.text_encoder, self.decoder, self.mapper]:
            module.requires_grad_(False)
            if any(p.dtype != torch.float32 for p in module.parameters()):
                raise RuntimeError("Inference precision differs from frozen float32")
        write_json(self.root/"model_interface.json", dict(
            scheduler=self.scheduler_config, precision="float32", tf32=False,
            native_shape=[1, 48, 2], base_restore="strict clean UNet state before each trajectory",
            extractor_initialization="none; full published state_dict strict=True",
            mapper_source="official class extracted with AST; forward unmodified",
            gpu=torch.cuda.get_device_name(), torch=torch.__version__))

    def extract(self, name):
        with Image.open(self.media/name) as image:
            result = decode(self.decoder, image)
        self.decoded[name] = result
        append_json(self.root/"readouts.jsonl", dict(media=name, **result))
        return result

    def trajectory(self, slot):
        index, tid = slot["slot"], slot["trajectory_id"]
        self.check()
        self.ledger.start(index)
        try:
            def switch():
                self.pipe.unload_lora_weights()
                self.pipe.unet.load_state_dict(self.clean_unet, strict=True)
                if slot["bits"] is not None:
                    converted = convert_message(self.mapper, self.source_lora, slot["bits"])
                    self.pipe.load_lora_weights(converted)
                    self.pipe.fuse_lora(lora_scale=1.0)
                    self.pipe.unload_lora_weights()
                    del converted
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler_config)
            self.stage("message_switch", switch, trajectory=tid)
            counts = dict(steps=0, unet_calls=0, branch_evaluations=0)
            def unet_hook(module, args):
                counts["unet_calls"] += 1
                counts["branch_evaluations"] += int(args[0].shape[0])
            def callback(step, timestep, latents):
                counts["steps"] += 1
                if not torch.isfinite(latents).all().item():
                    raise ValueError("Non-finite denoising latents")
                self.check()
            handle = self.pipe.unet.register_forward_pre_hook(unet_hook)
            try:
                def generate():
                    with torch.inference_mode():
                        return self.pipe(
                            prompt=slot["prompt"], negative_prompt=None, height=512, width=512,
                            num_inference_steps=25, guidance_scale=7.5,
                            generator=torch.Generator(device="cuda").manual_seed(slot["noise_seed"]),
                            callback=callback, callback_steps=1).images[0]
                image = self.stage("generation", generate, trajectory=tid)
            finally:
                handle.remove()
            if counts != dict(steps=25, unet_calls=25, branch_evaluations=50):
                raise RuntimeError(f"Unexpected denoising workload: {counts}")
            path = self.media/f"{tid}.png"
            self.stage("media_io", lambda: image.convert("RGB").save(path), trajectory=tid)
            # Decode only the saved and read-back uint8 image.
            readout = self.stage("extraction", lambda: self.extract(path.name), trajectory=tid)
            bits = slot["bits"]
            correct = None if bits is None else sum(a == b for a, b in zip(bits, readout["decoded_bits"]))
            self.ledger.end(index, "COMPLETED", media=path.name, correct_bits=correct, **counts)
        except BaseException as exc:
            self.ledger.end(index, "FAILED", error=f"{type(exc).__name__}: {exc}")
            raise

    def work_media(self, work):
        wid = work["work_id"]
        base = read_rgb(self.media/f"{wid}_base.png")
        pairs = {}
        for source in ["registered", "foreign"]:
            pair = [read_rgb(self.media/f"{wid}_{source}_{i}.png") for i in range(2)]
            pairs[source] = pair
            for i, img in enumerate(pair):
                tid = f"{wid}_{source}_{i}"
                slot = next(x for x in self.manifest["trajectories"] if x["trajectory_id"] == tid)
                bits = self.decoded[f"{tid}.png"]["decoded_bits"]
                row = dict(trajectory_id=tid, source=source, work_id=wid,
                           correct_bits=sum(a == b for a, b in zip(bits, slot["bits"])), **quality(base, img))
                self.native_rows.append(row)
                append_json(self.root/"native_quality.jsonl", row)
            native_pair = dict(work_id=wid, comparison=f"{source}_copy_pair", **quality(*pair))
            self.pair_rows.append(native_pair)
            append_json(self.root/"pair_quality.jsonl", native_pair)
            mixed_name = f"{wid}_{source}_mix.png"
            Image.fromarray(mix_rgb(*pair)).save(self.media/mixed_name)
            mixed = read_rgb(self.media/mixed_name)
            row = dict(work_id=wid, comparison=f"{source}_mix_vs_base", **quality(base, mixed))
            self.pair_rows.append(row)
            append_json(self.root/"pair_quality.jsonl", row)
            self.extract(mixed_name)
            jpg = f"{wid}_{source}_mix_jpeg85.jpg"
            Image.fromarray(mixed).save(self.media/jpg, quality=85, subsampling=0, optimize=False, progressive=False)
            self.extract(jpg)
            row = dict(work_id=wid, comparison=f"{source}_jpeg85_mix_vs_png_base", **quality(base, read_rgb(self.media/jpg)))
            self.pair_rows.append(row)
            append_json(self.root/"pair_quality.jsonl", row)
            # Decoder nonlinearity and common-bit loss are descriptive only here.
            x0 = self.decoded[f"{wid}_{source}_0.png"]
            x1 = self.decoded[f"{wid}_{source}_1.png"]
            slots = [next(x for x in self.manifest["trajectories"] if x["trajectory_id"] == f"{wid}_{source}_{i}") for i in range(2)]
            common = [j for j in range(48) if slots[0]["bits"][j] == slots[1]["bits"][j]]
            for medium in [mixed_name, jpg]:
                mix = self.decoded[medium]
                native_correct = [(.5*(x0["decoded_bits"][j] == slots[0]["bits"][j]) +
                                  .5*(x1["decoded_bits"][j] == slots[0]["bits"][j])) for j in common]
                mix_correct = [mix["decoded_bits"][j] == slots[0]["bits"][j] for j in common]
                delta = np.asarray(mix["logits"])-.5*(np.asarray(x0["logits"])+np.asarray(x1["logits"]))
                append_json(self.root/"mix_diagnostics.jsonl", dict(work_id=wid, source=source, media=medium,
                    common_bit_denominator=len(common), common_positions=common,
                    native_common_accuracy=None if not common else float(np.mean(native_correct)),
                    mixed_common_accuracy=None if not common else float(np.mean(mix_correct)),
                    logit_nonlinearity_rms=float(np.sqrt(np.mean(delta**2))), claim="DESCRIPTIVE_DEV_ONLY"))
        for i, a in enumerate(pairs["registered"]):
            for j, b in enumerate(pairs["foreign"]):
                append_json(self.root/"pair_quality.jsonl", dict(work_id=wid,
                    comparison=f"registered_{i}_vs_foreign_{j}", **quality(a, b)))
        base_jpg = f"{wid}_base_jpeg85.jpg"
        Image.fromarray(base).save(self.media/base_jpg, quality=85, subsampling=0, optimize=False, progressive=False)
        self.extract(base_jpg)
        # Anonymous, deterministic panel ordering; all groups must be reviewed.
        import hashlib
        label = hashlib.sha256(("visual|"+wid).encode()).hexdigest()[:10]
        names = [f"{wid}_base.png", *[f"{wid}_{s}_{i}.png" for s in pairs for i in range(2)],
                 *[f"{wid}_{s}_mix.png" for s in pairs]]
        order = sorted(names[1:], key=lambda n: hashlib.sha256(("panel|"+n).encode()).digest())
        sheet = Image.new("RGB", (4*256, 2*280), "white")
        draw = ImageDraw.Draw(sheet)
        for i, name in enumerate([names[0]]+order):
            with Image.open(self.media/name) as img:
                sheet.paste(img.resize((256, 256), Image.Resampling.LANCZOS), ((i%4)*256, (i//4)*280))
            draw.text(((i%4)*256+4, (i//4)*280+258), "Reference" if i == 0 else f"Panel {i}", fill="black")
        sheet.save(self.root/"visual_review"/f"{label}.png")
        append_json(self.root/"visual_mapping.jsonl", dict(label=label, work_id=wid, panels=[names[0]]+order))

    def finish(self):
        gates = native_gate(self.native_rows, self.manifest["gates"])
        wid = self.manifest["works"][0]["work_id"]
        names = ["probe_k", "probe_j", "probe_h", "probe_jh"]
        imgs = {n: read_rgb(self.media/f"{wid}_{n}.png") for n in names}
        zs = {n: np.asarray(self.decoded[f"{wid}_{n}.png"]["logits"]) for n in names}
        probe = interaction(imgs, zs)
        probe["target_bits"] = [dict(role=n, targets=[0, 17],
            expected=[next(t for t in self.manifest["trajectories"] if t["role"] == n)["bits"][j] for j in [0, 17]],
            observed=[self.decoded[f"{wid}_{n}.png"]["decoded_bits"][j] for j in [0, 17]]) for n in names]
        write_json(self.root/"probe_diagnostic.json", probe)
        # Extrapolate observed worst work and probe costs; this cannot certify
        # unimplemented full-baseline scoring, calibration or future data costs.
        cost = {}
        for key in ["wall_seconds", "cpu_seconds"]:
            cost[key] = dict(max_work=max(x[key] for x in self.work_costs),
                             max_probe=max(x[key] for x in self.probe_costs))
            cost[key]["remaining_observed_components"] = 3352*cost[key]["max_work"]+256*cost[key]["max_probe"]
        write_json(self.root/"throughput_projection.json", dict(
            full_trajectories=17060, completed_prefix=44, observed_costs=cost,
            unmeasured_full_baseline_overheads="CAL scoring, final trace statistics and future data preparation",
            reserve_fraction=.2, full_baseline_authorized=False))
        labels = sorted(p.stem for p in (self.root/"visual_review").glob("*.png"))
        write_json(self.root/"visual_verdicts.json", [dict(label=l, same_subject="UNREVIEWED",
                   same_layout="UNREVIEWED", no_severe_ghosting="UNREVIEWED") for l in labels])
        complete = all(x["status"] == "COMPLETED" for x in self.ledger.rows)
        result = dict(status="NUMERIC_COMPLETE_VISUAL_UNRESOLVED" if all(x["passed"] for x in gates.values()) else "QUALIFICATION_NUMERIC_GATE_FAILED",
                      complete_44=complete, trajectories_started=sum(x["status"] != "NOT_STARTED" for x in self.ledger.rows),
                      trajectories_completed=sum(x["status"] == "COMPLETED" for x in self.ledger.rows),
                      native=gates, visual_review="UNRESOLVED", qualification_pass=False,
                      full_baseline_authorized=False, scientific_pass=None,
                      checkpoint_history="UNVERIFIED", unseen_content="USER_ACCEPTED_ASSUMPTION",
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      claim="No catch-one/FWER/anti-collusion or novelty inference from qualification")
        write_json(self.root/"numeric_result.json", result)
        return result

    def run(self):
        self.stage("model_loading", self.load)
        for work in self.manifest["works"]:
            wall, cpu = time.monotonic(), time.process_time()
            probes_wall = probes_cpu = 0.
            for slot in [x for x in self.manifest["trajectories"] if x["work_id"] == work["work_id"]]:
                start, cstart = time.monotonic(), time.process_time()
                self.trajectory(slot)
                if slot["role"].startswith("probe_"):
                    row = dict(wall_seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cstart)
                    self.probe_costs.append(row)
                    probes_wall += row["wall_seconds"]
                    probes_cpu += row["cpu_seconds"]
            self.stage("work_media_quality_and_extract", lambda: self.work_media(work), work=work["work_id"])
            self.work_costs.append(dict(wall_seconds=time.monotonic()-wall-probes_wall,
                                       cpu_seconds=time.process_time()-cpu-probes_cpu))
            write_json(self.root/"work_costs.json", self.work_costs)
            print(f"Saved {work['work_id']}: {sum(r['status']=='COMPLETED' for r in self.ledger.rows)}/44 trajectories", flush=True)
        self.finish()


def main():
    root, assets = sys.argv[1:3]
    try:
        Runner(root, assets).run()
    except BaseException as exc:
        write_json(Path(root)/"run_failure.json", dict(type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
