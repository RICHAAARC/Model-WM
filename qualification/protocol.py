"""Freeze inputs before any model output exists. Standard-library only."""
import hashlib
import json
import random
from pathlib import Path

SEED = 2026110902
RUN_ID = "aqualora-qualification-20261108-attempt01"
CODE_REV = "62fb5cf186050dd4e3bce0bfe15068872fc57d62"
MODEL_REV = "98688b2a1e762339593ee8fe96ed13762f06b732"
BASE_REV = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

# Created for this DEV qualification on 2026-11-08, before generation.
# These are not a sampled dataset or evidence of unknown training-history exclusion.
PROMPTS = [
    "A documentary photograph of a red enamel kettle and two empty ceramic cups on a wooden kitchen table beside a rain covered window, natural morning light.",
    "A photograph of a narrow stone footbridge over a shallow mountain stream, yellow autumn trees and grey rocks, overcast daylight, no people.",
    "A studio photograph of a single blue running shoe next to a coiled white shoelace on a pale grey surface, soft side lighting.",
    "A photograph inside a small bicycle repair workshop, one green bicycle on a stand and organized tools on a brick wall, daylight.",
    "A photograph of a brown owl resting on a weathered fence post above a grassy field, distant misty hills, early evening.",
    "An architectural photograph of a white circular staircase in a quiet library with wooden bookshelves and warm ceiling lights.",
    "A close photograph of sliced purple cabbage, a whole lemon and a steel kitchen knife on a dark chopping board, overhead view.",
    "A photograph of a small orange sailboat moored beside a wooden pier in a calm lake, pine forest on the far shore, cloudy sky.",
]


def stream(label):
    payload = f"Model-WM/AquaLoRA/qualification/v1|{SEED}|{label}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))


def build_manifest():
    bias = [min(.95, max(.05, stream_rng.betavariate(.7, .7)))
            for stream_rng in [stream("bias")] for _ in range(192)]
    def codebook(label):
        rng = stream(label)
        return [[int(rng.random() < p) for p in bias] for _ in range(16)]
    reg, foreign = codebook("registered"), codebook("foreign")
    cr, cf, noise = stream("coalition"), stream("foreign_coalition"), stream("noise")
    # Whole scene groups stay together. No CAL/TEST content is selected here.
    order = list(range(8))
    stream("source").shuffle(order)
    works, slots = [], []
    probe = [stream_rng.randrange(2) for stream_rng in [stream("probe")] for _ in range(48)]
    for bundle in range(2):
        registered_pair, foreign_pair = sorted(cr.sample(range(16), 2)), sorted(cf.sample(range(16), 2))
        for position in range(4):
            wid = f"dev{bundle:02d}_work{position:02d}"
            work = dict(work_id=wid, bundle=bundle, position=position,
                        source_group=f"authored_scene_{order[4*bundle+position]:02d}",
                        prompt=PROMPTS[order[4*bundle+position]],
                        noise_seed=noise.randrange(2**63),
                        registered_pair=registered_pair, foreign_pair=foreign_pair)
            works.append(work)
            messages = [("base", None, None)]
            for group, pair, book in [("registered", registered_pair, reg), ("foreign", foreign_pair, foreign)]:
                for index, user in enumerate(pair):
                    messages.append((f"{group}_{index}", user, book[user][48*position:48*(position+1)]))
            if not bundle and not position:
                for name, flips in [("probe_k", []), ("probe_j", [0]), ("probe_h", [17]), ("probe_jh", [0, 17])]:
                    bits = probe.copy()
                    for j in flips:
                        bits[j] ^= 1
                    messages.append((name, None, bits))
            for role, user, bits in messages:
                slots.append(dict(slot=len(slots), trajectory_id=f"{wid}_{role}", work_id=wid,
                                  role=role, user=user, bits=bits, **{k: work[k] for k in ["prompt", "noise_seed"]}))
    return dict(
        schema=1, run_id=RUN_ID, purpose="DEV_QUALIFICATION_ONLY", master_seed=SEED,
        unseen_content_assumption="USER_ACCEPTED_2026_11_08",
        checkpoint_training_selection_evaluation_history="UNVERIFIED",
        source_note="Eight newly authored scene groups; unknown prior use is assumed absent, not verified.",
        rng="CPython random.Random, SHA256 domain label to big-endian integer; frozen manifest is authoritative",
        source_revision=CODE_REV, model_revision=MODEL_REV, base_revision=BASE_REV,
        model_repo="georgefen/AquaLoRA-Models", model_folder="ppft_trained",
        base_repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
        generation=dict(width=512, height=512, steps=25, cfg=7.5, sampler="dpms_m",
                        precision="float32", mapper_scale=1.03, lora_scale=1.0, negative_prompt=None, batch_size=1),
        budget=dict(trajectories=44, gpu_seconds=1800, wall_seconds=3600, cpu_seconds=900,
                    gpu_bytes=24*1024**3, ram_bytes=32*1024**3, download_bytes=10*1024**3,
                    result_bytes=1024**3),
        gates=dict(native_bit_accuracy=.95, mean_psnr_db=23., mean_ssim=.75,
                   each_native_psnr_db=20., visual_review_required=True),
        n=16, k=2, q=48, t=4, length=192, biases=bias, registered=reg, foreign=foreign,
        works=works, trajectories=slots,
        full_baseline=dict(trajectories_with_dev_probe=17060, works=3360, original_probe_trajectories=256,
                           gpu_seconds=14400, wall_seconds=21600, cpu_seconds=7200,
                           result_bytes=32*1024**3, reserve_fraction=.2, authorized=False),
    )


def write_manifest(path):
    Path(path).write_text(json.dumps(build_manifest(), ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    import sys
    write_manifest(sys.argv[1])
