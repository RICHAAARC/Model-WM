"""Construct the single preregistered draw before any image is generated."""
import hashlib
import json
import random
from pathlib import Path

RUN_ID = "wam-qualification-20261109-attempt01"
SEED = 2026110902
SOURCE_COMMIT = "2c08af04d037d5667c02f6ddebbda9ff04581c3e"
BASE_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
PROMPTS = [
    "A documentary photograph of a brass telescope on a tripod inside an empty observatory dome, the shutter open to a clear blue afternoon sky.",
    "A photograph of a striped beach umbrella on a deserted sandy shore, turquoise sea and a few white clouds, midday sunlight.",
    "A studio photograph of a transparent glass aquarium containing water plants and small pebbles, plain white background, no fish.",
    "A photograph of a freight locomotive beside a grain silo in open farmland, low evening sunlight and a clear horizon.",
    "A close photograph of a mechanical pocket watch with its case open on dark velvet, visible gears, soft studio lighting.",
    "An architectural photograph of an empty indoor swimming pool with blue tiles and tall rectangular windows, morning light.",
    "A photograph of a cactus garden beside an adobe wall in a dry desert town, hard afternoon shadows.",
    "A still life photograph of three spools of colored sewing thread, metal scissors and folded linen on a neutral cloth.",
    "A photograph of a red fire hydrant beside an empty asphalt street, pale apartment buildings in the background, cloudy daylight.",
    "A studio photograph of a silver trumpet standing next to its black instrument case, dark background and side lighting.",
    "A landscape photograph of layered red sandstone cliffs above a dry canyon, distant ridges and clear dawn sky.",
    "A photograph of an empty greenhouse aisle with trays of young seedlings and condensation on the glass roof.",
    "A close photograph of a polished purple mineral crystal on a small white display plinth, soft even museum lighting.",
    "A photograph of a yellow excavator parked at an inactive construction site with piles of gravel, flat daylight, no workers.",
    "An interior photograph of an empty bowling alley with polished lanes and pins at the far end, warm ceiling lights.",
    "A photograph of a large satellite dish on a grassy hill beneath a grey sky, a narrow service road in the foreground.",
]


def rng(label):
    digest = hashlib.sha256(f"Model-WM/WAM/qualification/v1|{SEED}|{label}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def build_manifest(carry):
    order = list(range(16))
    rng("source_order").shuffle(order)
    bundles, works, operations = [], [], []
    for b in range(4):
        pr = rng(f"bias/{b}")
        bias = [min(.95, max(.05, pr.betavariate(.7, .7))) for _ in range(128)]
        def book(label):
            rr = rng(f"{label}/{b}")
            return [[int(rr.random() < p) for p in bias] for _ in range(16)]
        reg, foreign = book("registered"), book("foreign")
        pair = sorted(rng(f"pair/{b}").sample(range(16), 2))
        fp = sorted(rng(f"foreign_pair/{b}").sample(range(16), 2))
        bundles.append(dict(bundle=b, biases=bias, registered=reg, foreign=foreign,
                            registered_pair=pair, foreign_pair=fp,
                            key=rng(f"key/{b}").getrandbits(256).to_bytes(32, "big").hex()))
        for t in range(4):
            wid = f"qualification{b:02d}_work{t:02d}"
            messages = []
            for group, members, cb in [("registered", pair, reg), ("foreign", fp, foreign)]:
                for j, u in enumerate(members):
                    messages.append(dict(role=f"{group}_{j}", source=group, user=u, bits=cb[u][t*32:(t+1)*32]))
            work = dict(work_id=wid, bundle=b, position=t, source_group=f"wam_authored_{order[4*b+t]:02d}",
                        prompt=PROMPTS[order[4*b+t]], noise_seed=rng(f"noise/{wid}").randrange(2**63), messages=messages)
            works.append(work)
            operations.append(dict(id=f"{wid}/generate", kind="generate"))
            operations.extend(dict(id=f"{wid}/write/{m['role']}", kind="write") for m in messages)
            media = ["null_png", "null_jpeg85"]+[m["role"]+"_png" for m in messages]+[
                f"{g}_mix_{a}" for g in ["registered", "foreign"] for a in ["png", "jpeg85"]]
            work["readouts"] = media
            operations.extend(dict(id=f"{wid}/extract/{name}", kind="extract") for name in media)
    return dict(schema=1, run_id=RUN_ID, master_seed=SEED, authorized="USER_2026_11_09_同意推进_QUALIFICATION_ONLY",
                purpose="DEV_QUALIFICATION_ONLY", n=16, k=2, q=32, t=4, length=128,
                checkpoint_history="UNVERIFIED", unseen_content="USER_ACCEPTED_ASSUMPTION",
                source_commit=SOURCE_COMMIT, checkpoint_url="https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth",
                checkpoint_license="MIT; official SA-1B checkpoint", base_repo="stable-diffusion-v1-5/stable-diffusion-v1-5",
                base_revision=BASE_REVISION, generation=dict(width=512, height=512, steps=25, cfg=7.5,
                scheduler="DPMSolverMultistepScheduler", precision="float32", batch_size=1, negative_prompt=None),
                wam=dict(scaling_w=2.0, scaling_i=1.0, full_image=True, nbits=32, input_normalization="official ImageNet",
                         representation="raw 33x256x256 float32 and 4x4 average pooling", native_decoder="official semihard"),
                jpeg=dict(quality=85, subsampling=2, optimize=False), mixing="uint8 RGB equal mean then numpy.rint",
                budget=dict(gpu_seconds=900, wall_seconds=1800, cpu_seconds=600, ram_bytes=32*1024**3,
                            device_bytes=24*1024**3, result_bytes=32*1024**3, global_gpu_seconds=14400,
                            global_wall_seconds=21600, global_cpu_seconds=7200, download_cap=None),
                inherited_resources=carry, physical_limits=dict(generate=16, write=64, extract=160),
                gates=dict(native_bit_accuracy=.95, mean_psnr_db=23., mean_ssim=.75, each_native_psnr_db=20.),
                full_baseline_authorized=False, training_authorized=False, cal_test_authorized=False,
                bundles=bundles, works=works, operations=operations)


def freeze(path, carry):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())  # A second invocation never redraws.
    manifest = build_manifest(carry)
    with path.open("x", encoding="utf8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest
