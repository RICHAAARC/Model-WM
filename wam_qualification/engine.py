"""Fixed media and paired readouts; no fitting, calibration, selection or retry."""
import json
import math
import os
import resource
import time
from pathlib import Path

from .core import Ledger, append, save, native_gates


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime+r.ru_stime


def quality(original, result):
    import numpy as np
    from skimage.metrics import structural_similarity
    mse = float(np.mean((original.astype(np.float64)-result.astype(np.float64))**2))
    return dict(psnr_db=None if mse == 0 else 10*math.log10(255**2/mse), psnr_infinite=mse == 0,
                ssim=float(structural_similarity(original, result, channel_axis=2, data_range=255,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False, win_size=11, K1=.01, K2=.03)))


def save_array(path, array):
    import numpy as np
    with Path(path).open("xb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush(); os.fsync(f.fileno())


def blind_scores(pooled, codewords, position):
    import numpy as np
    # No coalition identity, clean image, correctness mask, or foreign labels.
    logits = pooled[1:].mean(axis=(1, 2), dtype=np.float64)
    bits = np.asarray(codewords, dtype=np.float64)[:, position*32:(position+1)*32]
    return ((2*bits-1)@logits/32).tolist()


def run(root, manifest, backend, before_work=None, precreated=False):
    import numpy as np
    from PIL import Image, ImageOps, ImageDraw
    root = Path(root)
    ledger = Ledger(root, manifest, precreated=precreated)
    rows, paired, timing = [], [], []
    for name in ("media", "logits", "pooled", "visual_review"):
        (root/name).mkdir()

    def physical(op, fn):
        ledger.start(op)
        t, c = time.monotonic(), cpu()
        try:
            value = fn()
            backend.synchronize()
            ledger.end(op, elapsed_seconds=time.monotonic()-t, cpu_seconds=cpu()-c)
            return value
        except BaseException as exc:
            ledger.end(op, status="FAILED", error=f"{type(exc).__name__}: {exc}")
            raise

    def media(wid, name, rgb):
        path = root/"media"/(wid+"_"+name+(".jpg" if "jpeg85" in name else ".png"))
        opts = manifest["jpeg"] if "jpeg85" in name else {}
        Image.fromarray(rgb).save(path, **opts)
        with Image.open(path) as im:
            result = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
        if result.shape != (512, 512, 3):
            raise RuntimeError("Unexpected media shape")
        return result

    for index, work in enumerate(manifest["works"]):
        if before_work:
            before_work(index, timing)
        t, c = time.monotonic(), cpu()
        wid = work["work_id"]
        bundle = manifest["bundles"][work["bundle"]]
        original = physical(wid+"/generate", lambda: backend.generate(work))
        clean = media(wid, "null_png", original)
        actual = {"null_png":clean, "null_jpeg85":media(wid, "null_jpeg85", clean)}
        truth, qualities = {}, {}
        for msg in work["messages"]:
            role = msg["role"]
            out = physical(wid+"/write/"+role, lambda: backend.write(clean, msg["bits"]))
            actual[role+"_png"] = media(wid, role+"_png", out)
            truth[role+"_png"] = msg
        for group in ("registered", "foreign"):
            mix = np.rint((actual[group+"_0_png"].astype(np.float32)+actual[group+"_1_png"].astype(np.float32))/2).astype(np.uint8)
            for attack in ("png", "jpeg85"):
                name = group+"_mix_"+attack
                actual[name] = media(wid, name, mix)
        for name in work["readouts"]:
            raw, pooled, native = physical(wid+"/extract/"+name, lambda: backend.extract(actual[name]))
            if raw.shape != (33, 256, 256) or pooled.shape != (33, 4, 4) or not np.isfinite(raw).all() or not np.isfinite(pooled).all():
                raise RuntimeError("Continuous evidence schema failed")
            save_array(root/"logits"/(wid+"_"+name+".npy"), raw.astype(np.float32))
            save_array(root/"pooled"/(wid+"_"+name+".npy"), pooled.astype(np.float32))
            scores = blind_scores(pooled, bundle["registered"], work["position"])
            record = dict(id=wid+"/"+name, work_id=wid, bundle=work["bundle"], media=name,
                native=native, scores=scores, score_rule="mean signed raw bit logit; no fitted threshold",
                quality=quality(clean, actual[name]))
            append(root/"readouts.jsonl", record)
            paired.append(record)
            if name in truth:
                msg = truth[name]
                native_row = dict(id=record["id"], source=msg["source"], native_status=native["status"],
                    correct_bits=sum(int(a==b) for a,b in zip(native["bits"], msg["bits"])) if native["bits"] is not None else 0,
                    pooled_semihard_correct_bits=sum(int(a==b) for a,b in zip(native["pooled_semihard_bits"], msg["bits"]))
                        if native.get("pooled_semihard_bits") is not None else None,
                    native_pool_disagreements=native.get("native_pool_disagreements"),
                    **record["quality"])
                rows.append(native_row)
                append(root/"native_quality.jsonl", native_row)
        # Fixed 16 panels, all seven PNG views. No selection by quality or decoding.
        panel = Image.new("RGB", (7*192, 220), "white")
        draw = ImageDraw.Draw(panel)
        names = [n for n in work["readouts"] if n.endswith("_png")]
        for j, name in enumerate(names):
            panel.paste(Image.fromarray(actual[name]).resize((192, 192)), (j*192, 24))
            draw.text((j*192+2, 4), name, fill="black")
        panel.save(root/"visual_review"/(wid+".png"))
        timing.append(dict(work_id=wid, wall_seconds=time.monotonic()-t, cpu_seconds=cpu()-c))
        save(root/"work_timing.json", timing)

    gates = native_gates(rows, manifest["gates"])
    resolved = all(r["native_status"] == "VALID" for r in rows)
    passed = resolved and all(x["passed"] for x in gates.values())
    diagnostics = []
    for bundle in manifest["bundles"]:
        members = bundle["registered_pair"]
        for condition in ("null_png", "null_jpeg85", "registered_mix_png", "registered_mix_jpeg85", "foreign_mix_png", "foreign_mix_jpeg85"):
            evidence = [r for r in paired if r["bundle"]==bundle["bundle"] and r["media"]==condition]
            scores = np.mean([r["scores"] for r in evidence], axis=0)
            # Coalition truth enters only this retrospective diagnostic, after blind scores are saved.
            margin = max(scores[u] for u in members)-max(scores[u] for u in range(16) if u not in members)
            diagnostics.append(dict(bundle=bundle["bundle"], condition=condition, works_observed=len(evidence),
                scores=scores.tolist(), registered_pair=members, retrospective_margin=float(margin),
                interpretation="descriptive only; no rejection threshold or catch-one PASS"))
    save(root/"paired_diagnostics.json", diagnostics)
    result = dict(status="NUMERIC_QUALIFICATION_PASSED_REVIEW_REQUIRED" if passed else "QUALIFICATION_NUMERIC_GATE_FAILED",
        numeric_qualification_pass=passed, qualification_pass=False, scientific_pass=None,
        native=gates, visual_review="UNRESOLVED", full_baseline_authorized=False,
        training_authorized=False, cal_test_authorized=False)
    save(root/"numeric_result.json", result)
    return result
