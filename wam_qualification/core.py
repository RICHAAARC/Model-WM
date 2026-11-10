"""Write-ahead accounting and fixed-denominator summaries."""
import json
import math
import os
import time
from pathlib import Path


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name+".tmp")
    with temp.open("w", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def append(path, value):
    with Path(path).open("a", encoding="utf8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        f.flush(); os.fsync(f.fileno())


def effective_limits(manifest, inherited):
    b = manifest["budget"]
    return {key: min(b[key], b["global_"+key]-inherited.get(hist, 0))
            for key, hist in [("gpu_seconds", "gpu_occupancy_seconds"), ("wall_seconds", "wall_seconds"), ("cpu_seconds", "cpu_seconds")]}


class Ledger:
    def __init__(self, root, manifest, precreated=False):
        self.root, self.manifest = Path(root), manifest
        self.path = self.root/"operation_status.json"
        self.rows = [dict(op, status="NOT_STARTED") for op in manifest["operations"]]
        if self.path.exists():
            if not precreated or json.loads(self.path.read_text()) != self.rows:
                raise RuntimeError("Existing operation ledger is read-only; no retries")
        elif precreated:
            raise RuntimeError("Missing supervisor reservation")
        self.by_id = {r["id"]:r for r in self.rows}
        if not precreated:
            save(self.path, self.rows)

    def start(self, op_id):
        row = self.by_id[op_id]
        if row["status"] != "NOT_STARTED":
            raise RuntimeError("Consumed physical operation")
        used = sum(r["kind"] == row["kind"] and r["status"] != "NOT_STARTED" for r in self.rows)
        if used >= self.manifest["physical_limits"][row["kind"]]:
            raise RuntimeError("Physical operation limit")
        row.update(status="STARTED", started_unix=time.time())
        save(self.path, self.rows)

    def end(self, op_id, status="COMPLETED", **kwargs):
        row = self.by_id[op_id]
        if row["status"] != "STARTED":
            raise RuntimeError("No prior reservation")
        row.update(status=status, finished_unix=time.time(), **kwargs)
        save(self.path, self.rows)


def native_gates(rows, gates):
    results = {}
    for source in ["registered", "foreign"]:
        group = [r for r in rows if r["source"] == source]
        complete = len(group) == 32 and len({r["id"] for r in group}) == 32
        psnrs = [math.inf if r.get("psnr_infinite") else r["psnr_db"] for r in group]
        accuracy = sum(r["correct_bits"] for r in group)/(32*32)
        mean_psnr = sum(psnrs)/32 if complete else None
        mean_ssim = sum(r["ssim"] for r in group)/32 if complete else None
        results[source] = dict(denominator=32, observed=len(group), bit_denominator=1024, bit_accuracy=accuracy,
            mean_psnr_db=None if mean_psnr == math.inf else mean_psnr, mean_psnr_infinite=mean_psnr == math.inf,
            mean_ssim=mean_ssim, all_native_at_least20=complete and min(psnrs)>=gates["each_native_psnr_db"],
            passed=bool(complete and accuracy>=gates["native_bit_accuracy"] and mean_psnr>=gates["mean_psnr_db"]
                        and mean_ssim>=gates["mean_ssim"] and min(psnrs)>=gates["each_native_psnr_db"]))
    return results
