"""Write-ahead trajectory reservations and inspectable terminal records."""
import json
import os
import time
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        f.flush()
        os.fsync(f.fileno())


class Ledger:
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.manifest = manifest
        self.rows = [dict(trajectory_id=t["trajectory_id"], slot=t["slot"], status="NOT_STARTED")
                     for t in manifest["trajectories"]]
        self.path = self.root / "trajectory_status.json"
        if self.path.exists():
            raise RuntimeError("Existing trajectory ledger: read-only; no retry or regeneration")
        self.save()

    def save(self):
        write_json(self.path, self.rows)

    def start(self, slot):
        if self.rows[slot]["status"] != "NOT_STARTED":
            raise RuntimeError("A STARTED trajectory cannot be repeated")
        if sum(r["status"] != "NOT_STARTED" for r in self.rows) >= 44:
            raise RuntimeError("44 trajectory budget exhausted")
        self.rows[slot].update(status="STARTED", started_unix=time.time())
        self.save()  # Durable reservation BEFORE conversion, loading, or generation.

    def end(self, slot, status, **details):
        if self.rows[slot]["status"] != "STARTED":
            raise RuntimeError("Terminal transition requires a reservation")
        self.rows[slot].update(status=status, finished_unix=time.time(), **details)
        self.save()


def total_bytes(root):
    return sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())
