"""Recover infrastructure, then execute the existing frozen 44-trajectory plan.

Software/assets persist in Drive. Setup failures are resumable. A started model
trajectory is never repeated. Original attempts/receipts stay untouched.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import colab_bootstrap as boot
from .durable_download import fetch
from .environment_check import pins
from .python_runtime import acquire, ASSET_API, ASSET_NAME
from .records import append_json, write_json, total_bytes
from .runtime_cache import wheel_identity

STAGE = "first-run"
RERUN_STAGE = "rerun-02"
RESOURCE_AMENDMENT = "2026-11-08 user authorizes full rerun and old output cleanup: CPU and download reporting-only; model phase <=30 min; entry <=60 min"


def read_started(attempt, manifest):
    """Inspect real per-slot state; never replace it with a stale zero summary."""
    expected = [(t["trajectory_id"], t["slot"]) for t in manifest["trajectories"]]
    used = []
    for path in sorted(Path(attempt).rglob("trajectory_status.json")):
        rows = json.loads(path.read_text())
        if [(r["trajectory_id"], r["slot"]) for r in rows] != expected:
            raise RuntimeError(f"Different trajectory plan in {path}")
        if any(r["status"] not in {"NOT_STARTED", "STARTED", "FAILED", "COMPLETED"} for r in rows):
            raise RuntimeError(f"Unknown saved trajectory state in {path}")
        if any(r["status"] != "NOT_STARTED" for r in rows):
            used.append((path.parent, rows))
    if len(used) > 1:
        raise RuntimeError("Multiple consumed trajectory ledgers; no automatic generation")
    return used[0] if used else None


def history_resources(attempt, new_root):
    """Prior summaries already contain cumulative charges; take their maxima."""
    keys = ["wall_seconds", "gpu_occupancy_seconds", "cpu_seconds", "download_charge_bytes",
            "peak_ram_bytes", "peak_device_used_bytes", "result_bytes"]
    values = {k: 0 for k in keys}
    for path in Path(attempt).rglob("summary.json"):
        if new_root in path.parents:
            continue
        saved = json.loads(path.read_text())
        row = saved.get("cumulative_resources", saved.get("resources", {}))
        for k in keys:
            if isinstance(row.get(k), (int, float)):
                values[k] = max(values[k], row[k])
    return values


def import_paid_wheels(runtime_parent, cache, requirements):
    """Recover retained local wheel files before they disappear with a VM."""
    copied = 0
    expected = pins(requirements)
    log = cache/"imported_paid_wheels.jsonl"
    for group in ["wheels"]:
        destination = cache/group
        destination.mkdir(parents=True, exist_ok=True)
        for source in Path(runtime_parent).glob(f"**/downloads/{group}/*.whl"):
            target = destination/source.name
            if target.exists():
                continue
            name, version = wheel_identity(source)
            if expected.get(name) != version:
                continue
            shutil.copy2(source, target)
            copied += target.stat().st_size
            append_json(log, dict(source=str(source), target=str(target), bytes=target.stat().st_size))
    return copied


class FirstRunSupervisor(boot.Supervisor):
    def __init__(self, *args, cache, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = Path(cache)
        self.model_start = None
        self.cache_scan_time = 0
        self.cache_bytes = 0
        self.environment.update(PIP_RETRIES="2", UV_HTTP_RETRIES="2",
            PIP_NO_CACHE_DIR="1")

    def download_bytes(self):
        # This is retained artifact volume, not packet-level traffic. Per-request
        # network retry receipts remain separate. Copying to local disk adds no charge.
        if time.monotonic()-self.cache_scan_time > 30 and self.model_start is None:
            self.cache_bytes = total_bytes(self.cache)
            self.cache_scan_time = time.monotonic()
        return self.cache_bytes

    def usage(self, child=None):
        row = super().usage(child)
        row["model_phase_seconds"] = 0 if self.model_start is None else time.monotonic()-self.model_start
        row["download_measurement"] = "durable artifact volume; historical charges and transport retry receipts reported separately"
        return row

    def reason(self, row):
        b = self.manifest["budget"]
        for key, limit in [("wall_seconds", 3600-20), ("model_phase_seconds", b["gpu_seconds"]-20),
                           ("peak_ram_bytes", b["ram_bytes"]-512*1024**2),
                           ("peak_device_used_bytes", b["gpu_bytes"]-256*1024**2),
                           ("result_bytes", b["result_bytes"]-16*1024**2)]:
            if row[key] >= limit:
                return f"Resource stop: {key}"
        return None

    def setup(self, lock):
        self.phase = "SOFTWARE_SETUP"
        self.lock = self.root/"environment-lock.txt"
        self.lock.write_text(lock)
        python = self.scratch/"venv/bin/python"
        ready = self.scratch/"environment.ready.json"
        if python.exists() and ready.exists():
            self.command("environment_check", [str(python), "-m", "qualification.environment_check", str(self.lock)])
            self.command("dependency_check", [str(python), "-m", "pip", "check"])
            return str(python)
        archive = self.cache/ASSET_NAME
        if not (self.scratch/"python/python/bin/python3.10").exists():
            if not archive.exists():
                fetch(ASSET_API, archive, self.root/"download_events.jsonl",
                    headers={"Accept": "application/octet-stream", "X-GitHub-Api-Version": "2022-11-28",
                             "User-Agent": "Model-WM-first-run", "Cache-Control": "no-cache"})
            acquire(self.scratch, self.root, 2**63-1, cached_archive=archive)
        interpreter = self.scratch/"python/python/bin/python3.10"
        if not python.exists():
            # CPython's bundled ensurepip creates this environment offline.
            self.command("python_environment", [str(interpreter), "-m", "venv", str(self.scratch/"venv")])
        wheels = self.cache/"wheels"
        wheels.mkdir(exist_ok=True)
        marker = self.cache/"wheel_set_complete.json"
        complete = json.loads(marker.read_text()) if marker.exists() else {}
        if (complete.get("requirements") != lock or not complete.get("wheels") or
                not all((wheels/name).is_file() for name in complete["wheels"])):
            self.command("download_dependencies", [str(python), "-m", "pip", "download", "--retries", "2",
                "--only-binary=:all:", "--find-links", str(wheels), "--dest", str(wheels),
                "--extra-index-url", "https://download.pytorch.org/whl/cu118", "-r", str(self.lock)])
            write_json(marker, dict(requirements=lock, wheels=sorted(f.name for f in wheels.glob("*.whl"))))
        self.command("install_dependencies", [str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheels),
                     "-r", str(self.lock)])
        self.command("environment_check", [str(python), "-m", "qualification.environment_check", str(self.lock)])
        self.command("dependency_check", [str(python), "-m", "pip", "check"])
        self.command("environment_versions", [str(python), "-m", "pip", "freeze"])
        write_json(ready, dict(requirements=lock))
        return str(python)

    def models(self, python):
        self.command("cuda_availability", [python, "-c",
            "import torch; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available()}); assert torch.cuda.is_available(), 'CUDA unavailable in the selected runtime'"])
        self.phase = "MODEL_DOWNLOAD"
        self.command("download_models", [python, "-m", "qualification.durable_download", str(self.root), str(self.cache)])
        self.phase = "MODEL_EXECUTION"
        self.cache_bytes = total_bytes(self.cache)
        self.model_start = time.monotonic()
        self.command("qualification", [python, "-m", "qualification.run", str(self.root), str(self.cache/"assets")])


def run(started_unix=None, started_cpu=None, *, authorized_rerun=False):
    manifest = json.loads((boot.ROOT/"qualification/manifest.json").read_text())
    attempt = boot.DRIVE_PARENT/manifest["run_id"]
    if not (attempt/"manifest.json").exists():
        raise RuntimeError("Original attempt manifest missing; mount the same Google Drive")
    if json.loads((attempt/"manifest.json").read_text()) != manifest:
        raise RuntimeError("Frozen scientific inputs differ")
    stage = attempt/(RERUN_STAGE if authorized_rerun else STAGE)
    # Explicitly authorized second run has its own ledger. Never reset either
    # round or silently create a third run when this notebook is clicked again.
    used = read_started(stage if authorized_rerun else attempt, manifest)
    if used:
        path, rows = used
        result = boot.receipt(path)
        result.update(output=str(path), trajectories_started=sum(r["status"] != "NOT_STARTED" for r in rows),
                      trajectories_completed=sum(r["status"] == "COMPLETED" for r in rows),
                      note="Existing generated trajectories retained; no automatic regeneration")
        return result
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10)
    if gpu.returncode or len(gpu.stdout.strip().splitlines()) != 1:
        raise RuntimeError("Select one GPU runtime first. No model execution has started.")
    stage.mkdir(exist_ok=True)
    active = stage/"active"
    root = stage/f"session-{time.time_ns()}"
    root.mkdir()
    cache = boot.DRIVE_PARENT/"cache/aqualora-pinned-v1"
    cache.mkdir(parents=True, exist_ok=True)
    runtime_parent = boot.WORK_PARENT/manifest["run_id"]
    scratch = runtime_parent/"persistent-runtime"
    scratch.mkdir(parents=True, exist_ok=True)
    write_json(root/"manifest.json", manifest)
    previous = history_resources(attempt, stage)
    write_json(root/"historical_resources.json", previous)
    write_json(root/"resource_amendment.json", dict(reason=RESOURCE_AMENDMENT,
        original_download_limit_bytes=manifest["budget"]["download_bytes"], original_budget_pass_not_claimed=True,
        original_cpu_limit_seconds=manifest["budget"]["cpu_seconds"], cpu_limit_enforced=False,
        setup_and_model_costs_both_retained=True, model_phase_gpu_seconds=1800,
        run_stage=stage.name, authorized_full_rerun=authorized_rerun))
    lock_file = attempt/boot.RECOVERY_NAME/"environment_versions.log"
    lock = lock_file.read_text() if lock_file.exists() else (boot.ROOT/"requirements-colab.txt").read_text()
    required = pins((boot.ROOT/"requirements-colab.txt").read_text())
    if any(pins(lock).get(k) != v for k, v in required.items()):
        raise RuntimeError("Saved dependency versions differ from the fixed inference stack")
    # Only acquire after read-only checks and directory preparation. Every action
    # after this point is inside the finally block that releases the live lock.
    try:
        active.mkdir()
    except FileExistsError:
        raise RuntimeError("A first-run invocation is still active or interrupted; inspect it before starting another")
    supervisor, error = None, None
    try:
        # Cache excludes results and lives outside the 1GiB result folder.
        imported = import_paid_wheels(runtime_parent, cache, lock)
        write_json(root/"cache_entry.json", dict(path=str(cache), imported_paid_bytes=imported,
                                                retained_bytes=total_bytes(cache)))
        supervisor = FirstRunSupervisor(root, scratch, manifest,
            time.time() if started_unix is None else started_unix, cache=cache, account_root=attempt)
        if started_cpu is not None:
            supervisor.cpu_base = started_cpu
        supervisor.phase = "SOFTWARE_SETUP"
        write_json(root/"runtime_entry.json", dict(gpu=gpu.stdout.strip(),
            code_commit=subprocess.check_output(["git", "-C", str(boot.ROOT), "rev-parse", "HEAD"], text=True).strip(),
            resource_amendment=RESOURCE_AMENDMENT))
        with tarfile.open(root/"executed_source.tar.gz", "w:gz") as archive:
            for name in ["qualification", "colab_bootstrap.py", "requirements-colab.txt", "README.md", "AMENDMENT.md"]:
                archive.add(boot.ROOT/name, arcname=name, filter=lambda i: None if "__pycache__" in i.name else i)
        python = supervisor.setup(lock)
        supervisor.models(python)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_json(root/"execution_failure.json", dict(error=error, phase=getattr(supervisor, "phase", "ENTRY")))
    try:
        if supervisor is None:
            raise RuntimeError(error)
        result = supervisor.summarize(error)
        result["execution_phase"] = supervisor.phase
        if error and supervisor.logs:
            last_log = root/(supervisor.logs[-1]["stage"]+".log")
            if last_log.exists():
                result["error_detail"] = last_log.read_text(errors="replace")[-4000:]
        result["resource_amendment"] = RESOURCE_AMENDMENT
        result["historical_resources"] = previous
        result["original_budget_pass_claimed"] = False
        if error and result["trajectories_started"] == 0:
            result["status"] = "SETUP_BLOCKED_RESUMABLE"
        result["setup_can_resume"] = result["trajectories_started"] == 0
        # Session clocks are increments; old summaries were cumulative. Count the
        # persistent cache once, and never bill local cache copies as downloads.
        sessions = [result["resources"]]
        cached_at_entry = json.loads((root/"cache_entry.json").read_text())["retained_bytes"]
        result["new_cache_bytes"] = max(0, total_bytes(cache)-cached_at_entry)
        new_cache_bytes = result["new_cache_bytes"]
        for saved in stage.glob("session-*/summary.json"):
            if saved.parent != root:
                saved_result = json.loads(saved.read_text())
                sessions.append(saved_result.get("resources", {}))
                new_cache_bytes += saved_result.get("new_cache_bytes", 0)
        cumulative = dict(previous)
        for key in ["wall_seconds", "gpu_occupancy_seconds", "cpu_seconds"]:
            cumulative[key] = previous.get(key, 0)+sum(s.get(key, 0) for s in sessions)
        for key in ["peak_ram_bytes", "peak_device_used_bytes"]:
            cumulative[key] = max([previous.get(key, 0)]+[s.get(key, 0) for s in sessions])
        paid = cache/"imported_paid_wheels.jsonl"
        imported = sum(json.loads(line)["bytes"] for line in paid.read_text().splitlines()) if paid.exists() else 0
        if authorized_rerun:
            # The prior round already charged the existing durable cache.
            cumulative["download_charge_bytes"] = previous.get("download_charge_bytes", 0)+new_cache_bytes
        else:
            cumulative["download_charge_bytes"] = previous.get("download_charge_bytes", 0)+max(0, total_bytes(cache)-imported)
        cumulative["result_bytes"] = total_bytes(attempt)
        result["cumulative_resources"] = cumulative
        write_json(root/"summary.json", result)
        write_json(stage/"latest.json", result)
        (root/"report.md").write_text("# AquaLoRA first run\n\n"+json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        return result
    finally:
        active.rmdir()


if __name__ == "__main__":
    run()
