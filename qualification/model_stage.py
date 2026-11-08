"""One model-stage continuation of attempt01, authorized for user-run Colab only."""
import json
import math
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import colab_bootstrap as boot
from .environment_check import pins
from .python_runtime import ASSET_NAME, ASSET_API
from .records import write_json
from .runtime_cache import charged_bytes, paid_links

STAGE_NAME = "model-stage-01"
MINIMUM_READY = dict(wall_seconds=192.58095860481262, gpu_occupancy_seconds=192.58095860481262,
    cpu_seconds=129.291007, peak_ram_bytes=219705344, peak_device_used_bytes=0,
    download_charge_bytes=3465446284, result_bytes=302997)
ENV_STAGES = ["download_uv", "install_uv", "download_python", "python_environment",
              "download_dependencies", "install_dependencies", "dependency_check", "environment_versions"]


def inputs(attempt, manifest):
    original = boot.recovery_inputs(attempt, manifest)
    ready = attempt/boot.RECOVERY_NAME
    if any(p.is_dir() and p.name != boot.RECOVERY_NAME for p in attempt.iterdir()):
        raise RuntimeError("Additional attempt history exists; cumulative costs unresolved")
    if any((p/"trajectory_status.json").exists() for p in (attempt, ready)):
        raise RuntimeError("Existing prior trajectory ledger; no replacement or reset allowed")
    if any((ready/name).exists() for name in ("numeric_result.json", "run_failure.json", "readouts.jsonl", "media")):
        raise RuntimeError("Unexpected model evidence in environment recovery; source unresolved")
    if json.loads((ready/"manifest.json").read_text()) != manifest:
        raise RuntimeError("Recovery manifest differs from frozen inputs")
    previous = json.loads((ready/"summary.json").read_text())
    if (previous.get("status") != "ENVIRONMENT_READY" or previous.get("error") is not None or
        previous.get("resources_final") is not True or previous.get("environment_only") is not True or
        previous.get("trajectories_started") != 0 or previous.get("trajectories_completed") != 0 or
        previous.get("trajectory_denominator") != 44 or
        Path(previous.get("output", "")).name != boot.RECOVERY_NAME or
        Path(previous.get("attempt_output", "")).name != manifest["run_id"]):
        raise RuntimeError("A finalized zero-trajectory ENVIRONMENT_READY receipt is required")
    if json.loads((ready/"carry_forward.json").read_text()) != original:
        raise RuntimeError("Recovery receipt does not connect to the retained original failure")
    for key, observed in MINIMUM_READY.items():
        value = previous["resources"].get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
            value < max(observed, original["resources"][key])):
            raise RuntimeError(f"Invalid or regressed latest resource receipt: {key}")
    stages = json.loads((ready/"setup_stages.json").read_text())
    if [(s["stage"], s["returncode"]) for s in stages] != [(s, 0) for s in ENV_STAGES]:
        raise RuntimeError("Successful recovery stage history missing")
    downloads = [json.loads(line) for line in (ready/"python_download.jsonl").read_text().splitlines()]
    if not downloads or not (downloads[-1].get("status") == "COMPLETED" and
        downloads[-1].get("asset_api") == ASSET_API and downloads[-1].get("version") == "Python 3.10.14"):
        raise RuntimeError("Official Python acquisition history unresolved")
    lock = (ready/"environment_versions.log").read_text()
    recorded = pins(lock)
    required = pins((boot.ROOT/"requirements-colab.txt").read_text())
    if any(recorded.get(name) != version for name, version in required.items()):
        raise RuntimeError("Recovery dependency versions differ from frozen requirements")
    return previous, original, lock


def saved_stage(root, manifest):
    """A reserved, failed or interrupted stage can only be read, never restarted."""
    result = boot.receipt(root)
    ledger = root/"trajectory_status.json"
    if ledger.exists():
        rows = json.loads(ledger.read_text())
        if ([(r["trajectory_id"], r["slot"]) for r in rows] !=
            [(r["trajectory_id"], r["slot"]) for r in manifest["trajectories"]] or
            any(r["status"] not in {"NOT_STARTED", "STARTED", "FAILED", "COMPLETED"} for r in rows)):
            raise RuntimeError("Saved trajectory ledger unresolved; no restart allowed")
        result.update(trajectories_started=sum(r["status"] != "NOT_STARTED" for r in rows),
                      trajectories_completed=sum(r["status"] == "COMPLETED" for r in rows))
    result["trajectory_denominator"] = 44
    if result.get("resources_final") is not True:
        result.update(status="EXISTING_MODEL_STAGE_READ_ONLY", resources_final=False,
                      note="Interrupted or unfinalized stage retained; costs unresolved; no retry.")
        if (root/"resources.json").exists():
            result["resources"] = json.loads((root/"resources.json").read_text())
    return result


class ModelSupervisor(boot.Supervisor):
    def __init__(self, *args, **kwargs):
        self.prepaid = {}
        super().__init__(*args, **kwargs)

    def download_bytes(self):
        return charged_bytes(self.scratch/"downloads", self.prepaid)

    def budget_context(self):
        write_json(self.root/"download_budget.json", dict(carried_download_bytes=self.carry["download_charge_bytes"],
                                                         prepaid=self.prepaid))

    def plan_environment(self, old, original, lock):
        self.old = old
        self.lock = self.root/"environment-lock.txt"
        self.lock.write_text(lock)
        self.existing_python = old/"venv/bin/python"
        reuse = self.existing_python.exists()
        if reuse:
            # A failed version/provenance check stops; it does not trigger a reinstall.
            self.command("reuse_environment_check", [str(self.existing_python), "-m", "qualification.environment_check", str(self.lock)])
            self.command("reuse_dependency_check", [str(self.existing_python), "-m", "pip", "check"])
        else:
            for subdir, expected in (("bootstrap", {"uv": "0.9.5"}), ("wheels", pins(lock))):
                destination = self.scratch/"downloads"/subdir
                destination.mkdir()
                self.prepaid.update(paid_links(old/"downloads"/subdir, destination, expected))
        self.cached_archive = old/"downloads"/ASSET_NAME
        retained = sum(r["bytes"] for r in self.prepaid.values())
        if self.cached_archive.exists():
            retained += self.cached_archive.stat().st_size
        historical = self.carry["download_charge_bytes"]-original["resources"]["download_charge_bytes"]
        if retained > historical:
            raise RuntimeError("Retained software cache exceeds its paid history; source unresolved")
        # Historical software payload is a conservative planning estimate; actual
        # new bytes and all elapsed/CPU time are still metered during each command.
        estimated = 0 if reuse else max(0, historical-retained)
        write_json(self.root/"environment_plan.json", dict(mode="REUSE_READY_ENVIRONMENT" if reuse else "REBUILD_MISSING_ENVIRONMENT",
            previous_runtime=str(old), estimated_new_software_bytes=estimated, reused_paid_wheels=self.prepaid,
            software_estimate_basis="previous observed software charge minus retained paid wheel/archive bytes"))
        self.budget_context()
        return reuse

    def rebuild_environment(self):
        bootstrap = self.scratch/"downloads/bootstrap"
        old_uv = self.old/"tools/bin/uv"
        if old_uv.exists():
            uv = str(old_uv)
            self.command("reuse_uv_version", [uv, "--version"])
            if (self.root/"reuse_uv_version.log").read_text().strip() != "uv 0.9.5":
                raise RuntimeError("Cached uv version differs")
        else:
            self.command("download_uv", [sys.executable, "-m", "pip", "download", "--no-cache-dir", "--retries", "0",
                "--only-binary=:all:", "--no-deps", "--find-links", str(bootstrap), "--dest", str(bootstrap), "uv==0.9.5"])
            self.command("install_uv", [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(bootstrap),
                "--target", str(self.scratch/"tools"), "uv==0.9.5"])
            uv = str(self.scratch/"tools/bin/uv")
        interpreter = self.old/"python/python/bin/python3.10"
        if not interpreter.exists():
            allowance = self.manifest["budget"]["download_bytes"]-512*1024**2-self.carry["download_charge_bytes"]
            args = [sys.executable, "-m", "qualification.python_runtime", str(self.scratch), str(self.root), str(allowance)]
            if self.cached_archive.exists():
                args.append(str(self.cached_archive))
            self.command("restore_python", args)
            interpreter = self.scratch/"python/python/bin/python3.10"
        venv = self.scratch/"venv"
        self.command("python_environment", [uv, "venv", "--python", str(interpreter), "--no-python-downloads", "--seed", str(venv)])
        python = str(venv/"bin/python")
        wheels = self.scratch/"downloads/wheels"
        self.command("download_dependencies", [python, "-m", "pip", "download", "--no-cache-dir", "--retries", "0",
            "--only-binary=:all:", "--find-links", str(wheels), "--dest", str(wheels),
            "--extra-index-url", "https://download.pytorch.org/whl/cu118", "-r", str(self.lock)])
        self.command("install_dependencies", [python, "-m", "pip", "install", "--no-index", "--find-links", str(wheels), "-r", str(self.lock)])
        self.command("restored_environment_check", [python, "-m", "qualification.environment_check", str(self.lock)])
        self.command("restored_dependency_check", [python, "-m", "pip", "check"])
        return python

    def models(self, old, original, lock):
        reuse = self.plan_environment(old, original, lock)
        self.command("download_preflight", [sys.executable, "-m", "qualification.prepare", str(self.root), str(self.scratch/"downloads"), "--preflight"])
        python = str(self.existing_python) if reuse else self.rebuild_environment()
        self.budget_context()
        self.command("download_models", [python, "-m", "qualification.prepare", str(self.root), str(self.scratch/"downloads")])
        self.command("qualification", [python, "-m", "qualification.run", str(self.root), str(self.scratch/"downloads/assets")])


def continue_attempt(started_unix=None, started_cpu=None):
    started_unix = time.time() if started_unix is None else started_unix
    manifest = json.loads((boot.ROOT/"qualification/manifest.json").read_text())
    attempt = boot.DRIVE_PARENT/manifest["run_id"]
    root = attempt/STAGE_NAME
    if root.exists():
        return saved_stage(root, manifest)
    previous, original, lock = inputs(attempt, manifest)
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    if query.returncode or len(query.stdout.strip().splitlines()) != 1:
        raise RuntimeError("Select one GPU runtime; model stage has not been reserved")
    root.mkdir()  # Sole model-stage reservation, before setup, downloads or loading.
    initial = dict(status="MODEL_STAGE_RESERVED", output=str(root), attempt_output=str(attempt),
        resources=previous["resources"], resources_final=False, trajectories_started=0, trajectories_completed=0,
        trajectory_denominator=44, qualification_pass=False, scientific_pass=None, full_baseline_authorized=False)
    write_json(root/"summary.json", initial)
    write_json(root/"carry_forward.json", previous)
    write_json(root/"manifest.json", manifest)
    supervisor, error = None, None
    try:
        scratch = boot.WORK_PARENT/manifest["run_id"]/STAGE_NAME
        scratch.mkdir(parents=True)
        supervisor = ModelSupervisor(root, scratch, manifest, started_unix, carry=previous["resources"], account_root=attempt)
        if started_cpu is not None:
            supervisor.cpu_base = started_cpu
        write_json(root/"runtime_entry.json", dict(started_unix=started_unix, gpu=query.stdout.strip(),
            parent_attempt=manifest["run_id"], predecessor=boot.RECOVERY_NAME,
            code_commit=subprocess.check_output(["git", "-C", str(boot.ROOT), "rev-parse", "HEAD"], text=True).strip()))
        with tarfile.open(root/"executed_source.tar.gz", "w:gz") as archive:
            for name in ["qualification", "colab_bootstrap.py", "requirements-colab.txt", "README.md", "AMENDMENT.md"]:
                archive.add(boot.ROOT/name, arcname=name, filter=lambda info: None if "__pycache__" in info.name else info)
        old = boot.WORK_PARENT/manifest["run_id"]/boot.RECOVERY_NAME
        supervisor.models(old, original, lock)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_json(root/"execution_failure.json", dict(error=error))
    try:
        if supervisor is None:
            raise RuntimeError("Resource monitoring unavailable; costs unresolved")
        error = error or supervisor.reason(supervisor.usage())
        result = supervisor.summarize(error)
        result.update(attempt_output=str(attempt), resources_final=True, predecessor=boot.RECOVERY_NAME)
        write_json(root/"summary.json", result)
        return result
    except BaseException as exc:
        write_json(root/"summary.json", dict(initial, status="FINALIZATION_INCOMPLETE", error=error,
            finalization_error=f"{type(exc).__name__}: {exc}"))
        return saved_stage(root, manifest)
