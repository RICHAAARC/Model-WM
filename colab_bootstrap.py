"""Thin Colab supervisor; all results are written directly under mounted Drive."""
import json
import hashlib
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from qualification.records import write_json, total_bytes

ROOT = Path(__file__).resolve().parent
DRIVE_PARENT = Path("/content/drive/MyDrive/Model-WM")
WORK_PARENT = Path("/content/model-wm-runtime")
SETUP_DOWNLOAD_RESERVE = 256*1024**2
RECOVERY_NAME = "environment-recovery-01"
# Observed attempt01 receipt, not a fresh budget. Never round the carried values.
MINIMUM_CARRY = dict(wall_seconds=24.21792459487915, gpu_occupancy_seconds=24.21792459487915,
    cpu_seconds=2.950728999999999, peak_ram_bytes=174944256, peak_device_used_bytes=0,
    download_charge_bytes=289791730, result_bytes=130247)


def receipt(root):
    summary = root/"summary.json"
    if summary.exists():
        return json.loads(summary.read_text())
    path = root/"trajectory_status.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    return dict(status="EXISTING_ATTEMPT_READ_ONLY", output=str(root),
                trajectories_started=sum(x["status"] != "NOT_STARTED" for x in rows),
                trajectories_completed=sum(x["status"] == "COMPLETED" for x in rows),
                qualification_pass=False, scientific_pass=None,
                note="Previous output retained. No automatic retry, reset or generation.")


def cpu_usage():
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime


class Supervisor:
    def __init__(self, root, scratch, manifest, started_unix, *, carry=None, account_root=None, monitor_gpu=True):
        import psutil  # Included in standard Colab; validated before the attempt.
        self.psutil = psutil
        self.root, self.scratch, self.manifest = root, scratch, manifest
        self.started = started_unix
        self.cpu_base = cpu_usage()
        self.carry = carry or {}
        self.account_root = account_root or root
        self.monitor_gpu = monitor_gpu
        self.max_ram = self.max_gpu = self.max_download = 0
        self.logs = []
        self.environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                                PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                                HF_HUB_DISABLE_TELEMETRY="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                                PIP_RETRIES="0", UV_HTTP_RETRIES="0", UV_PYTHON_DOWNLOADS="never",
                                TMPDIR=str(scratch/"downloads/tmp"), UV_CACHE_DIR=str(scratch/"uv-cache"),
                                UV_PYTHON_INSTALL_DIR=str(scratch/"python"))
        Path(self.environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        if carry is not None:
            self.environment["UV_CACHE_DIR"] = str(scratch/"downloads/uv-cache")
        if not monitor_gpu:
            self.environment["CUDA_VISIBLE_DEVICES"] = ""

    def usage(self, child=None):
        cpu = cpu_usage()-self.cpu_base
        ram = self.psutil.Process().memory_info().rss
        if child is not None:
            try:
                procs = [self.psutil.Process(child.pid)]+self.psutil.Process(child.pid).children(recursive=True)
            except self.psutil.NoSuchProcess:
                procs = []
            for proc in procs:
                try:
                    c = proc.cpu_times()
                    cpu += c.user+c.system+c.children_user+c.children_system
                    ram += proc.memory_info().rss
                except self.psutil.NoSuchProcess:
                    pass
        self.max_ram = max(self.max_ram, ram)
        # Device memory usage includes context, libraries and CUDA allocator reserve.
        gpu = 0
        if self.monitor_gpu:
            query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10)
            if query.returncode:
                raise RuntimeError("GPU monitor unavailable during bounded attempt")
            gpu = sum(float(x)*1024**2 for x in query.stdout.splitlines() if x.strip())
        self.max_gpu = max(self.max_gpu, gpu)
        downloaded = self.download_bytes()+(0 if self.carry else SETUP_DOWNLOAD_RESERVE)
        self.max_download = max(self.max_download, downloaded)
        elapsed = time.time()-self.started
        return dict(wall_seconds=self.carry.get("wall_seconds", 0)+elapsed,
                    gpu_occupancy_seconds=self.carry.get("gpu_occupancy_seconds", 0)+elapsed,
                    cpu_seconds=self.carry.get("cpu_seconds", 0)+cpu,
                    peak_ram_bytes=max(self.carry.get("peak_ram_bytes", 0), self.max_ram),
                    peak_device_used_bytes=max(self.carry.get("peak_device_used_bytes", 0), self.max_gpu),
                    download_charge_bytes=self.carry.get("download_charge_bytes", 0)+self.max_download,
                    result_bytes=max(self.carry.get("result_bytes", 0), total_bytes(self.account_root)),
                    download_measurement="retained download payload high-water plus 256 MiB setup/metadata reserve; not packet bytes",
                    gpu_clock="conservative entire notebook entry elapsed time, including setup/download/CPU work")

    def download_bytes(self):
        return total_bytes(self.scratch/"downloads")

    def reason(self, stats):
        b = self.manifest["budget"]
        # Reserve margin for polling, cleanup and durable receipts.
        limits = [("wall_seconds", b["wall_seconds"]-20), ("gpu_occupancy_seconds", b["gpu_seconds"]-20),
                  ("cpu_seconds", b["cpu_seconds"]-20), ("peak_ram_bytes", b["ram_bytes"]-512*1024**2),
                  ("peak_device_used_bytes", b["gpu_bytes"]-256*1024**2),
                  ("download_charge_bytes", b["download_bytes"]-512*1024**2),
                  ("result_bytes", b["result_bytes"]-16*1024**2)]
        return next((f"Budget stopping margin reached: {key}" for key, limit in limits if stats[key] >= limit), None)

    def command(self, name, args):
        stats = self.usage()
        reason = self.reason(stats)
        if reason:
            write_json(self.root/"STOP_REQUESTED.json", dict(reason=reason, **stats))
            raise RuntimeError(reason)
        before = time.monotonic()
        log = self.root/f"{name}.log"
        print(f"{name} ...", flush=True)
        with log.open("w") as stream:
            child = subprocess.Popen(args, cwd=ROOT, env=self.environment,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while child.poll() is None:
                    stats = self.usage(child)
                    write_json(self.root/"resources.json", dict(stage=name, **stats))
                    reason = self.reason(stats)
                    if reason:
                        write_json(self.root/"STOP_REQUESTED.json", dict(reason=reason, **stats))
                        raise RuntimeError(reason)
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f"{name} exited {child.returncode}; see {log.name}")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                self.logs.append(dict(stage=name, wall_seconds=time.monotonic()-before, returncode=child.returncode))
                write_json(self.root/"setup_stages.json", self.logs)
        stats = self.usage()
        write_json(self.root/"resources.json", dict(stage=name, **stats))
        reason = self.reason(stats)
        if reason:
            raise RuntimeError(reason)

    def environment_only(self):
        tools_dir = self.scratch/"tools"
        uv_wheels = self.scratch/"downloads/bootstrap"
        uv_wheels.mkdir()
        self.command("download_uv", [sys.executable, "-m", "pip", "download", "--no-cache-dir", "--retries", "0",
                                     "--only-binary=:all:", "--no-deps", "--dest", str(uv_wheels), "uv==0.9.5"])
        self.command("install_uv", [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(uv_wheels),
                                    "--target", str(tools_dir), "uv==0.9.5"])
        uv = str(tools_dir/"bin/uv")
        venv = self.scratch/"venv"
        allowance = self.manifest["budget"]["download_bytes"]-512*1024**2-self.carry.get("download_charge_bytes", SETUP_DOWNLOAD_RESERVE)
        self.command("download_python", [sys.executable, "-m", "qualification.python_runtime",
                                        str(self.scratch), str(self.root), str(allowance)])
        interpreter = self.scratch/"python/python/bin/python3.10"
        self.command("python_environment", [uv, "venv", "--python", str(interpreter),
                                            "--no-python-downloads", "--seed", str(venv)])
        python = str(venv/"bin/python")
        wheels = self.scratch/"downloads/wheels"
        wheels.mkdir()
        self.command("download_dependencies", [python, "-m", "pip", "download", "--no-cache-dir", "--retries", "0",
                     "--only-binary=:all:", "--dest", str(wheels),
                     "--extra-index-url", "https://download.pytorch.org/whl/cu118", "-r", str(ROOT/"requirements-colab.txt")])
        self.command("install_dependencies", [python, "-m", "pip", "install", "--no-index", "--find-links", str(wheels),
                     "-r", str(ROOT/"requirements-colab.txt")])
        self.command("dependency_check", [python, "-m", "pip", "check"])
        self.command("environment_versions", [python, "-m", "pip", "freeze"])
        return python

    def execute(self):
        python = self.environment_only()
        self.command("download_models", [python, "-m", "qualification.prepare", str(self.root), str(self.scratch/"downloads")])
        self.command("qualification", [python, "-m", "qualification.run", str(self.root), str(self.scratch/"downloads/assets")])

    def summarize(self, error=None):
        result_path = self.root/"numeric_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(
            status="STOPPED_BEFORE_NUMERIC_COMPLETION", qualification_pass=False, scientific_pass=None)
        rows = json.loads((self.root/"trajectory_status.json").read_text()) if (self.root/"trajectory_status.json").exists() else []
        stats = self.usage()
        result.update(output=str(self.root), resources=stats, error=error,
                      trajectories_started=sum(x["status"] != "NOT_STARTED" for x in rows),
                      trajectories_completed=sum(x["status"] == "COMPLETED" for x in rows),
                      trajectory_denominator=44, full_baseline_authorized=False,
                      checkpoint_history="UNVERIFIED", unseen_content="USER_ACCEPTED_ASSUMPTION")
        if error:
            result["status"] = "STOPPED_INCOMPLETE"
            result["qualification_pass"] = False
        projection = self.root/"throughput_projection.json"
        if projection.exists():
            p = json.loads(projection.read_text())
            wall = stats["wall_seconds"]+p["observed_costs"]["wall_seconds"]["remaining_observed_components"]
            cpu = stats["cpu_seconds"]+p["observed_costs"]["cpu_seconds"]["remaining_observed_components"]
            # Qualifier already measures complete wall residency conservatively as GPU occupation.
            full = self.manifest["full_baseline"]
            exceeds = wall > full["gpu_seconds"]*.8 or wall > full["wall_seconds"]*.8 or cpu > full["cpu_seconds"]*.8
            max_media_work_bytes = max(sum(p.stat().st_size for p in (self.root/"media").glob(w["work_id"]+"*")) for w in self.manifest["works"])
            projected_bytes = stats["result_bytes"]+3352*max_media_work_bytes
            exceeds = exceeds or projected_bytes > full["result_bytes"]*.8
            result["full_baseline_budget"] = dict(status="PROJECTED_BUDGET_EXCEEDED" if exceeds else "UNRESOLVED_UNMEASURED_OVERHEAD",
                gpu_occupancy_seconds=wall, wall_seconds=wall, cpu_seconds=cpu, minimum_projected_result_bytes=projected_bytes,
                remaining_unmeasured_overhead=p["unmeasured_full_baseline_overheads"], may_start=False)
        write_json(self.root/"summary.json", result)
        text = ["# AquaLoRA qualification", "", f"Status: {result['status']}",
                f"Trajectories: {result['trajectories_completed']}/44 completed; {result['trajectories_started']}/44 started.",
                "", "Actual checkpoint history remains UNVERIFIED; unseen content is a user-accepted assumption.",
                "This is DEV qualification only. No training, CAL/TEST, catch-one/FWER claim or innovation PASS.",
                "", "Inspect summary.json, trajectory_status.json, readouts.jsonl and all eight visual_review panels.",
                "Missing visual review remains UNRESOLVED. Reopening this notebook only reads the saved attempt.",
                "", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```"]
        (self.root/"report.md").write_text("\n".join(text)+"\n")
        return result


def recovery_inputs(attempt, manifest):
    """Admit only the observed pre-trajectory environment stop; never repair history."""
    if json.loads((attempt/"manifest.json").read_text()) != manifest:
        raise RuntimeError("Frozen manifest differs; recovery stopped")
    summary = json.loads((attempt/"summary.json").read_text())
    if (summary.get("status") != "STOPPED_INCOMPLETE" or
        summary.get("error") != "RuntimeError: python_environment exited 2; see python_environment.log" or
        summary.get("trajectories_started") != 0 or summary.get("trajectories_completed") != 0 or
        summary.get("trajectory_denominator") != 44 or
        Path(summary.get("output", "")).name != manifest["run_id"]):
        raise RuntimeError("Not the observed zero-trajectory environment failure; recovery stopped")
    stages = json.loads((attempt/"setup_stages.json").read_text())
    if [(s["stage"], s["returncode"]) for s in stages] != [("install_uv", 0), ("python_environment", 2)]:
        raise RuntimeError("Setup history is not the observed Python installation stop")
    for name in ("python_environment.log", "install_uv.log", "runtime_entry.json", "execution_failure.json", "resources.json", "executed_source.tar.gz"):
        if not (attempt/name).is_file() or not (attempt/name).stat().st_size:
            raise RuntimeError(f"Missing original evidence: {name}")
    rows_path = attempt/"trajectory_status.json"
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
        expected = [(t["trajectory_id"], t["slot"], "NOT_STARTED") for t in manifest["trajectories"]]
        if [(t["trajectory_id"], t["slot"], t["status"]) for t in rows] != expected:
            raise RuntimeError("Trajectory ledger is not entirely NOT_STARTED; no reset allowed")
    if any((attempt/name).exists() for name in ("numeric_result.json", "readouts.jsonl", "media")):
        raise RuntimeError("Model-stage evidence exists; environment-only recovery stopped")
    carry = summary["resources"]
    for key, minimum in MINIMUM_CARRY.items():
        value = carry.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
            raise RuntimeError(f"Missing, invalid or regressed resource receipt: {key}")
    return summary


def recover_environment(started_unix=None, started_cpu=None):
    """One authorized CPU-only recovery segment, inside the same consumed attempt."""
    started_unix = time.time() if started_unix is None else started_unix
    manifest = json.loads((ROOT/"qualification/manifest.json").read_text())
    attempt = DRIVE_PARENT/manifest["run_id"]
    root = attempt/RECOVERY_NAME
    if root.exists():
        return receipt(root)
    previous = recovery_inputs(attempt, manifest)  # Missing/unclear history is a stop.
    carry = previous["resources"]
    root.mkdir()  # Reserve before any recovery downloads; interruption consumes it.
    initial = dict(status="RECOVERY_RESERVED", output=str(root), attempt_output=str(attempt),
        qualification_pass=False, scientific_pass=None, trajectories_started=previous["trajectories_started"],
        trajectories_completed=previous["trajectories_completed"], trajectory_denominator=44,
        resources=carry, resources_final=False, environment_only=True, full_baseline_authorized=False,
        checkpoint_history="UNVERIFIED", unseen_content="USER_ACCEPTED_ASSUMPTION")
    write_json(root/"summary.json", initial)
    write_json(root/"carry_forward.json", previous)
    write_json(root/"manifest.json", manifest)
    supervisor = None
    error = None
    try:
        scratch = WORK_PARENT/manifest["run_id"]/RECOVERY_NAME
        scratch.mkdir(parents=True)  # Never reuse or erase an old recovery runtime.
        supervisor = Supervisor(root, scratch, manifest, started_unix,
                                carry=carry, account_root=attempt, monitor_gpu=False)
        if started_cpu is not None:
            supervisor.cpu_base = started_cpu
        # Inventory is evidence of preservation, not a new source admission gate.
        inventory = {str(p.relative_to(attempt)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in attempt.rglob("*") if p.is_file() and root not in p.parents}
        write_json(root/"preserved_files.json", inventory)
        write_json(root/"runtime_entry.json", dict(started_unix=started_unix, mode="CPU_ENVIRONMENT_ONLY",
            code_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            parent_attempt=manifest["run_id"], original_resources=carry))
        with tarfile.open(root/"executed_source.tar.gz", "w:gz") as archive:
            for name in ["qualification", "colab_bootstrap.py", "requirements-colab.txt", "README.md", "AMENDMENT.md"]:
                archive.add(ROOT/name, arcname=name, filter=lambda info: None if "__pycache__" in info.name else info)
        supervisor.environment_only()  # No prepare, model loading, Runner, ledger or CUDA call.
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_json(root/"execution_failure.json", dict(error=error))
    result = dict(initial, status="STOPPED_INCOMPLETE" if error else "ENVIRONMENT_READY", error=error)
    try:
        if supervisor is None:
            raise RuntimeError("Resource monitor did not initialize; segment costs unresolved")
        stats = supervisor.usage()
        result.update(resources=stats, resources_final=True)
        reason = supervisor.reason(stats)
        if reason:
            result.update(status="STOPPED_INCOMPLETE", error=error or reason)
        write_json(root/"resources.json", stats)
    except BaseException as exc:
        result.update(status="FINALIZATION_INCOMPLETE", resources_final=False,
                      finalization_error=f"{type(exc).__name__}: {exc}")
    write_json(root/"summary.json", result)
    (root/"report.md").write_text("# AquaLoRA environment recovery\n\n"
        "Original attempt01 failure files are unchanged. This segment only restores the software environment.\n"
        "No model weights, GPU execution, trajectories or CAL/TEST were started by this entry.\n"
        "ENVIRONMENT_READY is not qualification PASS. Reopening only reads this saved segment.\n"
        "Costs carry forward from the original receipt; all new elapsed time also consumes the conservative GPU-time allowance, even on CPU.\n\n"
        "```json\n"+json.dumps(result, ensure_ascii=False, indent=2)+"\n```\n")
    return result


def run_model_stage(started_unix=None, started_cpu=None):
    # Import only the metadata/supervisor entry; the model runner is a later child.
    from qualification.model_stage import continue_attempt
    return continue_attempt(started_unix=started_unix, started_cpu=started_cpu)


def run_first_run(started_unix=None, started_cpu=None):
    from qualification.first_run import run
    return run(started_unix=started_unix, started_cpu=started_cpu)


def run_authorized_rerun(started_unix=None, started_cpu=None):
    from qualification.first_run import run
    return run(started_unix=started_unix, started_cpu=started_cpu, authorized_rerun=True)


def run_colab(started_unix=None):
    import psutil
    started_unix = started_unix or time.time()
    manifest = json.loads((ROOT/"qualification/manifest.json").read_text())
    root = DRIVE_PARENT/manifest["run_id"]
    if root.exists():
        return receipt(root)
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Mount Google Drive using the first cell")
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    if query.returncode or not query.stdout.strip():
        raise RuntimeError("Select a Colab GPU runtime before running; no attempt has started")
    if len(query.stdout.strip().splitlines()) != 1:
        raise RuntimeError("This qualification requires one GPU")
    DRIVE_PARENT.mkdir(parents=True, exist_ok=True)
    root.mkdir()  # Exclusive attempt reservation; never erase or reuse its outputs.
    scratch = WORK_PARENT/manifest["run_id"]
    if scratch.exists():
        write_json(root/"summary.json", dict(status="STALE_LOCAL_RUNTIME", qualification_pass=False,
                    scientific_pass=None, output=str(root), note="Existing local runtime retained; no automatic reset."))
        return receipt(root)
    scratch.mkdir(parents=True)
    write_json(root/"manifest.json", manifest)
    write_json(root/"runtime_entry.json", dict(started_unix=started_unix, gpu=query.stdout.strip(),
               code_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
               source_history="UNVERIFIED", unseen_content_assumption="USER_ACCEPTED_2026_11_08",
               previous_entrance_stop="preserved; superseded only for source-assumption and Colab execution environment"))
    with tarfile.open(root/"executed_source.tar.gz", "w:gz") as archive:
        for name in ["qualification", "colab_bootstrap.py", "requirements-colab.txt", "README.md", "AMENDMENT.md"]:
            archive.add(ROOT/name, arcname=name, filter=lambda info: None if "__pycache__" in info.name else info)
    supervisor = Supervisor(root, scratch, manifest, started_unix)
    error = None
    try:
        supervisor.execute()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_json(root/"execution_failure.json", dict(error=error))
    try:
        return supervisor.summarize(error)
    except BaseException as exc:
        # Even monitoring/finalization failure must leave an explicit non-pass receipt.
        write_json(root/"summary.json", dict(status="FINALIZATION_INCOMPLETE", error=error,
                   finalization_error=f"{type(exc).__name__}: {exc}", qualification_pass=False,
                   scientific_pass=None, output=str(root)))
        return receipt(root)
