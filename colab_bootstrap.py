"""Thin Colab supervisor; all results are written directly under mounted Drive."""
import json
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
    def __init__(self, root, scratch, manifest, started_unix):
        import psutil  # Included in standard Colab; validated before the attempt.
        self.psutil = psutil
        self.root, self.scratch, self.manifest = root, scratch, manifest
        self.started = started_unix
        self.cpu_base = cpu_usage()
        self.max_ram = self.max_gpu = self.max_download = 0
        self.logs = []
        self.environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                                PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                                HF_HUB_DISABLE_TELEMETRY="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                                TMPDIR=str(scratch/"downloads/tmp"), UV_CACHE_DIR=str(scratch/"uv-cache"),
                                UV_PYTHON_INSTALL_DIR=str(scratch/"python"))
        Path(self.environment["TMPDIR"]).mkdir(parents=True)

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
        query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
        if query.returncode:
            raise RuntimeError("GPU monitor unavailable during bounded attempt")
        gpu = sum(float(x)*1024**2 for x in query.stdout.splitlines() if x.strip())
        self.max_gpu = max(self.max_gpu, gpu)
        downloaded = total_bytes(self.scratch/"downloads")+SETUP_DOWNLOAD_RESERVE
        self.max_download = max(self.max_download, downloaded)
        elapsed = time.time()-self.started
        return dict(wall_seconds=elapsed, gpu_occupancy_seconds=elapsed,
                    cpu_seconds=cpu, peak_ram_bytes=self.max_ram, peak_device_used_bytes=self.max_gpu,
                    download_charge_bytes=self.max_download, result_bytes=total_bytes(self.root),
                    download_measurement="retained download payload high-water plus 256 MiB setup/metadata reserve; not packet bytes",
                    gpu_clock="conservative entire notebook entry elapsed time, including setup/download/CPU work")

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

    def execute(self):
        tools_dir = self.scratch/"tools"
        self.command("install_uv", [sys.executable, "-m", "pip", "install", "--no-cache-dir",
                                    "--target", str(tools_dir), "uv==0.9.5"])
        uv = str(tools_dir/"bin/uv")
        venv = self.scratch/"venv"
        self.command("python_environment", [uv, "venv", "--python", "3.10.14", "--seed", str(venv)])
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
