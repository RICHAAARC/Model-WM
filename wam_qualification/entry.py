"""User-run Colab entry; independent accounting and immutable old attempts."""
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from .core import Ledger, effective_limits, save
from .protocol import RUN_ID

PACKAGE = Path(__file__).resolve().parent
OLD_NAME = "aqualora-qualification-20261108-attempt01"
PARENT = Path("/content/drive/MyDrive/Model-WM")
RUNTIME_ROOT = Path("/content/model-wm-runtime")


def cpu():
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def size(root):
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file() and not p.is_symlink())


def inherit(summary, floor):
    if summary.get("trajectories_completed") != 44 or summary.get("trajectories_started") != 44:
        raise RuntimeError("Previous completed 44-trajectory history is unresolved")
    carry = summary.get("cumulative_resources", {})
    for key, minimum in floor.items():
        value = carry.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < minimum:
            raise RuntimeError("Missing or regressed cumulative resource: "+key)
    return carry


def read_only(root):
    path = root/"summary.json"
    result = json.loads(path.read_text()) if path.exists() else dict(status="INTERRUPTED_BEFORE_RECEIPT")
    result = dict(result, reopened_read_only=True)
    if not result.get("resources_final"):
        progress = root/"resources.json"
        if progress.exists():
            result["last_recorded_resources"] = json.loads(progress.read_text())
        result.update(status="UNFINISHED_OR_INTERRUPTED_RESOURCE_TOTAL_UNRESOLVED", qualification_pass=False,
            note="May still be running in the original runtime. Last receipt is a lower bound; no retry, repair, continuation or new budget admitted.")
    return result


class Supervisor:
    def __init__(self, root, manifest, carry, start, start_cpu, cache, limits=None, cache_baseline=0):
        import psutil
        self.psutil, self.root, self.m = psutil, root, manifest
        self.carry, self.start, self.cpu_base, self.cache = carry, start, start_cpu, cache
        self.limits = effective_limits(manifest, carry) if limits is None else limits
        self.cache_baseline = cache_baseline
        self.peak_ram = self.peak_device = self.download_high = 0
        self.env = dict(os.environ, PYTHONPATH=str(PACKAGE.parent), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
            PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_RETRIES="0", PIP_DEFAULT_TIMEOUT="30",
            MPLCONFIGDIR=str(root/"matplotlib_cache"))
        self.last = None

    def usage(self, process=None):
        ram, cores = self.psutil.Process().memory_info().rss, cpu()-self.cpu_base
        if process is not None and process.poll() is None:
            try:
                children = [self.psutil.Process(process.pid)]+self.psutil.Process(process.pid).children(recursive=True)
            except self.psutil.NoSuchProcess:
                children = []
            for p in children:
                try:
                    t = p.cpu_times()
                    # Include reaped grandchildren of a live child without counting
                    # still-running descendant CPU twice.
                    cores += t.user+t.system+t.children_user+t.children_system
                    ram += p.memory_info().rss
                except self.psutil.NoSuchProcess:
                    pass
        self.peak_ram = max(self.peak_ram, ram)
        query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if query.returncode:
            raise RuntimeError("Device accounting unavailable")
        self.peak_device = max(self.peak_device, sum(float(v)*1024**2 for v in query.stdout.split()))
        self.download_high = max(self.download_high, max(0, size(self.cache)-self.cache_baseline))
        elapsed = time.time()-self.start
        self.last = dict(wall_seconds=elapsed, gpu_occupancy_seconds=elapsed, cpu_seconds=cores,
            peak_ram_bytes=self.peak_ram, peak_device_used_bytes=self.peak_device,
            download_charge_bytes=self.download_high, result_bytes=size(self.root),
            download_measurement="new durable cache high-water, excludes reused paid cache; reporting-only, not network packets",
            gpu_clock="conservative entire notebook execution cell elapsed, including setup/download/CPU/IO")
        return self.last

    def reason(self, s):
        pairs = [("gpu_occupancy_seconds", self.limits["gpu_seconds"]-20),
            ("wall_seconds", self.limits["wall_seconds"]-20), ("cpu_seconds", self.limits["cpu_seconds"]-20),
            ("peak_ram_bytes", self.m["budget"]["ram_bytes"]-512*1024**2),
            ("peak_device_used_bytes", self.m["budget"]["device_bytes"]-256*1024**2),
            ("result_bytes", self.m["budget"]["result_bytes"]-self.carry["result_bytes"]-64*1024**2)]
        return next(("Stopping margin reached: "+k for k,v in pairs if s[k]>=v), None)

    def command(self, stage, args):
        reason = self.reason(self.usage())
        if reason:
            raise RuntimeError(reason)
        print(stage+" ...", flush=True)
        with (self.root/(stage+".log")).open("x") as log:
            p = subprocess.Popen(list(map(str, args)), env=self.env, cwd=PACKAGE.parent,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    stats = self.usage(p)
                    save(self.root/"resources.json", dict(stage=stage, resources=stats, resources_final=False))
                    reason = self.reason(stats)
                    if reason:
                        raise RuntimeError(reason)
                    time.sleep(.5)
                if p.returncode:
                    raise RuntimeError(f"{stage} exited {p.returncode}; see {stage}.log")
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait(timeout=3)
        stats = self.usage()
        save(self.root/"resources.json", dict(stage=stage, resources=stats, resources_final=False))
        reason = self.reason(stats)
        if reason:
            raise RuntimeError(reason)

    def finish(self, error):
        numeric = self.root/"numeric_result.json"
        result = json.loads(numeric.read_text()) if numeric.exists() else dict(status="STOPPED_INCOMPLETE", qualification_pass=False)
        rows = json.loads((self.root/"operation_status.json").read_text())
        for row in rows:
            if row["status"] == "STARTED":
                row["status"] = "INTERRUPTED"
        save(self.root/"operation_status.json", rows)
        if error:
            result.update(status="STOPPED_INCOMPLETE", qualification_pass=False, error=error)
        totals = {}
        for kind, limit in self.m["physical_limits"].items():
            subset = [r for r in rows if r["kind"]==kind]
            totals[kind] = dict(denominator=limit, started=sum(r["status"]!="NOT_STARTED" for r in subset),
                completed=sum(r["status"]=="COMPLETED" for r in subset))
        stats = self.usage()
        cumulative = {k:max(self.carry[k], stats[k]) if k.startswith("peak_") else self.carry[k]+stats[k]
            for k in self.carry if isinstance(self.carry[k], (int, float)) and k in stats}
        result.update(output=str(self.root), operations=totals, resources=stats, cumulative_resources=cumulative,
            inherited_resources=self.carry, inherited_trajectories_started=44, inherited_trajectories_completed=44,
            resources_final=True, scientific_pass=None, full_baseline_authorized=False, training_authorized=False,
            cal_test_authorized=False, checkpoint_history="UNVERIFIED", unseen_content="USER_ACCEPTED_ASSUMPTION")
        if self.reason(stats):
            result.update(status="STOPPED_RESOURCE_MARGIN", qualification_pass=False,
                resource_stop=self.reason(stats))
        save(self.root/"summary.json", result)
        return result


def run_colab(started_unix=None, started_cpu=None, recovery=False):
    start = time.time() if started_unix is None else started_unix
    start_cpu = cpu() if started_cpu is None else started_cpu
    original = PARENT/RUN_ID
    from .recovery import SEGMENT, admit
    root = original/SEGMENT if recovery else original
    if root.exists():
        return read_only(root)
    if not PARENT.is_dir():
        raise RuntimeError("Mount the existing Google Drive first")
    m = json.loads((PACKAGE/"manifest.json").read_text())
    admission = admit(original, m, PACKAGE) if recovery else None
    root.mkdir()  # Exclusive claim before setup, downloads, models or physical operations.
    save(root/"manifest.json", m)
    Ledger(root, m)
    save(root/"summary.json", dict(status="RESERVED", qualification_pass=False, resources_final=False))
    supervisor = None
    try:
        old = json.loads((PARENT/OLD_NAME/"rerun-02/latest.json").read_text())
        aqua = inherit(old, m["inherited_resources"])
        carry = admission["carry"] if admission else aqua
        if admission:
            # A newer Aqua receipt must not silently disappear behind the sealed WAM receipt.
            if any(aqua[k] > json.loads((original/"summary.json").read_text())["inherited_resources"][k] for k in aqua):
                raise RuntimeError("Prior Aqua history changed after WAM failure; reconcile before recovery")
            save(root/"recovery_admission.json", admission)
        save(root/"carry_forward.json", old)
        save(root/"runtime_entry.json", dict(started_unix=start, started_cpu=start_cpu,
            effective_limits=admission["effective_limits"] if admission else effective_limits(m, carry), parent_attempt=str(original) if recovery else OLD_NAME,
            old_files="read-only", checkpoint_history="UNVERIFIED"))
        cache = PARENT/"cache"/RUN_ID
        if recovery:
            if not cache.is_dir():
                raise RuntimeError("Paid WAM cache missing; no cold redownload recovery")
            baseline = size(cache)
            if baseline > admission["original_resources"]["download_charge_bytes"]:
                raise RuntimeError("Cache exceeds sealed paid volume; provenance unresolved")
        else:
            cache.mkdir(parents=True, exist_ok=False)
            baseline = 0
        supervisor = Supervisor(root, m, carry, start, start_cpu, cache,
            limits=admission["effective_limits"] if admission else None, cache_baseline=baseline)
        scratch = RUNTIME_ROOT/RUN_ID
        if recovery:
            scratch = scratch/SEGMENT
        scratch.mkdir(parents=True, exist_ok=False)
        hardware = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3)
        if hardware.returncode or len(hardware.stdout.strip().splitlines()) != 1:
            raise RuntimeError("Exactly one readable GPU is required")
        save(root/"hardware.json", dict(nvidia_smi=hardware.stdout.strip()))
        # Keep exact entry sources beside every failed or successful attempt.
        import shutil
        shutil.copytree(PACKAGE, root/"executed_source", ignore=shutil.ignore_patterns("__pycache__"))
        oldcache = PARENT/"cache/aqualora-pinned-v1"
        args = [root, scratch, oldcache, cache]
        supervisor.command("asset_preflight", [sys.executable, "-m", "wam_qualification.transport", root, oldcache, cache])
        supervisor.command("environment", [sys.executable, "-m", "wam_qualification.setup", "environment", *args])
        python = json.loads((root/"python.json").read_text())["python"]
        supervisor.command("official_assets", [python, "-m", "wam_qualification.setup", "models", *args])
        supervisor.command("qualification", [python, "-m", "wam_qualification.worker", root])
        return supervisor.finish(None)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        save(root/"execution_failure.json", dict(error=error))
        if supervisor:
            try:
                return supervisor.finish(error)
            except BaseException as final:
                save(root/"finalization_failure.json", dict(error=str(final), last_resources=supervisor.last))
        result = dict(status="STOPPED_INCOMPLETE_ACCOUNTING_UNRESOLVED", qualification_pass=False,
            scientific_pass=None, resources_final=False, error=error, output=str(root),
            elapsed_lower_bound=time.time()-start, full_baseline_authorized=False)
        save(root/"summary.json", result)
        return result
