"""One explicitly authorized continuation of the known pre-load import failure."""
import json
import math
from pathlib import Path
from .core import effective_limits
from .recovery import admit as admit_first, KEYS

SEGMENT = "technical-recovery-02"
PREVIOUS = "technical-recovery-01"


def admit(original, manifest, package):
    original, package = Path(original), Path(package)
    first = admit_first(original, manifest, package)
    previous = original/PREVIOUS
    prior = json.loads((previous/"summary.json").read_text())
    if (prior.get("status") != "STOPPED_INCOMPLETE" or prior.get("resources_final") is not True
        or prior.get("error") != "RuntimeError: qualification exited 1; see qualification.log"):
        raise RuntimeError("Only the sealed known pre-load import failure is admitted")
    if json.loads((previous/"manifest.json").read_text()) != manifest:
        raise RuntimeError("Frozen manifest changed")
    if json.loads((previous/"operation_status.json").read_text()) != [dict(op, status="NOT_STARTED") for op in manifest["operations"]]:
        raise RuntimeError("A physical operation was consumed or its ledger is unresolved")
    expected = {k:dict(denominator=v, started=0, completed=0) for k,v in manifest["physical_limits"].items()}
    if prior.get("operations") != expected or any(prior.get(k) != 44 for k in
            ("inherited_trajectories_started", "inherited_trajectories_completed")):
        raise RuntimeError("Operation or inherited trajectory counts do not reconcile")
    reservation = json.loads((previous/"worker_started.json").read_text())
    started = reservation.get("started_unix")
    if isinstance(started, bool) or not isinstance(started, (int,float)) or not math.isfinite(started) or started <= 0:
        raise RuntimeError("Previous worker reservation is unresolved")
    log = (previous/"qualification.log").read_text()
    if (not log.startswith("Traceback (most recent call last):") or log.count("Traceback (") != 1
        or 'from inference_utils import load_model_from_checkpoint' not in log
        or 'import matplotlib.pyplot as plt' not in log
        or not log.rstrip().splitlines()[-1].startswith("ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value for backend;")):
        raise RuntimeError("Failure is not the verified Colab Matplotlib import error")
    if not (previous/"python.json").is_file() or not (previous/"model_paths.json").is_file():
        raise RuntimeError("Environment or asset-stage receipt absent")
    for name in ("requirements-colab.txt", "requirements-cpu.txt"):
        if (previous/"executed_source"/name).read_bytes() != (package/name).read_bytes():
            raise RuntimeError("Frozen environment lock differs")
    carry, used, inherited = prior["cumulative_resources"], prior["resources"], prior["inherited_resources"]
    total = {}
    for key in KEYS:
        values = [d.get(key) for d in (carry, used, inherited)]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in values):
            raise RuntimeError("Missing/nonfinite accounting: "+key)
        if inherited[key] != first["carry"][key]:
            raise RuntimeError("Broken accounting chain: "+key)
        expected = max(inherited[key], used[key]) if key.startswith("peak_") else inherited[key]+used[key]
        if not math.isclose(carry[key], expected, rel_tol=1e-12, abs_tol=1e-6):
            raise RuntimeError("Cumulative resources do not reconcile: "+key)
        total[key] = max(first["original_resources"][key], used[key]) if key.startswith("peak_") else first["original_resources"][key]+used[key]
    limits = effective_limits(manifest, carry)
    for cap,cost in (("gpu_seconds","gpu_occupancy_seconds"),("wall_seconds","wall_seconds"),("cpu_seconds","cpu_seconds")):
        limits[cap] = min(limits[cap], manifest["budget"][cap]-total[cost])
        if limits[cap] <= 20:
            raise RuntimeError("Original allowance exhausted")
    return dict(carry=carry, original_resources=total, effective_limits=limits,
        parent_attempt=str(original), previous_segment=str(previous), segment=SEGMENT,
        original_files="read-only", physical_operations_previously_started=0,
        authorization="2026-11-12 user authorizes one same-attempt technical continuation; local content weight storage",
        experiment_parameters_changed=False)
