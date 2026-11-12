"""One technical continuation, only before any physical operation ever started."""
import json
import math
from pathlib import Path
from .core import effective_limits

SEGMENT = "technical-recovery-01"
KEYS = ("wall_seconds", "gpu_occupancy_seconds", "cpu_seconds", "download_charge_bytes",
        "peak_ram_bytes", "peak_device_used_bytes", "result_bytes")


def admit(original, manifest, package):
    original, package = Path(original), Path(package)
    prior = json.loads((original/"summary.json").read_text())
    if (prior.get("status") != "STOPPED_INCOMPLETE" or prior.get("resources_final") is not True
        or prior.get("error") != "RuntimeError: official_assets exited 1; see official_assets.log"
        or (original/"worker_started.json").exists()):
        raise RuntimeError("Only the sealed pre-model official-assets failure is recoverable")
    if json.loads((original/"manifest.json").read_text()) != manifest:
        raise RuntimeError("Frozen manifest changed")
    rows = json.loads((original/"operation_status.json").read_text())
    if rows != [dict(op, status="NOT_STARTED") for op in manifest["operations"]]:
        raise RuntimeError("Consumed/missing/changed operations; no recovery allowed")
    expected = {k:dict(denominator=v, started=0, completed=0) for k,v in manifest["physical_limits"].items()}
    if prior.get("operations") != expected:
        raise RuntimeError("Operation summary does not match the untouched ledger")
    if prior.get("inherited_trajectories_started") != 44 or prior.get("inherited_trajectories_completed") != 44:
        raise RuntimeError("Historical 44 trajectories unresolved")
    if not (original/"python.json").is_file():
        raise RuntimeError("Successful environment receipt absent")
    for name in ("requirements-colab.txt", "requirements-cpu.txt"):
        if (original/"executed_source"/name).read_bytes() != (package/name).read_bytes():
            raise RuntimeError("Frozen environment lock differs")
    downloads = [json.loads(x) for x in (original/"downloads.jsonl").read_text().splitlines()]
    from .transport import ORIGINAL
    failures = [x for x in downloads if x.get("status") == "FAILED"]
    if len(failures) != 1 or failures[0].get("url") != ORIGINAL or failures[0].get("received_bytes") != 0 or "certificate has expired" not in failures[0].get("error", ""):
        raise RuntimeError("Failure is not the verified zero-byte expired-certificate case")
    carry, used, inherited = prior["cumulative_resources"], prior["resources"], prior["inherited_resources"]
    for key in KEYS:
        values = [d.get(key) for d in (carry, used, inherited)]
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x < 0 for x in values):
            raise RuntimeError("Missing/nonfinite resource accounting: "+key)
        expected = max(inherited[key],used[key]) if key.startswith("peak_") else inherited[key]+used[key]
        if not math.isclose(carry[key],expected,rel_tol=1e-12,abs_tol=1e-6):
            raise RuntimeError("Cumulative resources do not reconcile: "+key)
        if inherited[key] < manifest["inherited_resources"][key]:
            raise RuntimeError("Historical cost regressed: "+key)
    limits = effective_limits(manifest, carry)
    for cap, cost in (("gpu_seconds","gpu_occupancy_seconds"),("wall_seconds","wall_seconds"),("cpu_seconds","cpu_seconds")):
        limits[cap] = min(limits[cap], manifest["budget"][cap]-used[cost])
        if limits[cap] <= 20:
            raise RuntimeError("Original allowance exhausted; no recovery admitted")
    return dict(carry=carry, original_resources=used, effective_limits=limits,
        parent_attempt=str(original), segment=SEGMENT, original_files="read-only",
        authorization="2026-11-12 user requests verified runnable notebook; technical repair only",
        physical_operations_previously_started=0, experiment_parameters_changed=False)
