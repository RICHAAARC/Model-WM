"""Read-only closure of attempt01, with additive correction for lost CPU usage."""
import json
import math
from pathlib import Path
from .recovery2 import admit as old_admit
from .recovery import KEYS


def close_original(original,manifest,package):
    original=Path(original)
    before=old_admit(original,manifest,package)
    last=original/"technical-recovery-02"
    s=json.loads((last/"summary.json").read_text())
    progress=json.loads((last/"resources.json").read_text())
    if s.get("status")!="STOPPED_RESOURCE_MARGIN" or s.get("resources_final") is not True or progress.get("stage")!="environment":
        raise RuntimeError("Original attempt is not the sealed observed environment stop")
    if (last/"worker_started.json").exists() or (last/"model_paths.json").exists():
        raise RuntimeError("Original model stage may have started")
    if json.loads((last/"manifest.json").read_text())!=manifest:
        raise RuntimeError("Frozen original manifest differs")
    if json.loads((last/"operation_status.json").read_text())!=[dict(op,status="NOT_STARTED") for op in manifest["operations"]]:
        raise RuntimeError("Original physical operations were consumed")
    expected={k:dict(denominator=v,started=0,completed=0) for k,v in manifest["physical_limits"].items()}
    if s.get("operations")!=expected or s.get("inherited_trajectories_started")!=44 or s.get("inherited_trajectories_completed")!=44:
        raise RuntimeError("Historical count mismatch")
    carry={}
    corrections={}
    for key in KEYS:
        values=[s["resources"].get(key),s["inherited_resources"].get(key),s["cumulative_resources"].get(key),progress["resources"].get(key)]
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in values):
            raise RuntimeError("Historical resource is not readable: "+key)
        final,inherited,cumulative,observed=values
        if inherited!=before["carry"][key]:raise RuntimeError("Broken historical resource chain: "+key)
        raw=max(inherited,final) if key.startswith("peak_") else inherited+final
        if not math.isclose(raw,cumulative,rel_tol=1e-12,abs_tol=1e-6):raise RuntimeError("Raw receipt does not reconcile: "+key)
        used=max(final,observed)
        carry[key]=max(inherited,used) if key.startswith("peak_") else inherited+used
        if used>final:corrections[key]=dict(reported=final,observed_floor=used,added_minimum=used-final)
    budget_carry=dict(carry)
    aqua_cpu=json.loads((original/"summary.json").read_text())["inherited_resources"]["cpu_seconds"]
    # Do not invent a precise historical CPU total. Keep the old WAM CPU
    # allowance fully debited for global-budget admission until its unknown tail
    # can be reconciled; measured cost and conservative reservation stay separate.
    budget_carry["cpu_seconds"]=max(carry["cpu_seconds"],aqua_cpu+manifest["budget"]["cpu_seconds"])
    return dict(carry=carry,budget_carry=budget_carry,
        historical_cpu_uncertainty_reserve=budget_carry["cpu_seconds"]-carry["cpu_seconds"],
        corrections=corrections,original_attempt=str(original),
        disposition="ENVIRONMENT_STOPPED_NO_MODEL_OPERATIONS",original_files="unchanged",
        historical_cpu_exact=False,cumulative_cpu_is_lower_bound=True,
        unknown_tail="CPU after last live-process sample cannot be recovered exactly; no fabricated exact total",
        original_budget="closed, not reset",physical_operations_previously_started=0)
