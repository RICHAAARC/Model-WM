"""One user-authorized closure entry after the four sealed attempt02 failures."""
import copy
import json
from .core import Ledger, save

MARKER = "engineering_closure_01.started.json"
MODEL_MARKER = "engineering_closure_01.model_started.json"


def effective_budget(original, amendment):
    budget = copy.deepcopy(original)
    for field, delta in amendment["additional_preparation"].items():
        if delta <= 0 or original["preparation"][field] + delta != amendment["effective_preparation"][field]:
            raise RuntimeError("Invalid additive preparation amendment")
    budget["preparation"] = dict(amendment["effective_preparation"])
    budget["preparation_amendment"] = copy.deepcopy(amendment)
    budget["cleanup_margin_seconds"] = amendment["cleanup_margin_seconds"]
    return budget


def admit(root, manifest, original, summary):
    if original["execution_id"] != "wam-qualification-20261112-attempt02":
        raise RuntimeError("Closure is scoped to attempt02")
    if sorted(p.name for p in (root/"invocations").iterdir()) != ["0001", "0002", "0003", "0004"]:
        raise RuntimeError("Expected exactly four retained failures")
    if not summary.get("resources_final") or summary.get("status") != "ENVIRONMENT_STOPPED":
        raise RuntimeError("Previous environment stop is not sealed")
    if not summary.get("technical_continuation_consumed") or not (root/"technical_continuation_01.started.json").is_file():
        raise RuntimeError("Previous authorized continuation is unresolved")
    if (root/"technical_continuation_01.model_started.json").exists():
        raise RuntimeError("Unexpected model work in previous continuation")
    if json.loads((root/"model_phase_started.json").read_text()).get("invocation") != "0003":
        raise RuntimeError("Unexpected original model reservation")
    Ledger(root, manifest, precreated=True)
    inv = root/"invocations/0004"
    latest = json.loads((inv/"summary.json").read_text())
    for key in ("phases", "phase_budget_debits"):
        if latest.get(key) != summary.get(key):
            raise RuntimeError("Latest sealed phase ledger differs: " + key)
    previous=json.loads((root/"invocations/0003/summary.json").read_text())
    if summary["phases"]["model"] != previous["phases"]["model"]:
        raise RuntimeError("Model charges changed during environment-only continuation")
    if (inv/"qualification.usage.json").exists():
        raise RuntimeError("Unexpected previous model stage")
    if "preparation wall_seconds" not in (summary.get("error") or ""):
        raise RuntimeError("Unexpected prior stop; reconcile before execution")
    for stage in ("asset_preflight", "environment", "official_assets"):
        r = json.loads((inv/(stage+".usage.json")).read_text())
        expected = -15 if stage == "official_assets" else 0
        if not (r.get("resources_final") and r.get("all_descendants_reaped") and r.get("exit_code") == expected):
            raise RuntimeError("Previous stage unresolved: " + stage)
    if summary.get("execution_budget") != original:
        raise RuntimeError("Original budget history differs")
    return dict(allowed_invocation="0005", old_invocations=["0001","0002","0003","0004"],
        additional_invocations=1, tail_uncertainty_reserves={}, old_files="retained",
        physical_operations_started=0, no_extra_trial=True)


def admit_cost(supervisor, kind, environment=None):
    """Cost gate before expensive install/copy; hits are not zero-cost work."""
    amendment = supervisor.budget.get("preparation_amendment")
    if not amendment:
        return
    stats = supervisor.usage()
    reason = supervisor.reason(stats)
    need = amendment["admission"][kind]
    remaining = {k: supervisor.budget["preparation"][k] - supervisor.last_phase_debit[k] for k in need}
    global_debit = getattr(supervisor, "budget_carry", supervisor.carry)
    prior = supervisor.prior.get("budget_resources", supervisor.prior["resources"])
    global_remaining = {k: supervisor.m["budget"]["global_"+k] - global_debit[k] - prior.get(k,0) - stats[k] for k in need}
    global_gpu = supervisor.m["budget"]["global_gpu_seconds"] - global_debit["gpu_occupancy_seconds"] - prior.get("gpu_occupancy_seconds",0) - stats["gpu_occupancy_seconds"]
    admitted = not reason and all(remaining[k] >= v and global_remaining[k] >= v for k,v in need.items()) and global_gpu >= need["wall_seconds"]
    save(supervisor.invocation/(kind+"_cost_admission.json"), dict(admitted=admitted,
        required=need, remaining=remaining, global_remaining=global_remaining,
        global_gpu_remaining=global_gpu, environment=environment,
        basis="Observed cold preparation/copy cost with reserve; not a timing guarantee"))
    if not admitted:
        raise RuntimeError(reason or "Insufficient remaining preparation budget for " + kind)
