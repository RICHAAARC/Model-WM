"""One explicitly authorized technical continuation of the consumed attempt02."""
import json
from .core import Ledger

MARKER="technical_continuation_01.started.json"
MODEL_MARKER="technical_continuation_01.model_started.json"
AUTHORIZATION="2026-11-14 user authorizes one same-attempt02 technical continuation, unchanged phase budgets and operations, GPU user-started"


def admit(root,manifest,budget,summary):
    if budget["execution_id"]!="wam-qualification-20261112-attempt02":
        raise RuntimeError("Technical continuation is scoped to attempt02")
    if sorted(p.name for p in (root/"invocations").iterdir())!=["0001","0002","0003"]:
        raise RuntimeError("Expected exactly the three retained failures; reconcile history")
    if not summary.get("resources_final") or summary.get("status")!="MODEL_STOPPED":
        raise RuntimeError("Prior model failure is not sealed")
    Ledger(root,manifest,precreated=True)  # All 240 rows must still be NOT_STARTED.
    inv=root/"invocations/0003"
    latest=json.loads((inv/"summary.json").read_text())
    if latest.get("phases")!=summary.get("phases"):
        raise RuntimeError("Phase history differs from the latest sealed invocation")
    if json.loads((root/"model_phase_started.json").read_text()).get("invocation")!="0003":
        raise RuntimeError("Unexpected consumed model reservation")
    r=json.loads((inv/"qualification.usage.json").read_text())
    if not (r.get("resources_final") and r.get("all_descendants_reaped") and r.get("exit_code")==1 and not r.get("interrupted")):
        raise RuntimeError("Prior model process accounting is unresolved")
    log=(inv/"qualification.log").read_text()
    if "OSError:" not in log or "does not appear to have a file named preprocessor_config.json" not in log:
        raise RuntimeError("Continuation is limited to the audited missing-component failure")
    for stage in ("asset_preflight","environment","official_assets"):
        r=json.loads((inv/(stage+".usage.json")).read_text())
        if not (r.get("resources_final") and r.get("all_descendants_reaped") and r.get("exit_code")==0):
            raise RuntimeError("Previous preparation stage is unresolved: "+stage)
    # Old self-write tails cannot be reconstructed as exact usage. Retain the
    # lower-bound observations and debit one existing stopping-margin reserve
    # for each old seal. This is a disclosed uncertainty provision, not a
    # measurement or a proof that historical usage was bounded by this value.
    margin=budget["stop_margin_seconds"]
    return dict(authorization=AUTHORIZATION,allowed_invocation="0004",additional_invocations=1,
        old_invocations=["0001","0002","0003"],historical_sealing_tail="UNMEASURED",
        tail_uncertainty_reserves=dict(preparation=2*margin,model=margin),
        reserve_basis="one unchanged 20-second stopping margin per prior seal; charged inside existing budgets, not new allowance",
        historical_budget_exact=False,old_files="retained",physical_operations_started=0)


def debit_history(prior,admission):
    previous=prior.get("budget_resources",{})
    prior["budget_resources"]={k:max(v,previous.get(k,0)) for k,v in prior["resources"].items()}
    previous_phases=prior.get("phase_budget_debits",{})
    prior["phase_budget_debits"]={p:{k:max(v,previous_phases.get(p,{}).get(k,0)) for k,v in row.items()}
        for p,row in prior["phases"].items()}
    for phase,reserve in admission["tail_uncertainty_reserves"].items():
        for key in ("wall_seconds","gpu_occupancy_seconds","cpu_seconds"):
            prior["phase_budget_debits"][phase][key]+=reserve
            prior["budget_resources"][key]+=reserve
