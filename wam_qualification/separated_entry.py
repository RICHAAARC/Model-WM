"""New explicitly budgeted execution; immutable history, resumable setup, one model run."""
import json
import shutil
import time
import traceback
from pathlib import Path
from .core import Ledger,save
from .entry import PARENT,RUNTIME_ROOT,PACKAGE,OLD_NAME,inherit,read_only,size
from .protocol import RUN_ID
from .history import close_original
from .phases import Supervisor,own_cpu,combine
from .receipts import tree_bytes


def run_colab(started_unix=None,started_cpu=None):
    start=time.time() if started_unix is None else started_unix
    cpu_start=own_cpu() if started_cpu is None else started_cpu
    budget=json.loads((PACKAGE/"execution_budget.json").read_text())
    manifest=json.loads((PACKAGE/"manifest.json").read_text())
    root=PARENT/budget["execution_id"]
    original=PARENT/RUN_ID
    if not PARENT.is_dir():raise RuntimeError("Mount existing Drive first")
    history=close_original(original,manifest,PACKAGE)
    aqua=json.loads((PARENT/OLD_NAME/"rerun-02/latest.json").read_text())
    latest=inherit(aqua,manifest["inherited_resources"])
    original_aqua=json.loads((original/"summary.json").read_text())["inherited_resources"]
    if any(latest[k]>original_aqua[k] for k in latest):raise RuntimeError("Aqua history advanced; reconcile before starting")
    if root.exists():
        if (root/"active.lock").exists():
            return dict(status="ACTIVE_OR_UNSEALED_INVOCATION",qualification_pass=False,
                resources_final=False,reopened_read_only=True,output=str(root),
                note="No new execution: retain the reservation and inspect the existing runtime/receipts.")
        if (root/"model_phase_started.json").exists():return read_only(root)
        if json.loads((root/"manifest.json").read_text())!=manifest or json.loads((root/"execution_budget.json").read_text())!=budget:
            raise RuntimeError("Frozen execution configuration changed")
        prior_summary=json.loads((root/"summary.json").read_text())
        if prior_summary.get("resources_final") is not True or not prior_summary.get("environment_resume_allowed"):
            return read_only(root)
        if prior_summary["inherited_resources"]!=history["carry"]:raise RuntimeError("Historical cost changed")
        Ledger(root,manifest,precreated=True)
        prior=dict(resources=prior_summary["resources"],phases=prior_summary["phases"])
        # The sealed invocation chain must not disappear behind an older root
        # summary. Never invent a new budget when an interrupted tail is unknown.
        observed={}
        for previous in sorted((root/"invocations").iterdir()):
            receipt=json.loads((previous/"summary.json").read_text())
            if receipt.get("resources_final") is not True:
                raise RuntimeError("Unsealed prior invocation; no automatic continuation")
            observed=combine(observed,receipt["resources"])
        for key,value in observed.items():
            if key!="result_bytes" and prior["resources"].get(key,0)+1e-8<value:
                raise RuntimeError("Regressed prior resource ledger: "+key)
        for key in ("wall_seconds","gpu_occupancy_seconds","cpu_seconds"):
            if prior["phases"].get("preparation",{}).get(key,0)+1e-8<observed.get(key,0):
                raise RuntimeError("Regressed preparation budget debit: "+key)
    else:
        root.mkdir()
        save(root/"manifest.json",manifest);save(root/"execution_budget.json",budget)
        save(root/"history_correction.json",history);Ledger(root,manifest)
        prior=dict(resources={},phases={})
        prior_summary=None
    prior_bytes=tree_bytes(root)
    with (root/"active.lock").open("x") as f:
        json.dump(dict(started_unix=start),f)
    invocations=root/"invocations";invocations.mkdir(exist_ok=True)
    invocation=invocations/f"{len(list(invocations.iterdir()))+1:04d}"
    invocation.mkdir()  # exclusive; concurrent clicks cannot both claim this step
    if prior_summary is not None:
        # Preserve the raw root receipts that this invocation will supersede.
        for name in ("summary.json","resources.json","operation_status.json"):
            if (root/name).exists():shutil.copyfile(root/name,invocation/("prior_"+name))
    save(invocation/"volume_reconciliation.json",dict(previous_reported_bytes=prior["resources"].get("result_bytes",0),
        observed_prior_tree_bytes=prior_bytes,correction_bytes=prior_bytes-prior["resources"].get("result_bytes",0),
        rule="Retain old receipts verbatim; final result volume measures the entire sealed attempt tree"))
    save(root/"summary.json",dict(status="PREPARING",resources_final=False,qualification_pass=False))
    cache_root=RUNTIME_ROOT/budget["execution_id"]/"assets"
    cache=cache_root/"new";cache.mkdir(parents=True,exist_ok=True)
    for n in ("wam","aqua"):(cache/n).mkdir(exist_ok=True)
    scratch=RUNTIME_ROOT/budget["execution_id"]/"invocations"/invocation.name
    scratch.mkdir(parents=True,exist_ok=False)
    work=scratch/"records";work.mkdir()
    local_invocation=work/"invocations"/invocation.name;local_invocation.mkdir(parents=True)
    save(work/"manifest.json",manifest);Ledger(work,manifest)
    mappings=[]
    old_assets=RUNTIME_ROOT/RUN_ID/"technical-recovery-02-assets"
    for name,drive in (("wam",PARENT/"cache"/RUN_ID),("aqua",PARENT/"cache/aqualora-pinned-v1")):
        mappings.append(dict(new=str(cache/name),paid=str(cache_root/"paid"/name),drive=str(drive),
            local=[str(old_assets/"paid"/name),str(old_assets/"new"/name)]))
    save(local_invocation/"local_storage.json",dict(mappings=mappings,weights_destination="/content only",
        live_records=str(work),sealed_results=str(root)))
    save(local_invocation/"manifest.json",manifest)
    save(local_invocation/"recovery_admission.json",history)
    sup=Supervisor(work,local_invocation,manifest,history["carry"],prior,budget,start,cpu_start,cache,size(cache),archive_root=root)
    sup.budget_carry=history["budget_carry"]
    error=None
    try:
        shutil.copytree(PACKAGE,local_invocation/"executed_source",ignore=shutil.ignore_patterns("__pycache__"))
        oldcache,assets=cache/"aqua",cache/"wam"
        args=[local_invocation,scratch,oldcache,assets]
        import sys
        sup.command("asset_preflight",[sys.executable,"-m","wam_qualification.transport",local_invocation,oldcache,assets])
        sup.command("environment",[sys.executable,"-m","wam_qualification.prepare",*args])
        python=json.loads((local_invocation/"python.json").read_text())["python"]
        sup.command("official_assets",[python,"-m","wam_qualification.setup","models",*args])
        save(work/"model_paths.json",json.loads((local_invocation/"model_paths.json").read_text()))
        save(local_invocation/"environment_ready.json",dict(status="READY_IN_CURRENT_RUNTIME",no_model_loaded=True))
        sup.enter_model()
        sup.command("qualification",[python,"-m","wam_qualification.worker",work])
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        failure=dict(error=error,traceback=traceback.format_exc(),phase=sup.phase,
            live_records=str(work),last_resources=sup.last_total,last_phase=sup.last_phase)
        save(local_invocation/"execution_failure.json",failure)
        # Write a sealed failure document even if subsequent settlement fails.
        save(invocation/"execution_failure.json",failure)
    try:
        result=sup.finish(error)
        if result["resources_final"]:(root/"active.lock").unlink()
        return result
    except BaseException as exc:
        result=dict(status="ACCOUNTING_UNRESOLVED",resources_final=False,qualification_pass=False,
            error=error,finalization_error=str(exc),traceback=traceback.format_exc(),
            last_resources=sup.last_total,prior_resources=prior["resources"],prior_phases=prior["phases"],
            inherited_resources=history["carry"],output=str(root),live_records=str(work))
        save(invocation/"finalization_failure.json",result)
        save(root/"summary.json",result);return result
