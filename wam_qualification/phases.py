"""Independent preparation/model envelopes with one conserved total ledger."""
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from .core import save
from .entry import Supervisor as Legacy, size, PACKAGE
from .receipts import read_receipt, publish_tree, tree_bytes


def own_cpu():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime+r.ru_stime


def combine(a,b):
    return {k:max(a.get(k,0),v) if k.startswith("peak_") else a.get(k,0)+v
        for k,v in b.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}


class Supervisor(Legacy):
    def __init__(self, root, invocation, manifest, carry, prior, budget, start, cpu_start, cache, baseline, archive_root=None):
        super().__init__(root,manifest,carry,start,cpu_start,cache,limits=budget["model"],cache_baseline=baseline)
        self.invocation,self.prior,self.budget=invocation,prior,budget
        self.phase="preparation";self.phase_started=start
        self.phase_cpu_base=0.;self.model_start=None
        self.cpu_high=0.;self.query_cpu=0.;self.stages=[]
        self.preparation_fixed=None
        self.last_total=None;self.last_phase=None
        self.last_phase_debit=None;self.preparation_debit=None
        self.model_reservation_name="model_phase_started.json"
        self.archive_root=Path(archive_root) if archive_root is not None else root
        self.archive_invocation=self.archive_root/"invocations"/invocation.name if archive_root is not None else invocation
        self.archive_bytes=tree_bytes(self.archive_root)

    def usage(self, process=None):
        core=max(0,own_cpu()-self.cpu_base)+self.query_cpu
        ram=self.psutil.Process().memory_info().rss
        for path in self.stages:
            if path.exists():
                r=read_receipt(path);core+=r["cpu_seconds"]
                # Per-stage peak is conservative while only one stage runs.
                ram=max(ram,r["peak_ram_bytes"]+self.psutil.Process().memory_info().rss)
        self.cpu_high=max(self.cpu_high,core)
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
        query=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=3)
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        self.query_cpu+=max(0,after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
        if query.returncode: raise RuntimeError("Device accounting unavailable")
        if len(query.stdout.split())!=1:raise RuntimeError("Exactly one GPU is required")
        self.peak_ram=max(self.peak_ram,ram)
        self.peak_device=max(self.peak_device,sum(float(v)*1024**2 for v in query.stdout.split()))
        self.download_high=max(self.download_high,max(0,size(self.cache)-self.cache_baseline))
        now=time.time()
        total=dict(wall_seconds=now-self.start,gpu_occupancy_seconds=now-self.start,
            cpu_seconds=self.cpu_high,peak_ram_bytes=self.peak_ram,peak_device_used_bytes=self.peak_device,
            download_charge_bytes=self.download_high,result_bytes=size(self.invocation))
        prior_phase=self.prior["phases"].get(self.phase,{})
        phase=dict(wall_seconds=prior_phase.get("wall_seconds",0)+now-self.phase_started,
            gpu_occupancy_seconds=prior_phase.get("gpu_occupancy_seconds",0)+now-self.phase_started,
            cpu_seconds=prior_phase.get("cpu_seconds",0)+max(0,self.cpu_high-self.phase_cpu_base))
        self.last_total,self.last_phase=total,phase
        prior_debit=self.prior.get("phase_budget_debits",self.prior["phases"]).get(self.phase,{})
        self.last_phase_debit={k:v+max(0,prior_debit.get(k,0)-prior_phase.get(k,0)) for k,v in phase.items()}
        save(self.root/"resources.json",dict(stage=self.phase,resources=self.last_phase_debit,
            measured_phase_resources=phase,total_resources=total,
            resources_final=False,invocation=str(self.invocation),gpu_clock="total occupancy proxy includes preparation; model allowance uses model phase only"))
        save(self.invocation/"resources.json",dict(phase=self.phase,resources=total,phase_resources=phase,resources_final=False))
        return total

    def reason(self, stats):
        phase=self.last_phase_debit or self.last_phase or {}
        limits=self.budget[self.phase]
        # New closure reserves cancellation + ordinary settlement separately
        # from the terminal receipt's 20-second debit. Both stay inside caps.
        margin=self.budget.get("cleanup_margin_seconds",self.budget["stop_margin_seconds"])
        for cap,field in (("wall_seconds","wall_seconds"),("cpu_seconds","cpu_seconds"),("gpu_seconds","gpu_occupancy_seconds")):
            if cap in limits and phase.get(field,0)>=limits[cap]-margin:
                return "Stopping margin reached: "+self.phase+" "+field
        debit=getattr(self,"budget_carry",self.carry)
        for cap,field in (("global_wall_seconds","wall_seconds"),("global_cpu_seconds","cpu_seconds"),("global_gpu_seconds","gpu_occupancy_seconds")):
            if debit[field]+self.prior.get("budget_resources",self.prior["resources"]).get(field,0)+stats[field]>=self.m["budget"][cap]-margin:
                return "Stopping margin reached: "+cap
        for field,cap,reserve in (("peak_ram_bytes","ram_bytes",512*1024**2),
                                  ("peak_device_used_bytes","device_bytes",256*1024**2)):
            if stats[field]>=self.m["budget"][cap]-reserve:return "Stopping margin reached: "+field
        # Conservative live bound; no Drive enumeration in the polling loop.
        if self.archive_bytes+size(self.root)+self.carry["result_bytes"]>=self.m["budget"]["result_bytes"]-64*1024**2:
            return "Stopping margin reached: result_bytes"
        return None

    def stop(self,p):
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)  # Owner reaps all descendants before sealing.
            try:p.wait(timeout=12)
            except subprocess.TimeoutExpired:
                raise RuntimeError("Accounting owner failed to seal; no automatic continuation")

    def command(self,stage,args):
        reason=self.reason(self.usage())
        if reason:raise RuntimeError(reason)
        receipt=self.invocation/(stage+".usage.json")
        self.stages.append(receipt)
        print(stage+" ...",flush=True)
        with (self.invocation/(stage+".log")).open("x") as log:
            child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
            query_start=self.query_cpu
            p=subprocess.Popen([sys.executable,"-m","wam_qualification.metered",str(receipt),*map(str,args)],
                env=self.env,cwd=PACKAGE.parent,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while p.poll() is None:
                    reason=self.reason(self.usage())
                    if reason:raise RuntimeError(reason)
                    time.sleep(.5)
                if p.returncode:raise RuntimeError(f"{stage} exited {p.returncode}; see invocations/{self.invocation.name}/{stage}.log")
            finally:
                self.stop(p)
                if receipt.exists():
                    r=read_receipt(receipt)
                    if r.get("resources_final"):
                        child_end=resource.getrusage(resource.RUSAGE_CHILDREN)
                        waited=child_end.ru_utime+child_end.ru_stime-child_start.ru_utime-child_start.ru_stime-(self.query_cpu-query_start)
                        r["cpu_seconds"]=max(r["cpu_seconds"],waited)
                        r["owner_exit_cpu_included"]=True
                        save(receipt,r)
                        self.publish_stage(stage)
        r=read_receipt(receipt)
        if not r.get("resources_final") or not r.get("all_descendants_reaped"):
            raise RuntimeError("Stage resource receipt is not sealed")
        reason=self.reason(self.usage())
        if reason:raise RuntimeError(reason)

    def publish_stage(self, stage):
        # The stage owner has exited and all descendants have been reaped.
        # Only a sealed checkpoint crosses Drive; no worker reads it back.
        stats=self.usage()
        save(self.invocation/(stage+".checkpoint.json"),dict(
            sealed_checkpoint=True,execution_resources_final=False,
            resources=stats,phase_resources=self.last_phase,
            prior_resources=self.prior["resources"],phase=self.phase))
        publish_tree(self.invocation,self.archive_invocation,exclude=("resources.json",))
        self.archive_bytes=tree_bytes(self.archive_root)

    def enter_model(self):
        reason=self.reason(self.usage())
        if reason:raise RuntimeError(reason)
        self.preparation_fixed=dict(self.last_phase)
        self.preparation_debit=dict(self.last_phase_debit)
        self.phase="model";self.phase_started=self.start+self.last_total["wall_seconds"];self.phase_cpu_base=self.cpu_high
        self.model_start=self.phase_started
        # Reserve before loading a single real checkpoint, even when no physical
        # image operation has yet started. A model phase is never retried.
        with (self.archive_root/self.model_reservation_name).open("x") as f:
            json.dump(dict(started_unix=self.model_start,invocation=self.invocation.name),f)
            f.flush();os.fsync(f.fileno())
        if self.root!=self.archive_root:
            save(self.root/self.model_reservation_name,dict(started_unix=self.model_start,invocation=self.invocation.name))
        save(self.root/"runtime_entry.json",dict(effective_limits=self.budget["model"],
            started_unix=self.model_start,phase="model",preparation_excluded_from_model_allowance=True))
        self.usage()

    def finish(self,error):
        sealed=all(p.exists() and read_receipt(p).get("resources_final") and read_receipt(p).get("all_descendants_reaped") for p in self.stages)
        rows=json.loads((self.root/"operation_status.json").read_text())
        for row in rows:
            if row["status"]=="STARTED":row["status"]="INTERRUPTED"
        save(self.root/"operation_status.json",rows)
        # Publish bulk results/logs first, so their IO is included in final time.
        publish_tree(self.root,self.archive_root,exclude=("resources.json","summary.json"))
        stats=self.usage()
        error=error or self.reason(stats)
        totals={k:dict(denominator=v,started=sum(r["kind"]==k and r["status"]!="NOT_STARTED" for r in rows),
            completed=sum(r["kind"]==k and r["status"]=="COMPLETED" for r in rows)) for k,v in self.m["physical_limits"].items()}
        phases=dict(self.prior["phases"])
        phases[self.phase]=dict(self.last_phase)
        if self.preparation_fixed is not None:phases["preparation"]=self.preparation_fixed
        invocation=dict(status="ENVIRONMENT_STOPPED" if self.phase=="preparation" else "MODEL_STOPPED",
            resources=stats,resources_final=sealed,error=error,phase=self.phase,phases=phases)
        resources=combine(self.prior["resources"],stats)
        resources["result_bytes"]=0  # Filled after every final receipt is present.
        cumulative=combine(self.carry,resources)
        numeric=self.root/"numeric_result.json"
        result=json.loads(numeric.read_text()) if numeric.exists() else dict(qualification_pass=False)
        if not error and sealed and numeric.exists():invocation["status"]=result["status"]
        if error or not sealed:
            result.update(status=invocation["status"] if sealed else "ACCOUNTING_UNRESOLVED",qualification_pass=False)
        result.update(error=error,output=str(self.archive_root),operations=totals,resources=resources,
            cumulative_resources=cumulative,inherited_resources=self.carry,phases=phases,
            resources_final=sealed,historical_cpu_exact=False,cumulative_cpu_is_lower_bound=True,
            inherited_trajectories_started=44,inherited_trajectories_completed=44,
            scientific_pass=None,full_baseline_authorized=False,training_authorized=False,cal_test_authorized=False,
            checkpoint_history="UNVERIFIED",unseen_content="USER_ACCEPTED_ASSUMPTION",
            environment_resume_allowed=bool(sealed and self.phase=="preparation" and not self.reason(stats)),
            model_phase_may_retry=False)
        result.update(execution_budget=self.budget,original_budget_pass_claimed=False,
            cumulative_budget_debits=combine(getattr(self,"budget_carry",self.carry),resources),
            gpu_clock="cumulative allocation proxy includes all preparation; only model phase consumes model GPU allowance")
        if "preparation_amendment" in self.budget:
            result.update(preparation_amendment=self.budget["preparation_amendment"],
                engineering_closure_consumed=True,environment_resume_allowed=False)
        # First settle the full ordinary receipts while still metering. Their
        # complete IO is included in the following measurement, unlike v1.
        result["result_volume_measurement"]="sealed attempt tree bytes, excludes removed active.lock; historical invocations retained"
        result["resources_final"]=False
        invocation["resources_final"]=False
        self.write_settlement(result,stats,invocation)
        stats.update(self.usage())
        phases[self.phase]=dict(self.last_phase)
        resources.update(combine(self.prior["resources"],stats))
        cumulative.update(combine(self.carry,resources))
        phase_debits=dict(self.prior.get("phase_budget_debits",self.prior["phases"]))
        phase_debits[self.phase]=dict(self.last_phase_debit)
        if self.preparation_debit is not None:phase_debits["preparation"]=self.preparation_debit
        budget_resources=combine(self.prior.get("budget_resources",self.prior["resources"]),stats)
        # A receipt cannot contain the measured duration of writing itself.
        # Debit the existing stopping margin before the terminal commit, then
        # verify its wall/CPU duration before releasing active.lock. This is a
        # reserve, not fabricated measured CPU. Exceeding it is unresolved.
        reserve=self.budget["stop_margin_seconds"]
        for field in ("wall_seconds","gpu_occupancy_seconds","cpu_seconds"):
            budget_resources[field]+=reserve
            phase_debits[self.phase][field]+=reserve
        result.update(resources_final=sealed,budget_resources=budget_resources,phase_budget_debits=phase_debits,
            cumulative_budget_debits=combine(getattr(self,"budget_carry",self.carry),budget_resources),
            terminal_seal=dict(reserved_wall_seconds=reserve,reserved_cpu_seconds=reserve,
                measurement="bulk results and ordinary settlement measured; terminal receipt self-write reserved",
                completion_proof="active.lock is released only after terminal commit fits its reserved wall and CPU"))
        if getattr(self,"technical_continuation",False):
            result.update(technical_continuation_consumed=True,environment_resume_allowed=False,
                historical_sealing_tail="UNMEASURED; explicit uncertainty reserves retained in budget debits")
        final_reason=self.reason(stats)
        for cap,field in (("wall_seconds","wall_seconds"),("cpu_seconds","cpu_seconds"),("gpu_seconds","gpu_occupancy_seconds")):
            if cap in self.budget[self.phase] and phase_debits[self.phase][field]>self.budget[self.phase][cap]:
                final_reason=final_reason or "Terminal seal reserve exceeds phase budget: "+field
        for cap,field in (("global_wall_seconds","wall_seconds"),("global_cpu_seconds","cpu_seconds"),("global_gpu_seconds","gpu_occupancy_seconds")):
            if result["cumulative_budget_debits"][field]>self.m["budget"][cap]:
                final_reason=final_reason or "Terminal seal reserve exceeds global budget: "+field
        if final_reason:
            result.update(error=error or final_reason,status="MODEL_STOPPED" if self.phase=="model" else "ENVIRONMENT_STOPPED",
                qualification_pass=False,environment_resume_allowed=False)
        invocation.update(resources_final=sealed,status=result.get("status",invocation["status"]),
            error=result["error"],phase_budget_debits=phase_debits,terminal_seal=result["terminal_seal"])
        self.terminal_start=time.monotonic();self.terminal_cpu=own_cpu();self.terminal_reserve=reserve
        self.write_settlement(result,stats,invocation)
        self.verify_terminal()
        return result

    def verify_terminal(self):
        self.terminal_observed=dict(wall_seconds=time.monotonic()-self.terminal_start,
            cpu_seconds=max(0,own_cpu()-self.terminal_cpu))
        if max(self.terminal_observed.values())>self.terminal_reserve:
            raise RuntimeError("Terminal seal exceeded reserved margin; retain lock and observed costs, no retry")

    def write_settlement(self,result,stats,invocation):
        resources=result["resources"]
        cumulative=result["cumulative_resources"]
        for _ in range(16):
            save(self.archive_invocation/"resources.json",dict(phase=self.phase,resources=stats,
                phase_resources=self.last_phase,resources_final=result["resources_final"]))
            save(self.archive_invocation/"summary.json",invocation)
            save(self.archive_root/"resources.json",dict(stage=self.phase,resources=resources,
                phase_resources=result["phases"],resources_final=result["resources_final"]))
            save(self.archive_root/"summary.json",result)
            actual=tree_bytes(self.archive_root)
            inv_bytes=tree_bytes(self.archive_invocation)
            if actual==resources["result_bytes"] and inv_bytes==stats["result_bytes"]:
                return
            stats["result_bytes"]=inv_bytes
            resources["result_bytes"]=actual
            cumulative["result_bytes"]=self.carry["result_bytes"]+actual
            result["cumulative_budget_debits"]["result_bytes"]=getattr(self,"budget_carry",self.carry)["result_bytes"]+actual
            if "budget_resources" in result:result["budget_resources"]["result_bytes"]=actual
        raise RuntimeError("Final result volume did not converge; accounting remains unresolved")
