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


def own_cpu():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime+r.ru_stime


def combine(a,b):
    return {k:max(a.get(k,0),v) if k.startswith("peak_") else a.get(k,0)+v
        for k,v in b.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}


class Supervisor(Legacy):
    def __init__(self, root, invocation, manifest, carry, prior, budget, start, cpu_start, cache, baseline):
        super().__init__(root,manifest,carry,start,cpu_start,cache,limits=budget["model"],cache_baseline=baseline)
        self.invocation,self.prior,self.budget=invocation,prior,budget
        self.phase="preparation";self.phase_started=start
        self.phase_cpu_base=0.;self.model_start=None
        self.cpu_high=0.;self.query_cpu=0.;self.stages=[]
        self.preparation_fixed=None
        self.last_total=None;self.last_phase=None

    def usage(self, process=None):
        core=max(0,own_cpu()-self.cpu_base)+self.query_cpu
        ram=self.psutil.Process().memory_info().rss
        for path in self.stages:
            if path.exists():
                r=json.loads(path.read_text());core+=r["cpu_seconds"]
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
        save(self.root/"resources.json",dict(stage=self.phase,resources=phase,total_resources=total,
            resources_final=False,invocation=str(self.invocation),gpu_clock="total occupancy proxy includes preparation; model allowance uses model phase only"))
        save(self.invocation/"resources.json",dict(phase=self.phase,resources=total,phase_resources=phase,resources_final=False))
        return total

    def reason(self, stats):
        phase=self.last_phase or {}
        limits=self.budget[self.phase]
        for cap,field in (("wall_seconds","wall_seconds"),("cpu_seconds","cpu_seconds"),("gpu_seconds","gpu_occupancy_seconds")):
            if cap in limits and phase.get(field,0)>=limits[cap]-20:
                return "Stopping margin reached: "+self.phase+" "+field
        debit=getattr(self,"budget_carry",self.carry)
        for cap,field in (("global_wall_seconds","wall_seconds"),("global_cpu_seconds","cpu_seconds"),("global_gpu_seconds","gpu_occupancy_seconds")):
            if debit[field]+self.prior["resources"].get(field,0)+stats[field]>=self.m["budget"][cap]-20:
                return "Stopping margin reached: "+cap
        for field,cap,reserve in (("peak_ram_bytes","ram_bytes",512*1024**2),
                                  ("peak_device_used_bytes","device_bytes",256*1024**2)):
            if stats[field]>=self.m["budget"][cap]-reserve:return "Stopping margin reached: "+field
        if size(self.root)+self.carry["result_bytes"]>=self.m["budget"]["result_bytes"]-64*1024**2:
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
                    r=json.loads(receipt.read_text())
                    if r.get("resources_final"):
                        child_end=resource.getrusage(resource.RUSAGE_CHILDREN)
                        waited=child_end.ru_utime+child_end.ru_stime-child_start.ru_utime-child_start.ru_stime-(self.query_cpu-query_start)
                        r["cpu_seconds"]=max(r["cpu_seconds"],waited)
                        r["owner_exit_cpu_included"]=True
                        save(receipt,r)
        r=json.loads(receipt.read_text())
        if not r.get("resources_final") or not r.get("all_descendants_reaped"):
            raise RuntimeError("Stage resource receipt is not sealed")
        reason=self.reason(self.usage())
        if reason:raise RuntimeError(reason)

    def enter_model(self):
        reason=self.reason(self.usage())
        if reason:raise RuntimeError(reason)
        self.preparation_fixed=dict(self.last_phase)
        self.phase="model";self.phase_started=self.start+self.last_total["wall_seconds"];self.phase_cpu_base=self.cpu_high
        self.model_start=self.phase_started
        # Reserve before loading a single real checkpoint, even when no physical
        # image operation has yet started. A model phase is never retried.
        with (self.root/"model_phase_started.json").open("x") as f:
            json.dump(dict(started_unix=self.model_start,invocation=self.invocation.name),f)
        save(self.root/"runtime_entry.json",dict(effective_limits=self.budget["model"],
            started_unix=self.model_start,phase="model",preparation_excluded_from_model_allowance=True))
        self.usage()

    def finish(self,error):
        stats=self.usage()
        sealed=all(p.exists() and json.loads(p.read_text()).get("resources_final") for p in self.stages)
        rows=json.loads((self.root/"operation_status.json").read_text())
        for row in rows:
            if row["status"]=="STARTED":row["status"]="INTERRUPTED"
        save(self.root/"operation_status.json",rows)
        totals={k:dict(denominator=v,started=sum(r["kind"]==k and r["status"]!="NOT_STARTED" for r in rows),
            completed=sum(r["kind"]==k and r["status"]=="COMPLETED" for r in rows)) for k,v in self.m["physical_limits"].items()}
        phases=dict(self.prior["phases"])
        phases[self.phase]=dict(self.last_phase)
        if self.preparation_fixed is not None:phases["preparation"]=self.preparation_fixed
        invocation=dict(status="ENVIRONMENT_STOPPED" if self.phase=="preparation" else "MODEL_STOPPED",
            resources=stats,resources_final=sealed,error=error,phase=self.phase,phases=phases)
        save(self.invocation/"summary.json",invocation)
        resources=combine(self.prior["resources"],stats)
        # Report all outputs, including files in the attempt root, exactly once.
        resources["result_bytes"]=size(self.root)
        cumulative=combine(self.carry,resources)
        numeric=self.root/"numeric_result.json"
        result=json.loads(numeric.read_text()) if numeric.exists() else dict(qualification_pass=False)
        if error or not sealed:
            result.update(status=invocation["status"] if sealed else "ACCOUNTING_UNRESOLVED",qualification_pass=False)
        result.update(error=error,output=str(self.root),operations=totals,resources=resources,
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
        save(self.root/"summary.json",result)
        return result
