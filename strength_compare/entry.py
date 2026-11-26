"""Colab owner. Runtime files stay local; Drive receives immutable observations/results."""
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

from latent_control.io import save
from lawa_ab.entry import gpu_ready, owner_cpu, read, tree_bytes
from probe_selection.owner import add
PREVIOUS='lawa-strength-bracket-dev-20261125-attempt01'

PACKAGE=Path(__file__).resolve().parent
RUN_ID='lawa-strength-bracket-compare-20261126-attempt01'
LIMITS=dict(generate=184,write=4416,mix=4416,read=18032)


def check_freeze():
    seal=read(PACKAGE/'receipt.json')
    for name,expected in seal['sha256'].items():
        if hashlib.sha256((PACKAGE.parent/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Preregistration file changed: '+name)
    return read(PACKAGE/'allocation.json')['budget']


def prior(budget,parent):
    from .gate import validate_dev
    path=parent/PREVIOUS
    old=read(path/'summary.json')
    validate_dev(old,read(path/'quality_records.json'),read(path/'allocation.json'))
    baseline=read(PACKAGE/'dev_summary.json')
    result=dict(source='verified sealed DEV and audited historical snapshot')
    for key in ('cumulative_resources','cumulative_budget_debits'):
        result[key]={k:max(v,old[key][k]) for k,v in baseline[key].items()}
    return result


def counts(out):
    return dict(works_reserved=len(list((out/'works').glob('*/reserved.json'))),
                works_completed=len(list((out/'works').glob('*/completed.json'))),
                slots_completed=len(list((out/'slots').glob('*/*/completed.json'))),
                works_planned=184,cal_planned=24,test_planned=160,slots_planned=0)


def operation_counts(local, live=False):
    ops={k:dict(started=0,completed=0,denominator=v) for k,v in LIMITS.items()}
    ledger=local/'operations.jsonl'
    if ledger.is_file():
        with ledger.open('rb') as f:
            for line in f:
                if live and not line.endswith(bytes([10])):
                    break
                event=json.loads(line)
                if event['state'] in ('STARTED','COMPLETED'):
                    ops[event['kind']][event['state'].lower()]+=1
    return ops


def replay(out,budget):
    failure=out/'settlement_failure.json'
    if (out/'summary.json').is_file() and not failure.exists():
        return read(out/'summary.json')
    history=read(out/'history.json') if (out/'history.json').exists() else prior(budget,out.parent)
    receipts=sorted((out/'receipts').glob('*.json'))
    last=read(receipts[-1]) if receipts else {}
    cost=dict(last.get('resources',{}))
    if failure.exists():
        for k,v in read(failure).get('resources',{}).items():
            cost[k]=max(cost.get(k,0),v)
    cost['result_bytes']=max(cost.get('result_bytes',0),tree_bytes(out))
    debit=dict(cost)  # Unobserved tail stays unknown; no invented time-cap debit.
    return dict(status='SETTLEMENT_INCOMPLETE_READ_ONLY' if failure.exists() else 'ACTIVE_OR_INTERRUPTED_READ_ONLY',output=str(out),counts=counts(out),
                last_observation=last,unknown_tail=True,retry=False,
                cumulative_lower_bound=add(history['cumulative_resources'],cost),
                cumulative_conservative_debit=add(history['cumulative_budget_debits'],debit))


class Monitor:
    def __init__(self,start,cpu,local,out,cache):
        self.start,self.cpu,self.local,self.out,self.cache=start,cpu,local,out,cache
        self.child=0.
        self.peak_ram=self.peak_device=self.download=self.cpu_high=self.sequence=0
        self.stages={}
        self.complete=True

    def sample(self,live=None,scan=False):
        live=live or {}
        self.cpu_high=max(self.cpu_high,owner_cpu()-self.cpu+self.child+live.get('cpu_seconds',0))
        self.peak_ram=max(self.peak_ram,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+
                          live.get('peak_ram_bytes',0))
        if scan:
            self.download=max(self.download,tree_bytes(self.cache))
        elapsed=time.monotonic()-self.start
        return dict(wall_seconds=elapsed,gpu_occupancy_seconds=elapsed,cpu_seconds=self.cpu_high,
                    peak_ram_bytes=self.peak_ram,peak_device_used_bytes=self.peak_device,
                    download_charge_bytes=self.download)

    def snapshot(self,stage,r):
        self.sequence+=1
        progress=self.local/'progress.json'
        row=dict(stage=stage,resources=r,progress=read(progress) if progress.is_file() else {},
                 operations_observed=operation_counts(self.local,live=True))
        save(self.out/'receipts'/f'{self.sequence:06}.json',row)
        print(stage,row['progress'],f"{r['wall_seconds']:.0f}s",flush=True)

    def command(self,stage,cmd,env):
        usage=self.local/(stage+'.usage.json')
        begin=self.sample()
        if stage=='preparation':
            begin.update(wall_seconds=0.,gpu_occupancy_seconds=0.,cpu_seconds=0.)
        live={}
        p=None
        try:
            with (self.local/(stage+'.log')).open('x') as log:
                with nullcontext():
                    p=subprocess.Popen([sys.executable,'-m','wam_qualification.metered',str(usage),
                                       *map(str,cmd)],env=env,stdout=log,stderr=subprocess.STDOUT)
                    next_obs=0
                    while p.poll() is None:
                        if time.monotonic()>=next_obs:
                            live=read(usage) if usage.is_file() else {}
                            try:
                                mem=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                                                   capture_output=True,text=True,timeout=2)
                                if mem.returncode==0:
                                    self.peak_device=max(self.peak_device,sum(int(x)*1024**2 for x in mem.stdout.split()))
                            except (OSError,subprocess.TimeoutExpired):
                                pass
                            self.snapshot(stage,self.sample(live,True))
                            next_obs=time.monotonic()+15
                        time.sleep(.2)
                if p.returncode:
                    raise RuntimeError(f'{stage} exited {p.returncode}; see {stage}.log')
        finally:
            if p is not None and p.poll() is None:
                p.terminate()
                p.wait(timeout=12)
            live=read(usage) if usage.exists() else {}
            self.complete &= bool(live.get('resources_final') and live.get('all_descendants_reaped'))
            self.child+=live.get('cpu_seconds',0)
            end_resources=self.sample(scan=True)
            self.stages[stage]={k:end_resources[k]-begin[k] for k in
                                ('wall_seconds','gpu_occupancy_seconds','cpu_seconds','download_charge_bytes')}
            self.stages[stage]['all_descendants_reaped']=bool(live.get('all_descendants_reaped'))
            self.snapshot(stage,end_resources)
        if not self.complete:
            raise RuntimeError('Subprocess accounting incomplete')


def run_colab(parent='/content/drive/MyDrive/Model-WM',runtime='/content/model-wm-runtime',
              started=None,cpu_start=None):
    start=time.monotonic() if started is None else started
    cpu=owner_cpu() if cpu_start is None else cpu_start
    budget=check_freeze()
    caps=None  # Fixed work, no arbitrary wall-time stop
    parent=Path(parent)
    out=parent/RUN_ID
    if out.exists():
        return replay(out,budget)
    with nullcontext():
        if not gpu_ready():
            return dict(status='GPU_REQUIRED',reserved=False,output=str(out))
        history=prior(budget,parent)
        parent.mkdir(parents=True,exist_ok=True)
    try:
        out.mkdir()
    except FileExistsError:
        return replay(out,budget)
    scratch=Path(runtime)/RUN_ID
    local=scratch/'records'
    cache=scratch/'cache'
    meter=Monitor(start,cpu,local,out,cache)
    result=dict(status='FORMAL_STOPPED_INCOMPLETE',scientific_pass=None)
    settled=False
    try:
        with nullcontext():
            save(out/'reservation.json',dict(at=time.time(),run_id=RUN_ID,limits=LIMITS,
                                             phase_caps_seconds=caps,retry=False))
            save(out/'history.json',history)
            local.mkdir(parents=True,exist_ok=False)
            cache.mkdir()
            for f in ('receipt.json','allocation.json','PREREGISTRATION.md','SOURCE_AUDIT.md','dev_summary.json','dev_quality.json','dev_visual_review.json','FORMAL_ENTRY.md'):
                shutil.copyfile(PACKAGE/f,out/f)
            env=dict(os.environ,PYTHONPATH=str(PACKAGE.parent),HF_HOME=str(scratch/'hf'),
                     TORCH_HOME=str(scratch/'torch'),XDG_CACHE_HOME=str(scratch/'xdg'),
                     PIP_CACHE_DIR=str(scratch/'pip'),MPLBACKEND='Agg',OMP_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        meter.command('preparation',[sys.executable,'-m','strength_bracket.assets',local,scratch],
                      env)
        py=read(local/'python.json')['python']
        model_start=time.monotonic()
        meter.command('model',[py,'-m','strength_compare.worker',local,out,scratch],
                      env)
        result.update(read(local/'outcome.json'))
    except BaseException as exc:
        result.update(status='FORMAL_STOPPED_INCOMPLETE',
                      error=repr(exc),traceback=traceback.format_exc())
    settlement_start=time.monotonic()
    try:
        with nullcontext():
            for p in local.glob('*'):
                if p.is_file() and p.suffix in ('.log','.jsonl','.json'):
                    shutil.copyfile(p,out/p.name)
            if (local/'slot').is_dir():
                shutil.copytree(local/'slot',out/'interrupted_slot')
            result['operations']=operation_counts(local,live=True)
            result['counts']=counts(out)
            size=tree_bytes(out)
            r=meter.sample(scan=True)
            phases=dict(meter.stages)
            phases['settlement_and_entry_tail']={k:max(0,r[k]-sum(s[k] for s in meter.stages.values()))
                                                   for k in ('wall_seconds','gpu_occupancy_seconds','cpu_seconds','download_charge_bytes')}
            result.update(output=str(out),stages=phases,resources=r,resources_final=meter.complete,
                          retry=False,training_updates=0,cal_test_started=bool(list((out/'works').glob('*/reserved.json'))),holdout_unsealed=False,old_holdout_unsealed=False,dev_recomputed=False,preparation_resume_automatic=False,
                          historical_cpu_exact=False,phase_caps_seconds=caps,estimated_minutes=budget['estimated_minutes'],
                          accounting='Total allocation clock includes setup and I/O; settlement sampled through artifact scan. Download is local retained payload high-water, excluding reused symlinks.')
            r['result_bytes']=size
            for k in ('cumulative_resources','cumulative_budget_debits'):
                result[k]=add(history[k],r)
            for _ in range(20):
                volume=size+len((json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n').encode())
                diff=volume-r['result_bytes']
                if not diff:break
                for k in ('resources','cumulative_resources','cumulative_budget_debits'):
                    result[k]['result_bytes']+=diff
            save(out/'summary.json',result)
            settled=True
    except BaseException as exc:
        # Preserve the final observed floor even if a Drive write or the VM fails.
        save(out/'settlement_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
                                                resources=meter.sample(),result_status=result['status']))
        settled=False
    if not settled or (out/'settlement_failure.json').exists():
        return replay(out,budget)
    return result
