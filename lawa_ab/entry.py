"""Single user-started run, local metering IPC and durable no-retry reservation."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from latent_control.io import save
from wam_method.entry import Owner, owner_cpu, tree_bytes

PACKAGE=Path(__file__).resolve().parent
RUN_ID='lawa-ab-20261119-attempt01'
PRIOR='terminal-solver-20261119-attempt01/continuation-01/summary.json'


def read(path):
    return json.loads(path.read_text())


def add(old,new):
    return {k:max(old.get(k,0),v) if k.startswith('peak_') else old.get(k,0)+v for k,v in new.items()}


def prior_history(parent):
    snapshot=read(PACKAGE/'prior_receipt.json')
    path=parent/PRIOR
    if not path.exists():
        return dict(snapshot,source='published continuation-01 receipt supplied by user')
    actual=read(path)
    if not actual.get('resources_final'):
        raise ValueError('Previous project accounting is incomplete')
    result=dict(source='Drive continuation-01 with published historical lower bound')
    for k in ('cumulative_resources','cumulative_budget_debits'):
        recorded=actual[k]
        result[k]={field:max(value,recorded.get(field,0)) for field,value in snapshot[k].items()}
    return result


class Meter(Owner):
    def check(self,r,stage,elapsed,progress):
        pass  # Report resource use; there is no global time/volume ceiling.

    def receipt(self,r,progress):
        self.sequence+=1
        save(self.archive/'receipts'/f'{self.sequence:06}.json',
             dict(resources=r,stage=self.stage,progress=progress,observed=True))

    def command(self,stage,command,env):
        before=self.resources()
        try:
            super().command(stage,command,env)
        finally:
            self.volume_sample=-float('inf')
            after=self.resources()
            row=self.stage_resources.setdefault(stage,{})
            for field in ('wall_seconds','gpu_occupancy_seconds','cpu_seconds','download_charge_bytes'):
                row[field]=after[field]-before[field]


def gpu_ready():
    try:
        return bool(subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                                            timeout=5,text=True).strip())
    except (OSError,subprocess.SubprocessError):
        return False


def read_only(out):
    if (out/'summary.json').is_file():
        return read(out/'summary.json')
    observations=sorted((out/'receipts').glob('*.json'))
    last=read(observations[-1]) if observations else None
    history=read(out/'history.json') if (out/'history.json').exists() else {}
    counts=tallies(out)
    return dict(status='ACTIVE_OR_INTERRUPTED_READ_ONLY',output=str(out),last_observation=last,
                cumulative_lower_bound=add(history.get('cumulative_resources',{}),last['resources'])
                if last else history.get('cumulative_resources'),unknown_tail=True,counts=counts,retry=False)


def tallies(out):
    return dict(started=sum(1 for _ in (out/'work_receipts').glob('*/started.json')),
                completed=sum(1 for _ in (out/'work_receipts').glob('*/completed.json')),denominator=16)


def run_colab(parent='/content/drive/MyDrive/Model-WM',runtime='/content/model-wm-runtime',
              started=None,cpu_start=None):
    started=time.monotonic() if started is None else started
    cpu_start=owner_cpu() if cpu_start is None else cpu_start
    parent=Path(parent)
    out=parent/RUN_ID
    if out.exists():
        return read_only(out)
    if not gpu_ready():
        return dict(status='GPU_REQUIRED',output=str(out))
    parent.mkdir(parents=True,exist_ok=True)
    try:
        out.mkdir()
    except FileExistsError:
        return read_only(out)
    result=dict(status='STOPPED',retry=False)
    meter=None
    history={}
    scratch=Path(runtime)/RUN_ID
    local=scratch/'records'
    try:
        save(out/'reservation.json',dict(time=time.time(),run_id=RUN_ID,retry=False))
        history=prior_history(parent)
        save(out/'history.json',history)
        for n in ('freeze.json','PREREGISTRATION.md','schema.json'):
            shutil.copyfile(PACKAGE/n,out/n)
        local.mkdir(parents=True,exist_ok=False)
        cache=scratch/'cache'
        cache.mkdir()
        meter=Meter({},local,out,scratch,cache,started,cpu_start)
        env=dict(os.environ,PYTHONPATH=str(PACKAGE.parent),HF_HOME=str(scratch/'hf'),
                 TORCH_HOME=str(scratch/'torch'),XDG_CACHE_HOME=str(scratch/'xdg'),
                 PIP_CACHE_DIR=str(scratch/'pip'),MPLBACKEND='Agg',OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
        meter.stage='preparation'
        meter.command('preparation',[sys.executable,'-m','lawa_ab.assets',local,scratch],env)
        py=read(local/'python.json')['python']
        meter.stage='model'
        meter.command('model',[py,'-m','lawa_ab.worker',local,out,scratch],env)
        result=read(local/'outcome.json')
    except BaseException as exc:
        (out/'exception.txt').write_text(traceback.format_exc())
        result.update(error=repr(exc))
    if local.is_dir():
        for p in local.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.log'):
                shutil.copyfile(p,out/p.name)
    size=tree_bytes(out)
    if meter:
        meter.volume_sample=-float('inf')
        r=meter.resources()
    else:
        elapsed=time.monotonic()-started
        r=dict(wall_seconds=elapsed,gpu_occupancy_seconds=elapsed,cpu_seconds=owner_cpu()-cpu_start)
    r['result_bytes']=size
    phases={p.stem:read(p) for p in sorted((out/'phases').glob('*.json'))}
    stages=dict(meter.stage_resources) if meter else {}
    stages['entry_and_settlement']={k:max(0,r.get(k,0)-sum(s.get(k,0) for s in stages.values()))
                                    for k in ('wall_seconds','gpu_occupancy_seconds','cpu_seconds','download_charge_bytes')}
    result.update(output=str(out),retry=False,holdout_unsealed=False,training_updates=0,
                  scientific_pass=None,counts=tallies(out),stages=stages,
                  worker_phases=phases,resources=r,resources_final=bool(meter and meter.accounting_complete),
                  historical_cpu_exact=False,accounting='GPU seconds are allocation wall time. Owner includes preparation, model and settlement through the archive scan. Worker phases are nested in model. Download charge is new cache volume, not packet bytes.')
    for k in ('cumulative_resources','cumulative_budget_debits'):
        result[k]=add(history[k],r) if k in history else None
    for _ in range(16):
        measured=size+len((json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n').encode())
        delta=measured-r['result_bytes']
        if not delta:
            break
        for k in ('resources','cumulative_resources','cumulative_budget_debits'):
            if result[k] is not None:
                result[k]['result_bytes']+=delta
    save(out/'summary.json',result)
    return result
