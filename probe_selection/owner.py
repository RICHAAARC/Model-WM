"""Metered one-shot owner using the validated LaWa preparation/model supervisor."""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from latent_control.io import save
from lawa_ab.entry import Meter, gpu_ready, owner_cpu, read, tree_bytes

def add(old,new):
    totals=dict(old)
    for key,value in new.items():
        previous=old.get(key,0)
        totals[key]=max(previous,value) if key.startswith('peak_') else previous+value
    return totals


def read_only(out,denominator=16):
    if (out/'summary.json').is_file():
        return read(out/'summary.json')
    observations=sorted((out/'receipts').glob('*.json'))
    last=read(observations[-1]) if observations else None
    history=read(out/'history.json') if (out/'history.json').exists() else {}
    counts=tallies(out,denominator)
    lower=dict(last['resources']) if last else {}
    lower['result_bytes']=max(lower.get('result_bytes',0),tree_bytes(out))
    return dict(status='ACTIVE_OR_INTERRUPTED_READ_ONLY',output=str(out),last_observation=last,
                cumulative_lower_bound=add(history.get('cumulative_resources',{}),lower),
                cumulative_debit_lower_bound=add(history.get('cumulative_budget_debits',{}),lower),
                unknown_tail=True,counts=counts,retry=False)


def tallies(out,denominator=16):
    return dict(started=sum(1 for _ in (out/'work_receipts').glob('*/started.json')),
                completed=sum(1 for _ in (out/'work_receipts').glob('*/completed.json')),denominator=denominator)


def run_colab(parent='/content/drive/MyDrive/Model-WM',runtime='/content/model-wm-runtime',
              started=None,cpu_start=None,*,run_id,package,
              history_reader,denominator,
              artifacts=('freeze.json','PREREGISTRATION.md','schema.json'),
              usage_reader=None):
    started=time.monotonic() if started is None else started
    cpu_start=owner_cpu() if cpu_start is None else cpu_start
    parent=Path(parent)
    out=parent/run_id
    if out.exists():
        return read_only(out,denominator)
    if not gpu_ready():
        return dict(status='GPU_REQUIRED',output=str(out))
    parent.mkdir(parents=True,exist_ok=True)
    try:
        out.mkdir()
    except FileExistsError:
        return read_only(out,denominator)
    result=dict(status='STOPPED',retry=False)
    meter=None
    history={}
    scratch=Path(runtime)/run_id
    local=scratch/'records'
    try:
        save(out/'reservation.json',dict(time=time.time(),run_id=run_id,retry=False))
        history=history_reader(parent)
        save(out/'history.json',history)
        for n in artifacts:
            shutil.copyfile(package/n,out/n)
        local.mkdir(parents=True,exist_ok=False)
        cache=scratch/'cache'
        cache.mkdir()
        meter=Meter({},local,out,scratch,cache,started,cpu_start)
        env=dict(os.environ,PYTHONPATH=str(package.parent),HF_HOME=str(scratch/'hf'),
                 TORCH_HOME=str(scratch/'torch'),XDG_CACHE_HOME=str(scratch/'xdg'),
                 PIP_CACHE_DIR=str(scratch/'pip'),MPLBACKEND='Agg',OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
        meter.stage='preparation'
        meter.command('preparation',[sys.executable,'-m',package.name+'.assets',local,scratch],env)
        py=read(local/'python.json')['python']
        meter.stage='model'
        meter.command('model',[py,'-m',package.name+'.worker',local,out,scratch],env)
        result=read(local/'outcome.json')
    except BaseException as exc:
        (out/'exception.txt').write_text(traceback.format_exc())
        result.update(error=repr(exc))
    if local.is_dir():
        for p in local.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.log'):
                shutil.copyfile(p,out/p.name)
    if usage_reader is not None:
        result['work_resources']=usage_reader(out)
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
                  scientific_pass=None,counts=tallies(out,denominator),stages=stages,
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
