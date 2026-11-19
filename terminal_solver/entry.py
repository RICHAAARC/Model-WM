"""One DEV selection and one CAL→TEST segment with additive reservations."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_DEFLATED
from latent_control.io import save,BudgetStop,operation_summary
from latent_control.entry import PhaseOwner,add_resources
from wam_method.entry import owner_cpu,tree_bytes,publish_closed
from wam_method.io import reconcile
from .protocol import RUN_ID,PACKAGE,BUDGET,protocol,read_json,limits

HISTORY='latent-control-20261116-attempt02'

def summary(root):
    data=read_json(root/'history.json')
    base=data
    totals={}
    debits={}
    phase_debits={'preparation':0.,'model':0.}
    volume=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    phases={'preparation':0.,'model':0.}
    segments={}
    interrupted=None
    for name in ('dev','formal'):
        segment=root/name
        if not segment.exists():
            continue
        if not (segment/'closed.json').is_file():
            interrupted=name
            observations=sorted((segment/'receipts').glob('*.json'))
            if observations:
                latest=read_json(observations[-1])['resources']
                totals=add_resources(totals,latest)
            volume+=tree_bytes(segment)
            break
        receipt=read_json(segment/'closed.json')
        totals=add_resources(totals,receipt['resources'])
        debits=add_resources(debits,receipt['budget_resources'])
        volume+=receipt['resources']['result_bytes']
        for p in phases:
            phases[p]+=receipt['phase_seconds'][p]
            phase_debits[p]+=receipt['phase_budget_debits'][p]
        segments[name]=receipt['status']
        if (segment/'overrun.json').exists():
            tail=read_json(segment/'overrun.json')['seconds']
            charged_phase='model' if receipt['phase_seconds']['model'] else 'preparation'
            phases[charged_phase]+=tail
            phase_debits[charged_phase]+=max(0.,tail-30.)
            for key in ('wall_seconds','gpu_occupancy_seconds'):
                totals[key]+=tail
                debits[key]+=max(0.,tail-30.)
            volume+=(segment/'overrun.json').stat().st_size
            segments[name]='STOPPED_RESOURCE'
    totals['result_bytes']=debits['result_bytes']=volume
    if interrupted:
        debits=add_resources({},totals)
        for key,cap in [('wall_seconds',11700),('gpu_occupancy_seconds',11700),('cpu_seconds',14400)]:
            debits[key]=max(debits.get(key,0),cap)
        phase_debits=dict(preparation=900,model=10800)
    return dict(segments=segments,phase_seconds=phases,phase_budget_debits=phase_debits,resources=totals,budget_resources=debits,unsealed=interrupted,
                cumulative_resources=add_resources(base['cumulative_resources'],totals),
                cumulative_budget_debits=add_resources(base['cumulative_budget_debits'],debits))

def inherited(parent):
    h=read_json(parent/HISTORY/'summary.json')
    if not h.get('resources_final'):
        raise ValueError('Unsealed historical accounting')
    return reconcile(h,read_json(PACKAGE.parent/'latent_control/manifest.json')['inherited'])

def start(root,parent):
    history=inherited(parent)
    root.mkdir(parents=True,exist_ok=False)
    save(root/'history.json',history)
    for name in ('allocation.json','source_history.json','PREREGISTRATION.md'):
        shutil.copyfile(PACKAGE/name,root/name)
    with ZipFile(root/'executed_source.zip','x',compression=ZIP_DEFLATED) as archive:
        for name in ('terminal_solver','latent_control','wam_qualification','wam_method'):
            for p in sorted((PACKAGE.parent/name).iterdir()):
                if p.is_file() and p.suffix in ('.py','.json','.txt','.npz') and not p.name.startswith(('test_','check_')):
                    archive.write(p,str(p.relative_to(PACKAGE.parent)))
    save(root/'init_complete.json',dict(run_id=RUN_ID,history=HISTORY,train=0,quality_protocol='v2_proxy_only'))

def remaining(state):
    b=dict(BUDGET)
    for phase in ('preparation','model'):
        b[phase+'_seconds']-=state['phase_budget_debits'][phase]
    b['cpu_seconds']-=state['budget_resources'].get('cpu_seconds',0)
    b['result_bytes']-=state['resources']['result_bytes']
    if min(b.values())<=0:
        raise BudgetStop('Cumulative execution budget exhausted')
    return b

def execute(root,stage,runtime,started,cpu_start):
    m=protocol()
    prior=summary(root)
    m['budgets']=remaining(prior)
    workset=[w for w in m['works'] if (w['split']=='DEV')==(stage=='dev')]
    segment=root/stage
    segment.mkdir(exist_ok=False)
    scratch=runtime/RUN_ID
    local=scratch/stage/'records'
    local.mkdir(parents=True,exist_ok=False)
    cache=scratch/'cache'
    cache.mkdir(exist_ok=True)
    own=PhaseOwner(m,local,segment,scratch,cache,started,cpu_start)
    status='STOPPED'
    try:
        save(local/'manifest.json',m)
        env=dict(os.environ,PYTHONPATH=str(PACKAGE.parent)+os.pathsep+os.environ.get('PYTHONPATH',''),
                 MPLBACKEND='Agg',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 PYTHONUNBUFFERED='1',HF_HOME=str(scratch/'hf'),TORCH_HOME=str(scratch/'torch'))
        own.command('preparation',[sys.executable,'-m','latent_control.prepare','environment',local,scratch,cache],env)
        py=read_json(local/'python.json')['python']
        own.command('model',[py,'-m','terminal_solver.worker',local,segment,stage],env)
        done=read_json(local/'worker_complete.json')
        ops=operation_summary(local/'operations.jsonl',limits(workset))
        if done['works']!=len(workset) or any(v['started']!=v['completed'] or v['failed'] for v in ops.values()):
            raise RuntimeError('Incomplete physical operations')
        for k,v in ops.items():
            if k not in ('closure_O','backward_O') and v['completed']!=v['denominator']:
                raise RuntimeError('Missing operation '+k)
        if ops['closure_O']['completed']!=ops['backward_O']['completed']:
            raise RuntimeError('Ordinary closure/VJP accounting mismatch')
        save(local/'operations_summary.json',ops)
        if stage=='dev':
            status=read_json(local/'dev_decision.json')['status']
        else:
            status='PROXY_SCREEN_PASSED' if read_json(local/'paired_results.json')['passed'] else 'GATE_FAILED'
    except BaseException:
        (local/'exception.txt').write_text(traceback.format_exc())
    for artifact in local.iterdir():
        if artifact.name=='works' and status!='STOPPED':
            continue  # each complete bundle was already sealed during the measured arm call
        if artifact.is_dir():
            publish_closed(artifact,segment/artifact.name)
        elif artifact.is_file() and not (segment/artifact.name).exists():
            shutil.copyfile(artifact,segment/artifact.name)
    if not own.accounting_complete:
        status='STOPPED_ACCOUNTING_UNRESOLVED'
    close(segment,status,prior,own)

def close(segment,status,prior,owner):
    # Account through the archive volume scan. Reserve the last write separately from observations.
    segment_bytes=tree_bytes(segment)
    phase='model' if owner.model_started is not None else 'preparation'
    r=owner.resources()
    ph={k:v['wall_seconds'] for k,v in owner.phases(r).items()}
    ph.setdefault('model',0.)
    debits=dict(ph)
    debits[phase]+=30.  # two terminal writes share the 60s study sealing provision
    costs=dict(r)
    costs['wall_seconds']=costs['gpu_occupancy_seconds']=sum(debits.values())
    costs['cpu_seconds']+=30.
    if not owner.accounting_complete:
        debits={p:max(v,BUDGET[p+'_seconds']-prior['phase_budget_debits'][p]) for p,v in debits.items()}
        costs['cpu_seconds']=max(costs['cpu_seconds'],BUDGET['cpu_seconds']-prior['budget_resources'].get('cpu_seconds',0))
        costs['wall_seconds']=costs['gpu_occupancy_seconds']=sum(debits.values())
    record=dict(status=status,phase_seconds=ph,phase_budget_debits=debits,resources=r,budget_resources=costs,
                stages=owner.stage_resources,measured_through='archive settlement scan; terminal write reserved separately')
    if (any(prior['phase_budget_debits'][p]+v>BUDGET[p+'_seconds'] for p,v in debits.items())
        or prior['budget_resources'].get('cpu_seconds',0)+costs['cpu_seconds']>BUDGET['cpu_seconds']
        or r['peak_ram_bytes']>BUDGET['ram_bytes'] or r['peak_device_used_bytes']>BUDGET['device_bytes']):
        record['status']='STOPPED_RESOURCE'
    # Match the exact byte representation used by io.save, including the closed receipt itself.
    for _ in range(12):
        volume=segment_bytes+len((json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n').encode())
        if prior['resources']['result_bytes']+volume>BUDGET['result_bytes']:
            record['status']='STOPPED_RESOURCE'
        if r['result_bytes']==volume:
            break
        r['result_bytes']=costs['result_bytes']=volume
    else:
        raise RuntimeError('Receipt size did not settle')
    final_started=time.monotonic()
    save(segment/'closed.json',record)
    if time.monotonic()-final_started>30:
        save(segment/'overrun.json',dict(status='STOPPED_RESOURCE',seconds=time.monotonic()-final_started,
                                         reason='Final receipt exceeded settlement reserve'))

def projection(root,selected,prior,same_runtime=False):
    times=read_json(root/'dev/timings.json')
    seconds=[]
    sizes=[]
    cpu=[]
    for t in times:
        seconds.append(t['common_including_C_response']+t['archive_seconds']+
                       sum(t['arm_seconds'][a][str(selected[a])] for a in 'CO'))
        cpu.append(t['common_cpu']+sum(t['arm_cpu'][a][str(selected[a])] for a in 'CO'))
        sizes.append(sum(t['bytes'][a][str(selected[a])] for a in 'CO'))
    load=max(0.,prior['phase_seconds']['model']-sum(t['wall_seconds'] for t in times))
    model=prior['phase_budget_debits']['model']+load+128*max(seconds)+60
    volume=prior['resources']['result_bytes']+128*max(sizes)+max(2**20,prior['resources']['result_bytes']-sum(sum(sum(a.values()) for a in t['bytes'].values()) for t in times))*16
    cpu_load=max(0.,prior['resources']['cpu_seconds']-sum(t['cpu_seconds'] for t in times))
    cpu_total=prior['budget_resources']['cpu_seconds']+cpu_load+128*max(cpu)+60
    prep=prior['phase_seconds']['preparation']
    # A replacement VM needs a full cold estimate. The warm path is revalidated
    # within its own remaining budget before the model subprocess can start.
    cold_allowance=0. if same_runtime else prep
    return dict(model_seconds=model, cpu_seconds=cpu_total, result_bytes=volume,
                same_runtime=same_runtime,cold_preparation_allowance=cold_allowance,
                fits=model<BUDGET['model_seconds'] and cpu_total<BUDGET['cpu_seconds'] and volume<BUDGET['result_bytes']
                     and prior['phase_budget_debits']['preparation']+cold_allowance<BUDGET['preparation_seconds']-60)

def gpu_ready():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=5,check=True)
        return len(result.stdout.strip().splitlines())==1
    except (OSError,subprocess.SubprocessError):
        return False

def terminal_receipt(root,status):
    path=root/'summary.json'
    if path.exists():
        return read_json(path)
    state=summary(root)
    result=dict(status=status,output=str(root),**state,resources_final=not state['unsealed'],
                scientific_pass=None,training_updates=0,human_review='NOT_PERFORMED',
                checkpoint_history='UNVERIFIED',execution_budget=BUDGET)
    keys=('resources','budget_resources','cumulative_resources','cumulative_budget_debits')
    before={key:result[key]['result_bytes'] for key in keys}
    for _ in range(12):
        n=len((json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n').encode())
        if result['resources']['result_bytes']==before['resources']+n:
            break
        for key in keys:
            result[key]['result_bytes']=before[key]+n
        if result['resources']['result_bytes']>BUDGET['result_bytes']:
            result['status']='STOPPED_RESOURCE'
    else:
        raise RuntimeError('Final volume did not settle')
    save(path,result)
    return read_json(path)

def run_colab(parent='/content/drive/MyDrive/Model-WM',runtime='/content/model-wm-runtime',started=None,cpu_start=None):
    started=time.monotonic() if started is None else started
    cpu_start=owner_cpu() if cpu_start is None else cpu_start
    parent=Path(parent)
    root=parent/RUN_ID
    if not root.exists():
        if not gpu_ready():
            return dict(status='GPU_REQUIRED',output=str(root))
        start(root,parent)
    if not (root/'init_complete.json').exists():
        return dict(status='STOPPED_UNSEALED_INITIALIZATION',output=str(root))
    if (root/'summary.json').exists():
        return read_json(root/'summary.json')
    state=summary(root)
    if state['unsealed']:
        return dict(status='STOPPED_UNSEALED_READ_ONLY',output=str(root),**state)
    if 'formal' in state['segments']:
        return terminal_receipt(root,state['segments']['formal'])
    dev_in_this_call='dev' not in state['segments']
    if dev_in_this_call:
        if not gpu_ready():
            return dict(status='GPU_REQUIRED',output=str(root),**state)
        execute(root,'dev',Path(runtime),started,cpu_start)
        started,cpu_start=time.monotonic(),owner_cpu()
        state=summary(root)
    status=state['segments']['dev']
    if status=='SELECTED':
        forecast_file=root/'projection.json'
        if not forecast_file.exists():
            forecast=projection(root,read_json(root/'dev/dev_decision.json')['gains'],state,dev_in_this_call)
            save(forecast_file,forecast)
        if not read_json(forecast_file)['fits']:
            status='STOPPED_PROJECTION'
        elif not gpu_ready():
            status='GPU_REQUIRED'
        else:
            execute(root,'formal',Path(runtime),started,cpu_start)
            status=summary(root)['segments']['formal']
    if status=='GPU_REQUIRED':
        return dict(status=status,output=str(root),**summary(root))
    return terminal_receipt(root,status)
