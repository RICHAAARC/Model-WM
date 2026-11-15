"""User-started Colab owner with local IPC and additive historical accounting."""
import json
import math
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .io import save,reconcile,operation_summary
from .protocol import read,RUN_ID

def tree_bytes(root):
    size=0
    for p in Path(root).rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():size+=p.stat().st_size
        except FileNotFoundError:pass # atomic local rename during live sampling
    return size

def owner_cpu():
    r=resource.getrusage(resource.RUSAGE_SELF);return r.ru_utime+r.ru_stime

def unique_result_bytes(local,archive):
    files={}
    for root in [Path(local),Path(archive)]:
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink():
                key=str(p.relative_to(root));files[key]=max(files.get(key,0),p.stat().st_size)
    return sum(files.values())

def live_result_bytes(local,archive):
    # Worker archives are mirrors of local closed artifacts. Never walk thousands
    # of Drive evaluation files on the one-second accounting loop.
    local,archive=Path(local),Path(archive)
    extras=tree_bytes(archive/'receipts')
    extras+=sum(p.stat().st_size for p in archive.iterdir()
                if p.is_file() and not (local/p.name).exists())
    return tree_bytes(local)+extras

def publish_closed(local,archive):
    for p in Path(local).rglob('*'):
        if not p.is_file() or p.is_symlink() or p.name.endswith('.tmp'):continue
        dst=Path(archive)/p.relative_to(local);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            # Already sealed evaluation/step result is immutable, not a mailbox.
            if p.stat().st_size!=dst.stat().st_size:raise RuntimeError('Sealed artifact conflict: '+str(dst))
        else:shutil.copyfile(p,dst)

class Owner:
    def __init__(self,manifest,local,archive,scratch,cache,started,cpu_start):
        self.m=manifest;self.local=Path(local);self.archive=Path(archive)
        self.scratch=Path(scratch);self.cache=Path(cache);self.started=started;self.cpu_start=cpu_start
        self.child_cpu=0.;self.cpu_high=0.;self.ram=0.;self.device=0.;self.download=0
        self.initial_cache=tree_bytes(cache);self.stage_resources={};self.sequence=0
        self.volume_sample=-math.inf;self.result_size=0
        self.accounting_complete=True
    def resources(self,live=None):
        live=live or {}
        self.cpu_high=max(self.cpu_high,owner_cpu()-self.cpu_start+self.child_cpu+live.get('cpu_seconds',0))
        self.ram=max(self.ram,live.get('peak_ram_bytes',0)+resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        elapsed=time.monotonic()-self.started
        if time.monotonic()-self.volume_sample>=5:
            self.volume_sample=time.monotonic()
            self.download=max(self.download,tree_bytes(self.cache)-self.initial_cache+tree_bytes(self.scratch/'images'))
            self.result_size=live_result_bytes(self.local,self.archive)
        return dict(wall_seconds=elapsed,gpu_occupancy_seconds=elapsed,cpu_seconds=self.cpu_high,
            peak_ram_bytes=self.ram,peak_device_used_bytes=self.device,
            download_charge_bytes=max(0,self.download),result_bytes=self.result_size)
    def check(self,r,stage,stage_elapsed,progress):
        b=self.m['budgets'];margin=b['cleanup_margin_seconds']
        for key in ['gpu_occupancy_seconds','wall_seconds','cpu_seconds']:
            if r[key]>=b[key]-margin:raise RuntimeError('Resource margin: '+key)
        for key,limit in [('result_bytes','result_bytes'),('peak_ram_bytes','ram_bytes'),('peak_device_used_bytes','device_bytes')]:
            if r[key]>=b[limit]:raise RuntimeError('Resource limit: '+key)
        if stage=='preparation' and stage_elapsed>=b['preparation_seconds']-margin:
            raise RuntimeError('Preparation margin exhausted')
        if stage=='method':
            for arm,t in progress.get('train_seconds',{}).items():
                if progress.get('active_arm')==arm:
                    t+=max(0,time.monotonic()-progress['active_started'])
                if t>=b['train_arm_seconds']-margin:raise RuntimeError('Training budget: '+arm)
            if progress.get('phase') in ['CAL','TEST'] and time.monotonic()-progress['evaluation_started']>=b['evaluation_seconds']-margin:
                raise RuntimeError('Evaluation margin exhausted')
    def receipt(self,r,progress):
        self.sequence+=1
        save(self.archive/'receipts'/f'{self.sequence:06d}.json',dict(resources=r,progress=progress,
            resources_final=False,unknown_interruption_tail=True,
            abrupt_loss_budget_debit=self.m['budgets'],
            note='Sealed observation; no dynamic Drive reads. Catastrophic loss blocks resume.'))
    def command(self,stage,command,env):
        self.accounting_complete=False
        receipt=self.local/(stage+'.usage.json');log=self.local/(stage+'.log')
        before=time.monotonic();last_snapshot=0.;last_size=0.;live={};progress={};error=None
        with log.open('x') as out:
            p=subprocess.Popen([sys.executable,'-m','wam_qualification.metered',str(receipt),*map(str,command)],
                stdout=out,stderr=subprocess.STDOUT,env=env)
            try:
                while p.poll() is None:
                    if receipt.exists():live=json.loads(receipt.read_text())
                    if (self.local/'progress.json').exists():progress=json.loads((self.local/'progress.json').read_text())
                    # Device total is a conservative runtime allocation observation.
                    if time.monotonic()-last_size>5:
                        last_size=time.monotonic()
                        try:
                            got=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                                capture_output=True,text=True,timeout=3)
                            if got.returncode==0:self.device=max(self.device,sum(float(v) for v in got.stdout.split())*1024**2)
                        except (OSError,subprocess.TimeoutExpired):pass
                    r=self.resources(live);self.check(r,stage,
                        time.monotonic()-(self.started if stage=='preparation' else before),progress)
                    if time.monotonic()-last_snapshot>15:
                        self.receipt(r,progress);last_snapshot=time.monotonic()
                        print(stage,progress.get('phase',''),progress.get('updates',{}),progress.get('evaluation_bundles',0),flush=True)
                    time.sleep(1)
            except BaseException as exc:
                error=exc;p.terminate()
                # Metered owner reaps its complete subtree before returning.
                p.wait(timeout=20)
            finally:
                if p.poll() is None:p.terminate();p.wait(timeout=20)
                if receipt.exists():live=json.loads(receipt.read_text())
                self.accounting_complete=bool(live.get('resources_final') and live.get('all_descendants_reaped'))
                self.resources(live);self.child_cpu+=live.get('cpu_seconds',0)
                self.stage_resources[stage]={**live,'wall_seconds':time.monotonic()-before}
                if stage=='preparation':
                    self.stage_resources[stage]['entry_elapsed_through_preparation']=time.monotonic()-self.started
                self.receipt(self.resources(),progress)
        if error:raise error
        if p.returncode or not live.get('resources_final') or not live.get('all_descendants_reaped'):
            raise RuntimeError(f'{stage} failed ({p.returncode}); see {stage}.log')

def run_colab(started=None,cpu_start=None,parent='/content/drive/MyDrive/Model-WM',runtime='/content/model-wm-runtime'):
    started=time.monotonic() if started is None else started
    cpu_start=owner_cpu() if cpu_start is None else cpu_start
    m=read();parent=Path(parent);archive=parent/RUN_ID
    if archive.exists():
        if (archive/'summary.json').exists():return json.loads((archive/'summary.json').read_text())
        receipts=sorted((archive/'receipts').glob('*.json'))
        return dict(status='STOPPED_UNSEALED_READ_ONLY',scientific_pass=None,retry=False,
            output=str(archive),last_observation=json.loads(receipts[-1].read_text()) if receipts else None,
            budget_debit=m['budgets'],note='Runtime loss has an unknown tail; full package remains reserved; no rerun')
    archive.mkdir(parents=True,exist_ok=False)
    save(archive/'reservation.json',dict(run_id=RUN_ID,time=time.time(),budget=m['budgets'],no_retry=True))
    scratch=Path(runtime)/RUN_ID;scratch.mkdir(parents=True,exist_ok=False)
    local=scratch/'records';local.mkdir();cache=scratch/'cache';cache.mkdir()
    own=Owner(m,local,archive,scratch,cache,started,cpu_start)
    inherited=m['inherited'];error=None;result={};status='STOPPED_INCOMPLETE'
    try:
        old=json.loads((parent/inherited['history_path']).read_text())
        inherited=reconcile(old,inherited)
        if not old.get('resources_final'):raise RuntimeError('Previous qualification history unsealed')
        save(local/'inherited_history.json',old);save(local/'manifest.json',m)
        shutil.copyfile(Path(__file__).with_name('source_usage_history.json'),local/'source_usage_history.json')
        # Freeze the executed source before any preparation or model work.
        package=Path(__file__).resolve().parents[1]
        for name in ['wam_method','wam_qualification']:
            shutil.copytree(package/name,local/'executed_source'/name,ignore=shutil.ignore_patterns('__pycache__'))
        env=os.environ.copy();env.update(MPLBACKEND='Agg',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
            PYTHONPATH=str(package)+os.pathsep+env.get('PYTHONPATH',''),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            HF_HOME=str(scratch/'hf'),TORCH_HOME=str(scratch/'torch'))
        own.command('preparation',[sys.executable,'-m','wam_method.prepare','environment',local,scratch,cache],env)
        python=json.loads((local/'python.json').read_text())['python']
        own.command('method',[python,'-m','wam_method.worker',local,archive,scratch],env)
        result=json.loads((local/'method_result.json').read_text());status=result['status']
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        (local/'exception.txt').write_text(traceback.format_exc())
    # Workers are quiescent. Drive receives only completed files and final failure logs.
    try:
        if own.accounting_complete:publish_closed(local,archive)
        else:
            status='STOPPED_ACCOUNTING_UNRESOLVED'
            save(archive/'accounting_failure.json',dict(error=error,unknown_current_tail=True,
                no_retry=True,full_package_budget_reserved=True))
            if (local/'exception.txt').exists():shutil.copyfile(local/'exception.txt',archive/'exception.txt')
    except Exception as exc:
        error=(error or '')+'; sealing: '+repr(exc);status='STOPPED_SEALING_INCOMPLETE'
        (archive/'sealing_exception.txt').write_text(traceback.format_exc())
    resources=own.resources();margin=m['budgets']['cleanup_margin_seconds']
    resources['download_charge_bytes']=max(resources['download_charge_bytes'],
        tree_bytes(cache)-own.initial_cache+tree_bytes(scratch/'images'))
    resources['result_bytes']=unique_result_bytes(local,archive)
    debit=dict(resources)
    for key in ['wall_seconds','gpu_occupancy_seconds','cpu_seconds']:debit[key]+=margin
    if not own.accounting_complete:
        for key in ['wall_seconds','gpu_occupancy_seconds','cpu_seconds']:debit[key]=max(debit[key],m['budgets'][key])
    def total(base,extra):
        return {k:(max(base.get(k,0),v) if k.startswith('peak_') else base.get(k,0)+v) for k,v in extra.items()}
    summary=dict(status=status,error=error,output=str(archive),scientific_pass=None,
        numeric_pass=result.get('numeric_pass',False),visual_review=result.get('visual_review','UNRESOLVED'),
        resources=resources,budget_resources=debit,stage_resources=own.stage_resources,
        operations=operation_summary(local/'operations.jsonl',m['physical_limits']),
        inherited_resources=inherited['cumulative_resources'],
        cumulative_resources=total(inherited['cumulative_resources'],resources),
        cumulative_budget_debits=total(inherited['cumulative_budget_debits'],debit),
        resources_final=own.accounting_complete and status!='STOPPED_SEALING_INCOMPLETE',historical_cpu_exact=False,
        current_accounting_complete=own.accounting_complete,
        cumulative_cpu_is_lower_bound=True,execution_budget=m['budgets'],retry=False,
        original_budget_pass_claimed=False,checkpoint_history='UNVERIFIED',
        download_measurement='new local durable payload high-water, including partials; restored paid cache excluded; not network packet bytes',
        result_volume_measurement='sealed attempt tree bytes including final summary; local mirrors counted once',
        ram_measurement='conservative sum of owner and descendant peak RSS observations',
        pretrained_weights_destination='/content only',
        final_trained_writers='experiment results in Drive',
        visual_and_scientific_claim='Numeric decision only; no automatic visual or innovation PASS',
        terminal_seal=dict(reserved_seconds=margin,unknown_tail_if_interrupted=True,
            measured_through='quiescent artifact archival; final summary writes covered by conservative 60-second debit'))
    # Reconcile the self-containing size rather than reporting the pre-summary tree.
    for _ in range(8):
        save(archive/'summary.json',summary)
        size=tree_bytes(archive)
        if summary['resources']['result_bytes']==size:break
        for k in ['resources','budget_resources']:summary[k]['result_bytes']=size
        for k in ['cumulative_resources','cumulative_budget_debits']:
            summary[k]['result_bytes']=inherited[k]['result_bytes']+size
    save(archive/'summary.json',summary)
    if time.monotonic()-started>m['budgets']['gpu_occupancy_seconds'] or owner_cpu()-cpu_start+own.child_cpu>m['budgets']['cpu_seconds']:
        summary.update(status='STOPPED_RESOURCE_LIMIT',numeric_pass=False)
        save(archive/'summary.json',summary)
    return summary
