"""One user-started reservation. Separate phase envelopes, complete monotone history."""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from wam_method.entry import Owner,owner_cpu,tree_bytes,unique_result_bytes,publish_closed
from wam_method.io import reconcile
from .protocol import read,RUN_ID,PACKAGE
from .io import save,operation_summary,BudgetStop

def add_resources(base,extra):
    return {k:(max(base.get(k,0),extra.get(k,0)) if k.startswith('peak_') else base.get(k,0)+extra.get(k,0))
            for k in set(base)|set(extra)}

def full_debit(m):
    b=m['budgets'];return dict(wall_seconds=b['preparation_seconds']+b['model_seconds'],
        gpu_occupancy_seconds=b['preparation_seconds']+b['model_seconds'],cpu_seconds=b['cpu_seconds'])

class PhaseOwner(Owner):
    def __init__(self,*args,**kwargs):
        self.model_started=None;self.model_cpu_base=0.;self.preparation_final=None
        super().__init__(*args,**kwargs)
    def phases(self,r):
        if self.model_started is None:return dict(preparation={k:r[k] for k in ['wall_seconds','gpu_occupancy_seconds','cpu_seconds']})
        prep=self.preparation_final
        return dict(preparation=prep,model={k:max(0,r[k]-prep[k]) for k in prep})
    def resources(self,live=None):
        r=super().resources(live)
        save(self.local/'budget_state.json',dict(total=r,**self.phases(r)))
        return r
    def check(self,r,stage,stage_elapsed,progress):
        b=self.m['budgets'];margin=b['cleanup_margin_seconds'];ph=self.phases(r)
        phase='model' if self.model_started is not None else 'preparation'
        limit=b['model_seconds'] if phase=='model' else b['preparation_seconds']
        if ph[phase]['wall_seconds']>=limit-margin:raise BudgetStop(phase+' wall margin exhausted')
        if r['cpu_seconds']>=b['cpu_seconds']-margin:raise BudgetStop('Total CPU margin exhausted')
        for k,limitkey in [('peak_ram_bytes','ram_bytes'),('peak_device_used_bytes','device_bytes'),('result_bytes','result_bytes')]:
            if r[k]>=b[limitkey]:raise BudgetStop('Resource limit '+k)
    def command(self,stage,command,env):
        if stage=='model':
            r=self.resources();self.preparation_final={k:r[k] for k in ['wall_seconds','gpu_occupancy_seconds','cpu_seconds']}
            if r['wall_seconds']>=self.m['budgets']['preparation_seconds']:raise BudgetStop('Preparation already exceeded')
            self.model_started=time.monotonic();self.model_cpu_base=r['cpu_seconds']
        self.accounting_complete=False
        path=self.local/(stage+'.usage.json');before=time.monotonic();last_snapshot=0.;last_device=0.
        live={};progress={};error=None;p=None
        def stop_tree():
            if p is None or p.poll() is not None:return
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # A broken supervisor cannot leave model work running behind a stopped receipt.
                import psutil
                try:children=psutil.Process(p.pid).children(recursive=True)
                except psutil.NoSuchProcess:children=[]
                for child in reversed(children):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                p.kill();p.wait(timeout=5)
        with (self.local/(stage+'.log')).open('x') as log:
            try:
                p=subprocess.Popen([sys.executable,'-m','wam_qualification.metered',str(path),*map(str,command)],
                    stdout=log,stderr=subprocess.STDOUT,env=env)
                while p.poll() is None:
                    if path.exists():live=json.loads(path.read_text())
                    if (self.local/'progress.json').exists():progress=json.loads((self.local/'progress.json').read_text())
                    if time.monotonic()-last_device>5:
                        last_device=time.monotonic()
                        try:
                            got=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                            if got.returncode==0:self.device=max(self.device,sum(float(v) for v in got.stdout.split())*1024**2)
                        except (OSError,subprocess.TimeoutExpired):pass
                    r=self.resources(live);self.check(r,stage,time.monotonic()-before,progress)
                    if time.monotonic()-last_snapshot>=15:
                        self.receipt(r,progress);last_snapshot=time.monotonic()
                        print(stage,progress.get('phase',''),progress.get('active_work',''),
                              f"completed={progress.get('evaluation_bundles',0)}/136",flush=True)
                    time.sleep(1)
            except BaseException as exc:error=exc
            finally:
                stop_tree()
                if path.exists():live=json.loads(path.read_text())
                self.accounting_complete=bool(live.get('resources_final') and live.get('all_descendants_reaped'))
                self.resources(live);self.child_cpu+=live.get('cpu_seconds',0)
                self.stage_resources[stage]={**live,'wall_seconds':time.monotonic()-before}
                if stage=='preparation':self.stage_resources[stage]['entry_elapsed_through_preparation']=time.monotonic()-self.started
                self.receipt(self.resources(),progress)
        if error:raise error
        if p is None or p.returncode or not self.accounting_complete:
            raise RuntimeError(f'{stage} failed ({None if p is None else p.returncode}); see {stage}.log')

def settle_volume(archive,summary,inherited):
    # Count the summary itself, after the last status/cost mutation. The size must converge.
    for _ in range(20):
        save(archive/'summary.json',summary);n=tree_bytes(archive)
        if summary['resources']['result_bytes']==n:return
        for key in ['resources','budget_resources']:summary[key]['result_bytes']=n
        for key in ['cumulative_resources','cumulative_budget_debits']:
            summary[key]['result_bytes']=inherited[key]['result_bytes']+n
    raise RuntimeError('Self-containing result size did not converge')

def run_colab(started=None,cpu_start=None,parent='/content/drive/MyDrive/Model-WM',runtime='/content/model-wm-runtime'):
    started=time.monotonic() if started is None else started;cpu_start=owner_cpu() if cpu_start is None else cpu_start
    m=read();parent=Path(parent);archive=parent/RUN_ID
    if archive.exists():
        if (archive/'summary.json').exists():return json.loads((archive/'summary.json').read_text())
        receipts=sorted((archive/'receipts').glob('*.json'))
        observation=json.loads(receipts[-1].read_text()) if receipts else None
        inherited=m['inherited']
        if (archive/'inherited_totals.json').exists():
            inherited=reconcile(json.loads((archive/'inherited_totals.json').read_text()),inherited)
        observed=(observation or {}).get('resources',{})
        extra=dict(observed)
        for k,v in full_debit(m).items():extra[k]=max(extra.get(k,0),v)
        return dict(status='STOPPED_UNSEALED_READ_ONLY',output=str(archive),retry=False,scientific_pass=None,
            last_observation=observation,full_package_reserved=extra,
            cumulative_budget_debits=add_resources(inherited['cumulative_budget_debits'],extra),
            note='Incomplete reservation retained. Unknown tail consumes the full attempt allowance; no second execution.')
    archive.mkdir(parents=True,exist_ok=False)
    save(archive/'reservation.json',dict(run_id=RUN_ID,started_at=time.time(),no_retry=True,budgets=m['budgets'],
        inherited_floor=m['inherited']))
    scratch=Path(runtime)/RUN_ID;scratch.mkdir(parents=True,exist_ok=False)
    local=scratch/'records';local.mkdir();cache=scratch/'cache';cache.mkdir()
    own=PhaseOwner(m,local,archive,scratch,cache,started,cpu_start)
    inherited=m['inherited'];result={};error=None;status='ENVIRONMENT_STOPPED'
    try:
        old=json.loads((parent/inherited['history_path']).read_text())
        inherited=reconcile(old,inherited)
        if not old.get('resources_final'):raise RuntimeError('Historical result not sealed')
        save(local/'inherited_history.json',old);save(archive/'inherited_totals.json',inherited)
        for name in ['manifest.json','source_usage_history.json','historical_floor.json','frozen.npz','PREREGISTRATION.md']:
            shutil.copyfile(PACKAGE/name,local/name)
        package=PACKAGE.parent
        for name in ['latent_control','wam_method','wam_qualification']:
            shutil.copytree(package/name,local/'executed_source'/name,
                ignore=shutil.ignore_patterns('__pycache__','review','test_*','check_*'))
        env=os.environ.copy();env.update(PYTHONPATH=str(package)+os.pathsep+env.get('PYTHONPATH',''),
            MPLBACKEND='Agg',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HOME=str(scratch/'hf'),TORCH_HOME=str(scratch/'torch'))
        own.command('preparation',[sys.executable,'-m','latent_control.prepare','environment',local,scratch,cache],env)
        python=json.loads((local/'python.json').read_text())['python'];status='MODEL_STOPPED'
        own.command('model',[python,'-m','latent_control.worker',local,archive,scratch],env)
        result=json.loads((local/'method_result.json').read_text());status=result['status']
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';(local/'exception.txt').write_text(traceback.format_exc())
    try:
        # All completed and partial records survive even when the meter itself fails.
        publish_closed(local,archive)
        if not own.accounting_complete:
            status='STOPPED_ACCOUNTING_UNRESOLVED'
            save(archive/'accounting_failure.json',dict(unknown_tail=True,full_debit=full_debit(m),no_retry=True))
    except BaseException as exc:
        error=(error or '')+'; sealing: '+repr(exc);status='STOPPED_SEALING_INCOMPLETE'
        (archive/'sealing_exception.txt').write_text(traceback.format_exc())
    r=own.resources();r['download_charge_bytes']=max(r['download_charge_bytes'],tree_bytes(cache)-own.initial_cache)
    r['result_bytes']=unique_result_bytes(local,archive);debit=dict(r);margin=m['budgets']['cleanup_margin_seconds']
    for k in ['wall_seconds','gpu_occupancy_seconds','cpu_seconds']:debit[k]+=margin
    if not own.accounting_complete:
        for k,v in full_debit(m).items():debit[k]=max(debit[k],v)
    phases=own.phases(r);phase='model' if own.model_started is not None else 'preparation'
    phase_debits={p:dict(v) for p,v in phases.items()}
    for k in phase_debits[phase]:phase_debits[phase][k]+=margin
    numeric=result.get('numeric_pass',False)
    try:own.check(r,phase,0,{})
    except BudgetStop as exc:status='STOPPED_RESOURCE_LIMIT';numeric=False;error=(error or '')+'; '+str(exc)
    summary=dict(status=status,error=error,output=str(archive),numeric_pass=numeric,scientific_pass=None,
        visual_review='UNRESOLVED',innovation_pass=False,resources=r,budget_resources=debit,phases=phases,
        phase_budget_debits=phase_debits,stage_resources=own.stage_resources,
        operations=operation_summary(local/'operations.jsonl',m['physical_limits']),
        inherited_resources=inherited['cumulative_resources'],
        cumulative_resources=add_resources(inherited['cumulative_resources'],r),
        cumulative_budget_debits=add_resources(inherited['cumulative_budget_debits'],debit),
        resources_final=own.accounting_complete and status!='STOPPED_SEALING_INCOMPLETE',
        historical_cpu_exact=False,cumulative_cpu_is_lower_bound=True,execution_budget=m['budgets'],
        historical_failures_preserved=True,retry=False,checkpoint_history='UNVERIFIED',
        unseen_content='USER_ACCEPTED_ASSUMPTION',pretrained_weights_destination='/content only',
        gpu_clock='Total allocation proxy includes preparation; model allowance starts at model subprocess launch',
        download_measurement='New local durable cache high-water including partials; restored paid cache excluded; not packets',
        result_volume_measurement='Sealed attempt tree including summary; local mirrors counted once',
        terminal_seal=dict(reserved_seconds=margin,unknown_tail_if_interrupted=True))
    summary['unallocated_unknown_tail_debit']={k:max(0,debit[k]-sum(p[k] for p in phase_debits.values()))
        for k in ['wall_seconds','gpu_occupancy_seconds','cpu_seconds']}
    if 'costs' in result:
        # Add the complete owner/setup/terminal common cost equally. This preserves the predeclared arm comparison.
        extra={k:max(0,debit[k]-result['worker_cost'][k]) for k in ['wall_seconds','cpu_seconds']}
        summary['full_costs_with_common_overhead']={a:{k:result['costs'][a][k]+extra[k]/2 for k in extra} for a in 'BC'}
    settle_volume(archive,summary,inherited)
    return summary
