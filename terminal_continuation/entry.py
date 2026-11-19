"""User-launched continuation of the original DEV-only receipt."""
import fcntl
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from latent_control.entry import add_resources
from latent_control.io import save
from terminal_solver.entry import gpu_ready
from terminal_solver.protocol import PACKAGE, RUN_ID, protocol, read_json
from wam_method.entry import Owner, owner_cpu, tree_bytes
from .checkpoints import audit_dev


class ReportingOwner(Owner):
    model_at = None

    def check(self, resources, stage, elapsed, progress):
        # Physical per-work caps are enforced by the original solver's Ledger.
        if latest_session(self.archive.parent) != self.archive:
            raise RuntimeError('A newer invocation owns the continuation')

    def receipt(self, resources, progress):
        elapsed = resources['wall_seconds']
        prep = elapsed if self.model_at is None else self.model_at
        snapshot = dict(resources=resources, phase_seconds=dict(preparation=prep, model=elapsed-prep),
                        progress=progress, subprocess_accounting_complete=self.accounting_complete)
        self.sequence += 1
        save(self.archive/'receipts'/f'{self.sequence:06d}.json', snapshot)
        return snapshot


def latest_session(folder):
    return max((p for p in folder.glob('[0-9]*') if p.name.isdecimal()), default=None)


def claim_session(output):
    folder = output/'invocations'
    folder.mkdir(parents=True, exist_ok=True)
    previous = latest_session(folder)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if previous and not (previous/'closed.json').exists():
        owner = previous/'owner.json'
        same_vm = owner.exists() and read_json(owner)['boot'] == boot
        receipts = list((previous/'receipts').glob('*.json'))
        heartbeat = max([previous.stat().st_mtime, *(p.stat().st_mtime for p in receipts)])
        # An active VM publishes observations every 15 seconds. A recent peer gets
        # exclusive use. After VM loss a new epoch fences the old worker.
        if not same_vm and time.time()-heartbeat < 120:
            return None
    stage = folder/f'{int(previous.name)+1 if previous else 1:06d}'
    try:
        stage.mkdir()
    except FileExistsError:
        return None
    save(stage/'owner.json', dict(boot=boot))
    return stage


def totals(output, old, outcome, owner=None):
    observed = {}
    phases = dict(old['phase_seconds'])
    unknown = []
    snapshots = []
    for session in sorted((output/'invocations').iterdir()):
        if owner is not None and session == owner.archive:
            continue
        path = session/'closed.json'
        if not path.is_file():
            unknown.append(session.name)
            receipts = sorted((session/'receipts').glob('*.json'))
            if not receipts:
                continue
            path = receipts[-1]
        snapshot = read_json(path)
        if not snapshot['subprocess_accounting_complete'] and session.name not in unknown:
            unknown.append(session.name)
        snapshots.append(snapshot)
    # All physical result files under this continuation counted once, across all sessions.
    prev = output/'summary.json'
    size = tree_bytes(output) - (prev.stat().st_size if prev.exists() else 0)
    if owner is not None:
        owner.volume_sample = -float('inf')
        snap = owner.receipt(owner.resources(), outcome)
        last_receipt = owner.archive/'receipts'/f'{owner.sequence:06d}.json'
        close = owner.archive/'closed.json'
        save(close, snap)
        size += last_receipt.stat().st_size + close.stat().st_size
        snapshots.append(snap)
        if not snap['subprocess_accounting_complete']:
            unknown.append(owner.archive.name)
        if latest_session(owner.archive.parent) != owner.archive:
            return dict(status='SUPERSEDED', invocation=owner.archive.name, output=str(output))
    for snapshot in snapshots:
        observed = add_resources(observed, snapshot['resources'])
        for key, value in snapshot['phase_seconds'].items():
            phases[key] += value
    observed['result_bytes'] = size
    row = dict(**outcome, output=str(output), phase_seconds=phases, sealed_dev_gains=read_json(
        output/'migration.json')['gains'] if (output/'migration.json').exists() else None,
        resources=add_resources(old['resources'], observed), continuation_resources=observed,
        cumulative_resources=add_resources(old['cumulative_resources'], observed),
        cumulative_budget_debits=add_resources(old['cumulative_budget_debits'], observed),
        unknown_tail_invocations=unknown, resources_final=not unknown, checkpoint_history='UNVERIFIED',
        human_review='NOT_PERFORMED', scientific_pass=None, global_resource_limits=None,
        accounting='GPU seconds are allocation wall time. Interrupted invocations are lower bounds. '
                   'Time includes archive scanning through final observation; terminal receipt writes follow it.')
    for _ in range(16):
        new_size = size + len((json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n').encode())
        delta = new_size - observed['result_bytes']
        if not delta:
            break
        for key in ('resources','continuation_resources','cumulative_resources','cumulative_budget_debits'):
            row[key]['result_bytes'] += delta
    else:
        raise RuntimeError('Unable to settle result volume')
    save(prev, row)
    return row


def run_locked(root, scratch, started, cpu_start):
    output = root/'continuation-01'
    summary = output/'summary.json'
    if summary.exists():
        prior = read_json(summary)
        settled = prior['status'] in ('PROXY_SCREEN_PASSED', 'GATE_FAILED')
        if prior['status'] in ('INCOMPLETE_CAL', 'INCOMPLETE_TEST'):
            settled = not any((output/'work_receipts'/(wid.replace('/','-')+'.json')).is_file()
                              for wid in prior['missing'])
        if settled:
            if prior.get('unknown_tail_invocations'):
                fields = ('status','missing','completed','denominator','operations','operations_completed_in_sealed_works')
                return totals(output, read_json(root/'summary.json'), {k:prior[k] for k in fields if k in prior})
            return prior
    if not (root/'summary.json').is_file():
        return dict(status='MISSING_SEALED_DEV', output=str(root))
    if not gpu_ready():
        return dict(status='GPU_REQUIRED', output=str(output))
    old = read_json(root/'summary.json')
    session = claim_session(output)
    if session is None:
        return dict(status='ACTIVE_IN_OTHER_RUNTIME', output=str(output),
                    message='Another runtime is active. After a VM loss, its last heartbeat expires in 120 seconds.')
    key = session.name
    local = scratch/'continuations'/key
    local.mkdir(parents=True)
    cache = scratch/'cache'
    cache.mkdir(exist_ok=True)
    manifest = protocol()
    manifest['budgets'] = {}  # No global quotas in this entry.
    own = ReportingOwner(manifest, local, session, scratch, cache, started, cpu_start)
    own.receipt(own.resources(), dict(phase='AUDIT'))
    try:
        migration = audit_dev(root, manifest, heartbeat=lambda: own.receipt(own.resources(), dict(phase='AUDIT')))
        seal = output/'migration.json'
        if not seal.exists():
            save(seal, migration)
        if read_json(seal) != migration:
            raise ValueError('Saved continuation conflicts with frozen DEV')
        print('Sealed DEV verified:', migration['gains'], 'historical projection:', migration['original_projection'], flush=True)
        save(local/'manifest.json', manifest)
        env = dict(os.environ, PYTHONPATH=str(PACKAGE.parent)+os.pathsep+os.environ.get('PYTHONPATH',''),
                   MPLBACKEND='Agg', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1', HF_HOME=str(scratch/'hf'), TORCH_HOME=str(scratch/'torch'))
        own.command('preparation', [sys.executable,'-m','latent_control.prepare','environment',local,scratch,cache], env)
        own.model_at = time.monotonic() - started
        py = read_json(local/'python.json')['python']
        own.command('model', [py,'-m','terminal_continuation.worker',local,output], env)
        outcome = read_json(local/'outcome.json')
    except BaseException as exc:
        (session/'exception.txt').write_text(traceback.format_exc())
        outcome = dict(status='INTERRUPTED', error=repr(exc), invocation=key)
    for p in local.iterdir():
        if p.is_file() and not p.name.endswith('.tmp'):
            shutil.copyfile(p, session/p.name)
    return totals(output, old, outcome, owner=own)


def run_colab(parent='/content/drive/MyDrive/Model-WM', runtime='/content/model-wm-runtime',
              started=None, cpu_start=None):
    started = time.monotonic() if started is None else started
    cpu_start = owner_cpu() if cpu_start is None else cpu_start
    root = Path(parent)/RUN_ID
    scratch = Path(runtime)/RUN_ID
    if not (root/'summary.json').is_file():
        return dict(status='MISSING_SEALED_DEV', output=str(root))
    scratch.mkdir(parents=True, exist_ok=True)
    with (scratch/'continuation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='ACTIVE_IN_THIS_RUNTIME', output=str(root))
        return run_locked(root, scratch, started, cpu_start)
