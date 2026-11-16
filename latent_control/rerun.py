"""One user-authorized larger-GPU rerun; delete only the named old output."""
import json
import shutil
import subprocess
from pathlib import Path
from .io import save
from wam_method.io import reconcile

OLD = 'latent-control-20261116-attempt01'
NEW = 'latent-control-20261116-attempt02'
CARRY = NEW + '-carry.json'

def gpu_capacity():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    values = [int(x.strip()) * 1024**2 for x in result.stdout.splitlines() if x.strip()]
    if len(values) != 1 or values[0] <= 16 * 1024**3:
        raise RuntimeError('Select a larger GPU (over 16 GiB; A100 preferred), then Run all. No experiment started.')
    return values[0]

def bounded_remove(parent, name):
    parent = Path(parent).resolve()
    target = parent / name
    if name != OLD or target.is_symlink() or target.resolve().parent != parent:
        raise RuntimeError('Cleanup path is outside the explicitly authorized old attempt')
    if target.exists():
        if any(p.is_symlink() for p in target.rglob('*')):
            raise RuntimeError('Unexpected symlink in old output; cleanup stopped')
        shutil.rmtree(target)

def prepare_rerun(parent, package, device_bytes):
    parent, package = Path(parent), Path(package)
    already_reserved = (parent / NEW).exists()
    m = json.loads((package / 'manifest.json').read_text())
    carry_path = parent / CARRY
    if carry_path.exists():
        carry = json.loads(carry_path.read_text())
        if carry.get('source_attempt') != OLD or not carry.get('resources_final'):
            raise RuntimeError('Invalid cumulative-cost carry receipt')
    else:
        old = json.loads((parent / OLD / 'summary.json').read_text())
        if not old.get('resources_final') or old.get('status') != 'MODEL_STOPPED':
            raise RuntimeError('Expected sealed stopped attempt01 receipt')
        # Carry only aggregate costs/exposure, not old error logs, traces or media.
        carry = reconcile(old, m['inherited'])
        carry.update(resources_final=True, source_attempt=OLD,
            source_operations=old['operations'], old_output_deleted_by_user_request=True,
            historical_failures_preserved=False,
            result_bytes_semantics='Cumulative produced volume; deletion does not refund costs')
        save(carry_path, carry)
        if json.loads(carry_path.read_text()) != carry:
            raise RuntimeError('Carry receipt readback failed; old output not deleted')
    reconcile(carry, m['inherited'])  # validate all resource numbers before deletion
    if not already_reserved:bounded_remove(parent, OLD)
    m['run_id'] = NEW
    m['authorization'] = '2026-11-16 user authorizes one full rerun on larger GPU and deletion of attempt01 output'
    m['inherited'] = dict(carry, history_path=CARRY)
    m['budgets']['device_bytes'] = device_bytes
    m['historical_failures_preserved'] = False
    m['execution_amendment'] = dict(previous_attempt=OLD, previous_output_deleted=True,
        new_execution_allowance=True, unchanged_time_budgets=True,
        gpu_memory_limit='Detected single-GPU capacity', frozen_method_unchanged=True,
        prior_exposure='DEV/000 generated in attempt01; CAL/TEST not entered')
    save(package / 'manifest.json', m)
    save(package / 'historical_floor.json', carry)
    history = json.loads((package / 'source_usage_history.json').read_text())
    history['rerun_exposure'] = m['execution_amendment']['prior_exposure']
    save(package / 'source_usage_history.json', history)
    protocol = package / 'protocol.py'
    protocol.write_text(protocol.read_text().replace(OLD, NEW))
    # The IDs have equal lengths and extraction/rewrite can share one mtime second.
    # A worker in another interpreter must not reuse attempt01's timestamp-valid pyc.
    for cached in (package / '__pycache__').glob('protocol.*.pyc'):
        cached.unlink()
    return True

def run_colab(started=None, cpu_start=None, parent='/content/drive/MyDrive/Model-WM',
              runtime='/content/model-wm-runtime'):
    from . import protocol, entry
    parent = Path(parent)
    if not (parent / NEW).exists():
        capacity = gpu_capacity()  # before cleanup, downloads or model work
        prepare_rerun(parent, protocol.PACKAGE, capacity)
    # Reopening uses the same frozen wrapper manifest, even in a fresh runtime.
    if (parent / NEW).exists():
        summary = parent / NEW / 'summary.json'
        if summary.exists():
            return json.loads(summary.read_text())
        prepare_rerun(parent, protocol.PACKAGE, 16 * 1024**3)
    protocol.RUN_ID = entry.RUN_ID = NEW
    return entry.run_colab(started=started, cpu_start=cpu_start, parent=parent, runtime=runtime)
