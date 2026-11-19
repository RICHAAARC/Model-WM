"""Offline migration of a sealed synthetic DEV through the real per-work solver and scorer."""
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

from latent_control.io import Ledger, operation_summary, save
from latent_control.protocol import arrays
from latent_control.test_cpu import SyntheticPipeline
from terminal_solver import worker as previous
from terminal_solver.model import Solver
from terminal_solver.protocol import limits, protocol, read_json, RUN_ID
from wam_method.entry import publish_closed, tree_bytes
from . import entry, worker
from .checkpoints import audit_dev


def fixtures():
    m = protocol()
    m['generation'].update(width=32, height=32)
    m['amplitude'] = .18
    assets = arrays()
    assets.update(permutation=np.arange(128), signs=np.ones(128))
    torch.set_num_threads(1)
    return m, lambda: Solver(None, m, assets, None, 'cpu', SyntheticPipeline())


def make_original(path):
    m, factory = fixtures()
    dev = path/'dev'
    local = path.parent/'synthetic-dev'
    works = [w for w in m['works'] if w['split']=='DEV']
    model = factory()
    ledger = Ledger(local, limits(works))
    model.ledger = ledger
    previous.run(local, dev, 'dev', model, m, ledger)
    publish_closed(local, dev)
    # Reproduce the asymmetric production handoff even when this toy decoder's
    # DEV ranks differ. The continuation must consume the recorded decision.
    decision = read_json(dev/'dev_decision.json')
    decision.update(status='SELECTED', gains={'O':1,'C':.5})
    save(dev/'dev_decision.json', decision)
    save(dev/'closed.json', dict(status='SELECTED'))
    save(dev/'operations_summary.json', operation_summary(ledger.path, limits(works)))
    save(path/'allocation.json', read_json(entry.PACKAGE/'allocation.json'))
    with ZipFile(path/'executed_source.zip','w') as archive:
        for pkg in ('latent_control','terminal_solver'):
            for p in (entry.PACKAGE.parent/pkg).iterdir():
                if p.is_file():
                    archive.write(p,pkg+'/'+p.name)
    save(path/'projection.json', dict(fits=False, model_seconds=10**10, cpu_seconds=10**10, result_bytes=10**15))
    r = dict(wall_seconds=12000, cpu_seconds=20000, result_bytes=5000000000)
    save(path/'summary.json', dict(status='STOPPED_PROJECTION', segments={'dev':'SELECTED'}, resources_final=True,
         unsealed=None, resources=r, phase_seconds=dict(preparation=1000,model=11000),
         cumulative_resources=r, cumulative_budget_debits={**r, 'cpu_seconds':21000}))


def synthetic_worker(local, output, interrupt):
    m, factory = fixtures()
    def generate(*args):
        timing = previous.generate_work(*args)
        if interrupt:
            raise RuntimeError('simulated disconnection after first durable paired receipt')
        return timing
    save(local/'outcome.json', worker.run(local, output, m, factory, generate))


class Migration(unittest.TestCase):
    def test_migrate_reopen_finish(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            root = home/RUN_ID
            make_original(root)
            dev = root/'dev'
            def digest():
                return {p.relative_to(root):hashlib.sha256(p.read_bytes()).digest() for p in
                        [root/'summary.json',root/'projection.json',*(p for p in dev.rglob('*') if p.is_file())]}
            before = digest()
            original_command = entry.ReportingOwner.command
            first = True
            def command(owner, stage, cmd, env):
                if stage == 'preparation':
                    code = "from latent_control.io import save;import sys;save(sys.argv[1],dict(python=sys.executable))"
                    args = [sys.executable, '-c', code, owner.local/'python.json']
                else:
                    args = [sys.executable,'-m','terminal_continuation.test_cpu','worker',*cmd[-2:],str(first)]
                return original_command(owner,stage,args,dict(env,HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES=''))
            with patch.object(entry,'gpu_ready',return_value=True), patch.object(entry.ReportingOwner,'command',command):
                run = lambda: entry.run_colab(home,home/'runtime')
                stopped = run()
                self.assertEqual(stopped['status'], 'INTERRUPTED', stopped)
                output = Path(stopped['output'])
                self.assertEqual(len(list((output/'work_receipts').glob('*.json'))), 1)
                self.assertFalse((output/'calibration.json').exists())
                first_done, = (output/'work_receipts').glob('*.json')
                original_time = first_done.stat().st_mtime_ns
                first = False
                result = run()
                self.assertIn(result['status'], ('PROXY_SCREEN_PASSED','GATE_FAILED'), result)
                self.assertEqual(first_done.stat().st_mtime_ns,original_time)
                self.assertEqual(result, run())
                self.assertEqual(result['sealed_dev_gains'],{'C':.5,'O':1})
                self.assertEqual(digest(),before)
                self.assertGreater(result['cumulative_resources']['cpu_seconds'],stopped['cumulative_resources']['cpu_seconds'])
                self.assertAlmostEqual(result['cumulative_budget_debits']['cpu_seconds']-result['cumulative_resources']['cpu_seconds'],1000)
                self.assertEqual(result['continuation_resources']['result_bytes'],tree_bytes(output))
                self.assertEqual(len(list((output/'reservations').iterdir())),128)
                self.assertEqual(result['operations']['generate'],128)
                self.assertEqual(len(read_json(output/'paired_results.json')['paired']),64)
                self.assertTrue(all(len(a['work_maxima'])==64 for a in read_json(output/'calibration.json').values()))
                self.assertTrue((output/'gallery.html').is_file())
                self.assertEqual(len(read_json(output/'figure_selection.json')['all']),64)
                self.assertIsNone(result['scientific_pass'])
                (first_done).unlink()
                first_work = first_done.stem.replace('-','/')
                def forbidden():
                    raise AssertionError('Sealed or interrupted work replayed')
                m,_ = fixtures()
                outcome = worker.run(home/'unused', output, m, forbidden)
                self.assertEqual(outcome['status'],'INCOMPLETE_CAL')
                self.assertEqual(outcome['missing'],[first_work])
                with patch.object(worker.science, 'verdict', side_effect=AssertionError('TEST entered')):
                    self.assertEqual(worker.run(home/'unused',output,m,forbidden),outcome)
                # A damaged DEV image must block migration, including when gain files remain present.
                next((dev/'works').rglob('*.jpg')).write_bytes(b'invalid')
                with self.assertRaises(OSError):
                    audit_dev(root,m)

    def test_unknown_tail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            save(root/'invocations/orphan/receipts/000001.json', dict(resources=dict(wall_seconds=11),
                 phase_seconds=dict(preparation=7, model=4), subprocess_accounting_complete=False))
            old = dict(resources={}, cumulative_resources=dict(wall_seconds=31),
                       cumulative_budget_debits=dict(wall_seconds=42),phase_seconds=dict(preparation=0,model=0))
            result = entry.totals(root,old,dict(status='INTERRUPTED'))
            self.assertEqual(result['cumulative_resources']['wall_seconds'],42)
            self.assertEqual(result['cumulative_budget_debits']['wall_seconds'],53)
            self.assertFalse(result['resources_final'])
            self.assertEqual(result['unknown_tail_invocations'],['orphan'])

    def test_exclusive_peer_and_successor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = entry.claim_session(root)
            save(first/'owner.json', dict(boot='different-VM'))
            self.assertIsNone(entry.claim_session(root))
            save(first/'closed.json', {})
            second = entry.claim_session(root)
            self.assertEqual(second.name,'000002')
            self.assertEqual(entry.latest_session(first.parent),second)


if __name__=='__main__':
    if sys.argv[1:2] == ['worker']:
        synthetic_worker(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]=='True')
    else:
        unittest.main()
