"""Release regressions using CPU fixtures; no remote assets or model initialization."""
import ast
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from . import dev_entry as entry, dev_worker as worker
from .test_cpu import Dummy
from latent_control.io import save

ROOT=entry.PACKAGE.parent
ALLOCATION=entry.read(entry.PACKAGE/'allocation.json')


class ReleaseChecks(unittest.TestCase):
    def test_notebook_payload_is_exact_and_importable(self):
        nb=json.loads((ROOT/'notebooks/lawa_shortcode_dev_colab.ipynb').read_text())
        self.assertEqual(len(nb['cells']),3)
        self.assertEqual(''.join(nb['cells'][0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
        tree=ast.parse(''.join(nb['cells'][2]['source']))
        data=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
        with tempfile.TemporaryDirectory() as temp, ZipFile(io.BytesIO(base64.b64decode(data))) as z:
            for name in z.namelist():
                self.assertEqual(z.read(name),(ROOT/name).read_bytes())
            z.extractall(temp)
            subprocess.run([sys.executable,'-c',
                "from shortcode_channel.dev_entry import check_freeze;check_freeze();"
                "import shortcode_channel.dev_assets,shortcode_channel.dev_worker,wam_qualification.metered"],
                cwd=temp,env=dict(os.environ,PYTHONPATH=temp),check=True)

    def test_no_gpu_and_abrupt_reopen(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(entry,'gpu_ready',return_value=False):
            base=Path(temp)
            self.assertEqual(entry.run_colab(base)['status'],'GPU_REQUIRED')
            out=base/entry.RUN_ID
            self.assertFalse(out.exists())
            out.mkdir()
            result=entry.run_colab(base)
            self.assertTrue(result['unknown_tail'])
            self.assertFalse(result['retry'])
            debit=ALLOCATION['budget']['historical_debits']['wall_seconds']
            self.assertEqual(result['cumulative_conservative_debit']['wall_seconds'],debit+1560)

    def test_full_dummy_release(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);parent=base/'drive';runtime=base/'content'
            # Drive the same metered subprocess path with a CPU-only synthetic backend.
            original=entry.Monitor.command
            def command(m,stage,cmd,env,end):
                if stage=='preparation':
                    script="from latent_control.io import save;from pathlib import Path;import sys;save(Path(sys.argv[1])/'python.json',{'python':sys.executable})"
                else:
                    script=("from shortcode_channel.test_cpu import Dummy;from shortcode_channel import dev_worker as w;"
                        "from pathlib import Path;import json,sys;from latent_control.io import save;"
                        "loc,out,s=map(Path,sys.argv[1:]);"
                        "save(loc/'outcome.json',w.run(loc,out,json.loads((w.PACKAGE/'allocation.json').read_text()),Dummy()))")
                return original(m,stage,[sys.executable,'-c',script,*cmd[3:]],env,end)
            with patch.object(entry,'gpu_ready',return_value=True), patch.object(entry.Monitor,'command',command):
                result=entry.run_colab(parent,runtime)
                self.assertEqual(result['status'],'DEV_STOP_RULE',result.get('traceback'))
                self.assertEqual(entry.run_colab(parent,runtime),result)
            out=parent/entry.RUN_ID
            self.assertEqual(result['counts']['slots_completed'],48)
            self.assertEqual(result['counts']['works_completed'],8)
            self.assertTrue(result['resources_final'])
            for k,n in entry.LIMITS.items():
                self.assertEqual(result['operations'][k],dict(started=n,completed=n,denominator=n))
            self.assertEqual(entry.tree_bytes(out),result['resources']['result_bytes'])
            for k,t in result['stages'].items():self.assertGreaterEqual(t['wall_seconds'],0)
            self.assertAlmostEqual(sum(t['wall_seconds'] for t in result['stages'].values()),result['resources']['wall_seconds'])
            with ZipFile(out/'slots/DEV-00/slot-01/results.zip') as z:
                self.assertEqual(len(z.read('readouts.jsonl').splitlines()),26)
                self.assertEqual(len([n for n in z.namelist() if n.endswith('.png')]),13)
            self.assertFalse(result['cal_test_started'])
            self.assertFalse(any('TEST-' in str(p) or 'CAL-' in str(p) for p in out.rglob('*')))

    def test_stage_timeout_and_reap(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            local=base/'local';local.mkdir()
            mon=entry.Monitor(time.monotonic(),entry.owner_cpu(),local,base,base/'cache')
            env=dict(os.environ,PYTHONPATH=str(ROOT))
            with self.assertRaises(entry.Deadline):
                mon.command('model',[sys.executable,'-c','import time;time.sleep(30)'],env,time.monotonic()+1.1)
            self.assertTrue(entry.read(local/'model.usage.json')['all_descendants_reaped'])
            self.assertTrue(mon.complete)
            self.assertLess(mon.stages['model']['wall_seconds'],10)

    def test_partial_slot_retained(self):
        class Failed(Dummy):
            def read(self,rgb):
                raise ValueError('injected read failure')
        with tempfile.TemporaryDirectory() as temp:
            local=Path(temp)/'scratch';local.mkdir();out=Path(temp)/'drive'
            with self.assertRaisesRegex(ValueError,'injected'):
                worker.run(local,out,ALLOCATION,Failed())
            with ZipFile(out/'slots/DEV-00/slot-01/partial.zip') as z:
                self.assertEqual(len(z.namelist()),9)
            events=(local/'operations.jsonl').read_text()
            self.assertIn('"state": "ERROR"',events)
            self.assertFalse((out/'dev_decision.json').exists())

    def test_settlement_failure_precedes_stale_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);save(p/'summary.json',dict(status='dummy'))
            save(p/'settlement_failure.json',dict(resources=dict(wall_seconds=10)))
            r=entry.replay(p,ALLOCATION['budget'])
            self.assertEqual(r['status'],'SETTLEMENT_INCOMPLETE_READ_ONLY')
            self.assertTrue(r['unknown_tail'])


if __name__=='__main__':
    unittest.main(verbosity=2)
