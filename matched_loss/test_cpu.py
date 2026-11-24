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
import numpy as np
from . import entry, worker
from latent_control.io import save

ALLOCATION=entry.read(entry.PACKAGE/'allocation.json')

class Dummy:
    def generate(self,work):
        return None,None,np.full((512,512,3),110,dtype=np.uint8)
    def uniform(self,h,bits,gain):
        return np.full((512,512,3),round(110+10*gain),dtype=np.uint8)
    def write(self,h,bits,action):
        return self.uniform(h,bits,1.025)
    def read(self,*args):
        raise AssertionError('Forbidden read')

class Checks(unittest.TestCase):
    def test_rule_boundaries(self):
        rows=[dict(work=str(i),loss={'coarse':1.,'1.0':1.04,'1.025':1.,'1.05':1.08,'1.075':1.1}) for i in range(8)]
        self.assertEqual(worker.decide(rows)['selected_gain'],1.025)
        self.assertTrue(worker.decide(rows)['matching_gate_passed'])
        for r in rows:r['loss']['1.025']=1.03
        self.assertFalse(worker.decide(rows)['matching_gate_passed'])
        with self.assertRaises(ValueError):worker.decide(rows[:7])
    def test_full_owner_and_replay(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t)
            original=entry.Monitor.command
            def command(m,stage,cmd,env,end):
                if stage=='preparation':
                    script="from latent_control.io import save;from pathlib import Path;import sys;save(Path(sys.argv[1])/'python.json',{'python':sys.executable})"
                else:
                    script=("from matched_loss.test_cpu import Dummy;from matched_loss import worker as w;"
                            "from pathlib import Path;import sys,json;from latent_control.io import save;"
                            "loc,out,s=map(Path,sys.argv[1:]);d=Dummy();"
                            "save(loc/'outcome.json',w.run(loc,out,json.loads((w.PACKAGE/'allocation.json').read_text()),d,d))")
                return original(m,stage,[sys.executable,'-c',script,*cmd[3:]],env,end)
            with patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Monitor,'command',command):
                result=entry.run_colab(base/'drive',base/'content')
                self.assertEqual(result['status'],'DEV_MATCHING_GATE_MET',result)
                self.assertEqual(entry.run_colab(base/'drive',base/'content'),result)
            out=base/'drive'/entry.RUN_ID
            self.assertEqual(result['operations']['write']['completed'],240)
            self.assertEqual(result['operations']['generate']['completed'],8)
            self.assertEqual(result['operations']['read']['started'],0)
            self.assertEqual(result['counts']['works_completed'],8)
            self.assertEqual(entry.tree_bytes(out),result['resources']['result_bytes'])
            self.assertAlmostEqual(sum(x['wall_seconds'] for x in result['stages'].values()),result['resources']['wall_seconds'])
            self.assertEqual(result['cumulative_resources']['wall_seconds'],ALLOCATION['budget']['historical_resources']['wall_seconds']+result['resources']['wall_seconds'])
    def test_interruption_readonly_and_history(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);out=p/entry.RUN_ID;out.mkdir()
            r=entry.run_colab(p)
            self.assertTrue(r['unknown_tail'])
            self.assertFalse(r['retry'])
            self.assertEqual(r['cumulative_conservative_debit']['wall_seconds'],ALLOCATION['budget']['historical_debits']['wall_seconds']+2040)
            old=dict(resources_final=False)
            save(p/entry.PREVIOUS/'summary.json',old)
            with self.assertRaises(ValueError):entry.prior(ALLOCATION['budget'],p)
    def test_partial_failure(self):
        class Fail(Dummy):
            def uniform(self,*args):raise RuntimeError('Injected')
            def write(self,*args):raise RuntimeError('Injected')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);local=p/'local';local.mkdir();out=p/'out'
            with self.assertRaises(RuntimeError):worker.run(local,out,ALLOCATION,Fail(),Fail())
            self.assertTrue(list(out.rglob('clean.png')))
            self.assertFalse(list(out.rglob('completed.json')))
            self.assertIn('ERROR',(local/'operations.jsonl').read_text())
    def test_reap(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);loc=p/'local';loc.mkdir()
            m=entry.Monitor(time.monotonic(),entry.owner_cpu(),loc,p,p/'cache')
            with self.assertRaises(entry.Deadline):
                m.command('model',[sys.executable,'-c','import time;time.sleep(30)'],dict(os.environ),time.monotonic()+1.)
            self.assertTrue(entry.read(loc/'model.usage.json')['all_descendants_reaped'])
    def test_uniform_tensor_hook(self):
        import torch
        from .carrier import Carrier
        class Backend:
            def __init__(self):self.convs=[torch.nn.Linear(2,2,bias=False) for _ in range(6)]
            def write(self,h,bits):return torch.stack([m(h) for m in self.convs])
        b=Backend();c=Carrier(b,.1)
        c.branches=lambda:[(None,m,1) for m in b.convs]
        h=torch.ones(2);original=b.write(h,None)
        self.assertTrue(torch.equal(c.uniform(h,None,1.),original))
        for gain in worker.GAINS[1:]:
            torch.testing.assert_close(c.uniform(h,None,gain),original*gain)
            self.assertTrue(torch.equal(b.write(h,None),original))
        with patch.object(b,'write',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):c.uniform(h,None,1.025)
        self.assertTrue(all(not m._forward_hooks for m in b.convs))
    def test_embedded_notebook(self):
        nb=json.loads((entry.PACKAGE.parent/'notebooks/lawa_matched_loss_dev_colab.ipynb').read_text())
        self.assertEqual(''.join(nb['cells'][0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
        source=''.join(nb['cells'][2]['source'])
        tree=ast.parse(source)
        data=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
        with ZipFile(io.BytesIO(base64.b64decode(data))) as z:
            for name in z.namelist():
                self.assertEqual(z.read(name),(entry.PACKAGE.parent/name).read_bytes())
        entry.check_freeze()

if __name__=='__main__':unittest.main()

