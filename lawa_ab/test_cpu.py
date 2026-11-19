"""Run offline with pinned CPU dependencies. No real checkpoint or study work is consumed."""
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from . import assets,entry,worker
from .science import decision,diagnose,scores,signs
from latent_control.io import save

FROZEN=json.loads((worker.PACKAGE/'freeze.json').read_text())


class Toy:
    def generate(self,w):
        k=int(w['id'][-2:])
        clean=np.full((64,64,3),40+k*20,np.uint8)
        return np.zeros(4),clean,clean

    def write(self,h,bits):
        image=h.copy()
        image[0,:48,0]=[255*int(b) for b in bits]
        return image

    def read(self,image):
        return (image[0,:48,0].astype(float)-127.5).tolist()

    def probe(self,h,bits):
        return dict(gradient_rms=1.,analytic=1.,finite_difference=1.,passed=True)


def toy_main(local,out):
    # Synthetic content with precisely the same field shapes and denominators.
    with worker.phase(out,'load'):
        backend=Toy()
    save(local/'outcome.json',worker.run(backend,local,out,FROZEN))


class Trial(unittest.TestCase):
    def test_preparation_routes(self):
        from unittest.mock import Mock
        records=Path('/content/lawa-test/records')
        scratch=records.parent
        mkdir=Mock()
        with patch.object(Path,'mkdir',mkdir),patch.object(assets,'environment') as env,patch.object(assets,'unpack') as unpack,patch.object(assets,'fetch',side_effect=lambda url,dst,records:dst) as fetch:
            with self.assertRaises(ValueError):
                assets.prepare(records,Path('/content/drive/MyDrive/weights'))
            env.assert_not_called()
            assets.prepare(records,scratch)
            env.assert_called_once_with(records,scratch,scratch/'cache',scratch/'cache')
            unpack.assert_called_once_with(scratch/'cache/source.tar.gz',scratch/'official',strip=True,source_only=True)
            requests=[call.args for call in fetch.call_args_list]
            self.assertEqual(len(requests),len(assets.FILES)+2)
            self.assertEqual(requests[1],(assets.AUTHOR,scratch/'cache/last.ckpt',records))
            for name,(url,dst,_) in zip(assets.FILES,requests[2:]):
                self.assertEqual(dst,scratch/'cache/sd14'/name)
                self.assertIn('/'+assets.BASE+'/',url)

    def test_allocation(self):
        rows=FROZEN['works']
        self.assertEqual(len(set(w['prompt'] for w in rows)),16)
        self.assertEqual(len(set(w['seed'] for w in rows)),16)
        for family in ('registered','foreign'):
            a={n for w in rows[:8] for n in w[family]}
            b={n for w in rows[8:] for n in w[family]}
            self.assertEqual((len(a),len(b),len(a&b)),(8,8,0))
            self.assertEqual(len({tuple(sorted(w[family])) for w in rows}),16)
        self.assertEqual(len(set(FROZEN['codes'])),32)
        with self.assertRaises(ValueError):
            diagnose(rows[-1],{},FROZEN['codes'])

    def test_gate(self):
        def row(i,actual,endpoint,condition):
            r=dict(condition=condition,actual=actual,endpoint=endpoint)
            return dict(id=f'fit-{i:02}',soft=dict(rows=[r],all_separated=actual>0),
                        hard=dict(all_separated=actual>0))
        rows=[row(i,1,1,'1_png') for i in range(8)]
        self.assertEqual(decision(rows)['status'],'SOFT_SUFFICIENT_ON_FIT')
        rows[0]=row(0,0,1,'1_png')
        rows[1]=row(1,-1,1,'1_jpeg85')
        self.assertEqual(decision(rows)['status'],'NO_REPEATED_RESCUABLE_CONDITION')
        rows[1]=row(1,-1,1,'1_png')
        self.assertEqual(decision(rows)['status'],'C_PILOT_SIGNAL_ON_FIT')
        with self.assertRaises(ValueError):
            decision(rows+rows[:1])
        with self.assertRaises(ValueError):
            scores([float('nan')]*48,np.stack([signs('0'*48)]))

    def test_complete_entry_readonly_and_failure(self):
        original=entry.Meter.command
        def cmd(owner,stage,command,env):
            if stage=='preparation':
                command=[sys.executable,'-c','from latent_control.io import save;import sys;save(sys.argv[1],dict(python=sys.executable))',owner.local/'python.json']
            else:
                command=[sys.executable,'-m','lawa_ab.test_cpu','toy',*command[-3:-1]]
            return original(owner,stage,command,env)
        with tempfile.TemporaryDirectory() as t, patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Meter,'command',cmd):
            base=Path(t);parent=base/'drive'
            baseline=entry.prior_history(parent)
            self.assertEqual(baseline['cumulative_resources']['wall_seconds'],39821.25273105507)
            save(parent/entry.PRIOR,baseline)
            run=lambda:entry.run_colab(parent,base/'runtime')
            result=run()
            self.assertEqual(result['status'],'AB_COMPLETE',str(result))
            out=Path(result['output'])
            self.assertEqual(result['counts'],dict(started=16,completed=16,denominator=16))
            self.assertEqual(result['resources']['result_bytes'],entry.tree_bytes(out))
            for key in ('wall_seconds','gpu_occupancy_seconds','cpu_seconds','download_charge_bytes'):
                self.assertAlmostEqual(sum(s[key] for s in result['stages'].values()),result['resources'][key])
            for key in ('cumulative_resources','cumulative_budget_debits'):
                self.assertAlmostEqual(result[key]['cpu_seconds'],result['resources']['cpu_seconds']+baseline[key]['cpu_seconds'])
            self.assertEqual(set(result['worker_phases']),{'A','B_fit','B_sealed','load'})
            self.assertEqual(result,run())
            self.assertEqual(len(list((out/'sealed_holdout').glob('*.zip'))),8)
            self.assertEqual(len(list((out/'fit').glob('*/continuous.json'))),8)
            self.assertLessEqual((out/'fit_decision.json').stat().st_mtime_ns,
                                 (out/'work_receipts/holdout-00/started.json').stat().st_mtime_ns)
            self.assertEqual(result['holdout_decision_inputs'],0)
            (out/'summary.json').unlink()
            self.assertEqual(run()['status'],'ACTIVE_OR_INTERRUPTED_READ_ONLY')
            self.assertEqual(len(list((out/'work_receipts').iterdir())),16)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)
            with patch.object(Toy,'probe',return_value=dict(passed=False)):
                result=worker.run(Toy(),path/'local',path,FROZEN)
            self.assertEqual(result['status'],'A_INTERFACE_OR_CARRIER_FAILED')
            self.assertEqual(len(list((path/'work_receipts').iterdir())),2)
            self.assertFalse((path/'fit_decision.json').exists())

    def test_partial_holdout(self):
        class Broken(Toy):
            reads=0
            def read(self,image):
                self.reads+=1
                if self.reads==8*22+4:
                    raise ValueError('injected extraction failure')
                return super().read(image)
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)
            with self.assertRaisesRegex(ValueError,'injected'):
                worker.run(Broken(),path/'local',path,FROZEN)
            self.assertTrue((path/'fit_decision.json').exists())
            self.assertEqual(entry.tallies(path),dict(started=9,completed=8,denominator=16))
            self.assertTrue((path/'work_receipts/holdout-00/exception.txt').is_file())
            # Inspect only an invented fixture archive. No study images exist in a CPU check.
            with ZipFile(path/'sealed_holdout/holdout-00.zip') as archive:
                self.assertEqual(len(archive.read('readouts.jsonl').splitlines()),3)
                self.assertNotIn('diagnostic.json',archive.namelist())

    def test_packed_notebook(self):
        import nbformat
        from types import ModuleType,SimpleNamespace
        nb=nbformat.read(worker.PACKAGE.parent/'notebooks/lawa_ab_colab.ipynb',as_version=4)
        nbformat.validate(nb)
        mount,runner=[c.source for c in nb.cells if c.cell_type=='code']
        call,=[n for n in ast.walk(ast.parse(runner)) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode']
        with ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(call.args[0])))) as payload:
            for path in payload.namelist():
                self.assertEqual(payload.read(path),(worker.PACKAGE.parent/path).read_bytes())
        google=ModuleType('google')
        google.colab=ModuleType('google.colab')
        events=[]
        google.colab.drive=SimpleNamespace(mount=events.append)
        with tempfile.TemporaryDirectory() as t:
            with patch.dict(sys.modules,{'google':google,'google.colab':google.colab}),patch.dict(os.environ,{'PATH':''}),patch.object(tempfile,'mkdtemp',return_value=t):
                ns={}
                exec(mount,ns)
                exec(runner,ns)
                self.assertEqual(ns['result']['status'],'GPU_REQUIRED')
        self.assertEqual(events,['/content/drive'])


if __name__=='__main__':
    if sys.argv[1:2]==['toy']:
        toy_main(Path(sys.argv[2]),Path(sys.argv[3]))
    else:
        unittest.main()
