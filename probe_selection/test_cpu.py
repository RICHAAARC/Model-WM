"""Only fixtures and random CPU tensors; no study generation or model weights."""
import ast
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
from PIL import Image
from latent_control.io import save
from . import assets,entry,owner,rules as R,worker

M=worker.read(worker.PACKAGE/'freeze.json')

class Fixture:
    actions=R.ACTIONS
    def generate(self,w):
        rgb=np.full((16,16,3),100+w['seed']%32,dtype=np.uint8)
        return np.zeros((1,4,8,8)),rgb,rgb
    def restore(self,z):
        return np.full((16,16,3),120,np.uint8)
    def write(self,h,bits,action):
        x=h.copy()
        x.reshape(-1)[:48]=[30+190*int(b) for b in bits]
        return x
    def read(self,x):
        return (x.reshape(-1)[:48].astype(float)-126.).tolist()

def fixture_old(parent):
    old=parent/worker.PRIOR
    save(old/'freeze.json',worker.read(entry.entry.PACKAGE/'freeze.json'))
    save(old/'summary.json',dict(status='AB_COMPLETE',resources_final=True,
        cumulative_resources=dict(cpu_seconds=31),cumulative_budget_debits=dict(cpu_seconds=42)))
    toy=Fixture()
    for w in M['dev']:
        path=old/'fit'/w['id']
        path.mkdir(parents=True)
        z=np.zeros((1,4,8,8));np.save(path/'latent.npy',z)
        clean=toy.restore(z)
        Image.fromarray(clean).save(path/'null_png.png')
        for j in range(2):
            Image.fromarray(toy.write(clean,M['codes'][w['reg'][j]],'native')).save(path/f'reg_{j}_png.png')
    return old

class Checks(unittest.TestCase):
    def test_assets(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t); prior=base/worker.PRIOR/'cache'
            prior.mkdir(parents=True)
            (prior/'last.ckpt').write_bytes(b'fixture only')
            save(prior/'last.ckpt.complete.json',dict(url=assets.assets.AUTHOR))
            scratch=base/'new';local=scratch/'records'
            assets.reuse_assets(local,scratch)
            self.assertEqual((scratch/'cache/last.ckpt').resolve(),(prior/'last.ckpt').resolve())
            self.assertEqual(worker.read(local/'cache_reuse.json')['reused'],['last.ckpt'])
            with self.assertRaises(ValueError):
                assets.prepare(local,Path('/content/drive/MyDrive/weights'))

    def test_freeze(self):
        self.assertEqual([r['split'] for r in M['works']],['cal']*24+['explore']*8)
        self.assertEqual(len(set(M['codes']+M['probe_codes'])),48)
        self.assertEqual(len({r['seed'] for r in M['works']}),32)
        self.assertTrue(all(len(set(r['reg']+r['foreign']))==6 for r in M['works']))
        self.assertTrue(all(len(c)==48 and c.count('1')==24 for c in M['codes']+M['probe_codes']))

    def test_common_rule(self):
        self.assertEqual(worker.issue_action(lambda: self.fail('Unexpected probe'),'native',1.1,False)[1]['margin'],'native')
        self.assertEqual(owner.add(dict(peak_ram_bytes=100,cpu_seconds=2),dict(cpu_seconds=3)),
                         dict(peak_ram_bytes=100,cpu_seconds=5))
        stats={a:dict(margin=i,confidence=-i,ordinary=2-i,quality=1.) for i,a in enumerate(R.ACTIONS)}
        stats['coarse']['quality']=1.2
        self.assertEqual(R.pick(stats,'coarse',1.10),dict(native='native',margin='fine',ordinary='native',confidence='native',**{'global':'coarse'}))
        stats['native']['margin']=1
        self.assertEqual(R.pick(stats,'native',1.10)['margin'],'native')
        v={R.PRIMARY:[0.]*16};w={'reg':[0,1,2]}
        self.assertTrue(R.evaluate(w,v,0)[R.PRIMARY]['rejected'])
        v[R.PRIMARY][0]=v[R.PRIMARY][7]=1
        self.assertEqual(R.negative(w,v),1)
        self.assertFalse(R.evaluate(w,v,.5)[R.PRIMARY]['hit'])

    def test_verdict(self):
        def report(i):
            w=dict(reg=[0,1,2]);x={a:{R.PRIMARY:[0.]*16} for a in R.ACTIONS}
            best=R.ACTIONS[1+i%2]
            x[best][R.PRIMARY][0]=1.
            ch=dict(native='native',margin=best,ordinary='native',confidence='native',**{'global':'fine'})
            tau={k:.5 for k in list(ch)+['fixed_'+a for a in R.ACTIONS]}
            return worker.audit_row(dict(w,id=str(i)),x,{a:1. for a in R.ACTIONS},ch,tau,1.1)
        rows=[report(i) for i in range(8)]
        self.assertEqual(R.verdict(rows)['status'],'NUMERIC_SIGNAL_PENDING_VISUAL_REVIEW')
        rows[0]['arms']['margin'][R.PRIMARY]['false']=[3]
        self.assertNotEqual(R.verdict(rows)['status'],'NUMERIC_SIGNAL_PENDING_VISUAL_REVIEW')

    def test_partial_failure(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);old=fixture_old(base);out=base/'new';local=base/'runtime/records'
            with patch.object(Fixture,'read',side_effect=RuntimeError('injected read failure')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    worker.run(Fixture(),M,local,out,old)
            claim=out/'work_receipts/fit-00'
            self.assertTrue((claim/'exception.txt').exists())
            self.assertFalse((claim/'completed.json').exists())
            with ZipFile(out/'archives/fit-00.zip') as z:
                self.assertIn('events.jsonl',z.namelist())
            self.assertFalse((out/'thresholds.json').exists())

    def test_entry_and_full_synthetic(self):
        original=owner.Meter.command
        def commands(meter,stage,cmd,env):
            if stage=='preparation':
                cmd=[sys.executable,'-c','from latent_control.io import save;import sys;save(sys.argv[1],dict(python=sys.executable))',meter.local/'python.json']
            else:
                cmd=[sys.executable,'-m','probe_selection.test_cpu','toy',*cmd[-3:-1]]
            return original(meter,stage,cmd,env)
        with tempfile.TemporaryDirectory() as t,patch.object(owner,'gpu_ready',return_value=True),patch.object(owner.Meter,'command',commands):
            base=Path(t);drive=base/'drive'
            run=lambda:entry.run_colab(drive,base/'runtime')
            self.assertEqual(run()['status'],'MISSING_HISTORICAL_INPUT')
            fixture_old(drive)
            result=run()
            self.assertIn('operations',result,result)
            self.assertEqual(result['operations'],worker.LIMITS)
            self.assertEqual(result['counts'],dict(started=40,completed=40,denominator=40))
            out=Path(result['output'])
            self.assertEqual(result['resources']['result_bytes'],owner.tree_bytes(out))
            self.assertAlmostEqual(result['cumulative_resources']['cpu_seconds'],31+result['resources']['cpu_seconds'])
            for k in ('wall_seconds','cpu_seconds','gpu_occupancy_seconds','download_charge_bytes'):
                self.assertAlmostEqual(sum(s[k] for s in result['stages'].values()),result['resources'][k])
            self.assertEqual(len(worker.read(out/'thresholds.json')['cal']),24)
            self.assertLessEqual((out/'thresholds.json').stat().st_mtime_ns,(out/'work_receipts/explore-00/started.json').stat().st_mtime_ns)
            self.assertEqual(len(list((out/'exploration').glob('*.json'))),8)
            for row in M['works']:
                with ZipFile(out/'archives'/(row['id']+'.zip')) as z:
                    events=[json.loads(l) for l in z.read('events.jsonl').splitlines()]
                    kinds=[v['kind'] for v in events if v['state']=='started']
                    self.assertLess(max(j for j,v in enumerate(kinds) if v=='probe_extract'),kinds.index('official_write'))
                    self.assertIn('coarse/reg/equal3.jpg',z.namelist())
            self.assertEqual(result,run())
            (out/'summary.json').unlink()
            interrupted=run()
            self.assertEqual(interrupted['status'],'ACTIVE_OR_INTERRUPTED_READ_ONLY')
            self.assertTrue(interrupted['unknown_tail'])
            self.assertEqual(interrupted['counts']['started'],40)
            self.assertGreaterEqual(interrupted['cumulative_lower_bound']['result_bytes'],owner.tree_bytes(out))

    def test_notebook(self):
        import nbformat
        from types import ModuleType,SimpleNamespace
        nb=nbformat.read(worker.PACKAGE.parent/'notebooks/lawa_probe_selection_colab.ipynb',as_version=4)
        nbformat.validate(nb)
        cells=[c.source for c in nb.cells if c.cell_type=='code']
        for cell in cells:
            compile(cell,'<notebook>','exec')
        node,=[n for n in ast.walk(ast.parse(cells[1])) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode']
        with ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(node.args[0])))) as z:
            for p in z.namelist():
                self.assertEqual(z.read(p),(worker.PACKAGE.parent/p).read_bytes())
        with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules),patch.object(tempfile,'mkdtemp',return_value=t):
            google=ModuleType('google');colab=ModuleType('google.colab')
            seen=[];colab.drive=SimpleNamespace(mount=seen.append);google.colab=colab
            sys.modules.update({'google':google,'google.colab':colab})
            ns={}
            exec(cells[0],ns);exec(cells[1],ns)
            self.assertEqual(seen,['/content/drive'])
            self.assertEqual(ns['result']['status'],'MISSING_HISTORICAL_INPUT')

if __name__=='__main__':
    if sys.argv[1:2]==['toy']:
        local,out=map(Path,sys.argv[2:])
        save(local/'outcome.json',worker.run(Fixture(),M,local,out,out.parent/worker.PRIOR))
    else:
        unittest.main()
