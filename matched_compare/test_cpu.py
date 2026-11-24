import ast,base64,io,json,os,sys,tempfile,time,unittest,copy,hashlib
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
from latent_control.io import save
from . import rules as R,worker,entry

A=json.loads((entry.PACKAGE/'allocation.json').read_text())

class Dummy:
    def __init__(self,out=None):self.out=out;self.h=None
    def generate(self,w):
        if w['split']=='TEST':
            assert (self.out/'cal_seal.json').is_file()
            assert len(list((self.out/'works').glob('CAL-*/completed.json')))==24
        self.h=object()
        return None,self.h,np.full((512,512,3),110,np.uint8)
    def uniform(self,h,bits,gain):
        assert h is self.h and gain==1.025
        return np.full((512,512,3),120,np.uint8)
    def write(self,h,bits,action):
        assert h is self.h and action=='coarse'
        return np.full((512,512,3),120,np.uint8)
    def read(self,rgb):return [float(rgb[0,0,0]-110)/10]*48

def fixture(i,split='CAL'):
    w=dict(id=f'{split}-{i:03}',split=split,reg=[0,1,2],foreign=[16,17,18],quality=dict(ordinary=1.,coarse=1.))
    keys=[f'{g}/{v}/{c}' for g in ('reg','foreign') for v in ('single0','single1','single2','equal2','equal3','unequal3') for c in ('png','jpeg85')]+['null/clean/png','null/clean/jpeg85']
    w['raw']={a:{k:[0.]*16 for k in keys} for a in R.ARMS}
    return w

class Checks(unittest.TestCase):
    def test_calibration_family_and_strictness(self):
        rows=[fixture(i) for i in range(24)]
        rows[0]['raw']['coarse']['reg/equal3/png'][0]=99 # guilty excluded
        rows[1]['raw']['ordinary']['foreign/equal3/jpeg85'][9]=3
        seal=R.calibrate(rows);self.assertEqual(seal['tau'],3)
        r=fixture(0,'TEST');r['raw']['coarse'][R.PRIMARY][0]=3
        self.assertEqual(R.evaluate(r,3)['arms']['coarse'][R.PRIMARY]['returned'],[])
        r['raw']['coarse'][R.PRIMARY][0]=3.00001
        self.assertEqual(R.evaluate(r,3)['arms']['coarse'][R.PRIMARY]['returned'],[0])
        with self.assertRaises(ValueError):R.calibrate(rows[:-1])
    def test_joint_verdict(self):
        rows=[fixture(i,'TEST') for i in range(80)]
        for r in rows[:8]:r['raw']['coarse'][R.PRIMARY][0]=2
        reports=[R.evaluate(r,1) for r in rows]
        result=R.adjudicate(reports)
        self.assertTrue(all(result['gates'].values()));self.assertEqual(result['net_members'],8)
        self.assertEqual(result['one_sided_exact_sign_p'],1/256)
        rows[0]['raw']['coarse']['foreign/equal3/jpeg85'][5]=2
        self.assertFalse(R.adjudicate([R.evaluate(r,1) for r in rows])['gates']['no_new_false_attribution'])
        for r in reports:r['quality']['ordinary']=1.03
        self.assertFalse(R.adjudicate(reports)['gates']['quality_proxy_equivalent'])
        with self.assertRaises(ValueError):R.adjudicate(reports[:-1])
    def test_full_owner_cal_seal_test_and_accounting(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);drive=base/'drive'
            save(drive/entry.PREVIOUS/'summary.json',json.loads((entry.PACKAGE/'dev_summary.json').read_text()))
            save(drive/entry.PREVIOUS/'quality_records.json',json.loads((entry.PACKAGE/'dev_quality.json').read_text()))
            original=entry.Monitor.command
            def command(m,stage,cmd,env,end):
                if stage=='preparation':script="from latent_control.io import save;from pathlib import Path;import sys;save(Path(sys.argv[1])/'python.json',{'python':sys.executable})"
                else:script=("from matched_compare.test_cpu import Dummy;from matched_compare import worker as w;"
                    "from pathlib import Path;import sys,json;from latent_control.io import save;"
                    "loc,out,s=map(Path,sys.argv[1:]);d=Dummy(out);"
                    "save(loc/'outcome.json',w.run(loc,out,json.loads((w.PACKAGE/'allocation.json').read_text()),d,d))")
                return original(m,stage,[sys.executable,'-c',script,*cmd[3:]],env,end)
            with patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Monitor,'command',command):
                result=entry.run_colab(drive,base/'content')
                self.assertEqual(result['status'],'MATCHED_COMPARISON_NOT_SUPPORTED',result)
                self.assertEqual(entry.run_colab(drive,base/'content'),result)
            out=drive/entry.RUN_ID
            for k,v in R.LIMITS.items():self.assertEqual(result['operations'][k],dict(started=v,completed=v,denominator=v))
            self.assertEqual(result['counts']['works_completed'],104)
            self.assertEqual(entry.tree_bytes(out),result['resources']['result_bytes'])
            self.assertAlmostEqual(sum(s['wall_seconds'] for s in result['stages'].values()),result['resources']['wall_seconds'])
            self.assertEqual(result['cumulative_resources']['wall_seconds'],A['budget']['historical_resources']['wall_seconds']+result['resources']['wall_seconds'])
            self.assertAlmostEqual(result['family_budget_used_seconds']['model'],182.728355496+result['stages']['model']['wall_seconds'])
            events=[json.loads(x)['event'] for x in (out/'stage_events.jsonl').read_text().splitlines()]
            self.assertEqual(events,['SELECTION_SEALED','CAL_THRESHOLD_SEALED','TEST_ENTERED_AFTER_SEAL_VERIFIED'])
            first=json.loads((out/'works/TEST-000/reserved.json').read_text())['at']
            sealed=json.loads((out/'cal_seal.json').read_text())['at']
            self.assertGreaterEqual(first,sealed)
            self.assertEqual(len(json.loads((out/'paired_results.json').read_text())),80)
    def test_partial_cal_failure_no_test(self):
        class Fail(Dummy):
            def read(self,rgb):raise RuntimeError('injected read error')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);loc=p/'local';loc.mkdir();out=p/'out';d=Fail(out)
            with self.assertRaises(RuntimeError):worker.run(loc,out,A,d,d)
            self.assertFalse((out/'cal_seal.json').exists())
            self.assertFalse(list(out.glob('works/TEST-*')))
            self.assertIn('ERROR',(loc/'operations.jsonl').read_text())
            self.assertEqual(entry.operation_counts(loc)['generate']['started'],1)
    def test_seal_tamper_stops_before_test(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);loc=p/'local';loc.mkdir();out=p/'out';d=Dummy(out)
            original=worker.seal
            def sabotage(path,v):
                result=original(path,v)
                if path.name=='cal_seal.json':
                    value=copy.deepcopy(v);value['calibration']['tau']+=1;save(path,value)
                return result
            with patch.object(worker,'seal',sabotage),self.assertRaises(RuntimeError):worker.run(loc,out,A,d,d)
            self.assertFalse(list(out.glob('works/TEST-*')))
            self.assertEqual(entry.operation_counts(loc)['generate']['started'],24)
    def test_history_and_interruption(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            with self.assertRaises(FileNotFoundError):entry.prior(A['budget'],p)
            save(p/entry.PREVIOUS/'summary.json',dict(resources_final=False))
            with self.assertRaises(ValueError):entry.prior(A['budget'],p)
            out=p/entry.RUN_ID;out.mkdir()
            save(out/'history.json',dict(cumulative_resources=A['budget']['historical_resources'],cumulative_budget_debits=A['budget']['historical_debits']))
            r=entry.replay(out,A['budget'])
            self.assertTrue(r['unknown_tail']);self.assertFalse(r['retry'])
            self.assertAlmostEqual(r['cumulative_conservative_debit']['wall_seconds'],A['budget']['historical_debits']['wall_seconds']+sum(A['budget']['proposed_wall_caps_seconds'].values()))
    def test_invalid_history_is_sealed_without_work(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            old=json.loads((entry.PACKAGE/'dev_summary.json').read_text())
            old['cumulative_resources']['wall_seconds']+=10
            save(p/'drive'/entry.PREVIOUS/'summary.json',old)
            with patch.object(entry,'gpu_ready',return_value=True):r=entry.run_colab(p/'drive',p/'content')
            self.assertFalse(r['history_verified'])
            self.assertEqual(r['operations']['generate']['started'],0)
            self.assertIn('DEV receipt changed',r['error'])
            self.assertGreaterEqual(r['cumulative_resources']['wall_seconds'],old['cumulative_resources']['wall_seconds'])
            self.assertEqual(entry.run_colab(p/'drive',p/'content'),r)
            self.assertIn('preparation',r['stages'])
    def test_payload_allocation_and_carry_forward(self):
        entry.check_freeze()
        from tools.freeze_matched_compare import collect
        prior=set();seeds=set()
        for name,sha in A['history'].items():
            data=(entry.PACKAGE.parent/name).read_bytes();self.assertEqual(hashlib.sha256(data).hexdigest(),sha)
            collect(json.loads(data),prior,seeds)
        self.assertEqual(len(A['works']),104)
        self.assertEqual(len({w['seed'] for w in A['works']}),104)
        for w in A['works']:
            self.assertNotIn(w['prompt'].lower(),prior);self.assertNotIn(w['seed'],seeds)
        self.assertEqual(sum(w['split']=='CAL' for w in A['works']),24)
        nb=json.loads((entry.PACKAGE.parent/'notebooks/lawa_matched_compare_colab.ipynb').read_text())
        self.assertEqual(''.join(nb['cells'][0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
        tree=ast.parse(''.join(nb['cells'][2]['source']))
        b=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
        with ZipFile(io.BytesIO(base64.b64decode(b))) as z:
            for name in z.namelist():self.assertEqual(z.read(name),(entry.PACKAGE.parent/name).read_bytes())
        caps=A['budget']['proposed_wall_caps_seconds']
        self.assertAlmostEqual(caps['preparation']+420.641923073,1080)
        self.assertAlmostEqual(caps['model']+182.728355496,5400)

if __name__=='__main__':unittest.main()
