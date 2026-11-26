"""Offline tests: full scientific workload with synthetic interfaces, never real weights."""
import ast,base64,copy,hashlib,io,json,os,shutil,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
from latent_control.io import save
from . import entry,gate,rules as R,worker

ROOT=entry.PACKAGE.parent
A=entry.read(entry.PACKAGE/'allocation.json')

class Toy:
    """Constant RGB interface fixtures; no watermark/generator efficacy simulation."""
    def __init__(self,out):self.out=out;self.h=None
    def generate(self,w):
        assert w['split'] in ('CAL','TEST')
        if w['split']=='TEST':
            assert (self.out/'cal_seal.json').is_file()
            assert len(list((self.out/'works').glob('CAL-*/completed.json')))==24
            assert entry.read(self.out/'selection_seal.json')['gains']==gate.GAINS
        self.h=object()
        return None,self.h,np.full((512,512,3),100,np.uint8)
    def uniform(self,h,bits,gain):
        assert h is self.h and gain in gate.GAINS.values()
        return np.full((512,512,3),151+list(gate.GAINS.values()).index(gain),np.uint8)
    def write(self,h,bits,action):
        assert h is self.h and action=='coarse'
        return np.full((512,512,3),150,np.uint8)
    def read(self,rgb):
        assert rgb.shape==(512,512,3) and rgb.dtype==np.uint8
        return [float(rgb[0,0,0]-100)/100]*48

def quality(clean,image):return {150:.01,151:.0099,152:.01,153:.0101}[int(image[0,0,0])]

def fake_child(local,out,scratch):
    b=Toy(out);worker.R.distance=quality
    save(local/'outcome.json',worker.run(local,out,A,b,b))

def install_dev(parent):
    dev=parent/entry.PREVIOUS;dev.mkdir(parents=True)
    for src,dst in [('dev_summary.json','summary.json'),('dev_quality.json','quality_records.json'),('allocation.json','allocation.json')]:
        shutil.copyfile(entry.PACKAGE/src,dev/dst)
    return dev

def fixture(i,split='CAL'):
    return dict(id=f'{split}-{i:03}',split=split,reg=[0,1,2],foreign=[16,17,18],
                quality=dict(coarse=.01,low=.0099,near=.01,high=.0101),
                raw={a:{k:[0.]*16 for k in R.QUERIES} for a in R.ARMS})

class Checks(unittest.TestCase):
    def test_frozen_dev_gate_and_allocation_unchanged(self):
        s=entry.read(entry.PACKAGE/'dev_summary.json');q=entry.read(entry.PACKAGE/'dev_quality.json')
        self.assertEqual(gate.validate_dev(s,q,A),dict(low=1.025,near=1.03,high=1.04))
        self.assertEqual((entry.PACKAGE/'allocation.json').read_bytes(),(ROOT/'strength_bracket/allocation.json').read_bytes())
        self.assertEqual((entry.PACKAGE/'PREREGISTRATION.md').read_bytes(),(ROOT/'strength_bracket/PREREGISTRATION.md').read_bytes())
        for key,val in [('retry',True),('resources_final',False),('cal_test_started',True),('training_updates',1),('unknown_tail',True)]:
            bad=copy.deepcopy(s);bad[key]=val
            with self.assertRaises(ValueError):gate.validate_dev(bad,q,A)
        bad=copy.deepcopy(q);bad[0]['loss']['coarse']+=.001
        with self.assertRaises(ValueError):gate.validate_dev(s,bad,A)
        bad=copy.deepcopy(A);bad['works'][9]['seed']+=1
        with self.assertRaises(ValueError):gate.validate_dev(s,q,bad)

    def test_calibration_all_arms_codecs_queries_and_strictness(self):
        for arm in R.ARMS:
            for query in R.QUERIES:
                rows=[fixture(i) for i in range(24)]
                rows[0]['raw']['coarse']['reg/equal3/png'][0]=100 # true member excluded
                rows[2]['raw'][arm][query][15]=7
                self.assertEqual(R.calibrate(rows)['tau'],7,(arm,query))
        r=fixture(0,'TEST');r['raw']['high'][R.PRIMARY][0]=7
        self.assertEqual(R.evaluate(r,7)['arms']['high'][R.PRIMARY]['returned'],[])
        r['raw']['high'][R.PRIMARY][0]=7.0001
        self.assertEqual(R.evaluate(r,7)['arms']['high'][R.PRIMARY]['returned'],[0])
        with self.assertRaises(ValueError):R.calibrate(rows[:23])
        rows[-1]['split']='TEST'
        with self.assertRaises(ValueError):R.calibrate(rows)
        for invalid in (float('nan'),float('inf')):
            r=fixture(0);r['raw']['near'][R.PRIMARY][0]=invalid
            with self.assertRaises(ValueError):R.validate_raw(r,r['raw'])

    def test_primary_secondary_and_bidirectional_errors(self):
        rows=[fixture(i,'TEST') for i in range(160)]
        for r in rows[:5]:r['raw']['coarse'][R.PRIMARY][0]=2
        reports=[R.evaluate(r,1) for r in rows];s=R.summarize(reports)
        self.assertEqual(s['status'],'COARSE_GAIN_PERSISTS_PENDING_VISUAL')
        self.assertEqual((s['wins'],s['losses'],s['ties']),(5,0,155))
        self.assertEqual(s['one_sided_exact_sign_p'],1/32)
        self.assertEqual(s['absolute']['coarse'][R.PRIMARY]['true_members'],5)
        rows[0]['raw']['coarse']['foreign/equal3/jpeg85'][9]=2
        s=R.summarize([R.evaluate(r,1) for r in rows]);self.assertEqual(s['status'],'INCONCLUSIVE')
        self.assertEqual(s['errors']['high']['coarse_new_false'][0]['ids'],[9])
        rows=[fixture(i,'TEST') for i in range(160)]
        rows[0]['raw']['high']['null/clean/png'][10]=2
        self.assertEqual(R.summarize([R.evaluate(r,1) for r in rows])['status'],'INCONCLUSIVE')
        # JPEG or low-arm gains never rescue a failed high-control PNG comparison.
        rows=[fixture(i,'TEST') for i in range(160)]
        for r in rows:r['raw']['coarse']['reg/equal3/jpeg85'][0]=2
        self.assertEqual(R.summarize([R.evaluate(r,1) for r in rows])['status'],'ORDINARY_CATCHUP_IN_THIS_SAMPLE')
        rows[0]['raw']['coarse']['foreign/equal2/png'][8]=2
        rows[0]['raw']['high']['foreign/equal2/png'][8]=2
        s=R.summarize([R.evaluate(r,1) for r in rows])
        self.assertEqual(s['errors']['high']['common_false'][0]['ids'],[8])
        self.assertEqual(s['absolute']['coarse']['foreign/equal2/png']['false_works'],1)
        self.assertIsNone(s['scientific_pass']);self.assertFalse(s['fwer_certified'])
        with self.assertRaises(ValueError):R.summarize(reports[:159])

    def test_full_owner_184_works_and_sealed_order_and_readonly(self):
        original=entry.Monitor.command
        def command(m,stage,cmd,env):
            if stage=='preparation':
                script="from latent_control.io import save;from pathlib import Path;import sys;save(Path(sys.argv[1])/'python.json',{'python':sys.executable})"
            else:
                script="from strength_compare.test_cpu import fake_child;from pathlib import Path;import sys;fake_child(*map(Path,sys.argv[1:]))"
            return original(m,stage,[sys.executable,'-c',script,*cmd[3:]],env)
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Monitor,'command',command):
            p=Path(tmp);dev=install_dev(p/'drive');before={f.name:f.read_bytes() for f in dev.iterdir()}
            result=entry.run_colab(p/'drive',p/'content');out=Path(result['output'])
            self.assertNotEqual(result['status'],'FORMAL_STOPPED_INCOMPLETE',result)
            self.assertEqual(result['operations'],{k:dict(started=n,completed=n,denominator=n) for k,n in R.LIMITS.items()})
            self.assertEqual(result['counts']['works_completed'],184)
            self.assertEqual(len(list((out/'works').glob('CAL-*/completed.json'))),24)
            self.assertEqual(len(list((out/'works').glob('TEST-*/completed.json'))),160)
            self.assertFalse(list((out/'works').glob('DEV-*')))
            self.assertFalse(result['dev_recomputed']);self.assertTrue(result['new_test_entered_after_cal_seal'])
            self.assertFalse(result['retry']);self.assertIsNone(result['phase_caps_seconds'])
            self.assertEqual(result['resources']['result_bytes'],entry.tree_bytes(out))
            self.assertEqual(len(entry.read(out/'paired_results.json')),160)
            events=[json.loads(x) for x in (out/'stage_events.jsonl').read_text().splitlines()]
            self.assertEqual([e['event'] for e in events],['SELECTION_SEALED','CAL_THRESHOLD_SEALED','TEST_ENTERED_AFTER_SEAL_VERIFIED'])
            ops=[json.loads(x) for x in (out/'operations.jsonl').read_text().splitlines()]
            starts=[e for e in ops if e['state']=='STARTED']
            self.assertEqual(len(starts),sum(R.LIMITS.values()))
            self.assertEqual(len({(e['kind'],e['id']) for e in starts}),len(starts))
            self.assertGreater(min(e['at'] for e in starts if e['id'].startswith('TEST-')),events[-1]['at'])
            for key in ('wall_seconds','gpu_occupancy_seconds','cpu_seconds','download_charge_bytes'):
                self.assertAlmostEqual(sum(x[key] for x in result['stages'].values()),result['resources'][key])
                for label in ('cumulative_resources','cumulative_budget_debits'):
                    self.assertAlmostEqual(result[label][key],entry.read(dev/'summary.json')[label][key]+result['resources'][key])
            with patch.object(entry,'gpu_ready',side_effect=AssertionError('No GPU on replay')),patch.object(entry.Monitor,'command',side_effect=AssertionError('No subprocess on replay')):
                self.assertEqual(entry.run_colab(p/'drive',p/'content'),result)
            self.assertEqual(before,{f.name:f.read_bytes() for f in dev.iterdir()})
            print('FULL_FIXTURE_VERIFIED',json.dumps(dict(operations=result['operations'],cal=24,test=160,dev_recomputed=False,phases=result['stages'])),flush=True)

    def test_failure_consumed_and_preserved(self):
        class Fail(Toy):
            def read(self,*args):raise RuntimeError('Injected reader error')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);local=p/'local';local.mkdir();out=p/'out';b=Fail(out)
            with self.assertRaisesRegex(RuntimeError,'Injected reader'):worker.run(local,out,A,b,b)
            log=(local/'operations.jsonl').read_bytes()
            self.assertIn(b'ERROR',log);self.assertIn(b'Traceback',log)
            self.assertEqual(len(list(out.glob('works/*/reserved.json'))),1)
            self.assertFalse((out/'cal_seal.json').exists())
            with self.assertRaises(FileExistsError):worker.run(local,out,A,b,b)
            self.assertEqual(log,(local/'operations.jsonl').read_bytes())

    def test_corrupt_cal_readback_stops_before_test(self):
        original=worker.append
        def append(path,row):
            original(path,row)
            if row.get('event')=='CAL_THRESHOLD_SEALED':
                seal=entry.read(path.parent/'cal_seal.json');seal['calibration']['tau']+=1;save(path.parent/'cal_seal.json',seal)
        with tempfile.TemporaryDirectory() as tmp,patch.object(worker.R,'distance',quality),patch.object(worker,'append',append):
            p=Path(tmp);local=p/'local';local.mkdir();out=p/'out';b=Toy(out)
            with self.assertRaisesRegex(RuntimeError,'CAL seal'):worker.run(local,out,A,b,b)
            self.assertEqual(len(list(out.glob('works/CAL-*/completed.json'))),24)
            self.assertFalse(list(out.glob('works/TEST-*')))

    def test_historical_highwater_and_failed_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);dev=install_dev(p);s=entry.read(dev/'summary.json')
            s['cumulative_resources']['wall_seconds']-=50
            s['cumulative_budget_debits']['cpu_seconds']+=10
            save(dev/'summary.json',s);h=entry.prior(A['budget'],p)
            self.assertEqual(h['cumulative_resources']['wall_seconds'],gate.load('dev_summary.json')['cumulative_resources']['wall_seconds'])
            self.assertEqual(h['cumulative_budget_debits']['cpu_seconds'],s['cumulative_budget_debits']['cpu_seconds'])
            out=p/entry.RUN_ID;save(out/'history.json',h)
            save(out/'receipts/000001.json',dict(resources=dict(wall_seconds=12,cpu_seconds=3)))
            save(out/'settlement_failure.json',dict(resources=dict(wall_seconds=15,cpu_seconds=4)))
            with patch.object(entry.Monitor,'command',side_effect=AssertionError('No retry')):
                r=entry.run_colab(p)
            self.assertTrue(r['unknown_tail']);self.assertFalse(r['retry'])
            self.assertEqual(r['cumulative_lower_bound']['wall_seconds'],h['cumulative_resources']['wall_seconds']+15)

    def test_unfinished_dev_prevents_reservation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'gpu_ready',return_value=True):
            p=Path(tmp);dev=install_dev(p);s=entry.read(dev/'summary.json');s['resources_final']=False;save(dev/'summary.json',s)
            with self.assertRaises(ValueError):entry.run_colab(p)
            self.assertFalse((p/entry.RUN_ID).exists())

    def test_notebook_payload_and_stdlib_bootstrap(self):
        nb=entry.read(ROOT/'notebooks/lawa_strength_bracket_compare_colab.ipynb')
        self.assertEqual(''.join(nb['cells'][0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
        tree=ast.parse(''.join(nb['cells'][2]['source']))
        payload=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
        with tempfile.TemporaryDirectory() as tmp:
            with ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
                for name in z.namelist():
                    self.assertEqual(z.read(name),(ROOT/name).read_bytes(),name)
                    self.assertFalse(name.endswith(('.ckpt','.safetensors')))
                self.assertNotIn('strength_bracket/worker.py',z.namelist());z.extractall(tmp)
            env=dict(os.environ,PYTHONPATH=tmp)
            script='from strength_compare.entry import check_freeze; from strength_compare.gate import validate_dev,load; check_freeze(); validate_dev(load("dev_summary.json"),load("dev_quality.json"),load("allocation.json")); import sys; assert "torch" not in sys.modules and "scipy" not in sys.modules'
            subprocess.run([sys.executable,'-S','-c',script],cwd=tmp,env=env,check=True,capture_output=True,text=True)
        entry.check_freeze()

if __name__=='__main__':unittest.main()
