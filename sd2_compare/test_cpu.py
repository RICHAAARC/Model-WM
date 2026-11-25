"""Offline engineering checks: synthetic images, no real weights or GPU."""
import ast,base64,copy,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
from latent_control.io import save
from . import entry,gate,worker
from matched_compare import worker as shared
from matched_compare import rules
from matched_compare.test_cpu import fixture
P=entry.PACKAGE
A=json.loads((P/'allocation.json').read_text())

class Dummy:
    def __init__(self,out):self.out=out;self.h=None
    def generate(self,w):
        assert w['split'] in ('CAL','TEST')
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

def dev(parent):
    out=parent/entry.PREVIOUS
    for src,dst in [('dev_summary.json','summary.json'),('dev_quality.json','quality_records.json')]:
        save(out/dst,json.loads((P/src).read_text()))
    save(out/'allocation.json',json.loads((P.parent/'sd2_matched/allocation.json').read_text()))
    return out

class Checks(unittest.TestCase):
    def test_freeze_and_exact_original_rows(self):
        entry.check_freeze()
        frozen=json.loads((P.parent/'sd2_matched/allocation.json').read_text())
        self.assertEqual(A['works'],[x for x in frozen['works'] if x['split']!='DEV'])
        self.assertEqual(A['codes'],frozen['codes'])
        self.assertEqual((sum(w['split']=='CAL' for w in A['works']),sum(w['split']=='TEST' for w in A['works'])),(24,80))
        self.assertIs(worker.run,shared.run)
        self.assertEqual(entry.LIMITS,rules.LIMITS)
    def test_dev_gate_checks_selection_and_ledger(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);d=dev(p);s=gate.verify_dev(d)
            self.assertEqual(s['selected_gain'],1.025)
            changed=copy.deepcopy(s);changed['cumulative_resources']['wall_seconds']-=1
            save(d/'summary.json',changed)
            with self.assertRaisesRegex(ValueError,'cumulative_resources'):gate.verify_dev(d)
            save(d/'summary.json',s)
            q=json.loads((d/'quality_records.json').read_text());q[0]['loss']['1.025']*=.8
            save(d/'quality_records.json',q)
            with self.assertRaisesRegex(ValueError,'quality changed'):gate.verify_dev(d)
    def test_missing_dev_prevents_reservation_and_work(self):
        with tempfile.TemporaryDirectory() as t,patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Monitor,'command') as command:
            p=Path(t)
            with self.assertRaises(FileNotFoundError):entry.run_colab(p,p/'content')
            command.assert_not_called()
            self.assertFalse((p/entry.RUN_ID).exists())
    def test_full_owner_cal_seal_test_fees_and_readonly(self):
        original=entry.Monitor.command
        def command(m,stage,cmd,env):
            if stage=='preparation':
                script="from latent_control.io import save;from pathlib import Path;import sys;save(Path(sys.argv[1])/'python.json',{'python':sys.executable})"
            else:
                script=("from sd2_compare.test_cpu import Dummy;from sd2_compare import worker as w;"
                        "from pathlib import Path;import sys,json;from latent_control.io import save;"
                        "loc,out,s=map(Path,sys.argv[1:]);d=Dummy(out);"
                        "save(loc/'outcome.json',w.run(loc,out,json.loads((w.PACKAGE/'allocation.json').read_text()),d,d))")
            return original(m,stage,[sys.executable,'-c',script,*cmd[3:]],env)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);drive=p/'drive';d=dev(drive)
            before={x.name:x.read_bytes() for x in d.iterdir()}
            with patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Monitor,'command',command):
                r=entry.run_colab(drive,p/'content')
                self.assertEqual(r['status'],'MATCHED_COMPARISON_NOT_SUPPORTED',r)
                self.assertEqual(r,entry.run_colab(drive,p/'content'))
            out=Path(r['output'])
            self.assertEqual(before,{x.name:x.read_bytes() for x in d.iterdir()})
            self.assertEqual({k:v['completed'] for k,v in r['operations'].items()},entry.LIMITS)
            self.assertEqual(r['counts']['works_completed'],104)
            self.assertFalse(list(out.glob('works/DEV-*')))
            self.assertFalse(r['dev_recomputed'])
            self.assertTrue(r['cal_test_started'])
            self.assertTrue(r['new_test_entered_after_cal_seal'])
            self.assertFalse(r['old_holdout_unsealed'])
            self.assertIsNone(r['phase_caps_seconds'])
            self.assertEqual(r['resources']['result_bytes'],entry.tree_bytes(out))
            for k in ('wall_seconds','cpu_seconds','download_charge_bytes'):
                self.assertAlmostEqual(sum(v[k] for v in r['stages'].values()),r['resources'][k])
                for ledger,budget in [('cumulative_resources','historical_resources'),('cumulative_budget_debits','historical_debits')]:
                    self.assertAlmostEqual(r[ledger][k],A['budget'][budget][k]+r['resources'][k])
            events=[json.loads(x) for x in (out/'stage_events.jsonl').read_text().splitlines()]
            self.assertEqual([x['event'] for x in events],['SELECTION_SEALED','CAL_THRESHOLD_SEALED','TEST_ENTERED_AFTER_SEAL_VERIFIED'])
            self.assertGreaterEqual(json.loads((out/'works/TEST-000/reserved.json').read_text())['at'],events[1]['at'])
    def test_tampered_cal_stops_before_test(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);local=p/'local';local.mkdir();out=p/'out';d=Dummy(out)
            original=shared.seal
            def sabotage(path,v):
                result=original(path,v)
                if path.name=='cal_seal.json':
                    changed=copy.deepcopy(v);changed['calibration']['tau']+=1;save(path,changed)
                return result
            with patch.object(shared,'seal',sabotage),self.assertRaisesRegex(RuntimeError,'CAL seal'):
                worker.run(local,out,A,d,d)
            self.assertEqual(entry.operation_counts(local)['generate']['started'],24)
            self.assertFalse(list(out.glob('works/TEST-*')))
    def test_failure_retains_image_no_auto_retry(self):
        class Bad(Dummy):
            def read(self,rgb):raise RuntimeError('synthetic reader failure')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);local=p/'local';local.mkdir();out=p/'out';d=Bad(out)
            with self.assertRaisesRegex(RuntimeError,'synthetic'):worker.run(local,out,A,d,d)
            self.assertTrue((out/'works/CAL-000/clean.png').exists())
            self.assertIn('ERROR',(local/'operations.jsonl').read_text())
            before=(local/'operations.jsonl').read_bytes()
            with self.assertRaises(FileExistsError):worker.run(local,out,A,d,d)
            self.assertEqual(before,(local/'operations.jsonl').read_bytes())
            self.assertFalse((out/'cal_seal.json').exists())
    def test_interruption_preserves_history_without_fake_cap_debit(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);dev(p);out=p/entry.RUN_ID;out.mkdir()
            h=entry.prior(A['budget'],p);save(out/'history.json',h)
            save(out/'receipts/000001.json',dict(resources=dict(wall_seconds=99.,cpu_seconds=8.)))
            with patch.object(entry.Monitor,'command') as command:r=entry.run_colab(p)
            command.assert_not_called()
            self.assertTrue(r['unknown_tail']);self.assertFalse(r['retry'])
            self.assertEqual(r['cumulative_lower_bound']['wall_seconds'],h['cumulative_resources']['wall_seconds']+99)
            self.assertEqual(r['cumulative_conservative_debit']['wall_seconds'],h['cumulative_budget_debits']['wall_seconds']+99)
    def test_notebook_bundle_and_mount_only_first_cell(self):
        nb=json.loads((P.parent/'notebooks/lawa_sd21_matched_compare_colab.ipynb').read_text())
        self.assertEqual(''.join(nb['cells'][0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
        source=''.join(nb['cells'][-1]['source']);tree=ast.parse(source)
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
        payload=ast.literal_eval(call.args[0])
        with ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
            self.assertIn('sd2_compare/worker.py',z.namelist())
            self.assertIn('matched_compare/worker.py',z.namelist())
            for n in z.namelist():
                self.assertFalse(n.startswith('/') or '..' in Path(n).parts)
                self.assertEqual(z.read(n),(P.parent/n).read_bytes())
                if n.endswith('.py'):compile(z.read(n),n,'exec')
        self.assertIn('from sd2_compare.entry import run_colab',source)
        self.assertNotIn('from sd2_matched.entry import run_colab',source)
    def test_innocent_calibration_strict_threshold_and_primary_gate(self):
        rows=[fixture(i) for i in range(24)]
        rows[0]['raw']['coarse']['reg/equal3/png'][0]=100
        rows[2]['raw']['ordinary']['foreign/unequal3/jpeg85'][9]=2
        self.assertEqual(rules.calibrate(rows)['tau'],2)
        tests=[fixture(i,'TEST') for i in range(80)]
        tests[0]['raw']['coarse'][rules.PRIMARY][0]=2
        self.assertEqual(rules.evaluate(tests[0],2)['arms']['coarse'][rules.PRIMARY]['returned'],[])
        for r in tests[:6]:r['raw']['coarse'][rules.PRIMARY][0]=3
        result=rules.adjudicate([rules.evaluate(r,2) for r in tests])
        self.assertEqual(result['one_sided_exact_sign_p'],.015625)
        self.assertTrue(all(result['gates'].values()))
        tests[0]['raw']['coarse']['foreign/equal3/jpeg85'][5]=3
        result=rules.adjudicate([rules.evaluate(r,2) for r in tests])
        self.assertFalse(result['gates']['no_new_false_attribution'])
        self.assertEqual(result['status'],'MATCHED_COMPARISON_NOT_SUPPORTED')

if __name__=='__main__':unittest.main()
