import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from . import protocol,science
from .io import Ledger,reconcile,save
from .model import quantize,jpeg_ste,jpeg_rgb,to_rgb,leaf_tail
from .entry import Owner,run_colab

def example(ident='CAL/0'):
    cond={c:dict(scores=[0.]*16,eligible=True) for c in protocol.CONDITIONS}
    for c in ['registered_mix_png','registered_mix_jpeg85']:
        cond[c]['scores'][0]=10.;cond[c]['scores'][1]=9.
    return dict(id=ident,pair=[0,1],conditions=cond)

class FrozenProtocol(unittest.TestCase):
    def test_disjoint_fixed_data(self):
        m=protocol.read();sets=[set(m['splits'][s]) for s in ['TRAIN','CAL','TEST']]
        self.assertEqual(list(map(len,sets)),[256,256,512]);self.assertEqual(len(set.union(*sets)),1024)
        self.assertTrue(all(x['license']['id']==4 for x in m['images']))
        self.assertEqual(len(m['train']),500)
        self.assertTrue(all(set(b['images'])<=sets[0] for b in m['train']))
    def test_books_draw_once_reproducible(self):
        m=protocol.read();b=m['train'][0]
        self.assertEqual(protocol.bundle(b['id'],b['images']),b)
        self.assertNotEqual(m['evaluation']['CAL'][0]['registered'],m['evaluation']['TEST'][0]['registered'])
        self.assertEqual(len(set(protocol.messages(b,0)[0])),2)
    def test_old_experiment_not_mutated(self):
        old=json.loads(Path(__file__).parents[1].joinpath('wam_qualification/manifest.json').read_text())
        self.assertEqual(old['physical_limits'],dict(generate=16,write=64,extract=160))
        self.assertFalse(old['training_authorized'])

class Channels(unittest.TestCase):
    def test_uint8_and_rint_channel(self):
        x=torch.tensor([.5/255,1.5/255,2.5/255,-.1,1.1],requires_grad=True)
        torch.testing.assert_close(quantize(x)*255,torch.tensor([1.,2.,3.,0.,255.]))
        torch.testing.assert_close(quantize(x,True)*255,torch.tensor([0.,2.,2.,0.,255.]))
        quantize(x).sum().backward();torch.testing.assert_close(x.grad,torch.tensor([1.,1.,1.,0.,0.]))
    def test_jpeg_exact_forward_and_surrogate_gradient(self):
        x=torch.linspace(0,1,3*32*32).reshape(1,3,32,32).requires_grad_()
        y=jpeg_ste(x);self.assertTrue(np.array_equal(to_rgb(y),jpeg_rgb(to_rgb(x))))
        y.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))
    def test_tail_scale_and_truth(self):
        b=protocol.read()['train'][0];codes=[[int(x) for x in c] for c in b['registered']]
        z=torch.randn(128,requires_grad=True)
        a=leaf_tail(z,codes,b['registered_pair']);bigger=leaf_tail(3*z,codes,b['registered_pair'])
        self.assertAlmostEqual(float(a),float(bigger),places=5)
        a.backward();self.assertTrue(torch.isfinite(z.grad).all())
        zero=torch.zeros(128,requires_grad=True);leaf_tail(zero,codes,[0,1]).backward()
        self.assertTrue(torch.isfinite(zero.grad).all())
    def test_streaming_vjp_equals_whole_bundle_gradient(self):
        b=protocol.read()['train'][0];codes=[[int(x) for x in c] for c in b['registered']]
        w=torch.randn(8,requires_grad=True);x=torch.randn(128,8)
        y=x@w;direct=torch.autograd.grad(leaf_tail(y,codes,b['registered_pair']),w)[0]
        leaf=y.detach().requires_grad_();g=torch.autograd.grad(leaf_tail(leaf,codes,b['registered_pair']),leaf)[0]
        streamed=sum(torch.autograd.grad(((x[t:t+32]@w)*g[t:t+32]).sum(),w)[0] for t in range(0,128,32))
        torch.testing.assert_close(streamed,direct)

class Calibration(unittest.TestCase):
    def test_true_members_excluded_innocent_foreign_included(self):
        b=example();self.assertEqual(science.negative_max(b),0.)
        b['conditions']['registered_mix_png']['scores'][3]=4.
        self.assertEqual(science.negative_max(b),4.)
        b['conditions']['foreign_mix_png']['scores'][0]=8.
        self.assertEqual(science.negative_max(b),8.)
    def test_full_cal_and_strict_threshold(self):
        with self.assertRaises(RuntimeError):science.calibrate([example()])
        cal=[example(f'CAL/{i}') for i in range(64)]
        result=science.calibrate(cal);self.assertTrue(result['sealed'])
        b=example('TEST/0');b['conditions']['null_png']['scores'][3]=0.
        self.assertFalse(science.evaluate(b,0.)['false_attribution'])
        b['conditions']['null_png']['scores'][3]=1e-5
        self.assertTrue(science.evaluate(b,0.)['false_attribution'])
    def test_empty_mask_abstains_not_success(self):
        b=example('TEST/0');b['conditions']['registered_mix_png']['eligible']=False
        result=science.evaluate(b,0.)
        self.assertFalse(result['catch_all_media']);self.assertEqual(result['returned']['registered_mix_png'],[])
    def test_no_missing_bundle_average(self):
        with self.assertRaises(RuntimeError):science.aggregate([],dict(id='CAL/0'))
    def test_risk_union_not_conditions_denominator(self):
        b=example();b['conditions']['foreign_mix_png']['scores'][2]=1.
        b['conditions']['null_jpeg85']['scores'][3]=1.
        self.assertIs(science.evaluate(b,.5)['false_attribution'],True)
    def test_cp_upper_zero(self):
        self.assertAlmostEqual(science.risk_upper(0,128),1-.05**(1/128))
    def test_candidate_must_beat_B_with_quality_and_cost(self):
        gates=protocol.read()['gates']
        ev={a:[dict(id=f'TEST/{i}',catch_all_media=(a=='C' and i<13),false_attribution=False) for i in range(128)] for a in 'ABC'}
        q={a:{s:dict(denominator=1024,valid=1024,bit_accuracy=1.,mean_psnr_db=40.,min_psnr_db=35.,mean_ssim=.99) for s in ['registered','foreign']} for a in 'ABC'}
        result=science.decide(ev,q,dict(B=10.,C=10.),gates);self.assertTrue(result['numeric_pass']);self.assertIsNone(result['scientific_pass'])
        self.assertFalse(science.decide(ev,q,dict(B=10.,C=11.),gates)['numeric_pass'])
        q['C']['registered']['mean_psnr_db']=39.
        self.assertFalse(science.decide(ev,q,dict(B=10.,C=10.),gates)['numeric_pass'])

class Accounting(unittest.TestCase):
    def test_failed_work_reserved_not_retried(self):
        with tempfile.TemporaryDirectory() as d:
            l=Ledger(d)
            with self.assertRaises(ZeroDivisionError):l.call('x','write',lambda:1/0)
            with self.assertRaises(RuntimeError):l.call('x','write',lambda:1)
            rows=[json.loads(x) for x in Path(d,'operations.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in rows],['STARTED','FAILED'])
    def test_monotone_history_and_uncertainty(self):
        floor=protocol.read()['inherited'];h=copy.deepcopy(floor)
        h['cumulative_resources']['cpu_seconds']=0.
        h['cumulative_budget_debits']['wall_seconds']+=100
        r=reconcile(h,floor)
        self.assertEqual(r['cumulative_resources']['cpu_seconds'],floor['cumulative_resources']['cpu_seconds'])
        self.assertEqual(r['cumulative_budget_debits']['wall_seconds'],h['cumulative_budget_debits']['wall_seconds'])
    def test_no_dynamic_drive_receipt_read(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);[p.mkdir() for p in [d/'local',d/'archive',d/'cache']]
            o=Owner(protocol.read(),d/'local',d/'archive',d,d/'cache',time.monotonic(),0)
            o.receipt(o.resources(),{});o.receipt(o.resources(),{})
            self.assertEqual(len(list((d/'archive/receipts').glob('*.json'))),2)
    def test_phase_limits(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);o=Owner(protocol.read(),p,p,p,p,time.monotonic(),0)
            r={k:0 for k in ['wall_seconds','gpu_occupancy_seconds','cpu_seconds','result_bytes','peak_ram_bytes','peak_device_used_bytes']}
            with self.assertRaises(RuntimeError):o.check(r,'preparation',1740,{})
            with self.assertRaises(RuntimeError):o.check(r,'method',1,{'train_seconds':{'B':3540}})
            with self.assertRaises(RuntimeError):o.check(r,'method',1,{'train_seconds':{'B':3500},'active_arm':'B','active_started':time.monotonic()-41})
            with self.assertRaises(RuntimeError):o.check(r,'method',1,{'phase':'TEST','evaluation_started':time.monotonic()-10741})
    def test_reopen_never_reexecutes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root=p/protocol.RUN_ID;root.mkdir();save(root/'summary.json',{'status':'C_NOT_SUPPORTED'})
            with patch('subprocess.Popen',side_effect=AssertionError('No rerun')):
                self.assertEqual(run_colab(parent=p,runtime=p/'runtime')['status'],'C_NOT_SUPPORTED')
    def test_unsealed_attempt_readonly(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/protocol.RUN_ID).mkdir()
            self.assertEqual(run_colab(parent=p,runtime=p/'runtime')['status'],'STOPPED_UNSEALED_READ_ONLY')

if __name__=='__main__':unittest.main()
