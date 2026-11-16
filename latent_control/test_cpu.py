"""No real weights, GPU, actual experiment data or scientific runs."""
import copy
import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from .protocol import read,arrays,validate,RUN_ID
from .carrier import Reader,tensor,quantize,quality,quality_ok,mixed,jpeg
from .io import save,Ledger,BudgetStop,MechanismFailure,seal,operation_summary
from .model import Model,regularized_inverse
from . import science,worker
from .entry import PhaseOwner,run_colab,owner_cpu,settle_volume,tree_bytes,add_resources

torch.set_num_threads(1)

def fixture():
    m=copy.deepcopy(read());m['generation'].update(width=64,height=64)
    m['amplitude']=m['rho']*math.sqrt(3*64*64)/2
    f=arrays();f['permutation']=np.arange(512)[::-1].copy();f['signs']=np.ones(512,dtype=np.int8)
    return m,f

class LinearVAE(torch.nn.Module):
    config=SimpleNamespace(scaling_factor=1.)
    def decode(self,z,return_dict=False):return (2*z-1,)

class SyntheticPipeline:
    def __init__(self):
        self.vae=LinearVAE();self.unet=torch.nn.Identity();self.text_encoder=torch.nn.Identity()
    def __call__(self,prompt,height,width,num_inference_steps,**kwargs):
        z=torch.ones(1,3,height,width)*.5
        for _ in range(num_inference_steps):z=self.unet(z)
        return SimpleNamespace(images=z)

class MathAndFreeze(unittest.TestCase):
    def test_actual_freeze(self):self.assertTrue(validate(read(),arrays()))
    def test_mutated_foreign_pool_rejected(self):
        m=copy.deepcopy(read());m['foreign_pools']['TEST'][0]=16
        with self.assertRaises(RuntimeError):validate(m,arrays())
    def test_mutated_codebook_rejected(self):
        a=arrays();a['codes'][0]=a['codes'][1]
        with self.assertRaises(RuntimeError):validate(read(),a)
    def test_frozen_source_history_labels(self):
        h=json.loads(Path(__file__).with_name('source_usage_history.json').read_text())
        self.assertEqual(h['checkpoint_history'],'UNVERIFIED');self.assertEqual(h['train'],0)
        self.assertEqual(h['exact_old_prompt_overlap'],0)
    def test_reader_is_linear(self):
        _,f=fixture();r=Reader(f['permutation'],f['signs']);x=torch.randn(1,3,64,64);y=torch.randn_like(x)
        torch.testing.assert_close(r(.25*x+.75*y),.25*r(x)+.75*r(y),atol=1e-6,rtol=1e-5)
    def test_reader_rows_are_orthonormal_at_real_resolution(self):
        f=arrays();r=Reader(f['permutation'],f['signs']);x=torch.zeros(1,3,512,512,requires_grad=True)
        q=r(x)[0];rows=[]
        for k in [0,1,31,63]:rows.append(torch.autograd.grad(q[k],x,retain_graph=True)[0].flatten())
        g=torch.stack(rows);torch.testing.assert_close(g@g.T,torch.eye(4),atol=2e-6,rtol=2e-6)
    def test_reader_wrong_resolution_stops(self):
        f=arrays()
        with self.assertRaises(RuntimeError):Reader(f['permutation'],f['signs'])(torch.zeros(1,3,64,64))
    def test_mix_matches_uint8_ties_even(self):
        a=np.zeros((8,8,3),np.uint8);b=a+1
        self.assertTrue((mixed(a,b,.5)==0).all());self.assertTrue((mixed(a,b,.25)==1).all())
    def test_jpeg_roundtrip(self):
        x=np.arange(64*64*3,dtype=np.uint8).reshape(64,64,3)
        data,y=jpeg(x);self.assertTrue(data.startswith(b'\xff\xd8'));self.assertEqual(y.shape,x.shape)
    def test_infinite_psnr_serializes(self):
        x=np.zeros((64,64,3),np.uint8);q=quality(x,x)
        self.assertTrue(q['psnr_infinite']);self.assertIsNone(q['psnr_db']);json.dumps(q,allow_nan=False)
    def test_frozen_rho_quality(self):
        m,_=fixture();q=dict(mse=m['rho']**2,ssim=.98)
        self.assertTrue(quality_ok(q,m));q['mse']+=1e-8;self.assertFalse(quality_ok(q,m))
    def test_inverse_rank_and_ridge(self):
        b,info=regularized_inverse(torch.eye(64));torch.testing.assert_close(b,torch.eye(64)/1.0001)
        with self.assertRaises(MechanismFailure):regularized_inverse(torch.zeros(64,64))
    def test_mixture_margin_condition(self):
        codes=arrays()['codes'];q=.25*codes[2]+.75*codes[7]
        got,_=science.accuse(q,codes[:16],.2);self.assertEqual(got,[2,7])
        got,_=science.accuse(codes[20],codes[:16],.2);self.assertEqual(got,[])
    def test_receiver_strict_boundary_and_no_top1(self):
        codes=np.eye(64)[:16];q=np.zeros(64);q[0]=.2
        self.assertEqual(science.accuse(q,codes,.2)[0],[])
    def test_receiver_nonfinite_rejected(self):
        with self.assertRaises(RuntimeError):science.accuse([float('nan')]*64,np.eye(64)[:16],0)

class ReservationsAndAccounting(unittest.TestCase):
    def owner_fixture(self,d):
        p=Path(d)
        for name in ['local','archive','cache']:(p/name).mkdir()
        return PhaseOwner(read(),p/'local',p/'archive',p,p/'cache',time.monotonic(),owner_cpu())
    def test_real_failed_child_is_reaped_and_charged(self):
        with tempfile.TemporaryDirectory() as d:
            own=self.owner_fixture(d)
            with patch('latent_control.entry.subprocess.run',return_value=SimpleNamespace(returncode=1)):
                with self.assertRaises(RuntimeError):own.command('preparation',[sys.executable,'-c','raise RuntimeError("CPU child fixture")'],os.environ.copy())
            self.assertTrue(own.accounting_complete);self.assertGreater(own.child_cpu,0)
            self.assertIn('CPU child fixture',(Path(d)/'local/preparation.log').read_text())
    def test_real_interrupt_reaps_tree_and_keeps_cost(self):
        with tempfile.TemporaryDirectory() as d:
            own=self.owner_fixture(d);own.m=copy.deepcopy(own.m);own.m['budgets']['preparation_seconds']=61
            with patch('latent_control.entry.subprocess.run',return_value=SimpleNamespace(returncode=1)):
                with self.assertRaises(BudgetStop):own.command('preparation',[sys.executable,'-c','while True: sum(range(10000))'],os.environ.copy())
            self.assertTrue(own.accounting_complete);self.assertGreater(own.resources()['cpu_seconds'],0)
            self.assertTrue(own.stage_resources['preparation']['interrupted'])
    def test_started_failed_consumes_limit(self):
        with tempfile.TemporaryDirectory() as d:
            l=Ledger(d,{'x':1})
            with self.assertRaises(ValueError):l.call('1','x',lambda:(_ for _ in ()).throw(ValueError('fixture')))
            with self.assertRaises(BudgetStop):l.call('2','x',lambda:1)
            r=operation_summary(Path(d)/'operations.jsonl',{'x':1})['x'];self.assertEqual(r['started'],1);self.assertEqual(r['failed'],1)
            self.assertIn('traceback',(Path(d)/'operations.jsonl').read_text())
    def test_existing_ledger_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            l=Ledger(d,{'x':2});l.call('1','x',lambda:1)
            with self.assertRaises(RuntimeError):Ledger(d,{'x':2})
    def test_duplicate_id_prevents_work(self):
        with tempfile.TemporaryDirectory() as d:
            l=Ledger(d,{'x':2});l.call('1','x',lambda:1)
            with self.assertRaises(RuntimeError):l.call('1','x',lambda:self.fail('should not execute'))
    def test_concurrent_atomic_ipc_readers(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.json';save(p,{'n':0});errors=[]
            def writer(offset):
                try:
                    for i in range(80):save(p,{'n':offset+i,'payload':'x'*100})
                except Exception as e:errors.append(e)
            threads=[threading.Thread(target=writer,args=(i*100,)) for i in range(2)]
            for t in threads:t.start()
            while any(t.is_alive() for t in threads):self.assertIn('n',json.loads(p.read_text()))
            for t in threads:t.join()
            self.assertFalse(errors)
    def test_immutable_seal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_text('original');seal(p,Path(d)/'y')
            p.write_text('changed')
            with self.assertRaises(RuntimeError):seal(p,Path(d)/'y')
            self.assertEqual((Path(d)/'y').read_text(),'original')
    def test_phase_budget_separation_and_total_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'local').mkdir();(p/'archive').mkdir();(p/'cache').mkdir()
            o=PhaseOwner(read(),p/'local',p/'archive',p,p/'cache',time.monotonic(),owner_cpu())
            r=dict(wall_seconds=1700,gpu_occupancy_seconds=1700,cpu_seconds=10,peak_ram_bytes=1,peak_device_used_bytes=0,result_bytes=1)
            o.check(r,'preparation',0,{})
            o.preparation_final={k:r[k] for k in ['wall_seconds','gpu_occupancy_seconds','cpu_seconds']};o.model_started=1
            r.update(wall_seconds=1700+21000,gpu_occupancy_seconds=1700+21000)
            o.check(r,'model',0,{})
            self.assertEqual(o.phases(r)['model']['wall_seconds'],21000)
            r['cpu_seconds']=43200
            with self.assertRaises(BudgetStop):o.check(r,'model',0,{})
    def test_size_settlement_counts_final_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'data').write_bytes(b'x'*111)
            s={k:{'result_bytes':0} for k in ['resources','budget_resources','cumulative_resources','cumulative_budget_debits']}
            inherited={k:{'result_bytes':99} for k in ['cumulative_resources','cumulative_budget_debits']}
            settle_volume(p,s,inherited)
            self.assertEqual(s['resources']['result_bytes'],tree_bytes(p))
            self.assertEqual(s['cumulative_resources']['result_bytes'],tree_bytes(p)+99)
    def test_history_adds_not_resets_and_peaks_max(self):
        self.assertEqual(add_resources({'wall_seconds':10,'peak_ram_bytes':100},{'wall_seconds':3,'peak_ram_bytes':2}),
            {'wall_seconds':13,'peak_ram_bytes':100})
    def test_history_fields_survive_partial_new_receipt(self):
        r=add_resources({'wall_seconds':10,'download_charge_bytes':99,'peak_ram_bytes':100},{'wall_seconds':3})
        self.assertEqual(r['download_charge_bytes'],99);self.assertEqual(r['peak_ram_bytes'],100)
    def test_unsealed_uses_latest_history_and_observation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/RUN_ID;out.mkdir();inherited=copy.deepcopy(read()['inherited'])
            inherited['cumulative_budget_debits']['wall_seconds']+=1000
            save(out/'inherited_totals.json',inherited)
            save(out/'receipts/000001.json',dict(resources={'wall_seconds':24000,'download_charge_bytes':7}))
            r=run_colab(parent=d)
            self.assertEqual(r['cumulative_budget_debits']['wall_seconds'],inherited['cumulative_budget_debits']['wall_seconds']+24000)
            self.assertEqual(r['cumulative_budget_debits']['download_charge_bytes'],inherited['cumulative_budget_debits']['download_charge_bytes']+7)
    def test_repeat_click_reads_only(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/RUN_ID;out.mkdir();save(out/'summary.json',{'status':'FAILED','resources':{'wall_seconds':123}})
            with patch('latent_control.entry.PhaseOwner',side_effect=AssertionError('must not run')):
                self.assertEqual(run_colab(parent=d)['resources']['wall_seconds'],123)
    def test_unsealed_click_debits_full_attempt_without_run(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/RUN_ID).mkdir()
            with patch('latent_control.entry.PhaseOwner',side_effect=AssertionError('must not run')):
                r=run_colab(parent=d)
            self.assertFalse(r['retry']);self.assertEqual(r['full_package_reserved']['wall_seconds'],23400)
    def test_failed_history_load_seals_cost_and_exception(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=run_colab(parent=p/'drive',runtime=p/'runtime')
            self.assertEqual(r['status'],'ENVIRONMENT_STOPPED');self.assertGreater(r['resources']['wall_seconds'],0)
            self.assertTrue((p/'drive'/RUN_ID/'exception.txt').exists())
            self.assertGreaterEqual(r['cumulative_budget_debits']['wall_seconds'],read()['inherited']['cumulative_budget_debits']['wall_seconds'])
    def test_projection_stops_without_CAL(self):
        with tempfile.TemporaryDirectory() as d:
            row=dict(wall_seconds=1000,cpu_seconds=100,result_bytes=1)
            with self.assertRaises(BudgetStop):worker.project([row]*8,Path(d),read(),time.monotonic(),worker.cpu())
            self.assertFalse(json.loads((Path(d)/'dev_projection.json').read_text())['passed'])

class RealWriterSyntheticCarrier(unittest.TestCase):
    def make(self,d):
        m,f=fixture();l=Ledger(d,m['physical_limits']);return Model(None,m,f,l,'cpu',SyntheticPipeline()),l,m
    def test_generation_is_latent_and_counts_all_steps(self):
        with tempfile.TemporaryDirectory() as d:
            model,l,m=self.make(d);z=model.generate(m['works'][0])
            self.assertEqual(l.counts,{'generate':1,'denoise':25});self.assertTrue(z.requires_grad)
    def test_unexpected_extra_denoiser_step_stops_before_physical_call(self):
        original=SyntheticPipeline.__call__
        def extra(pipe,*args,**kwargs):
            kwargs['num_inference_steps']+=1;return original(pipe,*args,**kwargs)
        with tempfile.TemporaryDirectory() as d:
            model,l,m=self.make(d)
            with patch.object(SyntheticPipeline,'__call__',extra):
                with self.assertRaises(BudgetStop):model.generate(m['works'][0])
            self.assertEqual(l.counts['denoise'],25)
    def test_partial_shared_copy_is_retained_after_later_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'root';archive=Path(d)/'archive';root.mkdir();archive.mkdir()
            model,l,m=self.make(root);w=copy.deepcopy(m['works'][0]);w['arm_order']='CB'
            original=model.decode
            def fail_later(z,ident,kind):
                if '/copy/1/decode' in ident:raise MechanismFailure('synthetic later copy failure')
                return original(z,ident,kind)
            with patch.object(model,'decode',side_effect=fail_later):result,_=worker.work_once(model,w,root,archive,l,m)
            self.assertFalse(result['C']['complete'])
            self.assertTrue((archive/'works'/w['id']/'C/registered_0_written.png').exists())
            self.assertNotIn('registered0',result['C']['failed_copies'])
    def test_full_shared_and_ordinary_use_frozen_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            model,l,m=self.make(d);w=m['works'][0];z=model.generate(w);y=model.decode(z,'clean','forward_C')
            fd=model.finite_difference(z,y,'fd')
            outs,diag=model.shared(z,y,w['registered']+w['foreign'],'C')
            self.assertTrue(np.allclose(fd,diag['initial']['first_response_column'],atol=.0005,rtol=.05))
            self.assertEqual(l.counts['backward_C'],128);self.assertEqual(l.counts['forward_C'],6)
            out=model.ordinary(z,y,w['registered'][0],'B')
            self.assertLessEqual(out['evaluations'],32);self.assertEqual(l.counts['objective_B'],l.counts['backward_B'])
            self.assertEqual(l.counts['forward_B'],l.counts['backward_B'])
            self.assertEqual(len(outs),4);self.assertTrue(all(p.grad is None for p in model.pipe.vae.parameters()))
    def test_guard_prevents_LBFGS_internal_overrun(self):
        with tempfile.TemporaryDirectory() as d:
            model,l,m=self.make(d);z=model.generate(m['works'][0]);y=model.decode(z,'clean','forward_C')
            def greedy(opt,closure):
                for _ in range(100):closure()
            with patch.object(torch.optim.LBFGS,'step',greedy):out=model.ordinary(z,y,0,'B')
            self.assertEqual(out['evaluations'],32);self.assertEqual(l.counts['backward_B'],32)
            self.assertTrue(out['evaluation_cap_reached'])
    def test_all_saved_views_and_foreign_native_family(self):
        with tempfile.TemporaryDirectory() as d:
            model,l,m=self.make(d);p=Path(d);(p/'out').mkdir();w=m['works'][0]
            clean=np.full((64,64,3),128,np.uint8);outputs={}
            for group in ['registered','foreign']:
                for i in range(2):outputs[(group,i)]=dict(rgb=clean.copy(),quality=quality(clean,clean))
            b=worker.evaluation(model,clean,outputs,w,'B',p/'out',l,m)
            self.assertEqual(len(b['rows']),22);self.assertEqual(l.counts['extract'],22)
            self.assertEqual(sum(r['kind']=='foreign_native' for r in b['rows']),4)
            next(r for r in b['rows'] if r['kind']=='foreign_native')['scores'][0]=10
            self.assertEqual(science.negative_max(b),10)
    def test_missing_copy_retained_and_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            model,l,m=self.make(d);p=Path(d);(p/'out').mkdir()
            clean=np.full((64,64,3),128,np.uint8)
            b=worker.evaluation(model,clean,{},m['works'][0],'C',p/'out',l,m)
            self.assertFalse(b['complete']);self.assertEqual(len(b['failed_copies']),4)
            with self.assertRaises(RuntimeError):science.calibrate([b],expected=1)
            self.assertFalse(science.evaluate(b,1,arrays()['codes'][:16],m)['catch_all'])

class FlowIntegration(unittest.TestCase):
    def test_CAL_failure_cannot_reach_TEST(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'local';archive=Path(d)/'archive';root.mkdir();archive.mkdir()
            m,f=fixture();m['works']=[next(w for w in m['works'] if w['split']==s) for s in ['DEV','CAL','TEST']]
            save(root/'model_paths.json',dict(base='synthetic'));seen=[]
            class Checked(Model):
                def __init__(self,base,manifest,frozen,ledger):super().__init__(base,manifest,frozen,ledger,'cpu',SyntheticPipeline())
                def generate(self,w):seen.append(w['split']);return super().generate(w)
            with patch.object(worker,'read',return_value=m),patch.object(worker,'arrays',return_value=f),\
                 patch.object(worker,'validate',return_value=True),patch.object(worker,'Model',Checked),\
                 patch.object(science,'calibrate',side_effect=RuntimeError('missing risk evidence')):
                with self.assertRaisesRegex(RuntimeError,'missing risk evidence'):worker.main(root,archive,Path(d))
            self.assertEqual(seen,['DEV','CAL']);self.assertFalse((archive/'calibration.json').exists())
    def test_full_worker_CAL_readback_precedes_TEST_generation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root=p/'local';archive=p/'archive';root.mkdir();archive.mkdir()
            m,f=fixture();m['works']=[next(w for w in m['works'] if w['split']==s) for s in ['DEV','CAL','TEST']]
            save(root/'model_paths.json',dict(base='synthetic'))
            class CheckedModel(Model):
                def __init__(self,base,manifest,frozen,ledger):super().__init__(base,manifest,frozen,ledger,'cpu',SyntheticPipeline())
                def generate(self,w):
                    if w['split']=='TEST':
                        got=json.loads((archive/'calibration.json').read_text())
                        assert set(got)==set('BC') and all(got[a]['sealed'] for a in 'BC')
                    return super().generate(w)
            calibrate=science.calibrate;decide=science.decide
            with patch.object(worker,'read',return_value=m),patch.object(worker,'arrays',return_value=f),\
                 patch.object(worker,'validate',return_value=True),patch.object(worker,'Model',CheckedModel),\
                 patch.object(science,'calibrate',side_effect=lambda b:calibrate(b,expected=1)),\
                 patch.object(science,'decide',side_effect=lambda *a:decide(*a,expected=1)):
                worker.main(root,archive,p)
            r=json.loads((root/'method_result.json').read_text())
            self.assertEqual(r['operations']['generate'],3);self.assertEqual(r['operations']['extract'],132)
            self.assertIsNone(r['scientific_pass']);self.assertEqual(r['visual_review'],'UNRESOLVED')
            self.assertTrue((archive/'works/TEST/000/C/foreign_0_written.png').exists())

if __name__=='__main__':unittest.main()
