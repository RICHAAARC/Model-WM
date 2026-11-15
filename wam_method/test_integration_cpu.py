"""Synthetic CPU process/worker checks; never a scientific run."""
import copy
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .io import save,Ledger,operation_summary,reconcile
from .protocol import read,RUN_ID
from .model import rgb_tensor,to_rgb,quantize
from .entry import Owner,owner_cpu,run_colab,tree_bytes,live_result_bytes
from .worker import Progress,train,eval_bundle,summarize_quality
from . import worker,science
from .data import fetch_one
from PIL import Image
from urllib.error import URLError,HTTPError

class TinyModel:
    def __init__(self):
        self.device=torch.device('cpu');self.wam=torch.nn.Module()
        self.wam.embedder=torch.nn.Linear(32,3)
        self.wam.detector=torch.nn.Conv2d(3,33,1).requires_grad_(False)
    def write(self,x,bits):
        m=torch.tensor(bits,dtype=torch.float32)[None]
        return quantize(x+.02*torch.tanh(self.wam.embedder(m))[:,:,None,None])
    def detect(self,x):return self.wam.detector(x)
    def eval_write(self,x,bits):
        with torch.no_grad():return to_rgb(self.write(rgb_tensor(x,'cpu'),bits))
    def eval_extract(self,x):
        with torch.no_grad():v=self.detect(rgb_tensor(x,'cpu'))[0].numpy().mean((1,2))
        return np.tile(v[:,None,None],(1,4,4)),dict(status='VALID',bits=(v[1:]>.5).astype(int).tolist())

class WorkerIntegration(unittest.TestCase):
    def test_identical_native_image_quality_is_explicit_infinity(self):
        rows=[dict(source=s,valid=True,correct=32,psnr_db=None,psnr_infinite=True,ssim=1.) for s in ['registered','foreign']]
        q=summarize_quality(rows)
        self.assertTrue(q['registered']['mean_psnr_infinite'])
        self.assertEqual(science.psnr(q['registered'],'min'),float('inf'))
        json.dumps(q,allow_nan=False)
    def test_full_orchestration_seals_all_CAL_before_any_TEST(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'local';archive=Path(d)/'archive';root.mkdir();archive.mkdir()
            m=copy.deepcopy(read());m['train']=m['train'][:1];m['training']['updates_per_arm']=1
            m['evaluation']={s:m['evaluation'][s][:1] for s in ['CAL','TEST']}
            m['physical_limits']=dict(train_updates_B=1,train_updates_C=1,train_write=32,train_extract=48,
                evaluation_write=96,evaluation_extract=240,generate=0)
            save(root/'model_paths.json',{'source':'synthetic','checkpoint':'synthetic'})
            test_ids=set(m['evaluation']['TEST'][0]['images']);seen=set()
            rgb=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
            def image(ident,*args):
                if ident in test_ids:
                    sealed=json.loads((archive/'calibration.json').read_text())
                    self.assertEqual(set(sealed),set('ABC'));self.assertTrue(all(v['sealed'] for v in sealed.values()))
                    seen.add(ident)
                return rgb
            cal=science.calibrate;decision=science.decide
            with patch.object(worker,'read',return_value=m),patch.object(worker,'Model',side_effect=lambda *a:TinyModel()),\
                 patch.object(worker,'image_for',side_effect=image),patch('torch.cuda.is_available',return_value=True),\
                 patch.object(worker,'calibrate',side_effect=lambda xs:cal(xs,expected=1)),\
                 patch.object(worker,'decide',side_effect=lambda *args:decision(*args,expected=1)):
                worker.main(root,archive,Path(d))
            self.assertEqual(seen,test_ids)
            r=json.loads((archive/'method_result.json').read_text())
            self.assertEqual(r['operations']['evaluation_extract'],240)
            self.assertIsNone(r['scientific_pass'])
            self.assertTrue((archive/'evaluation/A/TEST/000/readouts.json').exists())

    def test_actual_worker_paired_training_and_saved_media(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'local';archive=Path(d)/'archive';root.mkdir();archive.mkdir()
            m=copy.deepcopy(read());m['train']=m['train'][:2];m['training']['updates_per_arm']=2
            rgb=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
            model=TinyModel();ledger=Ledger(root);p=Progress(root,archive)
            with patch('wam_method.worker.image_for',return_value=rgb):
                initial=train(model,m,root,root,p,ledger)
                result,q=eval_bundle(model,m['evaluation']['CAL'][0],'B',root,root,ledger)
            self.assertEqual(p.state['updates'],{'B':2,'C':2})
            self.assertTrue((archive/'checkpoints/B-final.pt').exists())
            self.assertEqual(ledger.counts,dict(train_update_B=2,train_update_C=2,train_write=64,
                train_extract=96,evaluation_write=16,evaluation_extract=40))
            self.assertEqual(len(result['conditions']),10)
            self.assertEqual(summarize_quality(q)['registered']['denominator'],8)
            self.assertTrue(all(x.grad is None for x in model.wam.detector.parameters()))
            saved=torch.load(root/'checkpoints/B-final.pt',weights_only=True)
            self.assertTrue(any(not torch.equal(saved[k],initial[k]) for k in saved))

class ProcessIntegration(unittest.TestCase):
    def test_atomic_IPC_under_concurrent_reads_and_volume_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'progress.json';save(p,{'n':0});errors=[]
            def writer():
                try:
                    for n in range(200):save(p,{'n':n,'payload':'x'*500})
                except BaseException as exc:errors.append(exc)
            thread=threading.Thread(target=writer);thread.start()
            while thread.is_alive():
                self.assertIn('n',json.loads(p.read_text()));tree_bytes(d)
            thread.join();self.assertFalse(errors)
            self.assertEqual(json.loads(p.read_text())['n'],199)
    def test_source_transport_recovery_retains_first_failure(self):
        with tempfile.TemporaryDirectory() as d:
            row=dict(id=123,file_name='000000000123.jpg',url='https://fixture.invalid/123',source_url='fixture',license={},width=8,height=8)
            stream=io.BytesIO();Image.new('RGB',(8,8)).save(stream,format='JPEG')
            with patch('urllib.request.urlopen',side_effect=[URLError('fixture EOF'),io.BytesIO(stream.getvalue())]) as url:
                p,n=fetch_one(row,d)
            self.assertEqual(url.call_count,2);self.assertTrue(p.exists());self.assertGreater(n,0)
            failed=json.loads(Path(d,'transport/000000000123.1.json').read_text())
            self.assertEqual(failed['status'],'FAILED');self.assertIn('traceback',failed)
            with patch('urllib.request.urlopen',side_effect=AssertionError('paid cache')):self.assertEqual(fetch_one(row,d)[1],0)
    def test_unknown_source_not_retried_or_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            row=dict(id=123,file_name='000000000123.jpg',url='https://fixture.invalid/123')
            with patch('urllib.request.urlopen',side_effect=HTTPError(row['url'],404,'fixture',None,None)) as url:
                with self.assertRaises(HTTPError):fetch_one(row,d)
            self.assertEqual(url.call_count,1)
            with self.assertRaisesRegex(RuntimeError,'consumed'):fetch_one(row,d)
    def owner(self,p):
        for name in ['local','archive','cache']:(p/name).mkdir()
        return Owner(read(),p/'local',p/'archive',p,p/'cache',time.monotonic(),owner_cpu())
    def test_actual_child_failure_accounted_and_retained(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);o=self.owner(p)
            with self.assertRaises(RuntimeError):
                o.command('preparation',[sys.executable,'-c','raise RuntimeError("fixture failure")'],os.environ.copy())
            usage=json.loads((p/'local/preparation.usage.json').read_text())
            self.assertTrue(usage['resources_final'] and usage['all_descendants_reaped'])
            self.assertGreater(o.resources()['cpu_seconds'],0)
            self.assertIn('fixture failure',(p/'local/preparation.log').read_text())
    def test_actual_child_interrupt_reaped_and_no_refund(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);o=self.owner(p);calls=0
            def cutoff(*args):
                nonlocal calls
                calls+=1
                if calls>1:raise RuntimeError('fixture interruption')
            with patch.object(o,'check',side_effect=cutoff),self.assertRaisesRegex(RuntimeError,'fixture interruption'):
                o.command('preparation',[sys.executable,'-c','import time; time.sleep(30)'],os.environ.copy())
            usage=json.loads((p/'local/preparation.usage.json').read_text())
            self.assertTrue(usage['all_descendants_reaped'])
            self.assertGreater(o.resources()['wall_seconds'],1.)
    def test_final_entry_failure_and_reopen_without_execution(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=read();history=p/m['inherited']['history_path'];history.parent.mkdir(parents=True)
            save(history,{**m['inherited'],'resources_final':True})
            def fail(owner,stage,cmd,env):
                save(owner.local/'failure.json',{'retained':True});raise RuntimeError('fixture preparation stop')
            with patch.object(Owner,'command',new=fail):r=run_colab(parent=p,runtime=p/'runtime')
            self.assertEqual(r['status'],'STOPPED_INCOMPLETE')
            self.assertEqual(r['operations']['evaluation_extract']['started'],0)
            self.assertEqual(r['resources']['result_bytes'],tree_bytes(p/RUN_ID))
            self.assertGreaterEqual(r['cumulative_budget_debits']['wall_seconds'],m['inherited']['cumulative_budget_debits']['wall_seconds']+60)
            with patch.object(Owner,'command',side_effect=AssertionError('retry forbidden')):
                self.assertEqual(run_colab(parent=p,runtime=p/'runtime'),r)
    def test_unknown_current_tail_reserves_full_budget_not_final(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m=read();history=p/m['inherited']['history_path'];history.parent.mkdir(parents=True)
            save(history,{**m['inherited'],'resources_final':True})
            def fail(owner,*args):
                owner.accounting_complete=False;raise RuntimeError('fixture owner lost final receipt')
            with patch.object(Owner,'command',new=fail):r=run_colab(parent=p,runtime=p/'runtime')
            self.assertEqual(r['status'],'STOPPED_ACCOUNTING_UNRESOLVED');self.assertFalse(r['resources_final'])
            self.assertEqual(r['budget_resources']['cpu_seconds'],m['budgets']['cpu_seconds'])
            self.assertEqual(r['budget_resources']['gpu_occupancy_seconds'],m['budgets']['gpu_occupancy_seconds'])
    def test_invalid_history_stops(self):
        m=read()['inherited'];h=copy.deepcopy(m);h['cumulative_resources']['wall_seconds']=float('nan')
        with self.assertRaises(RuntimeError):reconcile(h,m)
    def test_live_volume_does_not_double_count_closed_mirror(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'local/x').mkdir(parents=True);(p/'archive/x').mkdir(parents=True)
            for f in ['local/x/a','archive/x/a']:(p/f).write_bytes(b'abc')
            save(p/'archive/receipts/1.json',{})
            self.assertEqual(live_result_bytes(p/'local',p/'archive'),6)

if __name__=='__main__':unittest.main()
