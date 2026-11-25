"""Offline synthetic checks. No pretrained tensors or scientific works executed."""
import ast,base64,io,json,os,shutil,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
from latent_control.io import save
from . import entry,worker,assets
ROOT=entry.PACKAGE.parent
FROZEN=json.loads((entry.PACKAGE/'allocation.json').read_text())

class Toy:
    last_interface={'synthetic':True,'latent_shape':[1,4,64,64]}
    def generate(self,w):
        assert w['split']=='DEV'
        return np.zeros((1,4,64,64),np.float32),None,np.full((512,512,3),110,np.uint8)
    def uniform(self,h,bits,gain):return np.full((512,512,3),round(110+10*gain),np.uint8)
    def write(self,h,bits,action):return self.uniform(h,bits,1.025)
    def read(self,image):return [0.]*48

class Checks(unittest.TestCase):
    def test_freeze_and_history(self):
        from tools.freeze_matched_compare import collect
        prompts=set();seeds=set()
        for name in FROZEN['history']:
            collect(json.loads((ROOT/name).read_text()),prompts,seeds)
        rows=FROZEN['works']
        self.assertEqual([sum(w['split']==s for w in rows) for s in ['DEV','CAL','TEST']],[8,24,80])
        self.assertEqual(len({w['seed'] for w in rows}),112)
        self.assertFalse({w['seed'] for w in rows}&seeds)
        self.assertFalse({' '.join(w['prompt'].lower().split()) for w in rows}&prompts)
        self.assertEqual(FROZEN['codes'],json.loads((ROOT/'matched_loss/allocation.json').read_text())['codes'])
        self.assertIsNone(FROZEN['budget']['hard_time_caps'])
        for w in rows:
            self.assertEqual(len(set(w['reg']+w['foreign'])),6)
            self.assertTrue(all(i<16 for i in w['reg']) and all(i>=16 for i in w['foreign']))

    def test_actual_config_parsers_and_meta_architecture(self):
        import torch
        from accelerate import init_empty_weights
        from diffusers import UNet2DConditionModel,AutoencoderKL
        from transformers import CLIPTextModel,CLIPTextConfig
        from .preflight import check
        with tempfile.TemporaryDirectory() as tmp:
            s=Path(tmp);shutil.copytree(entry.PACKAGE/'source_evidence/sd21',s/'cache/sd21')
            shutil.copytree(entry.PACKAGE/'source_evidence/sd14',s/'cache/sd14')
            r=check(s);self.assertEqual(r['prediction_type'],'epsilon')
            with init_empty_weights():
                u=UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(s/'cache/sd21/unet'))
                t=CLIPTextModel(CLIPTextConfig.from_pretrained(s/'cache/sd21/text_encoder'))
                v=AutoencoderKL.from_config(AutoencoderKL.load_config(s/'cache/sd14/vae'))
            self.assertEqual(u.conv_in.weight.device.type,'meta')
            self.assertEqual(t.text_model.embeddings.token_embedding.weight.shape[1],1024)
            self.assertEqual(v.post_quant_conv.weight.shape,(4,4,1,1))

    def test_wrong_prediction_and_context_stop(self):
        from .model import validate_configs
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);shutil.copytree(entry.PACKAGE/'source_evidence/sd21',b/'sd21')
            shutil.copytree(entry.PACKAGE/'source_evidence/sd14',b/'sd14')
            p=b/'sd21/scheduler/scheduler_config.json';x=json.loads(p.read_text());x['prediction_type']='v_prediction';save(p,x)
            with self.assertRaises(ValueError):validate_configs(b/'sd21',b/'sd14')
            x['prediction_type']='epsilon';save(p,x)
            p=b/'sd21/unet/config.json';x=json.loads(p.read_text());x['cross_attention_dim']=768;save(p,x)
            with self.assertRaises(ValueError):validate_configs(b/'sd21',b/'sd14')

    def test_tiny_real_component_loading_pipeline_and_scale(self):
        import torch
        from torch import nn
        from diffusers import UNet2DConditionModel,AutoencoderKL
        from transformers import CLIPTextModel,CLIPTextConfig
        from .model import build_pipeline,Backend,SCALE
        torch.set_num_threads(1);torch.manual_seed(11)
        class Writer(nn.Module):
            def __init__(self):
                super().__init__();self.ae=nn.Module();self.ae.post_quant_conv=nn.Conv2d(4,4,1)
                with torch.no_grad():self.ae.post_quant_conv.weight.copy_(torch.eye(4)[:,:,None,None]);self.ae.post_quant_conv.bias.zero_()
            def forward(self,h,_,code):return None,torch.nn.functional.interpolate(h[:,:3],size=(512,512))+code.mean()*.001
        with tempfile.TemporaryDirectory() as tmp:
            s=Path(tmp);b=s/'sd21';c=s/'sd14'
            shutil.copytree(entry.PACKAGE/'source_evidence/sd21/tokenizer',b/'tokenizer')
            shutil.copytree(entry.PACKAGE/'source_evidence/sd21/scheduler',b/'scheduler')
            CLIPTextModel(CLIPTextConfig(vocab_size=49408,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=77)).save_pretrained(b/'text_encoder',safe_serialization=True)
            UNet2DConditionModel(sample_size=64,in_channels=4,out_channels=4,block_out_channels=(8,8),layers_per_block=1,
                down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),
                cross_attention_dim=32,attention_head_dim=2,norm_num_groups=4).save_pretrained(b/'unet',safe_serialization=True)
            AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,block_out_channels=(8,8,8,8),
                down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,
                layers_per_block=1,norm_num_groups=4,sample_size=512).save_pretrained(c/'vae',safe_serialization=True)
            pipe=build_pipeline(b,c);w=Writer();backend=Backend(pipe,w,'cpu')
            calls=[];hook=w.ae.post_quant_conv.register_forward_hook(lambda m,a,o:calls.append(1))
            z,h,image=backend.generate(dict(prompt='synthetic interface fixture',seed=123))
            self.assertEqual(len(calls),1);hook.remove()
            torch.testing.assert_close(h,torch.tensor(z)/SCALE)
            self.assertEqual(image.shape,(512,512,3));self.assertEqual(image.dtype,np.uint8)
            self.assertTrue(np.isfinite(z).all())
            with self.assertRaises(ValueError):backend.marked(h,'2'*48)
            self.assertEqual(backend.write(h,'01'*24).shape,(512,512,3))

    def test_full_owner_replay_and_costs(self):
        original=entry.Monitor.command
        def command(m,stage,cmd,env):
            if stage=='preparation':
                script="from latent_control.io import save;from pathlib import Path;import sys;save(Path(sys.argv[1])/'python.json',{'python':sys.executable})"
            else:
                script=("from sd2_matched.test_cpu import Toy;from sd2_matched import worker as w;"
                  "from pathlib import Path;import sys,json;from latent_control.io import save;"
                  "loc,out,s=map(Path,sys.argv[1:]);d=Toy();save(loc/'outcome.json',w.run(loc,out,json.loads((w.PACKAGE/'allocation.json').read_text()),d,d))")
            return original(m,stage,[sys.executable,'-c',script,*cmd[3:]],env)
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Monitor,'command',command):
            p=Path(tmp);r=entry.run_colab(p/'drive',p/'content')
            self.assertEqual(r['status'],'DEV_MATCHING_GATE_MET',r)
            self.assertEqual(r,entry.run_colab(p/'drive',p/'content'))
            self.assertEqual([r['operations'][k]['completed'] for k in ['generate','write','read']],[8,240,24])
            self.assertEqual(r['resources']['result_bytes'],entry.tree_bytes(Path(r['output'])))
            for k in ('wall_seconds','cpu_seconds','download_charge_bytes'):
                self.assertAlmostEqual(sum(v[k] for v in r['stages'].values()),r['resources'][k])
                self.assertAlmostEqual(r['cumulative_resources'][k],FROZEN['budget']['historical_resources'][k]+r['resources'][k])
            self.assertFalse(r['cal_test_started']);self.assertIsNone(r['phase_caps_seconds'])
            self.assertEqual(len(list(Path(r['output']).glob('works/DEV-*'))),8)
            self.assertFalse(list(Path(r['output']).glob('works/CAL-*')))

    def test_reader_failure_retains_image_and_no_retry(self):
        class Bad(Toy):
            def read(self,image):raise RuntimeError('Injected reader failure')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);local=p/'local';local.mkdir();b=Bad()
            with self.assertRaises(RuntimeError):worker.run(local,p/'out',FROZEN,b,b)
            self.assertTrue(list((p/'out').glob('works/DEV-000/*/0.png')))
            self.assertIn('ERROR',(local/'operations.jsonl').read_text())
            before=(local/'operations.jsonl').read_bytes()
            with self.assertRaises(FileExistsError):worker.run(local,p/'out',FROZEN,b,b)
            self.assertEqual(before,(local/'operations.jsonl').read_bytes())

    def test_interface_logits_cannot_select_gain(self):
        rows=[dict(work=str(i),loss={'coarse':1.,'1.0':1.06,'1.025':1.,'1.05':1.03,'1.075':1.1}) for i in range(8)]
        self.assertEqual(worker.decide(rows)['selected_gain'],1.025)
        for row in rows:row['interface_readouts']=[float('nan')]
        self.assertEqual(worker.decide(rows)['selected_gain'],1.025)
        for row in rows:row['loss']['1.025']=1.04
        self.assertFalse(worker.decide(rows)['matching_gate_passed'])

    def test_unknown_tail_no_cap_fabrication(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/entry.RUN_ID;out.mkdir()
            save(out/'receipts/000001.json',{'resources':{'wall_seconds':90,'cpu_seconds':4}})
            r=entry.run_colab(p)
            self.assertTrue(r['unknown_tail']);self.assertFalse(r['retry'])
            self.assertEqual(r['cumulative_lower_bound']['wall_seconds'],FROZEN['budget']['historical_resources']['wall_seconds']+90)
            self.assertEqual(r['cumulative_conservative_debit']['wall_seconds'],FROZEN['budget']['historical_debits']['wall_seconds']+90)

    def test_user_interrupt_reaps_tree_and_keeps_cost(self):
        import threading,signal
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);loc=p/'local';loc.mkdir()
            m=entry.Monitor(time.monotonic(),entry.owner_cpu(),loc,p,p/'cache')
            timer=threading.Timer(.5,lambda:os.kill(os.getpid(),signal.SIGINT))
            timer.start()
            try:
                with self.assertRaises(KeyboardInterrupt):
                    m.command('model',[sys.executable,'-c','import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);time.sleep(60)'],dict(os.environ))
            finally:timer.cancel()
            r=json.loads((loc/'model.usage.json').read_text())
            self.assertTrue(r['all_descendants_reaped'])
            self.assertGreater(m.stages['model']['wall_seconds'],0)

    def test_settlement_failure_preserves_high_water(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/entry.RUN_ID;out.mkdir()
            save(out/'receipts/000001.json',{'resources':{'wall_seconds':90,'cpu_seconds':4}})
            save(out/'settlement_failure.json',{'resources':{'wall_seconds':110,'cpu_seconds':6}})
            r=entry.run_colab(p)
            self.assertEqual(r['status'],'SETTLEMENT_INCOMPLETE_READ_ONLY')
            self.assertEqual(r['cumulative_lower_bound']['wall_seconds'],FROZEN['budget']['historical_resources']['wall_seconds']+110)

    def test_asset_routes_and_cache(self):
        routes=assets.targets()
        self.assertNotIn('sd21/vae/diffusion_pytorch_model.safetensors',routes)
        self.assertNotIn('sd14/unet/diffusion_pytorch_model.safetensors',routes)
        self.assertEqual(sum(k.endswith(('.ckpt','.safetensors')) for k in routes),4)
        with self.assertRaises(ValueError):assets.prepare(Path('/tmp/test'),Path('/content/drive/weights'))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);old=p/'previous/cache/last.ckpt';old.parent.mkdir(parents=True);old.write_bytes(b'synthetic')
            save(old.with_name('last.ckpt.complete.json'),{'url':assets.old.AUTHOR,'bytes':9})
            unrelated=p/'a-different-project/cache/last.ckpt';unrelated.parent.mkdir(parents=True);unrelated.write_bytes(b'other')
            save(unrelated.with_name('last.ckpt.complete.json'),{'url':'https://example.invalid/unrelated'})
            s=p/'current';s.mkdir();loc=s/'records';loc.mkdir();assets.reuse(s,loc)
            self.assertTrue((s/'cache/last.ckpt').is_symlink())
            self.assertEqual(entry.tree_bytes(s/'cache'),0)
            self.assertEqual(json.loads((loc/'cache_reuse.json').read_text())['skipped'][0]['reason'],'Different asset source; not reused')

    def test_notebook_bundle(self):
        nb=json.loads((ROOT/'notebooks/lawa_sd21_matched_dev_colab.ipynb').read_text())
        self.assertEqual(''.join(nb['cells'][0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
        tree=ast.parse(''.join(nb['cells'][2]['source']))
        payload=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
        with ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
            for name in z.namelist():self.assertEqual(z.read(name),(ROOT/name).read_bytes())
            self.assertNotIn('matched_compare/worker.py',z.namelist())
        entry.check_freeze()

if __name__=='__main__':unittest.main()
