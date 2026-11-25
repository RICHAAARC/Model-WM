"""Offline CPU checks for the frozen strength bracket. No trained tensors or GPU calls."""
import ast,base64,copy,hashlib,io,json,math,os,shutil,signal,sys,tempfile,threading,time,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np
from latent_control.io import save
from . import entry,worker,rules,assets

ROOT=entry.PACKAGE.parent
FROZEN=json.loads((entry.PACKAGE/'allocation.json').read_text())
ALLOCATION_BYTES=(entry.PACKAGE/'allocation.json').read_bytes()
RATIOS=(.980,.990,1.000,1.010,1.015,1.025)

def quality_rows():
    return [dict(work=f'DEV-{i:03}',loss={'coarse':.01,**{str(g):.01*r for g,r in zip(rules.GAINS,RATIOS)}}) for i in range(8)]

def fixture_distance(clean,image):
    value=int(image[0,0,0])
    return .01 if value==150 else .01*RATIOS[value-151]

class Toy:
    """Image file/ledger fixture, deliberately not a watermark or generation simulation."""
    def generate(self,w):
        assert w['split']=='DEV'
        return None,None,np.full((512,512,3),100,np.uint8)
    def uniform(self,h,bits,gain):
        return np.full((512,512,3),151+rules.GAINS.index(gain),np.uint8)
    def write(self,h,bits,action):
        assert action=='coarse'
        return np.full((512,512,3),150,np.uint8)
    def read(self,*args):raise AssertionError('Reader calls forbidden in DEV')

class Checks(unittest.TestCase):
    def test_allocation_fixed_and_historical_disjointness(self):
        from tools.freeze_matched_compare import collect
        prompts=set();seeds=set()
        for name,digest in FROZEN['history'].items():
            data=(ROOT/name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),digest,name)
            collect(json.loads(data),prompts,seeds)
        works=FROZEN['works']
        self.assertEqual(Counter(w['split'] for w in works),{'DEV':8,'CAL':24,'TEST':160})
        self.assertEqual(len({w['id'] for w in works}),192)
        self.assertEqual(len({w['seed'] for w in works}),192)
        self.assertEqual(len({' '.join(w['prompt'].lower().split()) for w in works}),192)
        self.assertFalse({w['seed'] for w in works}&seeds)
        self.assertFalse({' '.join(w['prompt'].lower().split()) for w in works}&prompts)
        self.assertEqual(FROZEN['codes'],entry.read(ROOT/'matched_loss/allocation.json')['codes'])
        self.assertEqual(FROZEN['master_seed'],2026112502)
        for w in works:
            self.assertEqual(len(set(w['reg']+w['foreign'])),6)
            self.assertTrue(all(0<=i<16 for i in w['reg']))
            self.assertTrue(all(16<=i<32 for i in w['foreign']))
            self.assertEqual(set(w['action_order']),{'coarse'}|{str(g) for g in rules.GAINS})
            self.assertEqual(set(w['arm_order']),set(rules.ARMS))
        self.assertIsNone(FROZEN['budget']['hard_time_caps'])
        self.assertEqual(FROZEN['budget']['fixed_family_operations'],dict(generate=192,write=4752,mix=4416,read=18032))
        self.assertEqual((entry.PACKAGE/'allocation.json').read_bytes(),ALLOCATION_BYTES)

    def test_quality_only_global_choice(self):
        rows=quality_rows();r=rules.choose(rows)
        self.assertEqual(r['status'],'DEV_BRACKET_GATE_MET')
        self.assertEqual(r['selected_gains'],dict(low=1.025,near=1.03,high=1.035))
        self.assertIsNone(r['scientific_pass']);self.assertFalse(r['cal_test_authorized'])
        for row in rows:row['invented_reader_logits']=[float('nan')]
        self.assertEqual(rules.choose(rows),r)

    def test_selection_failures_and_overlapping_boundary(self):
        rows=quality_rows()
        with self.assertRaises(ValueError):rules.choose(rows[:7])
        rows[1]['work']=rows[0]['work']
        with self.assertRaises(ValueError):rules.choose(rows)
        for bad in (0.,-1.,float('nan'),float('inf')):
            rows=quality_rows();rows[0]['loss']['1.03']=bad
            with self.assertRaises(ValueError):rules.choose(rows)
        rows=quality_rows();del rows[0]['loss']['1.03']
        with self.assertRaises(ValueError):rules.choose(rows)
        rows=quality_rows()
        for row in rows:row['loss']['1.045']=.01
        self.assertEqual(rules.choose(rows)['status'],'DEV_BRACKET_STOP')
        rows=quality_rows()
        for row in rows:
            row['loss'].update({str(g):.01*r for g,r in zip(rules.GAINS,(.97,.975,.98,.985,.99,.995))})
        self.assertFalse(rules.choose(rows)['bracket_gate_passed'])
        # One boundary candidate cannot count as distinct low and near choices.
        rows=quality_rows()
        for row in rows:row['loss'].update({str(g):.01*r for g,r in zip(rules.GAINS,(.97,.98,.995,1.01,1.015,1.02))})
        self.assertFalse(rules.choose(rows)['bracket_gate_passed'])

    def test_quality_interval_and_judgment_boundaries(self):
        q={a:[v]*160 for a,v in [('low',.99),('near',1.),('high',1.01)]}
        for invalid in ([1.]*159,[1.]*159+[0.],[1.]*159+[float('nan')]):
            with self.assertRaises(ValueError):rules.quality_interval(invalid)
        for n,status in [(4,'INCONCLUSIVE'),(5,'COARSE_GAIN_PERSISTS_PENDING_VISUAL')]:
            r=rules.adjudicate([1]*n+[0]*(160-n),[0]*160,False,False,q)
            self.assertEqual(r['status'],status)
            self.assertEqual(r['one_sided_exact_sign_p'],.5**n)
            self.assertIsNone(r['zero_win_positive_gain_upper95'])
        r=rules.adjudicate([0]*160,[0]*160,False,False,q)
        self.assertEqual(r['status'],'ORDINARY_CATCHUP_IN_THIS_SAMPLE')
        self.assertAlmostEqual(r['zero_win_positive_gain_upper95'],3*(1-.05**(1/160)))
        self.assertFalse(r['population_equivalence_claim'])
        self.assertIsNone(r['scientific_pass'])
        self.assertEqual(rules.adjudicate([0]*160,[1]+[0]*159,False,False,q)['status'],'INCONCLUSIVE')
        self.assertEqual(rules.adjudicate([0]*160,[0]*160,False,True,q)['status'],'INCONCLUSIVE')
        self.assertEqual(rules.adjudicate([1]*5+[0]*155,[0]*160,True,False,q)['status'],'INCONCLUSIVE')
        self.assertEqual(rules.adjudicate([0]*160,[0]*160,False,False,q,complete=False)['status'],'BRACKET_NOT_ESTABLISHED')
        for arm,value in [('low',1.001),('near',1.006),('high',.999),('high',1.021)]:
            bad=copy.deepcopy(q);bad[arm]=[value]*160
            self.assertEqual(rules.adjudicate([1]*6+[0]*154,[0]*160,False,False,bad)['status'],'BRACKET_NOT_ESTABLISHED')
        with self.assertRaises(ValueError):rules.adjudicate([4]+[0]*159,[0]*160,False,False,q)

    def test_hook_restoration_and_grid_guard(self):
        import torch
        from .carrier import Carrier
        class Backend:
            def __init__(self):self.convs=[torch.nn.Linear(2,2,bias=False) for _ in range(6)]
            def write(self,h,bits):return torch.stack([m(h) for m in self.convs])
        b=Backend();c=Carrier(b,.1);c.branches=lambda:[(None,m,1) for m in b.convs]
        h=torch.ones(2);original=b.write(h,None)
        for gain in rules.GAINS:
            torch.testing.assert_close(c.uniform(h,None,gain),original*gain)
            torch.testing.assert_close(b.write(h,None),original)
        with self.assertRaises(ValueError):c.uniform(h,None,1.05)
        with patch.object(b,'write',side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(RuntimeError):c.uniform(h,None,1.025)
        self.assertTrue(all(not m._forward_hooks for m in b.convs))

    def test_actual_config_parsers_and_meta_components(self):
        from .preflight import check
        from accelerate import init_empty_weights
        from diffusers import UNet2DConditionModel,AutoencoderKL
        from transformers import CLIPTextModel,CLIPTextConfig
        with tempfile.TemporaryDirectory() as tmp:
            s=Path(tmp);shutil.copytree(entry.PACKAGE/'source_evidence/sd14',s/'cache/sd14')
            with (patch.object(UNet2DConditionModel,'from_pretrained',side_effect=AssertionError('No weights')),
                  patch.object(AutoencoderKL,'from_pretrained',side_effect=AssertionError('No weights'))):
                result=check(s)
            self.assertFalse(result['weights_loaded'])
            self.assertEqual(result['status'],'CPU_COMPONENT_PARSERS_PASSED')
            with init_empty_weights():
                u=UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(s/'cache/sd14/unet'))
                v=AutoencoderKL.from_config(AutoencoderKL.load_config(s/'cache/sd14/vae'))
                t=CLIPTextModel(CLIPTextConfig.from_pretrained(s/'cache/sd14/text_encoder',local_files_only=True))
            self.assertEqual(u.conv_in.weight.device.type,'meta')
            self.assertEqual(v.post_quant_conv.weight.shape,(4,4,1,1))
            self.assertEqual(t.text_model.embeddings.token_embedding.weight.shape[1],768)

    def test_wrong_actual_component_semantics_stop(self):
        from .preflight import check
        for relative,key,value in [('unet/config.json','cross_attention_dim',1024),('vae/config.json','scaling_factor',.13),('scheduler/scheduler_config.json','prediction_type','v_prediction')]:
            with tempfile.TemporaryDirectory() as tmp:
                s=Path(tmp);shutil.copytree(entry.PACKAGE/'source_evidence/sd14',s/'cache/sd14')
                p=s/'cache/sd14'/relative;x=entry.read(p);x[key]=value;save(p,x)
                with self.assertRaises(AssertionError):check(s)

    def test_full_dev_owner_and_readonly_replay(self):
        original=entry.Monitor.command
        def command(m,stage,cmd,env):
            if stage=='preparation':
                script="from latent_control.io import save;from pathlib import Path;import sys;save(Path(sys.argv[1])/'python.json',{'python':sys.executable})"
            else:
                script=("from strength_bracket.test_cpu import Toy,fixture_distance;from strength_bracket import worker as w;"
                    "from pathlib import Path;import sys,json;from latent_control.io import save;"
                    "loc,out,s=map(Path,sys.argv[1:]);d=Toy();w.distance=fixture_distance;"
                    "save(loc/'outcome.json',w.run(loc,out,json.loads((w.PACKAGE/'allocation.json').read_text()),d,d))")
            return original(m,stage,[sys.executable,'-c',script,*cmd[3:]],env)
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry,'gpu_ready',return_value=True),patch.object(entry.Monitor,'command',command):
            p=Path(tmp);r=entry.run_colab(p/'drive',p/'content')
            self.assertEqual(r['status'],'DEV_BRACKET_GATE_MET',r)
            self.assertEqual([r['operations'][k]['completed'] for k in ('generate','write','read')],[8,336,0])
            self.assertEqual(r['counts']['works_completed'],8)
            self.assertEqual(entry.run_colab(p/'drive',p/'content'),r)
            out=Path(r['output'])
            self.assertEqual(len(list(out.glob('works/DEV-*'))),8)
            self.assertFalse(list(out.glob('works/CAL-*')) or list(out.glob('works/TEST-*')))
            self.assertEqual(r['resources']['result_bytes'],entry.tree_bytes(out))
            for key in ('wall_seconds','gpu_occupancy_seconds','cpu_seconds','download_charge_bytes'):
                self.assertAlmostEqual(sum(x[key] for x in r['stages'].values()),r['resources'][key])
                self.assertAlmostEqual(r['cumulative_resources'][key],FROZEN['budget']['historical_resources'][key]+r['resources'][key])
                self.assertAlmostEqual(r['cumulative_budget_debits'][key],FROZEN['budget']['historical_debits'][key]+r['resources'][key])
            self.assertIsNone(r['phase_caps_seconds']);self.assertFalse(r['cal_test_started'])
            self.assertFalse(r['retry']);self.assertEqual(r['training_updates'],0)
            self.assertEqual(len(list(out.glob('works/*/*/*.png'))),336)

    def test_write_failure_preserved_and_cannot_repeat(self):
        class Fail(Toy):
            def write(self,*args):raise RuntimeError('Injected write failure')
            def uniform(self,*args):raise RuntimeError('Injected write failure')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);local=p/'local';local.mkdir();b=Fail()
            with self.assertRaises(RuntimeError):worker.run(local,p/'out',FROZEN,b,b)
            self.assertTrue((p/'out/works/DEV-000/clean.png').is_file())
            self.assertFalse(list((p/'out').rglob('completed.json')))
            events=(local/'operations.jsonl').read_text();self.assertIn('ERROR',events);self.assertIn('Traceback',events)
            with self.assertRaises(FileExistsError):worker.run(local,p/'out',FROZEN,b,b)
            self.assertEqual((local/'operations.jsonl').read_text(),events)

    def test_invalid_rgb_stops_before_quality(self):
        class Bad(Toy):
            def write(self,*args):return np.zeros((1,1,3),np.uint8)
            def uniform(self,*args):return np.zeros((512,512,3),np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);local=p/'local';local.mkdir();b=Bad()
            with self.assertRaisesRegex(ValueError,'Invalid native RGB'):worker.run(local,p/'out',FROZEN,b,b)
            self.assertFalse(list((p/'out').rglob('quality.jsonl')))

    def test_prior_high_water_and_unknown_tail(self):
        budget=FROZEN['budget']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);old=dict(resources_final=True,unknown_tail=False)
            for key,source in [('cumulative_resources','historical_resources'),('cumulative_budget_debits','historical_debits')]:
                old[key]={k:v-1 for k,v in budget[source].items()};old[key]['wall_seconds']=budget[source]['wall_seconds']+10
            save(p/entry.PREVIOUS/'summary.json',old)
            h=entry.prior(budget,p)
            self.assertEqual(h['cumulative_resources']['wall_seconds'],budget['historical_resources']['wall_seconds']+10)
            self.assertEqual(h['cumulative_resources']['cpu_seconds'],budget['historical_resources']['cpu_seconds'])
            old['unknown_tail']=True;save(p/entry.PREVIOUS/'summary.json',old)
            with self.assertRaises(ValueError):entry.prior(budget,p)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);out=p/entry.RUN_ID;out.mkdir()
            save(out/'receipts/000001.json',dict(resources=dict(wall_seconds=90,cpu_seconds=4)))
            save(out/'settlement_failure.json',dict(resources=dict(wall_seconds=110,cpu_seconds=6)))
            r=entry.run_colab(p)
            self.assertTrue(r['unknown_tail']);self.assertFalse(r['retry'])
            self.assertEqual(r['status'],'SETTLEMENT_INCOMPLETE_READ_ONLY')
            self.assertEqual(r['cumulative_lower_bound']['wall_seconds'],budget['historical_resources']['wall_seconds']+110)
            self.assertEqual(r['cumulative_conservative_debit']['wall_seconds'],budget['historical_debits']['wall_seconds']+110)

    def test_partial_live_ledger_line_is_not_charged_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'operations.jsonl').write_bytes(b'{"kind":"write","state":"STARTED"}\n{"kind":"write"')
            r=entry.operation_counts(p,live=True)
            self.assertEqual(r['write']['started'],1);self.assertEqual(r['write']['completed'],0)

    def test_user_interrupt_reaps_descendants_and_keeps_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);local=p/'local';local.mkdir()
            m=entry.Monitor(time.monotonic(),entry.owner_cpu(),local,p,p/'cache')
            timer=threading.Timer(.5,lambda:os.kill(os.getpid(),signal.SIGINT));timer.start()
            try:
                with self.assertRaises(KeyboardInterrupt):
                    m.command('model',[sys.executable,'-c','import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);time.sleep(60)'],dict(os.environ))
            finally:timer.cancel()
            r=entry.read(local/'model.usage.json')
            self.assertTrue(r['all_descendants_reaped']);self.assertTrue(r['resources_final'])
            self.assertGreater(m.stages['model']['wall_seconds'],0)
            self.assertGreaterEqual(m.stages['model']['cpu_seconds'],0)

    def test_cache_source_identity_and_content_only(self):
        self.assertIn('sd14/unet/diffusion_pytorch_model.safetensors',assets.targets())
        self.assertFalse(any(x.startswith('sd21/') for x in assets.targets()))
        with self.assertRaises(ValueError):assets.prepare(Path('/tmp/test'),Path('/content/drive/not-allowed'))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);old=p/'previous/cache/last.ckpt';old.parent.mkdir(parents=True);old.write_bytes(b'synthetic')
            save(old.with_name('last.ckpt.complete.json'),dict(url=assets.base.AUTHOR,bytes=9))
            other=p/'a-unrelated/cache/last.ckpt';other.parent.mkdir(parents=True);other.write_bytes(b'other')
            save(other.with_name('last.ckpt.complete.json'),dict(url='https://example.invalid/different'))
            broken=p/'b-empty/cache/last.ckpt';broken.parent.mkdir(parents=True);broken.write_bytes(b'')
            save(broken.with_name('last.ckpt.complete.json'),dict(url=assets.base.AUTHOR,bytes=0))
            scratch=p/'current';scratch.mkdir();local=scratch/'records';local.mkdir();assets.reuse(scratch,local)
            self.assertTrue((scratch/'cache/last.ckpt').is_symlink())
            self.assertEqual(entry.tree_bytes(scratch/'cache'),0)
            r=entry.read(local/'cache_reuse.json')
            self.assertFalse(r['weights_on_drive']);self.assertEqual(len(r['reused']),1);self.assertTrue(r['skipped'])
            self.assertTrue(any('Empty cached artifact' in x['reason'] for x in r['skipped']))
            self.assertTrue(broken.is_file());self.assertEqual(broken.stat().st_size,0)
        for state in ('partial','empty','unresolved'):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);old=p/'previous/cache/last.ckpt';old.parent.mkdir(parents=True);old.write_bytes(b'synthetic')
                save(old.with_name('last.ckpt.complete.json'),dict(url=assets.base.AUTHOR,bytes=9))
                scratch=p/'current';dest=scratch/'cache/last.ckpt';dest.parent.mkdir(parents=True)
                local=scratch/'records';local.mkdir()
                artifact=dest.with_name('last.ckpt.part') if state=='partial' else dest
                artifact.write_bytes(b'' if state=='empty' else b'preserved')
                if state=='empty':save(dest.with_name('last.ckpt.complete.json'),dict(url=assets.base.AUTHOR))
                with self.assertRaises(RuntimeError):assets.reuse(scratch,local)
                self.assertFalse(dest.is_symlink());self.assertTrue(artifact.is_file())
                self.assertEqual(artifact.read_bytes(),b'' if state=='empty' else b'preserved')

    def test_embedded_notebook_exact_payload(self):
        nb=entry.read(ROOT/'notebooks/lawa_strength_bracket_dev_colab.ipynb')
        self.assertEqual(''.join(nb['cells'][0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
        source=''.join(nb['cells'][2]['source']);tree=ast.parse(source)
        payload=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
        with ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
            for name in z.namelist():self.assertEqual(z.read(name),(ROOT/name).read_bytes(),name)
            self.assertIn('strength_bracket/worker.py',z.namelist())
            self.assertNotIn('matched_compare/worker.py',z.namelist())
            self.assertFalse(any(n.endswith(('.ckpt','.safetensors')) for n in z.namelist()))
        entry.check_freeze()
        self.assertEqual((entry.PACKAGE/'allocation.json').read_bytes(),ALLOCATION_BYTES)

if __name__=='__main__':unittest.main()
