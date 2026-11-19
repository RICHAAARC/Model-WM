"""CPU integration uses a synthetic linear decoder and synthetic content only."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from latent_control.test_cpu import SyntheticPipeline
from latent_control.io import Ledger,save,BudgetStop
from latent_control.protocol import arrays
from . import entry,worker,quality,science
from .model import Solver
from .protocol import protocol,limits,read_json,RUN_ID,BUDGET

def synthetic(root,archive,stage):
    torch.set_num_threads(1)
    m=protocol()
    for i,w in enumerate(m['works']):
        w.update(prompt='synthetic constant array',noise_seed=i)
    m['generation'].update(width=32,height=32)
    m['amplitude']=.18
    f=arrays()
    f.update(permutation=np.arange(128),signs=np.ones(128))
    batch=[w for w in m['works'] if (w['split']=='DEV')==(stage=='dev')]
    ledger=Ledger(root,limits(batch))
    model=Solver(None,m,f,ledger,'cpu',SyntheticPipeline())
    worker.run(root,archive,stage,model,m,ledger)

class Integration(unittest.TestCase):
    def test_complete_entry(self):
        original=entry.PhaseOwner.command
        def cpu_command(owner,stage,cmd,env):
            if stage=='preparation':
                code="import json,sys;open(sys.argv[1],'w').write(json.dumps(dict(python=sys.executable)))"
                cmd=[sys.executable,'-c',code,str(owner.local/'python.json')]
            else:
                cmd=[sys.executable,'-m','terminal_solver.test_cpu','synthetic',*cmd[-3:]]
            return original(owner,stage,cmd,dict(env,CUDA_VISIBLE_DEVICES=''))
        with tempfile.TemporaryDirectory() as td, patch.object(entry.PhaseOwner,'command',cpu_command), patch.object(entry,'gpu_ready',return_value=True):
            parent=Path(td)/'drive'
            history=read_json(entry.PACKAGE.parent/'latent_control/manifest.json')['inherited']
            save(parent/entry.HISTORY/'summary.json',dict(history,resources_final=True))
            root=parent/RUN_ID
            def run():
                return entry.run_colab(parent,Path(td)/'runtime')
            final=run()
            self.assertIn(final['status'],('PROXY_SCREEN_PASSED','GATE_FAILED'),str(final)+
                          ''.join(p.read_text() for p in root.glob('*/exception.txt')))
            self.assertEqual(final['segments']['dev'],'SELECTED')
            self.assertTrue(read_json(root/'projection.json')['fits'])
            self.assertTrue(all(v['sealed'] and len(v['work_maxima'])==64 for v in read_json(root/'formal/calibration.json').values()))
            for stage,n in [('dev',8),('formal',128)]:
                ops=read_json(root/stage/'operations_summary.json')
                self.assertEqual(ops['backward_C']['completed'],n*64)
                self.assertEqual(ops['backward_center']['completed'],n*64)
                self.assertLessEqual(ops['backward_O']['completed'],ops['backward_O']['denominator'])
            result=read_json(root/'formal/paired_results.json')
            self.assertEqual(len(result['paired']),64)
            self.assertEqual(result['human_review'],'NOT_PERFORMED')
            self.assertEqual(len(read_json(root/'formal/figure_selection.json')['all']),64)
            self.assertEqual(final,run())
            self.assertEqual(final['resources']['result_bytes'],entry.tree_bytes(root))
            self.assertTrue(all(final['phase_budget_debits'][k]>=v for k,v in final['phase_seconds'].items()))

    def test_real_freeze_limits(self):
        works=protocol()['works']
        counts=limits(works)
        self.assertEqual(sum(v for k,v in counts.items() if k.startswith('backward_')),27136)
        self.assertEqual(sum(counts[k] for k in ('initial_decode','center_decode','final_C','closure_O')),10608)
        self.assertEqual(counts['extract'],6688)

    def test_unknown_tail_no_restart(self):
        with tempfile.TemporaryDirectory() as td:
            parent=Path(td)
            root=parent/RUN_ID
            save(root/'init_complete.json',{})
            save(root/'history.json',dict(cumulative_resources={},cumulative_budget_debits={}))
            (root/'dev').mkdir()
            result=entry.run_colab(parent)
            self.assertEqual(result['status'],'STOPPED_UNSEALED_READ_ONLY')
            self.assertGreaterEqual(result['cumulative_budget_debits']['wall_seconds'],11700)

    def test_budget_no_refund(self):
        state={'resources':{'cpu_seconds':300,'result_bytes':12},'phase_seconds':{'preparation':450,'model':5200}}
        state['phase_budget_debits']=state['phase_seconds']
        state['budget_resources']=state['resources']
        b=entry.remaining(state)
        self.assertEqual((b['preparation_seconds'],b['model_seconds'],b['cpu_seconds']),(450,5600,14100))
        state['phase_seconds']['model']=10800
        with self.assertRaises(BudgetStop):
            entry.remaining(state)

    def test_cal_failure_blocks_test(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'records'
            archive=Path(td)/'attempt/formal'
            save(archive.parent/'dev/dev_decision.json',dict(gains=dict(C=1,O=1)))
            m=protocol()
            seen=[]
            def fake(model,w,gains,local,target,ledger,m):
                seen.append(w['split'])
                for a in 'CO':
                    save(local/'works'/w['id']/a/'1/readouts.json',dict(id=w['id'],complete=False,rows=[]))
            with self.assertRaises(RuntimeError):
                worker.run(root,archive,'formal',None,m,None,fake)
            self.assertEqual(seen,['CAL']*64)
            self.assertFalse((archive/'calibration.json').exists())

    def test_objective_selection_and_stop(self):
        def bundle(i,g,cost):
            rows=[dict(kind='registered_mix' if j<6 else 'null',truth=[0] if j<6 else [],
                       scores=[g,0] if j<6 else [0,0]) for j in range(22)]
            return dict(id=str(i),complete=True,rows=rows,proxies=dict(layout=cost,artifact=cost))
        def build(scale):
            return {a:{g:[bundle(i,g,g*(scale if a=='C' else 1)) for i in range(8)]
                       for g in (.5,1,2)} for a in 'CO'}
        self.assertEqual(science.select(build(2))['gains'],{'C':.5,'O':1})
        self.assertEqual(science.select(build(3))['status'],'STOP_DEV')
        self.assertEqual(len(science.select(build(1))['grid']),9)

    def test_proxy_changes_and_codec(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)
            for name in quality.VIEWS:
                Image.fromarray(np.zeros((64,64,3),dtype='uint8')).save(path/name)
            equal=quality.measure(path)
            self.assertEqual((equal['layout'],equal['artifact']),(0,0))
            arr=np.zeros((64,64,3),dtype='uint8')
            arr[:32]=255
            Image.fromarray(arr).save(path/quality.NATIVE[1])
            changed=quality.measure(path)
            self.assertGreater(changed['layout'],0)
            self.assertGreater(changed['artifact'],0)
            self.assertEqual(len(changed['pairs']),6)

    def test_frozen_tracking_and_joint_false_attribution(self):
        def bundle(i,hit):
            rows=[dict(kind='registered_mix' if j<6 else 'null',name=str(j),truth=[0] if j<6 else [],
                       scores=[float(hit) if j<6 else 0.,0.]) for j in range(22)]
            return dict(id=str(i),rows=rows,complete=True,proxies=dict(layout=.02,artifact=.01))
        data={a:[bundle(i,(i<38 or (38<=i<51 if a=='C' else 51<=i<56))) for i in range(64)] for a in 'CO'}
        cal={a:dict(threshold=.5) for a in 'CO'}
        result=science.verdict(data,cal)
        self.assertTrue(result['passed'])
        self.assertEqual((result['wins'],result['losses']),(13,5))
        data['O'][0]['rows'][-1]['scores'][1]=.6
        result=science.verdict(data,cal)
        self.assertTrue(result['tracking_gate'])
        self.assertEqual(result['risk_works'],1)
        self.assertFalse(result['passed'])
        data['O'][0]['rows'][-1]['scores'][1]=.5
        data['C'][0]['proxies']['artifact']=.011
        result=science.verdict(data,cal)
        self.assertTrue(result['risk_gate'])
        self.assertFalse(result['proxy_gate'])

if __name__=='__main__':
    if sys.argv[1:2]==['synthetic']:
        synthetic(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4])
    else:
        unittest.main()
