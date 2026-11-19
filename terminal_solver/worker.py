import sys
import time
from pathlib import Path
import torch
from latent_control.io import Ledger,save,seal
from latent_control.protocol import arrays
from latent_control.worker import evaluation,size
from latent_control.carrier import quality
from .protocol import protocol,limits,read_json,GAINS
from .model import Solver
from . import quality as proxy, science

def generate_work(model,w,gains,root,archive,ledger,m):
    start=time.monotonic()
    cpu_start=time.process_time()
    wid=w['id']
    target=root/'works'/wid
    target.mkdir(parents=True,exist_ok=False)
    clean,center,y,rc=model.common_center(w)
    shared=time.monotonic()-start
    common_cpu=time.process_time()-cpu_start
    inverse=None
    times={a:{} for a in 'CO'}
    arm_cpu={a:{} for a in 'CO'}
    keys=[(kind,i) for kind in ('registered','foreign') for i in range(2)]
    for arm in w['arm_order']:
        if arm=='C':
            t=time.monotonic()
            ct=time.process_time()
            inverse=model.response(center,y,wid)
            common_cpu+=time.process_time()-ct
            shared+=time.monotonic()-t
        for g in gains[arm]:
            t=time.monotonic()
            ct=time.process_time()
            arm_id=f'{arm}/{g}'
            dest=target/arm_id
            dest.mkdir(parents=True)
            outputs={}
            for kind,i in keys:
                user=w[kind][i]
                ident=f'{wid}/{arm_id}/{kind}/{i}'
                def write():
                    if arm=='C':
                        return model.inverse_copy(center,rc,inverse,user,g,ident)
                    return model.ordinary_copy(center,user,g,ident)
                try:
                    rgb=model.call(ident,'write_'+arm,write)
                except BaseException:
                    from PIL import Image
                    for (group,index),partial in outputs.items():
                        Image.fromarray(partial['rgb']).save(dest/f'{group}_{index}_partial.png')
                    raise
                outputs[kind,i]=dict(rgb=rgb,quality=quality(clean,rgb))
            bundle=evaluation(model,clean,outputs,w,arm_id,dest,ledger,m)
            bundle["proxies"]=proxy.measure(dest)
            save(dest/"readouts.json",bundle)
            seal(dest,archive/'works'/wid/arm_id)
            times[arm][str(g)]=time.monotonic()-t
            arm_cpu[arm][str(g)]=time.process_time()-ct
    t=time.monotonic()
    ct=time.process_time()
    save(archive/'work_receipts'/f'{wid.replace("/","-")}.json',dict(id=wid,operations=dict(ledger.counts)))
    archive_elapsed=time.monotonic()-t
    return dict(id=wid,common_including_C_response=shared,arm_seconds=times,archive_seconds=archive_elapsed,
                bytes={a:{str(g):size(target/a/str(g)) for g in gains[a]} for a in 'CO'},
                common_cpu=common_cpu+time.process_time()-ct,arm_cpu=arm_cpu,cpu_seconds=time.process_time()-cpu_start,
                wall_seconds=time.monotonic()-start)

def bundles(stage_root,works,arm,g):
    results=[]
    for w in works:
        b=read_json(stage_root/'works'/w['id']/arm/str(g)/'readouts.json')
        if b.get('id')!=w['id'] or b.get('arm')!=f'{arm}/{g}' or not b.get('complete'):
            raise RuntimeError('Mismatched or incomplete work/arm/gain bundle')
        results.append(b)
    return results

def run(root,archive,stage,model,m,ledger,generate=generate_work):
    parent=archive.parent
    works=m['works']
    dev=stage=='dev'
    selected=None if dev else read_json(parent/'dev/dev_decision.json')['gains']
    gains={a:GAINS if dev else [selected[a]] for a in 'CO'}
    timing=[]
    for split in (['DEV'] if dev else ['CAL','TEST']):
        batch=[w for w in works if w['split']==split]
        if split=='TEST':
            cal=read_json(archive/'calibration.json')
            if cal!=read_json(root/'calibration.json') or any(not cal[a]['sealed'] for a in 'CO'):
                raise RuntimeError('CAL seal mismatch')
        for w in batch:
            save(root/'progress.json',dict(phase=split,active_work=w['id'],evaluation_bundles=len(timing)))
            timing.append(generate(model,w,gains,root,archive,ledger,m))
        if split=='CAL':
            cal={a:science.calibrate(bundles(root,batch,a,selected[a]),expected=64) for a in 'CO'}
            save(root/'calibration.json',cal)
            seal(root/'calibration.json',archive/'calibration.json')
        elif split=='DEV':
            decision=science.select({a:{g:bundles(root,batch,a,g) for g in GAINS} for a in 'CO'})
            save(root/'dev_decision.json',decision)
        else:
            result=science.verdict({a:bundles(root,batch,a,selected[a]) for a in 'CO'},cal)
            save(root/'paired_results.json',result)
            proxy.gallery(root,batch,selected,result)
    save(root/'timings.json',timing)
    save(root/'worker_complete.json',dict(phase=stage,works=len(timing),operations=ledger.counts))

def main():
    root,archive=map(Path,sys.argv[1:3])
    stage=sys.argv[3]
    if stage not in ('dev','formal'):
        raise ValueError('Unknown stage')
    m=protocol()
    batch=[w for w in m['works'] if (w['split']=='DEV')==(stage=='dev')]
    torch.set_num_threads(1)
    torch.manual_seed(m['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    ledger=Ledger(root,limits(batch))
    model=Solver(read_json(root/'model_paths.json')['base'],m,arrays(),ledger)
    run(root,archive,stage,model,m,ledger)

if __name__=='__main__':
    main()
