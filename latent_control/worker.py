"""DEV once, both CAL seals, then TEST once; every physical attempt is retained."""
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .protocol import read,arrays,validate
from .io import save,seal,Ledger,MechanismFailure,BudgetStop
from .model import Model
from .carrier import tensor,quantize,mixed,jpeg
from . import science

def cpu():
    r=resource.getrusage(resource.RUSAGE_SELF);return r.ru_utime+r.ru_stime

def size(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())

def evaluation(model,clean,outputs,work,arm,target,ledger,m):
    media=[('null','null',clean,[],np.zeros(64),[])]
    qualities=[];failed=[]
    for group in ['registered','foreign']:
        for i,user in enumerate(work[group]):
            out=outputs.get((group,i))
            if out is None:
                failed.append(group+str(i));continue
            qualities.append(out['quality'])
            media.append((f'{group}_{i}',group+'_native',out['rgb'],[user] if group=='registered' else [],
                          model.codes[user].cpu().numpy(),[1.]))
        for wi,weight in enumerate(m['mix_weights']):
            aa,bb=outputs.get((group,0)),outputs.get((group,1))
            if aa is None or bb is None:continue
            ideal=(weight*model.codes[work[group][0]]+(1-weight)*model.codes[work[group][1]]).cpu().numpy()
            media.append((f'{group}_mix_{wi}',group+'_mix',mixed(aa['rgb'],bb['rgb'],weight),
                          work[group] if group=='registered' else [],ideal,[weight,1-weight]))
    rows=[]
    for name,kind,rgb,truth,ideal,weights in media:
        baseline_q=None
        for codec in ['png','jpeg85']:
            condition=name+'_'+codec;path=target/(condition+('.png' if codec=='png' else '.jpg'))
            if codec=='png':Image.fromarray(rgb).save(path);observed=rgb
            else:
                data,observed=jpeg(rgb);path.write_bytes(data)
            with Image.open(path) as im:
                if not np.array_equal(np.asarray(im.convert('RGB')),observed):raise RuntimeError('Persisted media mismatch')
            def extract():
                with torch.no_grad():return (model.reader(tensor(observed,model.device))[0]/model.a).cpu().numpy()
            q=ledger.call(f'{work["id"]}/{arm}/read/{condition}','extract',extract)
            scores=(model.codes[:16].cpu().numpy().astype(np.float64)@q.astype(np.float64)).tolist()
            if not np.isfinite(scores).all():raise RuntimeError('Nonfinite readout')
            if codec=='png':baseline_q=q
            rows.append(dict(name=condition,kind=kind,truth=truth,weights=weights,q=q.tolist(),scores=scores,
                ideal_q=ideal.tolist(),endpoint_or_mixture_error=(q-ideal).tolist() if kind!='null' else None,
                codec_added_error=(q-baseline_q).tolist(),codec_error_l2=float(np.linalg.norm(q-baseline_q))))
    result=dict(id=work['id'],arm=arm,rows=rows,quality=qualities,failed_copies=failed,
                complete=len(rows)==22 and len(qualities)==4 and not failed)
    save(target/'readouts.json',result)
    return result

def write_outputs(target,outputs):
    for key,out in outputs.items():
        name=key[0]+'_'+str(key[1]);np.save(target/(name+'_control.npy'),out['control'],allow_pickle=False)
        Image.fromarray(out['rgb']).save(target/(name+'_written.png'))
        save(target/(name+'_writer.json'),{k:v for k,v in out.items() if k not in ['rgb','control']})

def work_once(model,work,root,archive,ledger,m):
    target=root/'works'/work['id'];target.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();started_cpu=cpu();arm_times={};results={}
    z=model.generate(work)
    clean_y=model.decode(z,work['id']+'/clean','forward_C');clean=quantize(clean_y)
    Image.fromarray(clean).save(target/'clean.png');np.save(target/'latent.npy',z.detach().cpu().numpy(),allow_pickle=False)
    fd=model.finite_difference(z,clean_y,work['id']) if work['split']=='DEV' else None
    users=work['registered']+work['foreign'];keys=[('registered',0),('registered',1),('foreign',0),('foreign',1)]
    for arm in work['arm_order']:
        model.sync();t=time.monotonic();c=cpu();at=target/arm;at.mkdir();outputs={};errors=[]
        if arm=='C':
            try:
                def retain(index,out):
                    outputs[keys[index]]=out;write_outputs(at,{keys[index]:out})
                copies,diag=model.shared(z,clean_y,users,work['id']+'/C',on_copy=retain)
                outputs=dict(zip(keys,copies));save(at/'response.json',diag)
                if fd is not None:
                    exact=np.asarray(diag['initial']['first_response_column']);f=m['finite_difference']
                    valid=bool(np.allclose(fd,exact,atol=f['absolute_tolerance'],rtol=f['relative_tolerance']))
                    save(at/'finite_difference.json',dict(analytic=exact.tolist(),measured=fd.tolist(),passed=valid,settings=f))
                    if not valid:raise RuntimeError('Real DEV finite-difference interface check failed; no parameter adjustment')
            except MechanismFailure as exc:
                errors.append(dict(error=str(exc),traceback=traceback.format_exc(),failed_copies=[k for k in keys if k not in outputs]))
        else:
            for key,user in zip(keys,users):
                try:
                    outputs[key]=model.ordinary(z,clean_y,user,work['id']+'/B/'+key[0]+str(key[1]))
                    write_outputs(at,{key:outputs[key]})
                except MechanismFailure as exc:errors.append(dict(copy=key,error=str(exc),traceback=traceback.format_exc()))
        save(at/'failures.json',errors);write_outputs(at,outputs)
        results[arm]=evaluation(model,clean,outputs,work,arm,at,ledger,m)
        model.sync();arm_times[arm]=dict(wall_seconds=time.monotonic()-t,cpu_seconds=cpu()-c)
    save(target/'arm_times.json',arm_times)
    # Publishing is counted in the shared work overhead; no live Drive IPC reads.
    seal(target,archive/'works'/work['id'])
    timing=dict(wall_seconds=time.monotonic()-started,cpu_seconds=cpu()-started_cpu,
                result_bytes=size(target),arms=arm_times)
    save(root/'timing'/f'{work["id"].replace("/","-")}.json',timing)
    return results,timing

def project(dev,root,m,started,started_cpu):
    live=json.loads((root/'budget_state.json').read_text()) if (root/'budget_state.json').exists() else {}
    used=live.get('model',dict(wall_seconds=time.monotonic()-started,cpu_seconds=cpu()-started_cpu))
    remaining=sum(w['split']!='DEV' for w in m['works']);factor=1.5
    projection={k:used.get(k,0)+remaining*max(t[k] for t in dev)*factor+60 for k in ['wall_seconds','cpu_seconds']}
    projection['result_bytes']=size(root)+remaining*max(t['result_bytes'] for t in dev)*factor
    limits=dict(wall_seconds=m['budgets']['model_seconds'],cpu_seconds=m['budgets']['cpu_seconds'],result_bytes=m['budgets']['result_bytes'])
    passed=all(projection[k]<limits[k] for k in limits)
    row=dict(passed=passed,projected=projection,limits=limits,remaining_works=remaining,factor=factor,
             basis='Maximum complete DEV work cost including both sequential arms, generation, media and archival',additional_warmups=0)
    save(root/'dev_projection.json',row)
    if not passed:raise BudgetStop('DEV completion projection exceeds fixed budget')

def main(root,archive,scratch):
    root,archive,scratch=map(Path,(root,archive,scratch));m=read();frozen=arrays();validate(m,frozen)
    torch.set_num_threads(1);torch.manual_seed(m['master_seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    started=time.monotonic();started_cpu=cpu();ledger=Ledger(root,m['physical_limits'])
    progress=dict(phase='LOAD',evaluation_bundles=0,train_updates=0)
    save(root/'progress.json',progress)
    paths=json.loads((root/'model_paths.json').read_text())
    model=Model(paths['base'],m,frozen,ledger)
    all_bundles={s:{a:[] for a in 'BC'} for s in ['DEV','CAL','TEST']};timings=[];cal=None
    for split in ['DEV','CAL','TEST']:
        if split=='TEST':
            # Require an exact readback of BOTH sealed CAL arms before even generating first TEST work.
            cal=json.loads((archive/'calibration.json').read_text())
            if cal!=json.loads((root/'calibration.json').read_text()) or any(not cal[a]['sealed'] for a in 'BC'):
                raise RuntimeError('CAL seal/readback mismatch')
        for work in [w for w in m['works'] if w['split']==split]:
            progress.update(phase=split,active_work=work['id']);save(root/'progress.json',progress)
            result,timing=work_once(model,work,root,archive,ledger,m)
            for arm in 'BC':all_bundles[split][arm].append(result[arm])
            timings.append(timing);progress['evaluation_bundles']+=1;save(root/'progress.json',progress)
        if split=='DEV':
            project(timings,root,m,started,started_cpu);seal(root/'dev_projection.json',archive/'dev_projection.json')
        elif split=='CAL':
            cal={a:science.calibrate(all_bundles['CAL'][a]) for a in 'BC'}
            save(root/'calibration.json',cal);seal(root/'calibration.json',archive/'calibration.json')
    evaluated={a:[science.evaluate(b,cal[a]['threshold'],frozen['codes'][:16],m) for b in all_bundles['TEST'][a]] for a in 'BC'}
    save(root/'paired_results.json',evaluated)
    # Everything outside arm-exclusive writing/evaluation is split equally, including load/generation and IO.
    exclusive={a:{k:sum(t['arms'][a][k] for t in timings) for k in ['wall_seconds','cpu_seconds']} for a in 'BC'}
    elapsed=dict(wall_seconds=time.monotonic()-started,cpu_seconds=cpu()-started_cpu)
    shared={k:max(0,elapsed[k]-sum(exclusive[a][k] for a in 'BC')) for k in elapsed}
    costs={a:{k:exclusive[a][k]+shared[k]/2 for k in elapsed} for a in 'BC'}
    result=science.decide(evaluated,all_bundles['TEST'],costs,m)
    result.update(shared_cost=shared,exclusive_cost=exclusive,worker_cost=elapsed,operations=ledger.counts,
        owner_and_terminal_overhead='Entry adds remaining common overhead equally; it cannot alter C<=B cost sign',
        checkpoint_history='UNVERIFIED',unseen_content='USER_ACCEPTED_ASSUMPTION')
    save(root/'method_result.json',result)
    progress.update(phase='COMPLETE',active_work=None);save(root/'progress.json',progress)

if __name__=='__main__':main(*sys.argv[1:4])
