"""Bounded quality-only DEV. Every physical call reserved before execution."""
import math
import sys
import time
import traceback
from pathlib import Path
from statistics import mean
import numpy as np
from PIL import Image
from latent_control.io import append, save
from lawa_ab.entry import read
from probe_selection.rules import distance

PACKAGE=Path(__file__).resolve().parent
GAINS=(1.,1.025,1.05,1.075)

def decide(rows):
    if len(rows)!=8 or len({r['work'] for r in rows})!=8:
        raise ValueError('Incomplete works')
    changes={}
    for gain in GAINS:
        values=[math.log(r['loss'][str(gain)]/r['loss']['coarse']) for r in rows]
        changes[str(gain)]=mean(values)
    best=min(GAINS,key=lambda g:(abs(changes[str(g)]),g))
    ratio=math.exp(changes[str(best)])
    passed=.98<=ratio<=1.02
    return dict(selected_gain=best,geometric_loss_ratio=ratio,log_ratios=changes,
                matching_gate_passed=passed,cal_test_authorized=False,
                status='DEV_MATCHING_GATE_MET' if passed else 'DEV_MATCHING_STOP',
                scientific_pass=None,visual_review='PENDING')

def run(local,out,allocation,backend,carrier):
    rows=[]
    count=dict(generate=0,write=0,read=0)
    def call(kind,tag,fn,*args):
        cap={'generate':8,'write':240,'read':24}[kind]
        if count[kind]>=cap:
            raise RuntimeError('Physical limit')
        count[kind]+=1
        event=dict(kind=kind,id=tag)
        append(local/'operations.jsonl',dict(event,state='STARTED',at=time.time()))
        try:
            result=fn(*args)
        except BaseException:
            append(local/'operations.jsonl',dict(event,state='ERROR',traceback=traceback.format_exc()))
            raise
        append(local/'operations.jsonl',dict(event,state='COMPLETED'))
        save(local/'progress.json',dict(started=count,works_completed=len(rows)))
        return result
    for work in [w for w in allocation['works'] if w['split']=='DEV']:
        dest=out/'works'/work['id']
        dest.mkdir(parents=True,exist_ok=False)
        save(dest/'reserved.json',work)
        z,h,clean=call('generate',work['id'],backend.generate,work)
        np.save(dest/'latent.npy',z)
        save(dest/'interface.json',backend.last_interface)
        Image.fromarray(clean).save(dest/'clean.png')
        losses={}
        for action in work['action_order']:
            values=[]
            for i,user in enumerate(work['reg']+work['foreign']):
                code=allocation['codes'][user]
                if action=='coarse':
                    image=call('write',f"{work['id']}/{action}/{i}",carrier.write,h,code,'coarse')
                else:
                    image=call('write',f"{work['id']}/{action}/{i}",carrier.uniform,h,code,float(action))
                if image.dtype!=np.uint8 or image.shape!=clean.shape or image.shape!=(512,512,3):
                    raise ValueError('Invalid native RGB')
                folder=dest/action
                folder.mkdir(exist_ok=True)
                Image.fromarray(image).save(folder/f'{i}.png')
                if work['id'] in ('DEV-000','DEV-001') and action in ('coarse','1.0'):
                    logits=call('read',f"{work['id']}/{action}/{i}",backend.read,image)
                    if np.asarray(logits).shape!=(48,) or not np.isfinite(logits).all():
                        raise ValueError('Reader interface failed')
                    append(dest/'interface_readouts.jsonl',dict(action=action,user=user,logits=logits))
                loss=distance(clean,image)
                if not math.isfinite(loss) or loss<=0:
                    raise ValueError('Invalid structural loss')
                values.append(loss)
                append(dest/'quality.jsonl',dict(action=action,user=user,loss=loss))
            losses[action]=mean(values)
        row=dict(work=work['id'],loss=losses)
        rows.append(row)
        save(dest/'completed.json',row)
        save(local/'progress.json',dict(started=count,works_completed=len(rows)))
    if count!={'generate':8,'write':240,'read':24}:
        raise RuntimeError('Incomplete operation family')
    save(out/'quality_records.json',rows)
    decision=decide(rows)
    decision.update(interface_numeric_passed=True,interface_works=2,interface_reads=24,
                    scope='DEV quality choice only; interface readouts never used to choose gain')
    save(out/'dev_decision.json',decision)
    return decision

if __name__=='__main__':
    from .model import Backend
    from matched_loss.carrier import Carrier
    local,out,scratch=map(Path,sys.argv[1:])
    backend=Backend.load(scratch)
    result=run(local,out,read(PACKAGE/'allocation.json'),backend,Carrier(backend,.10))
    save(local/'outcome.json',result)

