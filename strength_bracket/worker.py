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
from .rules import GAINS,choose

def run(local,out,allocation,backend,carrier):
    rows=[]
    count=dict(generate=0,write=0)
    def call(kind,tag,fn,*args):
        cap={'generate':8,'write':336}[kind]
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
    for work in (w for w in allocation['works'] if w['split']=='DEV'):
        dest=out/'works'/work['id']
        dest.mkdir(parents=True,exist_ok=False)
        save(dest/'reserved.json',work)
        _,h,clean=call('generate',work['id'],backend.generate,work)
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
                loss=distance(clean,image)
                if not math.isfinite(loss) or loss<=0:
                    raise ValueError('Invalid structural loss')
                values.append(loss)
                append(dest/'quality.jsonl',dict(action=action,user=user,loss=loss))
            losses[action]=mean(values)
        row=dict(work=work['id'],loss=losses)
        rows.append(row)
        save(dest/'completed.json',row)
    if count!={'generate':8,'write':336}:
        raise RuntimeError('Incomplete operation family')
    save(out/'quality_records.json',rows)
    decision=choose(rows)
    save(out/'dev_decision.json',decision)
    return decision

if __name__=='__main__':
    from lawa_ab.model import Backend
    from .carrier import Carrier
    local,out,scratch=map(Path,sys.argv[1:])
    backend=Backend.load(scratch)
    result=run(local,out,read(PACKAGE/'allocation.json'),backend,Carrier(backend,.10))
    save(local/'outcome.json',result)
