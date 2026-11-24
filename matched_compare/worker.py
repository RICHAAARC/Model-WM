"""One fixed CAL -> seal -> TEST execution. No retries or online selection."""
import json
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from latent_control.io import save, append
from probe_selection import rules as R
from .rules import ARMS,LIMITS,calibrate,evaluate,adjudicate

PACKAGE=Path(__file__).resolve().parent

def seal(path,value):
    if path.exists(): raise FileExistsError(path)
    save(path,value)
    if json.loads(path.read_text())!=value: raise RuntimeError('Seal readback mismatch')
    return value

def run(local,out,allocation,backend,carrier):
    if allocation['selected_gain']!=1.025: raise ValueError('Frozen gain changed')
    cal=[w for w in allocation['works'] if w['split']=='CAL']
    test=[w for w in allocation['works'] if w['split']=='TEST']
    if len(cal)!=24 or len(test)!=80 or len({w['id'] for w in cal+test})!=104:
        raise ValueError('Incomplete frozen allocation')
    counts={k:0 for k in LIMITS};done={k:0 for k in LIMITS}
    completed=0;clock={};calrows=[];reports=[]
    bank=R.signs(allocation['codes'][:16])
    def call(kind,tag,fn,*args):
        if counts[kind]>=LIMITS[kind]: raise RuntimeError('Physical limit '+kind)
        counts[kind]+=1
        append(local/'operations.jsonl',dict(kind=kind,id=tag,state='STARTED',at=time.time()))
        try: result=fn(*args)
        except BaseException:
            append(local/'operations.jsonl',dict(kind=kind,id=tag,state='ERROR',traceback=traceback.format_exc()))
            raise
        done[kind]+=1
        append(local/'operations.jsonl',dict(kind=kind,id=tag,state='COMPLETED',at=time.time()))
        save(local/'progress.json',dict(started=counts,completed=done,works_completed=completed,active=tag))
        return result
    selection=seal(out/'selection_seal.json',dict(gain=1.025,coarse_transfer=.1,
        source='completed independent DEV; quality-only selection',at=time.time()))
    append(out/'stage_events.jsonl',dict(event='SELECTION_SEALED',at=time.time()))
    threshold=None
    for stage,works in [('CAL',cal),('TEST',test)]:
        if stage=='TEST':
            # Verify from disk, including all completed CAL records, before any TEST reservation or generation.
            saved=json.loads((out/'cal_seal.json').read_text())
            reread=[json.loads((out/'works'/w['id']/'completed.json').read_text()) for w in cal]
            if saved['calibration']!=calibrate(reread): raise RuntimeError('CAL seal does not match completed evidence')
            if json.loads((out/'selection_seal.json').read_text())!=selection: raise RuntimeError('Selection changed')
            threshold=saved['calibration']['tau']
            append(out/'stage_events.jsonl',dict(event='TEST_ENTERED_AFTER_SEAL_VERIFIED',at=time.time()))
        begin=time.monotonic()
        for w in works:
            dest=out/'works'/w['id'];dest.mkdir(parents=True,exist_ok=False)
            save(dest/'reserved.json',dict(w,at=time.time()))
            _,h,clean=call('generate',w['id'],backend.generate,w)
            def valid(rgb):
                if rgb.dtype!=np.uint8 or rgb.shape!=(512,512,3): raise ValueError('Invalid RGB')
                return rgb
            def views(name,rgb):
                rgb=valid(rgb);p=dest/(name+'.png');p.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(rgb).save(p)
                rows={}
                for codec in ('png','jpeg85'):
                    f=p if codec=='png' else dest/(name+'.jpg')
                    if codec=='jpeg85': Image.fromarray(rgb).save(f,quality=85,subsampling=2)
                    with Image.open(f) as im: decoded=np.asarray(im.convert('RGB'))
                    q=call('read',w['id']+'/'+name+'/'+codec,carrier.read,decoded)
                    values=R.scores(q,bank)
                    append(dest/'readouts.jsonl',dict(name=name,codec=codec,logits=list(q),scores=values))
                    rows[codec]=values
                return rows
            null={f'null/clean/{c}':v for c,v in views('clean',clean).items()}
            raw={};quality={}
            for arm in w['arm_order']:
                rows=dict(null);losses=[]
                for group in ('reg','foreign'):
                    pics=[]
                    for i,uid in enumerate(w[group]):
                        name=f'{arm}/{group}/single{i}'
                        code=allocation['codes'][uid]
                        rgb=call('write',w['id']+'/'+name,
                                 carrier.uniform if arm=='ordinary' else carrier.write,
                                 h,code,1.025 if arm=='ordinary' else 'coarse')
                        valid(rgb);pics.append(rgb);loss=R.distance(clean,rgb)
                        if not np.isfinite(loss) or loss<=0: raise ValueError('Invalid quality')
                        losses.append(loss)
                        append(dest/'quality.jsonl',dict(arm=arm,group=group,user=uid,loss=loss))
                        for codec,v in views(name,rgb).items(): rows[f'{group}/single{i}/{codec}']=v
                    for view,weights in R.COMPOSITIONS.items():
                        name=f'{arm}/{group}/{view}'
                        rgb=call('mix',w['id']+'/'+name,R.mix,pics[:len(weights)],weights)
                        for codec,v in views(name,rgb).items(): rows[f'{group}/{view}/{codec}']=v
                raw[arm]=rows;quality[arm]=float(np.mean(losses))
            row=dict(w,raw=raw,quality=quality)
            if stage=='CAL': calrows.append(row)
            else:
                report=evaluate(row,threshold);reports.append(report);save(dest/'paired.json',report)
            save(dest/'completed.json',row);completed+=1
            save(local/'progress.json',dict(started=counts,completed=done,works_completed=completed,stage=stage))
        clock[stage]=time.monotonic()-begin
        if stage=='CAL':
            seal(out/'cal_seal.json',dict(calibration=calibrate(calrows),selection=selection,at=time.time()))
            append(out/'stage_events.jsonl',dict(event='CAL_THRESHOLD_SEALED',at=time.time()))
    if counts!=LIMITS or done!=LIMITS: raise RuntimeError('Incomplete operation family')
    save(out/'paired_results.json',reports)
    result=adjudicate(reports)
    result.update(model_components_wall_seconds=clock,cal_threshold=threshold,
                  cal_test_started=True,holdout_unsealed=False,old_holdout_unsealed=False,
                  new_test_entered_after_cal_seal=True,dev_recomputed=False)
    save(out/'comparison.json',result)
    return result

if __name__=='__main__':
    import sys
    from lawa_ab.model import Backend
    from matched_loss.carrier import Carrier
    local,out,scratch=map(Path,sys.argv[1:])
    b=Backend.load(scratch)
    save(local/'outcome.json',run(local,out,json.loads((PACKAGE/'allocation.json').read_text()),b,Carrier(b,.1)))
