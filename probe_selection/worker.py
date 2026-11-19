"""Single pass through DEV, independent CAL and paired exploration."""
from collections import Counter,defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from zipfile import ZipFile, ZIP_STORED

import numpy as np
from PIL import Image
from latent_control.io import append,save
from . import rules as R
from .worker_paths import PRIOR

PACKAGE=Path(__file__).resolve().parent
LIMITS=dict(generate=32,restore=8,probe_write=576,official_write=632,probe_extract=960,
            official_extract=2392,mix=792)

def read(path):
    return json.loads(path.read_text())

class Work:
    def __init__(self,path,counts):
        self.path=path
        self.counts=counts
        self.times=defaultdict(lambda:dict(wall=0.,cpu=0.,completed=0))
        self.started=Counter()
        self.done=Counter()

    @contextmanager
    def time(self,kind):
        start=time.monotonic(),time.process_time()
        try:
            yield
            self.times[kind]['completed']+=1
        finally:
            self.times[kind]['wall']+=time.monotonic()-start[0]
            self.times[kind]['cpu']+=time.process_time()-start[1]

    def call(self,kind,name,fn,*args):
        if self.counts[kind]>=LIMITS[kind]:
            raise RuntimeError('Physical operation limit '+kind)
        self.counts[kind]+=1
        self.started[kind]+=1
        append(self.path/'events.jsonl',dict(kind=kind,name=name,state='started'))
        with self.time(kind):
            value=fn(*args)
        self.done[kind]+=1
        append(self.path/'events.jsonl',dict(kind=kind,name=name,state='complete'))
        return value

    def put(self,name,img):
        p=self.path/(name+'.png')
        with self.time('storage'):
            p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(img).save(p)
        return img

    def views(self,carrier,kind,name,img,codecs=('png','jpeg85')):
        values=[]
        for codec in codecs:
            file=self.path/(name+('.png' if codec=='png' else '.jpg'))
            with self.time('storage'):
                if codec=='jpeg85':
                    Image.fromarray(img).save(file,quality=85,subsampling=2)
                with Image.open(file) as opened:
                    rgb=np.array(opened.convert('RGB'))
            q=self.call(kind,name+'/'+codec,carrier.read,rgb)
            if not np.isfinite(q).all():
                raise ValueError('Invalid readout '+name)
            append(self.path/'readouts.jsonl',dict(name=name,codec=codec,logits=q))
            values.append(q)
        return values

def copies(ctx,carrier,h,clean,codes,ids,action,group,old=None):
    pics=[]; quality=[]
    for i,uid in enumerate(ids):
        name=f'{action}/{group}/single{i}'
        if old is not None and action=='native' and i<2:
            with Image.open(old/f'reg_{i}_png.png') as original:
                rgb=np.array(original.convert('RGB'))
        else:
            kind='probe_write' if group.startswith('probe') else 'official_write'
            rgb=ctx.call(kind,name,carrier.write,h,codes[uid],action)
        pics.append(ctx.put(name,rgb))
        with ctx.time('quality'):
            quality.append(R.distance(clean,rgb))
    return pics,quality

def run_probes(ctx,carrier,h,clean,m):
    codes=m['probe_codes']
    bank=R.signs(codes)
    stats={}
    for action in carrier.actions:
        mixed=[];singles=[];quality=[]
        for t,ids in enumerate(m['probe_triples']):
            group=f'probe{t}'
            pics,dists=copies(ctx,carrier,h,clean,codes,ids,action,group)
            quality.extend(dists)
            for i,(uid,rgb) in enumerate(zip(ids,pics)):
                q,=ctx.views(carrier,'probe_extract',f'{action}/{group}/single{i}',rgb,('png',))
                singles.append((uid,q))
            name=f'{action}/{group}/equal3'
            rgb=ctx.put(name,ctx.call('mix',name,R.mix,pics,R.COMPOSITIONS['equal3']))
            mixed.append(ctx.views(carrier,'probe_extract',name,rgb))
        stats[action]=R.probe_stat(mixed,singles,m['probe_triples'],bank,float(np.mean(quality)))
    return stats

def issue_action(probe_provider,global_action,ratio,enabled=True):
    """Disabled M selects the native watermarked writer without any probe writes."""
    if not enabled:
        return {},{'native':'native','margin':'native'}
    stats=probe_provider()
    return stats,R.pick(stats,global_action,ratio)

def formal(ctx,carrier,h,clean,w,codes,old=None):
    bank=R.signs(codes[:16])
    raw={};quality={}
    null={}
    if old is None:
        for codec,q in zip(('png','jpeg85'),ctx.views(carrier,'official_extract','clean',clean)):
            null['null/clean/'+codec]=R.scores(q,bank)
    for action in carrier.actions:
        rows=dict(null)
        dists=[]
        for group in (('reg',) if old else ('reg','foreign')):
            ids=w[group]
            pics,d=copies(ctx,carrier,h,clean,codes,ids,action,group,old)
            dists.extend(d)
            inputs={}
            if old is None:
                inputs.update({f'single{i}':v for i,v in enumerate(pics)})
            for view in (('equal3',) if old else R.COMPOSITIONS):
                weights=R.COMPOSITIONS[view]
                name=f'{action}/{group}/{view}'
                inputs[view]=ctx.put(name,ctx.call('mix',name,R.mix,pics[:len(weights)],weights))
            codecs=('png',) if old else ('png','jpeg85')
            for view,rgb in inputs.items():
                qrows=ctx.views(carrier,'official_extract',f'{action}/{group}/{view}',rgb,codecs)
                for codec,q in zip(codecs,qrows):
                    rows[f'{group}/{view}/{codec}']=R.scores(q,bank)
        raw[action]=rows
        quality[action]=float(np.mean(dists))
    return raw,quality

def seal(path,value):
    if path.exists():
        raise FileExistsError(path)
    save(path,value)
    if read(path)!=value:
        raise RuntimeError('Seal readback mismatch '+str(path))
    return value

def audit_row(w,raw,quality,choices,thresholds,ratio):
    arms={arm:R.evaluate(w,raw[action],thresholds[arm]) for arm,action in choices.items()}
    fixed={action:R.evaluate(w,raw[action],thresholds['fixed_'+action])[R.PRIMARY] for action in R.ACTIONS}
    real={a:R.margin(raw[a][R.PRIMARY],w['reg']) for a in R.ACTIONS}
    return dict(id=w['id'],choices=choices,real_margins=real,
                best_actions=[a for a in R.ACTIONS if real[a]==max(real.values())],
                arms=arms,fixed_primary=fixed,quality=quality,
                quality_ok=quality[choices['margin']]<=ratio*quality['native']+1.e-12)

def run(carrier,m,local,out,oldroot):
    counts=Counter()
    dev=[];cal=[];reports=[]
    ratio=m['quality_ratio']
    for k,w in enumerate(m['dev']+m['works']):
        wid=w['id'];stage=w.get('split','dev')
        path=local.parent/'media'/wid
        claim=out/'work_receipts'/wid
        claim.mkdir(parents=True,exist_ok=False)
        save(claim/'started.json',dict(id=wid,stage=stage,time=time.time()))
        path.mkdir(parents=True,exist_ok=False)
        ctx=Work(path,counts)
        save(local/'progress.json',dict(phase=stage,evaluation_bundles=k,active_work=wid))
        try:
            if stage=='dev':
                if wid!=f'fit-{k:02}' or k>=8:
                    raise ValueError('Forbidden historical work')
                old=oldroot/'fit'/wid
                z=np.load(old/'latent.npy',allow_pickle=False)
                h=ctx.call('restore',wid,carrier.restore,z)
                with Image.open(old/'null_png.png') as im:
                    clean=np.array(im.convert('RGB'))
                raw,q=formal(ctx,carrier,h,clean,w,m['codes'],old)
                dev.append(dict(id=wid,quality=q,margin={a:R.margin(raw[a][R.PRIMARY],w['reg']) for a in R.ACTIONS}))
                save(path/'dev.json',dev[-1])
                if len(dev)==8:
                    allowed=[a for a in R.ACTIONS if all(r['quality'][a]<=ratio*r['quality']['native']+1.e-12 for r in dev)]
                    chosen=R.choose({a:sum(r['margin'][a] for r in dev) for a in R.ACTIONS},allowed)
                    seal(out/'global.json',dict(action=chosen,dev=dev))
            else:
                global_action=read(out/'global.json')['action']
                thresholds=read(out/'thresholds.json')['thresholds'] if stage=='explore' else None
                z,h,clean=ctx.call('generate',wid,carrier.generate,w)
                np.save(path/'latent.npy',z,allow_pickle=False)
                ctx.put('clean',clean)
                stats,choices=issue_action(lambda:run_probes(ctx,carrier,h,clean,m),global_action,ratio)
                choices=seal(claim/'selection.json',dict(stats=stats,choices=choices,time=time.time()))['choices']
                raw,q=formal(ctx,carrier,h,clean,w,m['codes'])
                save(path/'formal_scores.json',raw)
                if stage=='cal':
                    maxima={arm:R.negative(w,raw[a]) for arm,a in choices.items()}
                    maxima.update({'fixed_'+a:R.negative(w,raw[a]) for a in R.ACTIONS})
                    cal.append(dict(id=wid,maxima=maxima))
                    if len(cal)==24:
                        seal(out/'thresholds.json',dict(cal=cal,thresholds={arm:max(v['maxima'][arm] for v in cal) for arm in maxima}))
                else:
                    row=audit_row(w,raw,q,choices,thresholds,ratio)
                    reports.append(row)
                    save(out/'exploration'/(wid+'.json'),row)
                save(path/'formal_quality.json',q)
        except BaseException:
            (claim/'exception.txt').write_text(traceback.format_exc())
            raise
        finally:
            # Preserve complete and partial local files, even after a model exception.
            with ctx.time('storage'):
                folder=out/'archives'
                folder.mkdir(exist_ok=True)
                temp=folder/(wid+'.part')
                packed=local.parent/'media'/(wid+'.zip')
                with ZipFile(packed,'x',ZIP_STORED) as zf:
                    for item in path.rglob('*'):
                        if item.is_file():
                            zf.write(item,item.relative_to(path))
                shutil.copyfile(packed,temp)
                os.replace(temp,folder/(wid+'.zip'))
                packed.unlink()
            save(claim/'usage.json',dict(times=ctx.times,started=ctx.started,completed=ctx.done))
        save(claim/'completed.json',dict(id=wid,stage=stage,time=time.time()))
    if counts!=Counter(LIMITS):
        raise ValueError(f'Unexpected physical counts {counts}')
    result=R.verdict(reports)
    result.update(operations=dict(counts),work_count=40,new_works=32)
    save(out/'paired_decision.json',result)
    return result

def main():
    from .carrier import LaWaCarrier
    from lawa_ab.worker import phase
    local,out,scratch=map(Path,sys.argv[1:4])
    m=read(PACKAGE/'freeze.json')
    with phase(out,'load'):
        carrier=LaWaCarrier.load(scratch,m['transfer_fraction'])
    result=run(carrier,m,local,out,out.parent/PRIOR)
    save(local/'outcome.json',result)

if __name__=='__main__':
    main()
