"""One physical pass. Decision precedes collection of opaque holdout archives."""
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from latent_control.io import append, save
from .science import CODECS, WEIGHTS, diagnose, decision, signs

PACKAGE=Path(__file__).resolve().parent


@contextlib.contextmanager
def phase(folder,name):
    start=time.monotonic()
    cpu=time.process_time()
    completed=False
    try:
        yield
        completed=True
    finally:
        save(folder/'phases'/(name+'.json'),dict(phase=name,wall_seconds=time.monotonic()-start,
             process_cpu_seconds=time.process_time()-cpu,complete=completed))


def collect(backend,work,codes,path,probe=False):
    z,h,clean=backend.generate(work)
    np.save(path/'latent.npy',z,allow_pickle=False)
    media={'null':clean}
    Image.fromarray(clean).save(path/'null_png.png')
    for group,users in [('reg',work['registered']),('foreign',work['foreign'])]:
        for i,uid in enumerate(users):
            name=f'{group}_{i}'
            media[name]=backend.write(h,codes[uid])
            Image.fromarray(media[name]).save(path/(name+'_png.png'))
        x,y=[media[f'{group}_{i}'].astype(np.float64) for i in range(2)]
        for i,b in enumerate(WEIGHTS):
            media[f'{group}_mix_{i}']=np.rint(b*x+(1-b)*y).clip(0,255).astype('uint8')
    raw={}
    for name,rgb in media.items():
        for c in CODECS:
            file=path/(name+'_'+c+('.png' if c=='png' else '.jpg'))
            im=Image.fromarray(rgb)
            if c=='png' and not file.exists():
                im.save(file)
            elif c!='png':
                im.save(file,quality=85,subsampling=2)
            with Image.open(file) as observed:
                row=backend.read(np.asarray(observed.convert('RGB')))
            if len(row)!=48 or not np.isfinite(row).all():
                raise ValueError('Nonfinite or incomplete readout')
            raw[name+'_'+c]=row
            append(path/'readouts.jsonl',dict(view=name+'_'+c,logits=row))
    save(path/'continuous.json',raw)
    if probe:
        value=backend.probe(h,codes[work['registered'][0]])
        save(path/'gradient.json',value)
    return raw


def acceptance(output,works,codes):
    names=['null']+[f'{g}_{i}' for g in ('reg','foreign') for i in range(2)]
    grids=[]
    panel=Image.new('RGB',(5*192,2*218),'white')
    draw=ImageDraw.Draw(panel)
    for j,w in enumerate(works):
        path=output/'fit'/w['id']
        row=[]
        for col,name in enumerate(names):
            with Image.open(path/(name+'_png.png')) as im:
                draw.text((col*192+4,j*218+2),w['id']+' '+name,fill='black')
                panel.paste(im.resize((192,192)),(col*192,j*218+23))
                row.append(np.asarray(im.convert('L').resize((32,32),Image.Resampling.BOX),dtype=float))
        grids.append(row)
    panel.save(output/'A_panel.png')
    bank=np.stack([signs(c) for c in codes])
    checks=[]
    for j,w in enumerate(works):
        path=output/'fit'/w['id']
        raw=json.loads((path/'continuous.json').read_text())
        for k,uid in enumerate(w['registered']+w['foreign'],1):
            q=np.asarray(raw[names[k]+'_png'])
            hard=(bank==np.where(q>0,1,-1)).mean(1)
            soft=bank@q
            own=np.abs(grids[j][k]-grids[j][0]).mean()
            other=np.abs(grids[j][k]-grids[1-j][0]).mean()
            checks.append(dict(id=w['id'],uid=uid,bit_accuracy=float(hard[uid]),
                          hard_unique=bool(hard[uid]>np.delete(hard,uid).max()),
                          soft_unique=bool(soft[uid]>np.delete(soft,uid).max()),
                          within_work=float(own),across_work=float(other),layout=bool(own<other)))
    grad=[json.loads((output/'fit'/w['id']/'gradient.json').read_text()) for w in works]
    return dict(passed=all(r['layout'] and r['hard_unique'] and r['soft_unique'] for r in checks)
                and all(g['passed'] for g in grad),native=checks,gradient=grad,
                note='Low-resolution layout is a proxy. See the full-size native PNGs for subject/naturalness.')


def run(backend,local,output,manifest):
    works=manifest['works']
    if [w['id'] for w in works]!=[f'{s}-{i:02}' for s in ('fit','holdout') for i in range(8)]:
        raise ValueError('Invalid allocation order')
    fit=[]
    completed=0
    for name,batch in [('A',works[:2]),('B_fit',works[2:8]),('B_sealed',works[8:])]:
        with phase(output,name):
            for work in batch:
                wid=work['id']
                save(local/'progress.json',dict(phase=name,active_work=wid,evaluation_bundles=completed))
                print(name,wid,flush=True)
                claim=output/'work_receipts'/wid
                claim.mkdir(parents=True,exist_ok=False)
                save(claim/'started.json',dict(id=wid,stage=name))
                held=work['split']=='holdout'
                path=(local/'holdout' if held else output/'fit')/wid
                path.mkdir(parents=True,exist_ok=False)
                try:
                    raw=collect(backend,work,manifest['codes'],path,probe=name=='A')
                    if not held:
                        fit.append(diagnose(work,raw,manifest['codes']))
                        save(path/'diagnostic.json',fit[-1])
                except BaseException:
                    (claim/'exception.txt').write_text(traceback.format_exc())
                    raise
                finally:
                    if held:
                        # Archive even a partially produced work, without inspecting readouts.
                        target=output/'sealed_holdout'/(wid+'.zip')
                        target.parent.mkdir(exist_ok=True)
                        scratchfile=local/(wid+'.zip')
                        with zipfile.ZipFile(scratchfile,'x',zipfile.ZIP_STORED) as archive:
                            for p in sorted(path.iterdir()):
                                archive.write(p,p.name)
                        temp=target.with_suffix('.part')
                        shutil.copyfile(scratchfile,temp)
                        os.replace(temp,target)
                save(claim/'completed.json',dict(id=wid,generations=1,basic_outputs=5,views=22,sealed=held))
                completed+=1
                save(local/'progress.json',dict(phase=name,evaluation_bundles=completed))
            if name=='A':
                a=acceptance(output,batch,manifest['codes'])
                save(output/'A_acceptance.json',a)
                if not a['passed']:
                    return dict(status='A_INTERFACE_OR_CARRIER_FAILED',fit_decision=None)
            if name=='B_fit':
                selected=decision(fit)
                save(output/'fit_decision.json',selected)
                if json.loads((output/'fit_decision.json').read_text())!=selected:
                    raise RuntimeError('Fit decision seal mismatch')
    return dict(status='AB_COMPLETE',fit_decision=selected,holdout_decision_inputs=0,
                generated=16,basic_outputs=80,extractions=352,gradient_probes=2)


def main():
    from .model import Backend
    local,output,scratch=map(Path,sys.argv[1:4])
    with phase(output,'load'):
        backend=Backend.load(scratch)
    result=run(backend,local,output,json.loads((PACKAGE/'freeze.json').read_text()))
    save(local/'outcome.json',result)


if __name__=='__main__':
    main()
