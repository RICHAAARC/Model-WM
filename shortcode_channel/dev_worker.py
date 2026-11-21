"""One real DEV execution. Called by the budgeted owner after reservation."""
import gzip
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

import numpy as np
from PIL import Image
from latent_control.io import append, save
from .design import dev_decision
from .interface import LIMITS, acquire_dev

PACKAGE = Path(__file__).resolve().parent


def publish_file(src, dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_name(dst.name+'.partial')
    shutil.copyfile(src,tmp)
    os.replace(tmp,dst)


class Recorder:
    def __init__(self, local, out, allocation):
        self.local,self.out=local,out
        self.spool=local/'slot'
        self.seen=set()
        self.started=Counter()
        self.completed=Counter()
        self.slot=None
        self.work=None
        self.dev={w['id']:w for w in allocation['works'] if w['split']=='DEV'}
        self.slot_reads=0

    def reserve_slot(self,slot):
        work,position=slot.split('/')
        if work!=self.work or slot in self.seen:
            raise RuntimeError('Invalid or repeated slot')
        self.seen.add(slot)
        self.slot=slot
        self.slot_reads=0
        self.spool.mkdir()
        save(self.out/'slots'/slot/'reservation.json',dict(slot=position,write_limit=9,read_limit=26))
        self.progress()

    def progress(self):
        save(self.local/'progress.json',dict(work=self.work,slot=self.slot,started=dict(self.started),
                                             completed=dict(self.completed)))

    def __call__(self,kind,tag,fn,*args):
        if self.started[kind]>=LIMITS[kind]:
            raise RuntimeError('Physical operation limit: '+kind)
        if kind=='generate':
            if tag not in self.dev or self.slot is not None:
                raise RuntimeError('Unexpected work')
            self.work=tag
            path=self.out/'works'/tag
            path.mkdir(parents=True,exist_ok=False)
            save(path/'reserved.json',dict(work=tag))
        else:
            slot,name=tag.rsplit('/',1)
            if self.slot is None:
                self.reserve_slot(slot)
            if slot!=self.slot:
                raise RuntimeError('Slot changed before sealing')
        ident=tag if kind!='read' else tag+('/png' if self.slot_reads%2==0 else '/jpeg85')
        event=dict(kind=kind,id=ident)
        append(self.local/'operations.jsonl',dict(event,state='STARTED',at=time.time()))
        self.started[kind]+=1
        try:
            value=fn(*args)
            self.completed[kind]+=1
            append(self.local/'operations.jsonl',dict(event,state='COMPLETED'))
            if kind=='generate':
                p=self.local/'clean.png'
                Image.fromarray(value[2]).save(p)
                publish_file(p,path/'clean.png')
                self.progress()
            elif kind=='read':
                codec=ident.rsplit('/',1)[1]
                file=self.spool/(name+'.png')
                if codec=='png' and not file.exists():
                    Image.fromarray(args[0]).save(file)
                append(self.spool/'readouts.jsonl',dict(query=name,codec=codec,logits=np.asarray(value).tolist()))
                self.slot_reads+=1
                if self.slot_reads==26:
                    self.seal_slot()
            elif kind=='write':
                # Keep unqueried issued copies on a partial failure as well.
                Image.fromarray(value).save(self.spool/('u'+name+'.png'))
            return value
        except BaseException:
            append(self.local/'operations.jsonl',dict(event,state='ERROR',traceback=traceback.format_exc()))
            raise

    def seal_slot(self,partial=False):
        slot=self.slot
        if slot is None:
            return
        name='partial.zip' if partial else 'results.zip'
        bundle=self.local/name
        with ZipFile(bundle,'w',compression=ZIP_STORED) as archive:
            for f in sorted(self.spool.iterdir()):
                archive.write(f,f.name)
        publish_file(bundle,self.out/'slots'/slot/name)
        if not partial:
            save(self.out/'slots'/slot/'completed.json',dict(started=dict(self.started),completed=dict(self.completed)))
            if slot.endswith('/slot-06'):
                save(self.out/'works'/self.work/'completed.json',dict(work=self.work,slots=6))
        shutil.rmtree(self.spool)
        self.slot=None
        self.progress()


def run(local,out,allocation,backend):
    recorder=Recorder(local,out,allocation)
    try:
        rows=acquire_dev(allocation,backend,recorder)
        if recorder.started!=LIMITS or recorder.completed!=LIMITS:
            raise RuntimeError('Incomplete DEV family')
        with gzip.open(local/'dev_records.json.gz','wt',encoding='utf8') as f:
            json.dump(rows,f,default=lambda x:x.tolist(),allow_nan=False)
        publish_file(local/'dev_records.json.gz',out/'dev_records.json.gz')
        decision=dev_decision(rows,allocation)
        save(out/'dev_decision.json',decision)
        return dict(status='DEV_NUMERIC_GATE_MET' if decision['eligible_for_separate_CAL_authorization']
                    else 'DEV_STOP_RULE',visual_review='PENDING',scientific_pass=None)
    finally:
        if recorder.slot:
            recorder.seal_slot(partial=True)


if __name__=='__main__':
    from lawa_ab.model import Backend
    local,out,scratch=map(Path,sys.argv[1:])
    allocation=json.loads((PACKAGE/'allocation.json').read_text())
    save(local/'progress.json',dict(phase='load'))
    backend=Backend.load(scratch)
    save(local/'outcome.json',run(local,out,allocation,backend))
