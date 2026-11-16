"""Atomic local IPC, append-only operation reservations, immutable Drive snapshots."""
import json
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path

class BudgetStop(RuntimeError):pass
class MechanismFailure(RuntimeError):pass

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('x',encoding='utf8') as f:
            json.dump(value,f,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def seal(source,dest):
    source,dest=Path(source),Path(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise RuntimeError('Sealed file already exists: '+str(dest))
    if source.is_dir():shutil.copytree(source,dest)
    else:shutil.copyfile(source,dest)

class Ledger:
    def __init__(self, root, limits):
        self.path=Path(root)/'operations.jsonl';self.limits=limits;self.counts={};self.ids=set()
        if self.path.exists():raise RuntimeError('Consumed operation ledger: no resume or retry')
    def call(self,ident,kind,fn):
        if ident in self.ids:raise RuntimeError('Repeated physical operation: '+ident)
        if kind not in self.limits or self.counts.get(kind,0)>=self.limits[kind]:
            raise BudgetStop('Physical cap: '+kind)
        self.ids.add(ident);self.counts[kind]=self.counts.get(kind,0)+1
        append(self.path,dict(id=ident,kind=kind,status='STARTED',time=time.time()))
        try:out=fn()
        except BaseException as e:
            append(self.path,dict(id=ident,kind=kind,status='FAILED',error=repr(e),traceback=traceback.format_exc()))
            raise
        append(self.path,dict(id=ident,kind=kind,status='COMPLETED'))
        return out

def operation_summary(path,limits):
    rows={k:dict(denominator=v,started=0,completed=0,failed=0) for k,v in limits.items()}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            rows[r['kind']][r['status'].lower()]+=1
    return rows
