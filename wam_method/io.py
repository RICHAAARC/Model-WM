"""Local atomic IPC, exclusive reservations and sealed immutable snapshots."""
import json
import math
import os
import time
from pathlib import Path

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf8') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()

class Ledger:
    def __init__(self,root):
        self.root=Path(root);self.ids=set();self.counts={}
    def call(self,ident,kind,fn):
        if ident in self.ids:raise RuntimeError('No retry: '+ident)
        self.ids.add(ident);self.counts[kind]=self.counts.get(kind,0)+1
        append(self.root/'operations.jsonl',dict(id=ident,kind=kind,status='STARTED',time=time.time()))
        try:
            out=fn()
        except BaseException as exc:
            append(self.root/'operations.jsonl',dict(id=ident,kind=kind,status='FAILED',error=repr(exc)))
            raise
        append(self.root/'operations.jsonl',dict(id=ident,kind=kind,status='COMPLETED'))
        return out

def reconcile(history, floor):
    """Additive totals use monotone max; unknown historical CPU is not refunded."""
    out={}
    for key in ['cumulative_resources','cumulative_budget_debits']:
        if key not in history:raise RuntimeError('Historical receipt missing '+key)
        out[key]={}
        for k,v in floor[key].items():
            new=history[key].get(k)
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in [v,new]):
                raise RuntimeError('Invalid historical resource: '+key+'/'+k)
            out[key][k]=max(v,new)
    out['historical_cpu_exact']=False
    return out

def operation_summary(path,limits):
    counts={k:dict(denominator=v,started=0,completed=0,failed=0) for k,v in limits.items()}
    path=Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue # interrupted final line is retained verbatim
            key=r['kind'].replace('train_update_','train_updates_')
            if key in counts:counts[key][r['status'].lower()]+=1
    return counts
