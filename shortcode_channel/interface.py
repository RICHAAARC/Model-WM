"""DEV data acquisition contract. Importing this module cannot load a backend."""
from collections import Counter
from io import BytesIO

import numpy as np
from PIL import Image
from probe_selection.rules import distance, mix

BOOKS = ('H0','H1','H2','H3','random','tardos')
LIMITS = {'generate':8,'write':432,'read':1248}


def image(x):
    x = np.asarray(x)
    if x.ndim!=3 or x.shape[-1]!=3 or x.dtype!=np.uint8:
        raise ValueError('Expected RGB uint8 image')
    return x


def query(reader, rgb):
    rows={}
    for name,fmt,opts in [('png','PNG',{}),('jpeg85','JPEG',dict(quality=85,subsampling=2))]:
        buf=BytesIO()
        Image.fromarray(image(rgb)).save(buf,format=fmt,**opts)
        buf.seek(0)
        with Image.open(buf) as im:
            q=np.asarray(reader(np.array(im.convert('RGB'))),dtype=float)
        if q.shape!=(48,) or not np.isfinite(q).all():
            raise ValueError('Expected 48 finite real-valued logits')
        rows[name]=q
    return rows


def acquire_dev(allocation, backend, call):
    """call(kind, tag, fn, *args) must reserve before work. No CAL/TEST entry."""
    works=[w for w in allocation['works'] if w['split']=='DEV']
    if len(works)!=8:
        raise ValueError('Expected eight fixed DEV works')
    result=[]
    for w in works:
        _,h,clean=call('generate',w['id'],backend.generate,w)
        clean=image(clean)
        for book in w['book_order']:
            tag=blind_ticket(w,book)
            bank=allocation['books'][book]
            ids=sum(w['groups'],[])
            pictures={i:image(call('write',f'{tag}/{i}',backend.write,h,bank[i])) for i in ids}
            dist=[distance(clean,pictures[i]) for i in ids]
            queries={'clean':clean,**{f'u{i}':pic for i,pic in pictures.items()}}
            for j,group in enumerate(w['groups']):
                queries[f'm{j}']=mix([pictures[i] for i in group],[1/3]*3)
            reads={}
            for key,pic in queries.items():
                def reader(x):
                    return call('read',tag+'/'+key,backend.read,x)
                reads[key]=query(reader,pic)
            result.append(dict(work=w['id'],book=book,groups=w['groups'],reads=reads,dist=dist))
    return result


class CPUCounter:
    """In-memory reservation for synthetic checks only."""
    def __init__(self):
        self.started=Counter()
        self.done=Counter()

    def __call__(self,kind,tag,fn,*args):
        if self.started[kind]>=LIMITS[kind]:
            raise RuntimeError('Operation limit: '+kind)
        self.started[kind]+=1
        data=fn(*args)
        self.done[kind]+=1
        return data


def blind_ticket(work,book):
    """Anonymous within-work position for paired visual/quality exports."""
    index=work['book_order'].index(book)
    return f"{work['id']}/slot-{index+1:02}"
