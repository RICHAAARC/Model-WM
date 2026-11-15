"""Official static images; at most two logged transport attempts, no redraw."""
import hashlib
import json
import traceback
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from .io import save

def fetch_one(row,cache,receipts=None):
    target=Path(cache)/row['file_name']; receipt=target.with_suffix('.source.json')
    if target.exists():
        if not receipt.exists() or json.loads(receipt.read_text())['url']!=row['url']:
            raise RuntimeError('Unverified source cache: '+str(target))
        return target,0
    received=0
    transport=Path(cache)/'transport';transport.mkdir(exist_ok=True)
    receipts=Path(receipts) if receipts else transport;receipts.mkdir(parents=True,exist_ok=True)
    for attempt in [1,2]:
        part=transport/f'{row["id"]:012d}.{attempt}.part'
        record=receipts/f'{row["id"]:012d}.{attempt}.json'
        if part.exists() or record.exists():raise RuntimeError('Transport attempt already consumed; no automatic resume')
        save(record,dict(id=row['id'],url=row['url'],attempt=attempt,status='STARTED'))
        try:
            with urllib.request.urlopen(row['url'],timeout=30) as response,part.open('xb') as f:
                while block:=response.read(1024**2):f.write(block);received+=len(block)
            save(record,dict(id=row['id'],url=row['url'],attempt=attempt,status='DOWNLOADED',bytes=part.stat().st_size))
            break
        except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
            save(record,dict(id=row['id'],url=row['url'],attempt=attempt,status='FAILED',
                error=repr(exc),traceback=traceback.format_exc(),bytes=part.stat().st_size if part.exists() else 0))
            retryable=not isinstance(exc,urllib.error.HTTPError) or exc.code in [408,429,500,502,503,504]
            if attempt==2 or not retryable:raise
    with Image.open(part) as im:
        im.load()
        if im.size!=(row['width'],row['height']):raise RuntimeError('Metadata/image dimensions disagree')
    part.replace(target)
    save(receipt,dict(id=row['id'],url=row['url'],source_url=row['source_url'],license=row['license'],
        bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return target,received

def prepare_images(manifest,cache,output):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    rows=[];errors=[]
    # This only validates/decompresses sources, never runs a detector on TEST.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(fetch_one,row,cache,Path(output).parent/'data_transport'):row for row in manifest['images']}
        for f in as_completed(futures):
            row=futures[f]
            try:
                p,received=f.result()
                with Image.open(p) as im:
                    rgb=np.asarray(im.convert('RGB'))
                rows.append(dict(id=row['id'],path=str(p),downloaded=received,
                    decoded_sha256=hashlib.sha256(rgb.tobytes()+str(rgb.shape).encode()).hexdigest()))
            except Exception as exc:
                errors.append(dict(id=row['id'],error=repr(exc)))
    duplicates={}
    for row in rows:duplicates.setdefault(row['decoded_sha256'],[]).append(row['id'])
    duplicates=[v for v in duplicates.values() if len(v)>1]
    result=dict(rows=sorted(rows,key=lambda x:x['id']),errors=errors,duplicates=duplicates,
        status='READY' if len(rows)==1024 and not errors and not duplicates else 'STOPPED_DATA',
        downloaded_bytes=sum(r['downloaded'] for r in rows),checkpoint_overlap='UNVERIFIED',
        near_duplicate_independence='UNVERIFIED; exact decoded duplicate check only')
    save(output,result)
    if result['status']!='READY':raise RuntimeError('Sources incomplete/duplicate; no replacement or redraw')
    return result

def image_for(ident,cache):
    with Image.open(Path(cache)/f'{ident:012d}.jpg') as im:
        im=im.convert('RGB');w,h=im.size
        nw,nh=(512,round(h*512/w)) if w<=h else (round(w*512/h),512)
        im=im.resize((nw,nh),Image.Resampling.BICUBIC)
        left,top=(nw-512)//2,(nh-512)//2
        return np.asarray(im.crop((left,top,left+512,top+512))).copy()

if __name__=='__main__':
    import sys
    from .protocol import read
    print(prepare_images(read(),sys.argv[1],sys.argv[2])['status'])
