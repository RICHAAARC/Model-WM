"""Prespecified observable costs. Neither proxy measures human-perceived equivalence."""
from itertools import combinations
from pathlib import Path
import html
import numpy as np
from PIL import Image
from latent_control.carrier import quality

NATIVE=[f'{group}_{i}_png.png' for group in ('registered','foreign') for i in range(2)]
MIX=[f'registered_mix_{i}_{codec}.{ext}' for i in range(3) for codec,ext in [('png','png'),('jpeg85','jpg')]]
VIEWS=['null_png.png',*NATIVE,*MIX]

def luminance(rgb):
    return (.299*rgb[:,:,0]+.587*rgb[:,:,1]+.114*rgb[:,:,2])/255

def coarse(y):
    h,w=y.shape
    return y.reshape(32,h//32,32,w//32).mean(axis=(1,3))

def roughness(y,mask):
    lap=y[:-2,1:-1]+y[2:,1:-1]+y[1:-1,:-2]+y[1:-1,2:]-4*y[1:-1,1:-1]
    return float(np.abs(lap[mask]).mean())

def measure(folder):
    def image(name):
        with Image.open(folder/name) as im:
            return np.asarray(im.convert('RGB'))
    clean=luminance(image(VIEWS[0]))
    grad=np.abs(clean[2:,1:-1]-clean[:-2,1:-1])+np.abs(clean[1:-1,2:]-clean[1:-1,:-2])
    mask=grad<=np.quantile(grad,.25)
    base=roughness(clean,mask)
    native=[image(name) for name in NATIVE]
    grids=[coarse(luminance(rgb)) for rgb in native]
    pairs=[]
    for i,j in combinations(range(4),2):
        pairs.append(dict(pair=[NATIVE[i],NATIVE[j]],layout=float(np.abs(grids[i]-grids[j]).mean()),
                          **quality(native[i],native[j])))
    changes={name:abs(roughness(luminance(image(name)),mask)-base) for name in NATIVE+MIX}
    result=dict(layout=max(p['layout'] for p in pairs),artifact=max(changes.values()),pairs=pairs,
                roughness_change=changes,flat_pixel_count=int(mask.sum()))
    if not np.isfinite([result['layout'],result['artifact']]).all():
        raise ValueError('Unmeasurable image proxy')
    return result

def gallery(root,works,gains,result):
    ids=[w['id'] for w in works]
    summary=result['paired']
    worst={a:{k:max(ids,key=lambda wid: next(r for r in summary if r['id']==wid)[a]['proxies'][k])
              for k in ('layout','artifact')} for a in 'CO'}
    failures=[r['id'] for r in summary if any(not r[a]['catch'] or r[a]['errors'] for a in 'CO')]
    index=dict(representative=ids[:4],worst=worst,failures=failures,all=ids)
    from latent_control.io import save
    save(root/'figure_selection.json',index)
    def links(names):
        return ' '.join(f'<a href="#{n}">{n}</a>' for n in names) or 'none'
    parts=['<!doctype html><meta charset="utf-8"><title>Paired TEST review</title>',
           '<style>img{width:150px;height:auto} figure{display:inline-block;margin:3px} figcaption{font:11px sans-serif}</style>',
           '<h1>All 64 paired TEST works</h1>Representative: '+links(ids[:4]),
           '<p>Worst proxy cases: '+links([v for a in worst.values() for v in a.values()]),
           '<p>All tracking failures and false attributions: '+links(failures)]
    for w in works:
        parts.append(f'<h2 id="{w["id"]}">{w["id"]}</h2><p>'+html.escape(w['prompt']))
        for a in 'CO':
            parts.append('<h3>'+a+'</h3>')
            for name in VIEWS:
                url=f'works/{w["id"]}/{a}/{gains[a]}/{name}'
                parts.append(f'<figure><a href="{url}"><img loading="lazy" src="{url}"></a><figcaption>{name}</figcaption></figure>')
    (root/'gallery.html').write_text('\n'.join(parts),encoding='utf8')
