"""Reuse only verified official cache entries on this Colab VM."""
from pathlib import Path
import sys
from lawa_ab import assets
from latent_control.io import save
from wam_qualification.transport import cached
from .worker_paths import PRIOR

def reuse_assets(local,scratch):
    old=scratch.parent/PRIOR/'cache'
    targets={'last.ckpt':assets.AUTHOR,'source.tar.gz':f'https://codeload.github.com/vbdi/LaWa/tar.gz/{assets.SOURCE}'}
    targets.update({'sd14/'+n:f'https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/{assets.BASE}/{n}' for n in assets.FILES})
    reused=[]
    for name,url in targets.items():
        src=old/name
        if src.is_file() and src.resolve().is_relative_to(old.resolve()) and cached(url,src):
            dst=scratch/'cache'/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            for a in (src,src.with_name(src.name+'.complete.json')):
                (dst.parent/a.name).symlink_to(a)
            reused.append(name)
    save(local/'cache_reuse.json',dict(reused=reused,weights='/content'))

def prepare(local,scratch):
    if not scratch.resolve().is_relative_to(Path('/content')) or scratch.resolve().is_relative_to(Path('/content/drive')):
        raise ValueError('Weights require local /content')
    reuse_assets(local,scratch)
    assets.prepare(local,scratch)

if __name__=='__main__':
    prepare(*map(Path,sys.argv[1:3]))
