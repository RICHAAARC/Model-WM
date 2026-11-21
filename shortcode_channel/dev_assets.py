"""Reuse verified LaWa files from either previous local run; new assets stay local."""
from pathlib import Path
import sys
from lawa_ab import assets
from probe_selection.worker_paths import PRIOR, RUN_ID
from latent_control.io import save
from wam_qualification.transport import cached


def prepare(local,scratch):
    base=scratch.resolve()
    if not base.is_relative_to(Path('/content')) or base.is_relative_to(Path('/content/drive')):
        raise ValueError('Weights require local /content')
    targets={'last.ckpt':assets.AUTHOR,'source.tar.gz':f'https://codeload.github.com/vbdi/LaWa/tar.gz/{assets.SOURCE}'}
    targets.update({'sd14/'+n:f'https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/{assets.BASE}/{n}' for n in assets.FILES})
    used=[]
    for name,url in targets.items():
        for prior in (RUN_ID,PRIOR):
            src=scratch.parent/prior/'cache'/name
            if src.is_file() and src.resolve().is_relative_to(base.parent) and cached(url,src):
                dest=scratch/'cache'/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                for f in (src,src.with_name(src.name+'.complete.json')):
                    (dest.parent/f.name).symlink_to(f.resolve())
                used.append([prior,name])
                break
    save(local/'cache_reuse.json',dict(reused=used))
    assets.prepare(local,scratch)


if __name__=='__main__':
    prepare(*map(Path,sys.argv[1:]))
