"""Pinned source preparation; weights stay in /content and paid caches are reused."""
import sys,json,shutil
from pathlib import Path
from lawa_ab import assets as old
from wam_qualification.setup import fetch,unpack
from wam_qualification.prepare import environment
from wam_qualification.transport import cached
from latent_control.io import save

REV='4e63672c03103b6c636b8fb4119ba982469b2955'
REPO='sd2-community/stable-diffusion-2-1-base'
def targets():
    result={'last.ckpt':old.AUTHOR,'source.tar.gz':f'https://codeload.github.com/vbdi/LaWa/tar.gz/{old.SOURCE}'}
    for n in old.FILES:
        if not n.startswith('vae/'):
            result['sd21/'+n]=f'https://huggingface.co/{REPO}/resolve/{REV}/{n}'
    for n in ('vae/config.json','vae/diffusion_pytorch_model.safetensors'):
        result['sd14/'+n]=f'https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/{old.BASE}/{n}'
    return result

def reuse(scratch,local):
    used=[];skipped=[]
    for name,url in targets().items():
        dest=scratch/'cache'/name
        if dest.exists():continue
        for run in sorted(scratch.parent.iterdir()):
            src=run/'cache'/name
            if run==scratch or not src.is_file() or not src.resolve().is_relative_to(scratch.parent.resolve()):continue
            receipt=src.with_name(src.name+'.complete.json')
            try:source=json.loads(receipt.read_text()).get('url')
            except (OSError,ValueError):
                skipped.append(dict(source=str(src),reason='No readable source receipt; not reused'));continue
            if source!=url:
                skipped.append(dict(source=str(src),reason='Different asset source; not reused'));continue
            if cached(url,src):
                dest.parent.mkdir(parents=True,exist_ok=True)
                for p in (src,src.with_name(src.name+'.complete.json')):
                    (dest.parent/p.name).symlink_to(p.resolve())
                used.append(dict(source=str(src),target=name));break
    save(local/'cache_reuse.json',{'reused':used,'skipped':skipped,'weights_on_drive':False})

def prepare(local,scratch):
    base=scratch.resolve()
    if not base.is_relative_to(Path('/content')) or base.is_relative_to(Path('/content/drive')):
        raise ValueError('Weights require local /content')
    cache=scratch/'cache';cache.mkdir(parents=True,exist_ok=True)
    reuse(scratch,local)
    environment(local,scratch,cache,cache)
    for name,url in targets().items():fetch(url,cache/name,local)
    unpack(cache/'source.tar.gz',scratch/'official',strip=True,source_only=True)
    # Run actual parsers in the pinned environment before model construction.
    import subprocess
    py=json.loads((local/'python.json').read_text())['python']
    subprocess.run([py,'-m','sd2_matched.preflight',str(scratch),str(local)],check=True)

if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
