"""Reuse paid local SD1.4 assets and the validated environment; never store weights on Drive."""
import json,sys,subprocess
from pathlib import Path
from lawa_ab import assets as base
from wam_qualification.transport import cached
from latent_control.io import save

def targets():
    urls={'last.ckpt':base.AUTHOR,'source.tar.gz':f'https://codeload.github.com/vbdi/LaWa/tar.gz/{base.SOURCE}'}
    urls.update({'sd14/'+n:f'https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/{base.BASE}/{n}' for n in base.FILES})
    return urls

def reuse(scratch,local):
    used=[];skipped=[]
    for name,url in targets().items():
        dest=scratch/'cache'/name
        # Current-attempt artifacts are authoritative: unresolved/partial payloads
        # must stop rather than be bypassed by a paid cache from another run.
        if cached(url,dest):continue
        for run in sorted(scratch.parent.iterdir()):
            src=run/'cache'/name
            if run==scratch or not src.is_file() or not src.resolve().is_relative_to(scratch.parent.resolve()):continue
            try:
                receipt=json.loads(src.with_name(src.name+'.complete.json').read_text())
            except (OSError,ValueError):
                skipped.append(dict(source=str(src),reason='No readable source receipt'));continue
            if receipt.get('url')!=url:
                skipped.append(dict(source=str(src),reason='Different source; not reused'));continue
            try:
                valid=cached(url,src)
            except (OSError,ValueError,RuntimeError) as exc:
                skipped.append(dict(source=str(src),reason=f'Invalid old cache retained: {type(exc).__name__}: {exc}'));continue
            if valid:
                dest.parent.mkdir(parents=True,exist_ok=True)
                for f in (src,src.with_name(src.name+'.complete.json')):(dest.parent/f.name).symlink_to(f.resolve())
                used.append(dict(source=str(src),target=name));break
    save(local/'cache_reuse.json',dict(reused=used,skipped=skipped,weights_on_drive=False))

def prepare(local,scratch):
    root=scratch.resolve()
    if not root.is_relative_to(Path('/content')) or root.is_relative_to(Path('/content/drive')):raise ValueError('Weights require local /content')
    (scratch/'cache').mkdir(parents=True,exist_ok=True)
    reuse(scratch,local);base.prepare(local,scratch)
    py=json.loads((local/'python.json').read_text())['python']
    subprocess.run([py,'-m','strength_bracket.preflight',str(scratch),str(local)],check=True)
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
