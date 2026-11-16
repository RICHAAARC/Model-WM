"""Use existing pinned software/assets when verifiably present; all payloads stay /content."""
import json
import os
import subprocess
import sys
from pathlib import Path
from wam_qualification.setup import RUNTIME_ROOT,BASE_FILES,METADATA_FILES,fetch,component_preflight
from wam_qualification.prepare import environment
from .io import save

def mapping(root,scratch,cache):
    runtime=RUNTIME_ROOT
    save(root/'local_storage.json',dict(mappings=[
        dict(new=str(cache/'base'),paid=str(scratch/'paid/base'),
            drive='/content/drive/MyDrive/Model-WM/cache/aqualora-qualification-20261108-attempt01/assets/base',
            local=[str(runtime/p) for p in [
                'wam-qualification-20261112-attempt02/assets/paid/base',
                'wam-qualification-20261112-attempt02/assets/new/base',
                'wam-qualification-20261109-attempt01/assets/new/base',
                'aqualora-qualification-20261108-attempt01/persistent-runtime/assets/base']]),
        dict(new=str(cache),paid=str(scratch/'paid/environment'),
            drive='/content/drive/MyDrive/Model-WM/cache/wam-qualification-20261109-attempt01',
            local=[str(runtime/p) for p in [
                'wam-writer-abc-20261115-attempt01/cache','wam-writer-abc-20261115-attempt01/paid',
                'wam-qualification-20261112-attempt02/assets/new/wam',
                'wam-qualification-20261112-attempt02/assets/paid/wam']])]))

def get_assets(root,scratch,cache,metadata_only=False):
    m=json.loads((root/'manifest.json').read_text());base=scratch/'base';base.mkdir(exist_ok=True)
    for name in (METADATA_FILES if metadata_only else BASE_FILES):
        url=f"https://huggingface.co/{m['base_repo']}/resolve/{m['base_revision']}/{name}"
        src=fetch(url,cache/'base'/name,root);dest=base/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.symlink_to(src.resolve())
        elif dest.resolve()!=src.resolve():raise RuntimeError('Conflicting local component source')
    result=component_preflight(base,metadata_only=metadata_only,parse_components=not metadata_only)
    save(root/('metadata_structure.json' if metadata_only else 'component_preflight.json'),result)
    if not metadata_only:save(root/'model_paths.json',dict(base=str(base),pretrained_weights_destination='/content only'))

def main(root,scratch,cache):
    root,scratch,cache=map(Path,(root,scratch,cache));mapping(root,scratch,cache)
    # Source/config failures precede expensive installation and checkpoint bodies.
    print('asset_preflight ...',flush=True);get_assets(root,scratch,cache,True)
    from wam_qualification.transport import head,cached
    from wam_qualification.storage import locate
    from .io import append
    m=json.loads((root/'manifest.json').read_text())
    for name in BASE_FILES:
        if name in METADATA_FILES:continue
        url=f"https://huggingface.co/{m['base_repo']}/resolve/{m['base_revision']}/{name}"
        row=dict(url=url,status='PAID_CACHE') if cached(url,locate(root,cache/'base'/name)) else head(url)
        append(root/'asset_preflight.jsonl',row)
    print('environment ...',flush=True);environment(root,scratch,cache,cache)
    py=json.loads((root/'python.json').read_text())['python']
    print('official_assets ...',flush=True)
    subprocess.run([py,'-m','latent_control.prepare','assets',str(root),str(scratch),str(cache)],check=True)

if __name__=='__main__':
    os.environ['MPLBACKEND']='Agg'
    if sys.argv[1]=='assets':get_assets(*map(Path,sys.argv[2:5]))
    else:main(*sys.argv[2:5])
