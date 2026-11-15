"""Reuse the pinned environment and only the WAM assets; never load SD."""
import json
import os
import sys
from pathlib import Path
from wam_qualification.setup import RUNTIME_ROOT,fetch,unpack
from wam_qualification.prepare import environment
from .io import save
from .protocol import read

def main(root,scratch,cache):
    root,scratch,cache=map(Path,(root,scratch,cache));m=read()
    root.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    history=Path('/content/drive/MyDrive/Model-WM/cache')
    oldruntime=RUNTIME_ROOT/'wam-qualification-20261112-attempt02/assets'
    priorruntime=RUNTIME_ROOT/'wam-qualification-20261109-attempt01/assets'
    save(root/'local_storage.json',dict(mappings=[dict(new=str(cache),paid=str(scratch/'paid'),
        drive=str(history/'wam-qualification-20261109-attempt01'),
        local=[str(oldruntime/'paid/wam'),str(oldruntime/'new/wam'),
               str(priorruntime/'paid/wam'),str(priorruntime/'new/wam')])]))
    environment(root,scratch,cache,cache)
    python=Path(json.loads((root/'python.json').read_text())['python'])
    # Follow-on source/data validation runs in the pinned environment, not Colab's host Python.
    import subprocess
    subprocess.run([str(python),'-m','wam_method.prepare','assets',str(root),str(scratch),str(cache)],check=True)

def assets(root,scratch,cache):
    root,scratch,cache=map(Path,(root,scratch,cache));m=read()
    url=f"https://codeload.github.com/facebookresearch/watermark-anything/tar.gz/{m['source_commit']}"
    archive=fetch(url,cache/'official_source.tar.gz',root)
    source=scratch/'official_source'
    if not source.exists():unpack(archive,source,strip=True,source_only=True)
    p=json.loads((source/'checkpoints/params.json').read_text())
    expected=dict(embedder_model='vae_small',extractor_model='sam_base',nbits=32,
        img_size=256,attenuation='jnd_1_3_blue',scaling_w=2.,scaling_i=1.)
    if any(p.get(k)!=v for k,v in expected.items()):raise RuntimeError('Official WAM configuration mismatch')
    from .data import prepare_images
    prepare_images(m,scratch/'images',root/'data_receipt.json')
    checkpoint=fetch(m['checkpoint_url'],cache/'wam_mit.pth',root)
    save(root/'model_paths.json',dict(source=str(source),checkpoint=str(checkpoint),device='cuda'))

if __name__=='__main__':
    os.environ['MPLBACKEND']='Agg'
    (assets if sys.argv[1]=='assets' else main)(*sys.argv[2:5])
