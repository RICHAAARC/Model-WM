"""Fetch only small official component metadata; never model tensors."""
import json,urllib.request
from pathlib import Path
from lawa_ab.assets import BASE,FILES
ROOT=Path(__file__).resolve().parents[1]
dest=ROOT/'strength_bracket/source_evidence/sd14'
records=[]
for name in FILES:
    if not name.endswith(('.json','.txt')):continue
    p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/{BASE}/{name}'
    if not p.exists():
        with urllib.request.urlopen(url,timeout=45) as r: data=r.read(4*1024*1024)
        if name.endswith('.json'):json.loads(data)
        p.write_bytes(data)
    records.append(dict(path=name,url=url,bytes=p.stat().st_size,model_tensor=False))
(dest.parent/'metadata_receipt.json').write_text(json.dumps(records,indent=2))
print('metadata files',len(records),'no model tensors')
