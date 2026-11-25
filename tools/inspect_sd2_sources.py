"""Small public configuration/source downloads only; no checkpoint payloads."""
from pathlib import Path
import urllib.request,json,hashlib
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'sd2_matched/source_evidence'
REV='4e63672c03103b6c636b8fb4119ba982469b2955'
BASE='133a221b8aa7292a167afc5127cb63fb5005638b'
def get(url,name):
 p=P/name;p.parent.mkdir(parents=True,exist_ok=True)
 with urllib.request.urlopen(url,timeout=40) as r:data=r.read()
 p.write_bytes(data)
 return {'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def main():
 sources={}
 for repo,rev,target,names in [
  ('sd2-community/stable-diffusion-2-1-base',REV,'sd21',['model_index.json','unet/config.json','vae/config.json','scheduler/scheduler_config.json','text_encoder/config.json','tokenizer/tokenizer_config.json','tokenizer/vocab.json','tokenizer/merges.txt','tokenizer/special_tokens_map.json','README.md']),
  ('CompVis/stable-diffusion-v1-4',BASE,'sd14',['vae/config.json'])]:
  for name in names:
   sources[target+'/'+name]=get(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',target+'/'+name)
 sources['hf_mirror_guidance.html']=get('https://huggingface.co/docs/optimum-habana/tutorials/stable_diffusion','hf_mirror_guidance.html')
 sources['convert_from_ckpt.py']=get('https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/src/diffusers/pipelines/stable_diffusion/convert_from_ckpt.py','convert_from_ckpt.py')
 sources['v2-inference.yaml']=get('https://huggingface.co/ckpt/stable-diffusion-2-1-base/resolve/80b44465963d65d12b35d07a2ea789545df260b9/v2-inference.yaml','v2-inference.yaml')
 sources['lawa_tree.json']=get('https://api.github.com/repos/vbdi/LaWa/git/trees/251ac11aa223e6ff12553364a65a75add5b1aecf?recursive=1','lawa_tree.json')
 (P/'sources.json').write_text(json.dumps(sources,indent=2))
 print('Saved source/config files:',len(sources))
 print([x['path'] for x in json.loads((P/'lawa_tree.json').read_text())['tree'] if 'config' in x['path']])
if __name__=='__main__':main()
