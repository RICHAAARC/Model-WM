"""Only the preparation subprocess writes model and dependency assets, in /content."""
import sys
from pathlib import Path
from wam_qualification.setup import fetch, unpack
from wam_qualification.prepare import environment

SOURCE = '251ac11aa223e6ff12553364a65a75add5b1aecf'
BASE = '133a221b8aa7292a167afc5127cb63fb5005638b'
AUTHOR = 'https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/lawa/pretrained-checkpoints/last.ckpt'
FILES = ('model_index.json','scheduler/scheduler_config.json','tokenizer/vocab.json',
         'tokenizer/merges.txt','tokenizer/special_tokens_map.json','tokenizer/tokenizer_config.json',
         'unet/config.json','unet/diffusion_pytorch_model.safetensors','vae/config.json',
         'vae/diffusion_pytorch_model.safetensors','text_encoder/config.json','text_encoder/model.safetensors')


def prepare(records, scratch):
    cache = scratch/'cache'
    if not scratch.resolve().is_relative_to(Path('/content')) or scratch.resolve().is_relative_to(Path('/content/drive')):
        raise ValueError('Model assets must stay in the local Colab runtime')
    cache.mkdir(parents=True,exist_ok=True)
    environment(records,scratch,cache,cache)
    archive = fetch(f'https://codeload.github.com/vbdi/LaWa/tar.gz/{SOURCE}',cache/'source.tar.gz',records)
    unpack(archive,scratch/'official',strip=True,source_only=True)
    fetch(AUTHOR,cache/'last.ckpt',records)
    for name in FILES:
        fetch(f'https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/{BASE}/{name}',cache/'sd14'/name,records)


if __name__=='__main__':
    prepare(*map(Path,sys.argv[1:]))
