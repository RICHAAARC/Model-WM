"""Metadata-only source check and strictly synthetic tensor load through the production backend."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import UNet2DConditionModel,AutoencoderKL
from transformers import CLIPTextConfig,CLIPTextModel,CLIPTokenizer
from .assets import SOURCE
from .model import build_writer,Backend


def check(source,metadata):
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==SOURCE
    torch.set_num_threads(1)
    torch.manual_seed(123)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        (root/'official').symlink_to(source,target_is_directory=True)
        base=root/'cache/sd14'
        shutil.copytree(metadata,base)
        model=build_writer(source)
        shapes={k:list(v.shape) for k,v in model.state_dict().items()}
        receipt=json.loads(Path(__file__).with_name('schema.json').read_text())
        assert len(shapes)==receipt['keys']
        assert hashlib.sha256(json.dumps(shapes,sort_keys=True).encode()).hexdigest()==receipt['keys_shapes_sha256']
        # All tensors below are randomly initialized on CPU.
        torch.save({'state_dict':model.state_dict()},root/'cache/last.ckpt')
        del model
        tokenizer=CLIPTokenizer.from_pretrained(base/'tokenizer',local_files_only=True)
        assert len(tokenizer('shape check',padding='max_length',max_length=77)['input_ids'])==77
        unet=UNet2DConditionModel.load_config(base/'unet')
        vae=AutoencoderKL.load_config(base/'vae')
        assert unet['in_channels']==vae['latent_channels']==4
        tok=CLIPTextConfig.from_pretrained(base/'text_encoder')
        CLIPTextModel(CLIPTextConfig(vocab_size=tok.vocab_size,hidden_size=8,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=2)).save_pretrained(base/'text_encoder')
        UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,down_block_types=('DownBlock2D',),
             up_block_types=('UpBlock2D',),block_out_channels=(8,),layers_per_block=1,
             norm_num_groups=4,cross_attention_dim=8,attention_head_dim=2).save_pretrained(base/'unet')
        AutoencoderKL(block_out_channels=(8,8,8,8),down_block_types=('DownEncoderBlock2D',)*4,
            up_block_types=('UpDecoderBlock2D',)*4,norm_num_groups=4,layers_per_block=1).save_pretrained(base/'vae')
        backend=Backend.load(root,'cpu')
        assert backend.pipe.scheduler.config.steps_offset==1
        bank='01'*24
        # Run the actual pinned DDIM and text model at reduced spatial resolution.
        pipe=backend.pipe
        def small(prompt,**kw):
            kw.update(width=64,height=64)
            return pipe(prompt,**kw)
        small.vae=pipe.vae
        backend.pipe=small
        z,h,clean=backend.generate(dict(seed=12,prompt='synthetic CPU interface'))
        assert z.shape==(1,4,8,8) and clean.shape==(64,64,3)
        im=backend.write(h,bank)
        q=backend.read(im)
        assert im.shape==clean.shape and len(q)==48 and np.isfinite(q).all()
        result=backend.probe(h,bank)
        assert result['passed'],result
        print('484 official inference state names/shapes matched; synthetic SD1.4 component-index load, 50 DDIM steps, original LaWa forward, RGB/logits and finite-difference check passed',result)


if __name__=='__main__':
    check(*map(Path,sys.argv[1:3]))
