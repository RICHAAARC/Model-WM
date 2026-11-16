"""Official component parsers and random-state VAE/pipeline interfaces, CPU only."""
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from diffusers import AutoencoderKL,UNet2DConditionModel,StableDiffusionPipeline,PNDMScheduler
from transformers import CLIPTextConfig,CLIPTextModel,CLIPTokenizer,CLIPImageProcessor
from wam_qualification.environment import check as check_versions
from wam_qualification.setup import component_preflight
from .protocol import read,arrays,PACKAGE
from .model import Model,load_pipeline
from .io import Ledger,save

def check():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(1);torch.manual_seed(17) # engineering random state; not an experimental seed
    receipt=check_versions();metadata=PACKAGE/'review/official_metadata'
    receipt['metadata']=component_preflight(metadata,metadata_only=True,parse_components=True)
    config=json.loads((metadata/'vae/config.json').read_text())
    with tempfile.TemporaryDirectory() as directory:
        temp=Path(directory);m=read();f=arrays();ledger=Ledger(temp/'full_vae',m['physical_limits'])
        # Exactly the official VAE architecture, initialized locally; no checkpoint data.
        vae=AutoencoderKL.from_config(config).eval().requires_grad_(False)
        model=Model(None,m,f,ledger,'cpu',SimpleNamespace(vae=vae,unet=None,text_encoder=None))
        z=(torch.randn(1,4,64,64)*.01).requires_grad_(True)
        y=model.decode(z,'random-vae/512','forward_C');q=model.reader(y)
        grad=ledger.call('random-vae/vjp','backward_C',lambda:torch.autograd.grad(q[0,0],z)[0])
        assert y.shape==(1,3,512,512) and q.shape==(1,64)
        assert torch.isfinite(grad).all() and grad.norm()>0 and all(p.grad is None for p in vae.parameters())
        receipt['actual_vae_architecture']=dict(image_shape=list(y.shape),latent_shape=list(z.shape),
            latent_gradient_norm=float(grad.norm()),parameter_gradients=False,pretrained_weights=False)
        del model,vae,y,q,grad,z
        # Complete official pipeline serialization/loading uses synthetic small tensors,
        # real tokenizer/processor and the actual component-index schema.
        tok=CLIPTokenizer.from_pretrained(str(metadata/'tokenizer'),local_files_only=True)
        processor=CLIPImageProcessor.from_pretrained(str(metadata/'feature_extractor'),local_files_only=True)
        text=CLIPTextModel(CLIPTextConfig(vocab_size=len(tok),hidden_size=32,intermediate_size=64,
            num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=77))
        vae=AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,
            down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,
            block_out_channels=(16,16,16,16),layers_per_block=1,norm_num_groups=8,sample_size=64)
        unet=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,layers_per_block=1,
            block_out_channels=(16,32),down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),
            up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),cross_attention_dim=32,
            attention_head_dim=4,norm_num_groups=8)
        sched=PNDMScheduler.from_config(json.loads((metadata/'scheduler/scheduler_config.json').read_text()))
        pipe=StableDiffusionPipeline(vae=vae,unet=unet,text_encoder=text,tokenizer=tok,scheduler=sched,
            safety_checker=None,feature_extractor=processor,requires_safety_checker=False)
        fake=temp/'synthetic-base';pipe.save_pretrained(fake,safe_serialization=True)
        # Force the real published index, including explicitly disabled safety_checker.
        (fake/'model_index.json').write_bytes((metadata/'model_index.json').read_bytes())
        del pipe,unet,text,vae
        pipe=load_pipeline(fake,'cpu')
        m['generation'].update(width=64,height=64);f['permutation']=__import__('numpy').arange(512);f['signs']=__import__('numpy').ones(512)
        ledger=Ledger(temp/'pipeline',m['physical_limits']);model=Model(None,m,f,ledger,'cpu',pipe)
        z=model.generate(m['works'][0]);assert z.requires_grad and z.shape==(1,4,8,8)
        y=model.decode(z,'random-pipeline/decode','forward_C')
        grad=torch.autograd.grad(model.reader(y)[0,0],z)[0]
        assert torch.isfinite(grad).all() and grad.norm()>0 and all(p.grad is None for p in model.pipe.vae.parameters())
        receipt['pipeline']=dict(index='actual pinned SD1.5 index',scheduler=type(pipe.scheduler).__name__,
            processor=type(pipe.feature_extractor).__name__,latent_shape=list(z.shape),image_shape=list(y.shape),
            denoiser_calls=ledger.counts['denoise'],scaling_factor=pipe.vae.config.scaling_factor,
            dtype=str(y.dtype),device=str(y.device),latent_grad=True,pretrained_weights=False)
    receipt.update(status='OFFICIAL_RANDOM_STATE_CPU_INTERFACES_PASSED',real_weights_loaded=False,gpu_used=False,scientific_pass=None)
    save(PACKAGE/'review/official_cpu.json',receipt);print(receipt['status'])

if __name__=='__main__':check()
