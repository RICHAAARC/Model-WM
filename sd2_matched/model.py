"""SD2.1 denoiser with the unchanged SD1.4 clean VAE and LaWa writer.

No SD2 VAE weights are loaded. Fixed float32 and staged CPU/GPU transfers.
The cross-model latent-coordinate suitability is an explicit DEV GPU check.
"""
import json
from pathlib import Path
import numpy as np
import torch
from lawa_ab.model import Backend as OriginalBackend, build_writer, load_inference, rgb

SCALE=.18215
def validate_configs(base,clean):
    def read(p):return json.loads(p.read_text())
    u=read(base/'unet/config.json');t=read(base/'text_encoder/config.json')
    s=read(base/'scheduler/scheduler_config.json');v=read(clean/'vae/config.json')
    index=read(base/'model_index.json')
    expected={'unet':['diffusers','UNet2DConditionModel'],'text_encoder':['transformers','CLIPTextModel'],
              'tokenizer':['transformers','CLIPTokenizer'],'vae':['diffusers','AutoencoderKL']}
    if index['_class_name']!='StableDiffusionPipeline' or any(index.get(k)!=v for k,v in expected.items()):
        raise ValueError('SD2 component interface changed')
    if (u['in_channels'],u['out_channels'],u['sample_size'],u['cross_attention_dim'])!=(4,4,64,1024):
        raise ValueError('SD2 latent/conditioning interface changed')
    if t['hidden_size']!=1024 or t['max_position_embeddings']!=77 or t['num_hidden_layers']!=23:
        raise ValueError('Expected converted penultimate OpenCLIP representation')
    if s['prediction_type']!='epsilon' or s['num_train_timesteps']!=1000:
        raise ValueError('Expected 512 epsilon model, not 768 v-prediction')
    if v['latent_channels']!=4 or len(v['block_out_channels'])!=4 or v.get('scaling_factor',SCALE)!=SCALE:
        raise ValueError('Fixed clean KL-f8 interface changed')
    return dict(latent_scale=SCALE,prediction_type='epsilon',context_dim=1024,
                clean_decoder='fixed SD1.4',writer='fixed LaWa SD1.4',dtype='float32',
                construction='SD2.1 denoiser + fixed KL-f8/LaWa decoder',
                latent_coordinates='shared KL-f8 convention; actual suitability pending DEV',
                weights_loaded=False)

def build_pipeline(base,clean):
    from diffusers import StableDiffusionPipeline,DDIMScheduler,AutoencoderKL,UNet2DConditionModel
    from transformers import CLIPTextModel,CLIPTokenizer
    # Explicit component loading prevents downloading a second VAE or safety model.
    opts=dict(local_files_only=True,torch_dtype=torch.float32,use_safetensors=True)
    vae=AutoencoderKL.from_pretrained(str(clean/'vae'),**opts)
    vae.register_to_config(scaling_factor=SCALE)
    text=CLIPTextModel.from_pretrained(str(base/'text_encoder'),**opts)
    unet=UNet2DConditionModel.from_pretrained(str(base/'unet'),**opts)
    tokenizer=CLIPTokenizer.from_pretrained(str(base/'tokenizer'),local_files_only=True)
    s=json.loads((base/'scheduler/scheduler_config.json').read_text())
    scheduler=DDIMScheduler.from_config(s,clip_sample=False,set_alpha_to_one=False,
                                       steps_offset=1,timestep_spacing='leading')
    pipe=StableDiffusionPipeline(vae=vae,text_encoder=text,tokenizer=tokenizer,unet=unet,
                                scheduler=scheduler,safety_checker=None,feature_extractor=None,
                                requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    return pipe

class Backend(OriginalBackend):
    @classmethod
    def load(cls,scratch,device='cuda'):
        report=validate_configs(scratch/'cache/sd21',scratch/'cache/sd14')
        pipe=build_pipeline(scratch/'cache/sd21',scratch/'cache/sd14')
        w=build_writer(scratch/'official')
        state=torch.load(scratch/'cache/last.ckpt',weights_only=True,map_location='cpu',mmap=True)['state_dict']
        load_inference(w,state);del state
        b=cls(pipe,w,device);b.config_report=report
        return b

    @torch.no_grad()
    def generate(self,work):
        # One fixed transfer policy, no OOM-dependent precision/offload switching.
        self.writer.to('cpu');self.pipe.to(self.device)
        generator=torch.Generator(device=self.device).manual_seed(work['seed'])
        z=self.pipe(work['prompt'],height=512,width=512,num_inference_steps=50,guidance_scale=7.5,
                    eta=0.,generator=generator,output_type='latent').images
        if z.shape!=(1,4,64,64) or not torch.isfinite(z).all():
            raise ValueError('Invalid SD2 terminal latent')
        self.pipe.text_encoder.to('cpu');self.pipe.unet.to('cpu')
        self.pipe.vae.to(self.device);self.writer.to(self.device)
        unscaled=z/SCALE
        clean=self.pipe.vae.decode(unscaled,return_dict=False)[0]
        h=self.writer.ae.post_quant_conv(unscaled)
        if not torch.isfinite(clean).all() or not torch.isfinite(h).all():
            raise ValueError('Nonfinite decoder interface')
        self.last_interface=dict(latent_shape=list(z.shape),latent_rms=float(z.square().mean().sqrt()),
            unscaled_rms=float(unscaled.square().mean().sqrt()),scale=SCALE,
            clean_shape=list(clean.shape),h_shape=list(h.shape),unscale_calls=1,writer_post_quant_calls=1)
        return z.detach().cpu().numpy(),h,rgb(clean)

    def marked(self,h,bits):
        if len(bits)!=48 or set(bits)-{'0','1'}:
            raise ValueError('Expected frozen 48-bit string')
        result=super().marked(h,bits)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite marked RGB')
        return result

    @torch.no_grad()
    def read(self,image):
        values=super().read(image)
        if np.asarray(values).shape!=(48,) or not np.isfinite(values).all():
            raise ValueError('Invalid continuous reader output')
        return values
