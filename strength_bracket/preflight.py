"""Actual pinned configuration/tokenizer parsing before real model loading."""
import json,sys
from pathlib import Path
from latent_control.io import save
def check(scratch):
    from transformers import CLIPTokenizer,CLIPTextConfig
    from diffusers import UNet2DConditionModel,AutoencoderKL,DDIMScheduler
    p=scratch/'cache/sd14'
    index=json.loads((p/'model_index.json').read_text())
    assert index['unet'][1]=='UNet2DConditionModel' and index['vae'][1]=='AutoencoderKL'
    u=UNet2DConditionModel.load_config(str(p/'unet'),local_files_only=True)
    v=AutoencoderKL.load_config(str(p/'vae'),local_files_only=True)
    c=CLIPTextConfig.from_pretrained(str(p/'text_encoder'),local_files_only=True)
    assert u['in_channels']==4 and u['cross_attention_dim']==c.hidden_size==768
    assert v['latent_channels']==4 and v.get('scaling_factor',.18215)==.18215
    tok=CLIPTokenizer.from_pretrained(str(p/'tokenizer'),local_files_only=True)
    assert len(tok('component interface',padding='max_length',max_length=77,truncation=True)['input_ids'])==77
    s=DDIMScheduler.from_config(json.loads((p/'scheduler/scheduler_config.json').read_text()),clip_sample=False,set_alpha_to_one=False,steps_offset=1,timestep_spacing='leading')
    s.set_timesteps(50);assert len(s.timesteps)==50 and s.config.prediction_type=='epsilon'
    return dict(status='CPU_COMPONENT_PARSERS_PASSED',weights_loaded=False,source='SD1.4 fixed components')
if __name__=='__main__':save(Path(sys.argv[2])/'component_preflight.json',check(Path(sys.argv[1])))
