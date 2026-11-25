"""Parse actual component metadata and tokenizer, without real model tensors."""
import sys,json
from pathlib import Path
from .model import validate_configs
from latent_control.io import save
def check(scratch):
    from diffusers import DDIMScheduler,UNet2DConditionModel,AutoencoderKL
    from transformers import CLIPTextConfig,CLIPTokenizer
    b=scratch/'cache/sd21';c=scratch/'cache/sd14'
    r=validate_configs(b,c)
    tok=CLIPTokenizer.from_pretrained(str(b/'tokenizer'),local_files_only=True)
    assert len(tok('interface check',padding='max_length',max_length=77,truncation=True)['input_ids'])==77
    CLIPTextConfig.from_pretrained(str(b/'text_encoder'),local_files_only=True)
    UNet2DConditionModel.load_config(str(b/'unet'),local_files_only=True)
    AutoencoderKL.load_config(str(c/'vae'),local_files_only=True)
    s=DDIMScheduler.from_config(json.loads((b/'scheduler/scheduler_config.json').read_text()),
           clip_sample=False,set_alpha_to_one=False,steps_offset=1,timestep_spacing='leading')
    s.set_timesteps(50);assert len(s.timesteps)==50 and s.config.prediction_type=='epsilon'
    r.update(status='CPU_COMPONENT_PARSERS_PASSED',weights_loaded=False)
    return r
if __name__=='__main__':save(Path(sys.argv[2])/'component_preflight.json',check(Path(sys.argv[1])))
