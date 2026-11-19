"""SD1.4 DDIM carrier and official LaWa inference graph. Nothing is fitted."""
import ast
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
import yaml


def author_graph(source):
    # Keep the published marked decoder forward verbatim without importing its
    # training framework or initializing discriminator/LPIPS/ImageNet weights.
    tree = ast.parse((source/'models/modifiedAEDecoder.py').read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LaWa')
    methods = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('forward','nonlinearity')]
    scope = {'torch':torch}
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(source/'models/modifiedAEDecoder.py'),'exec'),scope)
    return type('LaWaInference',(nn.Module,),{n.name:scope[n.name] for n in methods})


def build_writer(source):
    sys.path.insert(0,str(source))
    from ldm.modules.diffusionmodules.model import Decoder
    p = yaml.safe_load((source/'configs/SD14_LaWa_inference.yaml').read_text())['model']['params']
    model = author_graph(source)()
    config = p['first_stage_config']['params']['ddconfig']
    model.ae = nn.Module()
    model.ae.decoder = Decoder(**config)
    model.ae.post_quant_conv = nn.Conv2d(4,4,1)
    model.decoder = nn.Module()
    model.decoder.decoder = resnet50(weights=None,num_classes=48)
    model.decoder.forward = model.decoder.decoder.forward
    model.noise_block_size = b = p['noise_block_size']
    def mapped(ch):
        return nn.Sequential(nn.Linear(48,ch*b*b),nn.SiLU(),nn.Unflatten(1,(ch,b,b)))
    def conv(ch):
        return nn.Conv2d(ch,ch,3,padding=1)
    model.watermark_initial_0,model.watermark_initial_0_conv = mapped(4),conv(4)
    model.watermark_initial,model.watermark_initial_conv = mapped(512),conv(512)
    channels = (128,256,512,512)
    model.watermark = nn.ModuleList(mapped(ch) for ch in channels)
    model.watermark_conv = nn.ModuleList(conv(ch) for ch in channels)
    return model


def load_inference(model, state):
    keys = model.state_dict().keys()
    model.load_state_dict({k:state[k] for k in keys},strict=True)
    return model.requires_grad_(False).eval()


def rgb(tensor):
    return ((tensor.detach()[0].permute(1,2,0).float().cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)


class Backend:
    def __init__(self,pipe,writer,device):
        self.pipe,self.writer,self.device = pipe,writer,device

    @classmethod
    def load(cls,scratch,device='cuda'):
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        w = build_writer(scratch/'official')
        state = torch.load(scratch/'cache/last.ckpt',weights_only=True,map_location='cpu',mmap=True)['state_dict']
        load_inference(w,state)
        del state
        pipe = StableDiffusionPipeline.from_pretrained(str(scratch/'cache/sd14'),local_files_only=True,
            safety_checker=None,feature_extractor=None,requires_safety_checker=False,
            torch_dtype=torch.float32,use_safetensors=True)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,clip_sample=False,
            set_alpha_to_one=False,steps_offset=1,timestep_spacing='leading')
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe.to(device),w.to(device),device)

    @torch.no_grad()
    def generate(self,work):
        generator = torch.Generator(device=self.device).manual_seed(work['seed'])
        z = self.pipe(work['prompt'],height=512,width=512,num_inference_steps=50,guidance_scale=7.5,
                      eta=0.,generator=generator,output_type='latent').images
        unscaled = z / 0.18215
        clean = self.pipe.vae.decode(unscaled,return_dict=False)[0]
        return z.detach().cpu().numpy(),self.writer.ae.post_quant_conv(unscaled),rgb(clean)

    def marked(self,h,bits):
        code = torch.tensor([[2*int(b)-1 for b in bits]],device=self.device,dtype=h.dtype)
        return self.writer(h,None,code)[1]

    @torch.no_grad()
    def write(self,h,bits):
        return rgb(self.marked(h,bits))

    @torch.no_grad()
    def read(self,image):
        x = torch.tensor(np.asarray(image).copy(),device=self.device).permute(2,0,1)[None].float()/127.5-1.
        return self.writer.decoder(x)[0].cpu().tolist()

    def probe(self,h,bits):
        code = torch.tensor([2*int(b)-1 for b in bits],device=self.device,dtype=h.dtype)
        def score(v):
            return (self.writer.decoder(self.marked(v,bits).clamp(-1,1))[0]*code).mean()
        h = h.detach().requires_grad_(True)
        grad, = torch.autograd.grad(score(h),h)
        with torch.no_grad():
            norm = grad.square().mean().sqrt()
            direction = grad / norm.clamp_min(1e-20)
            eps = .001*h.square().mean().sqrt().clamp_min(.01)
            pred = (grad*direction).sum()
            fd = (score(h+eps*direction)-score(h-eps*direction))/(2*eps)
        return dict(gradient_rms=float(norm),analytic=float(pred),finite_difference=float(fd),
                    passed=bool(torch.isfinite(fd) and norm>0 and fd>0 and
                                torch.abs(fd-pred)<=.35*pred.abs()+1e-5))
