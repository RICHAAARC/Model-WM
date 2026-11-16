"""Frozen VAE input control, bounded shared inverse and bounded ordinary L-BFGS."""
import math
import numpy as np
import torch
from .carrier import Reader,quantize,quality,quality_ok,tensor
from .io import MechanismFailure,BudgetStop,append,save

class EvaluationLimit(Exception):pass

def load_pipeline(base,device):
    from diffusers import StableDiffusionPipeline,DPMSolverMultistepScheduler
    pipe=StableDiffusionPipeline.from_pretrained(str(base),local_files_only=True,
        torch_dtype=torch.float32,safety_checker=None,requires_safety_checker=False)
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device);pipe.set_progress_bar_config(disable=True)
    return pipe

def regularized_inverse(j):
    # Solve small systems in float64; the expensive VAE remains the frozen float32 model.
    j=j.double()
    if not torch.isfinite(j).all():raise MechanismFailure('Nonfinite response Jacobian')
    s=torch.linalg.svdvals(j)
    if s[-1]<=s[0]*1e-6:
        raise MechanismFailure('Response rank deficient at fixed relative tolerance 1e-6',dict(singular_values=s.cpu().tolist()))
    gram=j@j.T;lam=1e-4*torch.trace(gram)/j.shape[0]
    b=torch.linalg.solve(gram+lam*torch.eye(j.shape[0],device=j.device,dtype=j.dtype),j).T
    return b.float(),dict(singular_values=s.cpu().tolist(),regularization=float(lam),rank_tolerance=1e-6)

class Model:
    def __init__(self,base,m,arrays,ledger,device='cuda',pipeline=None):
        self.device=torch.device(device);self.m=m;self.ledger=ledger
        if pipeline is None:
            if self.device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('User-started CUDA required')
            pipeline=load_pipeline(base,self.device)
        self.pipe=pipeline
        for name in ['vae','unet','text_encoder']:
            mod=getattr(pipeline,name,None)
            if mod is not None:mod.eval().requires_grad_(False)
        self.reader=Reader(arrays['permutation'],arrays['signs']).to(self.device)
        self.codes=torch.tensor(arrays['codes'],dtype=torch.float32,device=self.device)
        self.a=m['amplitude']
    def sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize()
    def call(self,ident,kind,fn):
        def complete():
            out=fn();self.sync();return out
        return self.ledger.call(ident,kind,complete)
    def generate(self,work):
        step=[0];old=self.pipe.unet.forward
        def counted(*args,**kwargs):
            k=step[0]
            if k>=self.m['generation']['steps']:raise BudgetStop('Per-work denoiser step cap')
            step[0]+=1
            return self.call(work['id']+f'/denoise/{k}','denoise',lambda:old(*args,**kwargs))
        self.pipe.unet.forward=counted
        def run():
            g=self.m['generation']
            z=self.pipe(work['prompt'],height=g['height'],width=g['width'],num_inference_steps=g['steps'],
                guidance_scale=g['cfg'],negative_prompt=None,output_type='latent',
                generator=torch.Generator(self.device).manual_seed(work['noise_seed'])).images
            if step[0]!=g['steps']:raise RuntimeError('Unexpected denoiser call count')
            # Clone outside the pipeline no_grad context; no encoder/PIL round trip.
            return (z.detach().clone()/self.pipe.vae.config.scaling_factor).requires_grad_(True)
        try:return self.call(work['id'],'generate',run)
        finally:self.pipe.unet.forward=old
    def decode(self,z,ident,kind):
        def run():
            y=(self.pipe.vae.decode(z,return_dict=False)[0]+1)/2
            if not torch.isfinite(y).all():raise MechanismFailure('Nonfinite VAE output')
            return y
        return self.call(ident,kind,run)
    def jacobian(self,z,y,prefix):
        r=self.reader(y)[0];rows=[]
        for k in range(64):
            grad=self.call(prefix+f'/vjp/{k}','backward_C',
                lambda k=k:torch.autograd.grad(r[k],z,retain_graph=k<63)[0])
            rows.append(grad.detach().flatten())
        return r.detach(),torch.stack(rows)
    def shared(self,z,clean_y,users,prefix,on_copy=None):
        r,j=self.jacobian(z,clean_y,prefix+'/initial')
        b,info0=regularized_inverse(j);info0['first_response_column']=j[:,0].cpu().tolist()
        u0=(-b@r).reshape_as(z)
        zc=(z.detach()+u0).requires_grad_(True)
        yc=self.decode(zc,prefix+'/center','forward_C')
        rc,jc=self.jacobian(zc,yc,prefix+'/center');bc,info1=regularized_inverse(jc)
        diagnostic=dict(initial=info0,center=info1,host_before=r.cpu().tolist(),host_after=rc.cpu().tolist(),
            common_control_norm=float(u0.norm()))
        save(self.ledger.path.parent/'response_traces'/f'{prefix}.json',diagnostic)
        clean=quantize(clean_y);outputs=[]
        for index,user in enumerate(users):
            ident=prefix+f'/copy/{index}'
            def write(user=user):
                u=u0+(bc@(self.a*self.codes[user]-rc)).reshape_as(z)
                with torch.no_grad():y=self.decode(z.detach()+u,ident+'/decode','forward_C')
                rgb=quantize(y);q=quality(clean,rgb)
                predicted=(rc+jc@(u-u0).flatten())/self.a
                return dict(rgb=rgb,control=u.detach().cpu().numpy(),quality=q,
                    predicted_q=predicted.cpu().tolist(),target=user,status='VALID')
            out=self.call(ident,'write_C',write);outputs.append(out)
            if on_copy is not None:on_copy(index,out)
        return outputs,diagnostic
    def ordinary(self,z,clean_y,user,prefix):
        def write():
            u=torch.zeros_like(z,requires_grad=True);n=[0];trace=[];best=[None]
            clean=quantize(clean_y);reference=tensor(clean,self.device)
            opt=torch.optim.LBFGS([u],lr=1.,max_iter=32,max_eval=32,history_size=10,
                tolerance_grad=1e-7,tolerance_change=1e-9,line_search_fn='strong_wolfe')
            def closure():
                if n[0]>=32:raise EvaluationLimit()
                k=n[0];n[0]+=1;ident=prefix+f'/eval/{k}'
                def evaluate():
                    y=self.decode(z.detach()+u,ident+'/decode','forward_B')
                    r=self.reader(y)[0]/self.a
                    mse=(y-reference).square().mean()
                    loss=(r-self.codes[user]).square().sum()+torch.relu(mse/self.m['rho']**2-1).square()
                    if not torch.isfinite(loss):raise MechanismFailure('Nonfinite ordinary objective')
                    grad=self.call(ident+'/gradient','backward_B',lambda:torch.autograd.grad(loss,u)[0])
                    if not torch.isfinite(grad).all():raise MechanismFailure('Nonfinite ordinary gradient')
                    u.grad=grad.detach()
                    rgb=quantize(y);q=quality(clean,rgb);value=float(loss.detach())
                    trace.append(dict(evaluation=k,objective=value,quality=q))
                    append(self.ledger.path.parent/'objective_traces'/f'{prefix}.jsonl',trace[-1])
                    if quality_ok(q,self.m) and (best[0] is None or value<best[0]['objective']):
                        best[0]=dict(rgb=rgb,control=u.detach().cpu().numpy().copy(),quality=q,
                            objective=value,target=user,status='VALID')
                    return loss
                return self.call(ident,'objective_B',evaluate)
            exhausted=False
            try:opt.step(closure)
            except EvaluationLimit:exhausted=True
            if best[0] is None:raise MechanismFailure('No quality-feasible evaluated ordinary point')
            best[0].update(evaluations=n[0],evaluation_cap_reached=exhausted,trace=trace)
            return best[0]
        return self.call(prefix,'write_B',write)
    def finite_difference(self,z,clean_y,prefix):
        # Fixed first coordinate; no random probes, model choices or search.
        h=self.m['finite_difference']['step'];direction=torch.zeros_like(z);direction.flatten()[0]=1
        values=[]
        with torch.no_grad():
            for sign in [-1,1]:
                y=self.decode(z.detach()+sign*h*direction,prefix+f'/fd/{sign}','finite_difference')
                values.append(self.reader(y)[0])
        return ((values[1]-values[0])/(2*h)).cpu().numpy()
