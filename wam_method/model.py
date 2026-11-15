"""Official writer/detector with an explicit differentiable saved-image channel."""
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from wam_qualification.backend import load_wam,features_and_native

def rgb_tensor(rgb, device):
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).unsqueeze(0).to(device=device,dtype=torch.float32)/255

def normalize(x):
    return (x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]

def unnormalize(x):
    return x*x.new_tensor([.229,.224,.225])[None,:,None,None]+x.new_tensor([.485,.456,.406])[None,:,None,None]

def quantize(x, mixture=False):
    clipped=x.clamp(0,1)
    exact=(torch.round(clipped*255) if mixture else torch.floor(clipped*255+.5))/255
    return clipped+(exact-clipped).detach()

def to_rgb(x):
    return x.detach()[0].mul(255).round().clamp(0,255).to('cpu',torch.uint8).permute(1,2,0).numpy()

def jpeg_rgb(rgb):
    stream=io.BytesIO();Image.fromarray(rgb).save(stream,format='JPEG',quality=85,subsampling=2,optimize=False)
    stream.seek(0)
    with Image.open(stream) as im:return np.asarray(im.convert('RGB')).copy()

def jpeg_ste(x):
    exact=rgb_tensor(jpeg_rgb(to_rgb(x)),x.device)
    return x+(exact-x).detach()

def leaf_tail(z, codes, pair):
    """Coalition truth is TRAIN supervision only, never a scoring input."""
    codes=torch.as_tensor(codes,dtype=z.dtype,device=z.device)
    scores=(2*codes-1)@z/128
    scale=torch.linalg.vector_norm(z)/(128**.5)+1e-6
    innocent=[i for i in range(16) if i not in pair]
    return F.softplus(scores[innocent].max()/scale-scores[pair].max()/scale)

def code_tensor(b):
    return [[int(x) for x in word] for word in b['registered']]

class Model:
    def __init__(self,source,checkpoint,device):
        self.device=torch.device(device)
        self.wam=load_wam(source,checkpoint,device)
        self.wam.detector.requires_grad_(False).eval()
        if self.wam.attenuation is not None:self.wam.attenuation.requires_grad_(False).eval()
    def write(self,x,bits):
        msg=torch.as_tensor(bits,device=self.device,dtype=torch.float32).reshape(1,32)
        return quantize(unnormalize(self.wam.embed(normalize(x),msg)['imgs_w']))
    def detect(self,x):
        return self.wam.detect(normalize(x))['preds']
    def eval_write(self,rgb,bits):
        with torch.inference_mode():return to_rgb(self.write(rgb_tensor(rgb,self.device),bits))
    def eval_extract(self,rgb):
        with torch.inference_mode():
            raw,pool,native=features_and_native(self.detect(rgb_tensor(rgb,self.device)))
        # Raw tensor is transient. All frozen receiver sufficient statistics persist.
        return pool,native

def train_work(model,rgb,messages,jpeg,ledger,prefix):
    x=rgb_tensor(rgb,model.device)
    ys=[ledger.call(prefix+f'/write/{i}','train_write',lambda bits=bits:model.write(x,bits))
        for i,bits in enumerate(messages)]
    mix=quantize((ys[0]+ys[1])/2,mixture=True)
    if jpeg:mix=jpeg_ste(mix)
    preds=[ledger.call(prefix+f'/extract/{i}','train_extract',lambda y=y:model.detect(y))
           for i,y in enumerate(ys+[mix])]
    z=[p[:,1:].mean((2,3)) for p in preds]
    targets=torch.as_tensor(messages,dtype=x.dtype,device=x.device)
    native=sum(F.binary_cross_entropy_with_logits(z[i][0],targets[i]) for i in range(2))/2
    mixed=F.binary_cross_entropy_with_logits(z[2][0],targets.mean(0))
    mask=sum(F.binary_cross_entropy_with_logits(p[:,0],torch.ones_like(p[:,0])) for p in preds)/3
    image=sum(F.mse_loss(y,x) for y in ys)/2
    return z[2][0],native+mixed+mask+100*image,dict(native=float(native.detach()),
        mixed=float(mixed.detach()),mask=float(mask.detach()),image_mse=float(image.detach()))
