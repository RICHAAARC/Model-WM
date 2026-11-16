"""A fixed, exactly linear block-DCT reader; no learned or truth-guided extraction."""
import io
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

FREQUENCIES=((1,2),(2,1),(2,2),(1,3),(3,1),(2,3),(3,2),(3,3))

class Reader(torch.nn.Module):
    def __init__(self,permutation,signs):
        super().__init__()
        n=torch.arange(8,dtype=torch.float32)
        dct=torch.stack([math.sqrt((1 if k==0 else 2)/8)*torch.cos(math.pi*(2*n+1)*k/16) for k in range(8)])
        luma=torch.tensor([.299,.587,.114]);luma=luma/luma.norm()
        kernels=torch.stack([torch.outer(dct[i],dct[j]) for i,j in FREQUENCIES])[:,None]*luma[None,:,None,None]
        self.register_buffer('kernels',kernels)
        self.register_buffer('permutation',torch.as_tensor(permutation,dtype=torch.long))
        self.register_buffer('signs',torch.as_tensor(signs,dtype=torch.float32))
    def forward(self,x):
        v=F.conv2d(x,self.kernels,stride=8).flatten(1)
        if v.shape[1]!=self.permutation.numel():raise RuntimeError('Reader resolution mismatch')
        return (v[:,self.permutation]*self.signs).reshape(x.shape[0],64,-1).sum(-1)/math.sqrt(v.shape[1]/64)

def tensor(rgb,device):
    return torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/255

def quantize(x):
    if not torch.isfinite(x).all():raise ValueError('Nonfinite image')
    return x.detach()[0].clamp(0,1).mul(255).add(.5).to(torch.uint8).permute(1,2,0).cpu().numpy()

def quality(clean,rgb):
    mse=float(np.mean(((clean.astype(np.float64)-rgb)/255)**2))
    return dict(mse=mse,psnr_db=None if mse==0 else -10*math.log10(mse),psnr_infinite=mse==0,
                ssim=float(structural_similarity(clean,rgb,channel_axis=2,data_range=255)))

def quality_ok(q,m):return q['mse']<=m['rho']**2 and q['ssim']>=m['gates']['native_ssim']

def mixed(a,b,weight):return np.rint(weight*a.astype(np.float64)+(1-weight)*b.astype(np.float64)).astype(np.uint8)

def jpeg(rgb):
    stream=io.BytesIO();Image.fromarray(rgb).save(stream,format='JPEG',quality=85,subsampling=2,optimize=False)
    data=stream.getvalue()
    with Image.open(io.BytesIO(data)) as im:decoded=np.asarray(im.convert('RGB')).copy()
    return data,decoded
