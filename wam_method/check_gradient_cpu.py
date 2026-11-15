"""Real official architecture, random engineering state, no pretrained weights."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from .io import Ledger,save
from .model import Model,train_work,normalize,unnormalize

def check(source):
    source=Path(source).resolve()
    sys.path[:0]=[str(source),str(source/'notebooks')]
    from watermark_anything.models import Wam,build_embedder,build_extractor
    from watermark_anything.augmentation.augmenter import Augmenter
    from watermark_anything.modules.jnd import JND
    from watermark_anything.data.transforms import normalize_img,unnormalize_img
    torch.set_num_threads(1);torch.manual_seed(7) # synthetic fixture, not protocol seed
    p=argparse.Namespace(**json.loads((source/'checkpoints/params.json').read_text()))
    emb=OmegaConf.load(source/p.embedder_config);ext=OmegaConf.load(source/p.extractor_config)
    aug=OmegaConf.load(source/p.augmentation_config);att=OmegaConf.load(source/p.attenuation_config)
    reference=Wam(build_embedder(p.embedder_model,emb[p.embedder_model],p.nbits),
        build_extractor(ext.model,ext[p.extractor_model],p.img_size,p.nbits),Augmenter(**aug),
        JND(**att[p.attenuation],preprocess=unnormalize_img,postprocess=normalize_img),p.scaling_w,p.scaling_i).eval()
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);fake=tmp/'random.pt';torch.save(reference.state_dict(),fake)
        model=Model(source,fake,'cpu');model.wam.embedder.requires_grad_(True).eval()
        for k,v in reference.embedder.state_dict().items():assert torch.equal(v,model.wam.embedder.state_dict()[k])
        del reference
        x=torch.linspace(0,1,3*512*512).reshape(1,3,512,512)
        torch.testing.assert_close(normalize(x),normalize_img(x))
        torch.testing.assert_close(unnormalize(normalize(x)),unnormalize_img(normalize_img(x)))
        rgb=np.arange(512*512*3,dtype=np.uint8).reshape(512,512,3)
        bits=[[0,1]*16,[1,0]*16];ledger=Ledger(tmp)
        for jpeg in [False,True]:
            model.wam.embedder.zero_grad(set_to_none=True)
            z,loss,_=train_work(model,rgb,bits,jpeg,ledger,str(jpeg))
            assert z.shape==(32,) and torch.isfinite(loss)
            loss.backward()
            grads=[v.grad for v in model.wam.embedder.parameters() if v.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(float(g.abs().sum()) for g in grads)>0
            assert all(v.grad is None for v in model.wam.detector.parameters())
            assert all(v.grad is None for v in model.wam.attenuation.parameters())
        assert ledger.counts=={'train_write':4,'train_extract':6}
    return dict(status='OFFICIAL_RANDOM_STATE_CPU_GRADIENT_PASSED',
        checks=['strict writer load','official normalization equality','512 native PNG channel',
                'actual JPEG85 forward with STE backward','nonzero finite writer gradients',
                'frozen detector and JND gradients','exact physical call counts'],
        real_weights_loaded=False,gpu_used=False,scientific_pass=None)

if __name__=='__main__':
    r=check(sys.argv[1]);save(sys.argv[2],r);print(r['status'])
