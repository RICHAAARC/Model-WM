"""Exact official architecture and synthetic state loading, without real weights."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def check(source):
    import torch
    from omegaconf import OmegaConf
    from .environment import check as versions
    from .backend import load_wam, features_and_native
    receipt=versions()
    source=Path(source).resolve()
    sys.path[:0]=[str(source),str(source/"notebooks")]
    from watermark_anything.models import Wam,build_embedder,build_extractor
    from watermark_anything.augmentation.augmenter import Augmenter
    from watermark_anything.modules.jnd import JND
    from watermark_anything.data.transforms import normalize_img,unnormalize_img
    torch.set_num_threads(1);torch.manual_seed(7) # synthetic engineering fixture, not a new experiment seed
    p=argparse.Namespace(**json.loads((source/"checkpoints/params.json").read_text()))
    emb=OmegaConf.load(source/p.embedder_config);ext=OmegaConf.load(source/p.extractor_config)
    aug=OmegaConf.load(source/p.augmentation_config);att=OmegaConf.load(source/p.attenuation_config)
    reference=Wam(build_embedder(p.embedder_model,emb[p.embedder_model],p.nbits),
        build_extractor(ext.model,ext[p.extractor_model],p.img_size,p.nbits),Augmenter(**aug),
        JND(**att[p.attenuation],preprocess=unnormalize_img,postprocess=normalize_img),p.scaling_w,p.scaling_i).eval()
    state=reference.state_dict()
    with tempfile.TemporaryDirectory() as d:
        fake=Path(d)/"synthetic-marker.pth";fake.write_text("synthetic state provided in memory")
        with patch("torch.load",return_value=state) as loader:
            wrapped=load_wam(source,fake,"cpu")
            assert loader.call_count==1
            assert loader.call_args.kwargs==dict(map_location="cpu")
        with patch("torch.load",side_effect=AssertionError("missing state must not be read")):
            try: load_wam(source,Path(d)/"absent.pth","cpu")
            except RuntimeError: pass
            else: raise AssertionError("Missing state did not fail closed")
        broken=dict(state);broken.pop(next(iter(broken)))
        with patch("torch.load",return_value=broken):
            try: load_wam(source,fake,"cpu")
            except RuntimeError: pass
            else: raise AssertionError("Mismatched state was accepted")
    x=normalize_img(torch.linspace(0,1,512*512*3).reshape(1,3,512,512))
    bits=torch.tensor([[0,1]*16],dtype=torch.float32)
    with torch.inference_mode():
        a=reference.embed(x,msgs=bits)["imgs_w"]
        b=wrapped.embed(x,msgs=bits)["imgs_w"]
        assert torch.equal(a,b) and a.shape==(1,3,512,512)
        pred=wrapped.detect(b)["preds"]
        raw,pool,native=features_and_native(pred)
        assert raw.shape==(33,256,256) and pool.shape==(33,4,4)
        torch.testing.assert_close(torch.from_numpy(pool).mean((1,2)),torch.from_numpy(raw).mean((1,2)))
        known=torch.ones(1,33,256,256);known[:,1:]=.25
        assert features_and_native(known)[2]["bits"]==[0]*32 # official final threshold is .5, not 0
        known[:,0]=-1
        assert features_and_native(known)[2]["status"]=="UNRESOLVED_EMPTY_MASK"
    receipt.update(status="CPU_SYNTHETIC_API_REGRESSION_PASSED",source=str(source),
        checks=["strict state match", "missing checkpoint rejection", "mismatched state rejection",
                "official embed equality at 512", "33-channel native detect", "4x4 global-mean preservation",
                "official .5 native threshold", "empty-mask unresolved"],
        real_weights_loaded=False,gpu_used=False,scientific_pass=None)
    return receipt


if __name__=="__main__":
    result=check(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
    print(result["status"])
