"""Pinned upstream interfaces on CPU. Only synthetic tensors, including the second reader."""
import importlib.util
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import torch
from lawa_ab.check_official_cpu import check
from lawa_ab.model import Backend
from .carrier import LaWaCarrier
from . import rules as R

GUIDANCE='d1f8e48946f830087baf32356e5ce0eeec68e27e'

def first(source,metadata):
    original_probe=Backend.probe
    def hooked(backend,h,bits):
        carrier=LaWaCarrier(backend,.10)
        energies=carrier.energies(h,bits)
        native_write=backend.write
        original=native_write(h,bits)
        native_raw=backend.marked(h,bits).detach()
        for action,sign in [('native',0),('fine',1),('coarse',-1)]:
            measured=[]
            def observe(h,bits):
                values={}
                with ExitStack() as stack:
                    for j,(_,conv,_) in enumerate(carrier.branches()):
                        def hook(module,args,out,j=j):
                            values[j]=float(out.square().sum())
                        stack.callback(conv.register_forward_hook(hook).remove)
                    rgb=native_write(h,bits)
                    measured.extend(sum(values[j] for j in (i,i+1)) for i in (0,2,4))
                    return rgb
            with patch.object(backend,'write',observe):
                rgb=carrier.write(h,bits,action)
            amount=sign*.10*min(energies[0],energies[2])
            assert np.allclose(measured,[energies[0]-amount,energies[1],energies[2]+amount],rtol=2e-6)
            assert np.isclose(sum(measured),sum(energies),rtol=2e-6)
            if sign:
                assert np.any(rgb!=original)
            else:
                assert np.array_equal(rgb,original)
            assert torch.equal(native_raw,backend.marked(h,bits))
        with patch.object(backend,'write',side_effect=ValueError('synthetic fault')):
            try:carrier.write(h,bits,'fine')
            except ValueError:pass
            else:raise AssertionError('Expected failure')
        assert torch.equal(native_raw,backend.marked(h,bits))
        print('LaWa native bypass, coarse/middle/fine physical energies, opposite RGB responses, exception cleanup: PASS')
        return original_probe(backend,h,bits)
    with patch.object(Backend,'probe',hooked):
        check(source,metadata)

def second(root):
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==GUIDANCE
    path=root/'guidance-watermarking-for-diffusion-models/detector/detector.py'
    spec=importlib.util.spec_from_file_location('official_detector',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    class Toy(torch.nn.Module):
        def forward(self,x):
            return x.mean((1,2,3))[:,None]+torch.linspace(-1,1,256)[None,:]
    reader=mod.VideoSealDetector(Toy(),M=256,whitener={'L':torch.eye(256),'mu':torch.zeros(256,1)})
    x=torch.ones(1,3,128,128)*.25
    q=reader.decode_message(reader.preprocess(x))
    codes=['01'*128,'10'*128]
    vals=R.scores(q[0].tolist(),R.signs(codes))
    assert q.shape==(1,256) and vals[0]!=vals[1] and reader.process_message(q).dtype==torch.bool
    print('Official Guidance/VideoSeal RGB transform, whitening and continuous 256-value scoring: PASS with a synthetic detector only')

if __name__=='__main__':
    a,b,c=map(Path,sys.argv[1:])
    first(a,b)
    second(c)
