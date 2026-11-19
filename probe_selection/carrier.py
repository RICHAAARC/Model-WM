"""LaWa adapter. The selector consumes ordinary RGB and real-valued bit evidence."""
from contextlib import ExitStack
import numpy as np
import torch
from lawa_ab.model import Backend

class LaWaCarrier:
    actions=('native','fine','coarse')

    def __init__(self, backend, fraction):
        self.backend=backend
        self.fraction=fraction
        self.energy_cache={}

    @classmethod
    def load(cls,scratch,fraction):
        return cls(Backend.load(scratch),fraction)

    def generate(self,work):
        return self.backend.generate(work)

    @torch.no_grad()
    def restore(self,z):
        x=torch.as_tensor(z,device=self.backend.device)
        return self.backend.writer.ae.post_quant_conv(x/0.18215)

    def read(self,rgb):
        return self.backend.read(rgb)

    def branches(self):
        w=self.backend.writer
        return [(w.watermark_initial_0,w.watermark_initial_0_conv,1),
                (w.watermark_initial,w.watermark_initial_conv,1)]+[
                (w.watermark[i],w.watermark_conv[i],scale) for i,scale in ((3,2),(2,4),(1,8),(0,8))]

    @torch.no_grad()
    def energies(self,h,bits):
        key=(tuple(h.shape[-2:]),bits)
        if key not in self.energy_cache:
            w=self.backend.writer
            code=h.new_tensor([[2*int(b)-1 for b in bits]])
            terms=[]
            for mapped,conv,scale in self.branches():
                tile=mapped(code)
                tiled=tile.repeat(1,1,h.shape[-2]*scale//w.noise_block_size,h.shape[-1]*scale//w.noise_block_size)
                terms.append(float(conv(tiled).square().sum()))
            self.energy_cache[key]=[sum(terms[i:i+2]) for i in (0,2,4)]
        return self.energy_cache[key]

    def write(self,h,bits,action):
        if action=='native':
            return self.backend.write(h,bits)
        if action not in self.actions:
            raise ValueError(action)
        coarse,_,fine=self.energies(h,bits)
        if min(coarse,fine)<=0 or not np.isfinite(coarse+fine):
            raise ValueError('Invalid residual branch energy')
        amount=self.fraction*min(coarse,fine)*(1 if action=='fine' else -1)
        gains=[(1-amount/coarse)**.5]*2+[1.]*2+[(1+amount/fine)**.5]*2
        with ExitStack() as stack:
            for (_,conv,_),gain in zip(self.branches(),gains):
                if gain!=1:
                    handle=conv.register_forward_hook(lambda module,args,out,g=gain:out*g)
                    stack.callback(handle.remove)
            return self.backend.write(h,bits)
