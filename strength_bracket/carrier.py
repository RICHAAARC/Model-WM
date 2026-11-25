"""Reuse the frozen six-branch coarse action; extend only the declared uniform grid."""
from contextlib import ExitStack
from probe_selection.carrier import LaWaCarrier
from .rules import GAINS
class Carrier(LaWaCarrier):
    def uniform(self,h,bits,gain):
        if gain not in GAINS:raise ValueError('Gain outside frozen grid')
        with ExitStack() as stack:
            for _,conv,_ in self.branches():
                stack.callback(conv.register_forward_hook(lambda module,args,out:out*gain).remove)
            return self.backend.write(h,bits)
