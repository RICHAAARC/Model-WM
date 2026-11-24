"""Six message branches scaled together. No model initialization on import."""
from contextlib import ExitStack
from probe_selection.carrier import LaWaCarrier

class Carrier(LaWaCarrier):
    def uniform(self, h, bits, gain):
        if gain not in (1.,1.025,1.05,1.075):
            raise ValueError("Unfrozen gain")
        if gain == 1.:
            return self.backend.write(h,bits)
        with ExitStack() as stack:
            for _,conv,_ in self.branches():
                stack.callback(conv.register_forward_hook(lambda module,args,out:out*gain).remove)
            return self.backend.write(h,bits)

