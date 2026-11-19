import torch
from latent_control.model import Model, EvaluationLimit, regularized_inverse
from latent_control.io import MechanismFailure
from latent_control.carrier import quantize

class Solver(Model):
    def common_center(self, work):
        ident=work['id']
        z=self.generate(work)
        clean=self.decode(z,ident+'/initial','initial_decode')
        r=self.reader(clean)[0]
        rows=[]
        for k in range(len(r)):
            grad=self.call(f'{ident}/host/vjp/{k}','backward_center',
                           lambda:torch.autograd.grad(r[k],z,retain_graph=k+1<len(r))[0])
            rows.append(grad.detach().flatten())
        j=torch.stack(rows)
        r=r.detach()
        inverse,_=regularized_inverse(j)
        center=(z.detach()-(inverse@r).reshape_as(z)).detach().requires_grad_(True)
        y=self.decode(center,ident+'/center','center_decode')
        return quantize(clean),center,y,self.reader(y)[0].detach()

    def response(self,center,y,ident):
        _,j=self.jacobian(center,y,ident+'/response')
        return regularized_inverse(j)[0]

    def inverse_copy(self,center,rc,inverse,user,gain,ident):
        with torch.no_grad():
            offset=inverse@(gain*self.a*self.codes[user]-rc)
            y=self.decode(center+offset.reshape_as(center),ident+'/decode','final_C')
            return quantize(y)

    def ordinary_copy(self,center,user,gain,ident):
        delta=torch.zeros_like(center,requires_grad=True)
        opt=torch.optim.LBFGS([delta],lr=1.,max_iter=16,max_eval=16,history_size=10,
                              line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-9)
        count=0
        best_loss=float('inf')
        best=None
        def closure():
            nonlocal count,best_loss,best
            if count==16:
                raise EvaluationLimit
            step=f'{ident}/eval/{count}'
            count+=1
            y=self.decode(center.detach()+delta,step+'/decode','closure_O')
            loss=(self.reader(y)[0]/self.a-gain*self.codes[user]).square().sum()
            if not torch.isfinite(loss):
                raise MechanismFailure('Nonfinite ordinary objective')
            grad=self.call(step+'/gradient','backward_O',lambda:torch.autograd.grad(loss,delta)[0])
            if not torch.isfinite(grad).all():
                raise MechanismFailure('Nonfinite ordinary gradient')
            delta.grad=grad.detach()
            if float(loss.detach())<best_loss:
                best_loss=float(loss.detach())
                best=quantize(y)
            return loss
        try:
            opt.step(closure)
        except EvaluationLimit:
            pass
        if best is None:
            raise MechanismFailure('No evaluated finite ordinary endpoint')
        return best
