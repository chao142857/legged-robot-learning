"""CPU gradient contracts of the new term, no native learner or real study I/O."""
import sys, types
sys.modules['telemetry']=types.SimpleNamespace(model_sha=lambda _:None,write_json=lambda *_:None)
import torch
from leg_retention import LegRetention,leg_kl,steady_goal_mask

torch.set_num_threads(1)
x=torch.zeros(4,210)
goals=x[:,189:210].reshape(4,3,7);goals[:,:,0]=.45
goals[1,:,0]=torch.tensor([.45,.45001,.45002])
goals[2,:,0]=torch.tensor([.46999753,.47,.46999753])  # actual8s/20mm peak scale
goals[3,:,3]=torch.tensor([1.,.99,.98])  # quaternion is not position
assert steady_goal_mask(x).tolist()==[True,False,False,True]

class Anchor:
    distribution=types.SimpleNamespace(std_type='log',log_std_param=torch.zeros(18))
    def __call__(self,obs,stochastic_output=False):return torch.zeros(len(obs['policy']),18)

def evaluate(extra):
    r=LegRetention.__new__(LegRetention)
    r.alg=types.SimpleNamespace(actor=types.SimpleNamespace(obs_groups=['policy']))
    r.anchor=Anchor();r.cfg=dict(weight=.02)
    if extra:r.cfg.update(steady_mean_weight=1.,steady_xyz_tolerance_m=1e-7)
    r.values=[];mean=torch.ones(2,18,requires_grad=True);std=torch.ones(2,18,requires_grad=True)
    loss=r.penalty(dict(policy=x[:2]),(mean,std))
    gm,gs=torch.autograd.grad(loss,(mean,std))
    if not extra:
        assert torch.allclose(loss,.02*leg_kl(torch.zeros_like(mean),torch.ones_like(std),mean,std))
    return loss.detach(),gm,gs,r.values[0]

old,gm0,gs0,_=evaluate(False);new,gm1,gs1,row=evaluate(True)
assert torch.allclose(new-old,torch.tensor(4.5))
assert torch.allclose(gm1[0]-gm0[0],torch.full((18,),.5))
assert torch.equal(gm1[1],gm0[1])  # moving target: no direct new mean gradient
assert torch.equal(gs1,gs0)  # current exploration std untouched by new term
assert row['steady_goal_fraction']==.5 and abs(row['steady_full_mean_weighted_loss']-4.5)<1e-6
assert not torch.cuda.is_initialized()
print('CPU steady retention PASS: original210 mask, dynamic extrema free, quaternion independence, all18 held-mean gradient, unchanged moving/std gradients and original default loss')
