"""CL1 recurrent macro-action PPO. No simulator/PPO-student gradients enter here."""
import copy, math
import numpy as np
import torch
from torch import nn

CONFIG = dict(hidden=64, action_dim=8, input_dim=34, sigma=.2, learn_sigma=False,
              lr=1e-4, clip=.2, gamma=.99, gae_lambda=.95, value_coef=.5,
              entropy_coef=0., max_grad_norm=.5, epochs=2, target_kl=.01,
              uniform_mix=.2, teacher_seed=20261109)
ERROR_KEYS = ('velocity_m_s','yaw_rad_s','ee_position_m','ee_orientation_rad')
ERROR_SCALES = np.array([.2,.3,.05,math.pi/12])

def observation(global_state, dev, previous=None, dynamic=None):
    """Fixed scales, explicit masks; no student seed, test score or future labels."""
    g=np.asarray(global_state,dtype=np.float64)
    e=np.array([dev['pooled_rmse'][k] for k in ERROR_KEYS])/ERROR_SCALES
    j=float(dev['continuous_joint_score'])
    if previous is None: delta=np.zeros(5); action=np.zeros(8); reward=0.; seen=0.
    else:
        old=previous['dev']; delta=np.r_[j-old['continuous_joint_score'],
           e-np.array([old['pooled_rmse'][k] for k in ERROR_KEYS])/ERROR_SCALES]
        action=np.asarray(previous['action']);reward=j-old['continuous_joint_score'];seen=1.
    # Dynamic evidence/age fields are already scaled by the validated V0.2 runtime.
    d=np.asarray(dynamic) if dynamic is not None else np.zeros((1,30))
    evidence=np.array([d[:,0].mean(),d[:,1].mean(),d[:,11].mean(),d[:,12].mean()])
    x=np.r_[g,j,np.clip(e,0,10)/10,np.clip(delta,-2,2),action,reward,seen,evidence]
    assert x.shape==(CONFIG['input_dim'],) and np.isfinite(x).all()
    return x.astype(np.float64)

class RecurrentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(CONFIG['input_dim'],64),nn.Tanh())
        self.gru=nn.GRUCell(64,64); self.mean=nn.Linear(64,8); self.value=nn.Linear(64,1)
        nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)
        self.double()
    def forward(self,x,h):
        h=self.gru(self.encoder(x),h)
        return self.mean(h),self.value(h).squeeze(-1),h
    def zero(self): return torch.zeros(64,dtype=torch.float64)

def replay(net, prefix):
    h=net.zero()
    with torch.no_grad():
        for x in prefix: _,_,h=net(torch.as_tensor(x,dtype=torch.float64),h)
    return h.detach()

def gae(rewards,values,next_value,terminal):
    adv=np.zeros(len(rewards));carry=0.;n=float(next_value)
    for i in reversed(range(len(rewards))):
        mask=0. if terminal and i==len(rewards)-1 else 1.
        delta=rewards[i]+CONFIG['gamma']*mask*n-values[i]
        carry=delta+CONFIG['gamma']*CONFIG['gae_lambda']*mask*carry
        adv[i]=carry;n=values[i]
    return adv,adv+np.asarray(values)

class Learner:
    def __init__(self):
        with torch.random.fork_rng():
            torch.manual_seed(CONFIG['teacher_seed']);self.net=RecurrentPolicy()
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=CONFIG['lr'])
        self.rng=torch.Generator().manual_seed(CONFIG['teacher_seed']+1)
        self.version=0
    def state(self):
        return copy.deepcopy(dict(net=self.net.state_dict(),optimizer=self.optimizer.state_dict(),
             rng=self.rng.get_state(),version=self.version,config=CONFIG))
    def load(self,s):
        assert s['config']==CONFIG
        self.net.load_state_dict(copy.deepcopy(s['net']))
        self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
        self.rng.set_state(s['rng'].clone());self.version=s['version']
    def act(self,x,prefix):
        h=replay(self.net,prefix)
        with torch.no_grad():
            mean,value,_=self.net(torch.as_tensor(x),h)
            action=mean+CONFIG['sigma']*torch.randn(8,generator=self.rng,dtype=torch.float64)
            lp=torch.distributions.Normal(mean,CONFIG['sigma']).log_prob(action).sum()
        return dict(action=action.numpy(),mean=mean.numpy(),value=float(value),logprob=float(lp),
                    version=self.version,hidden_before=h.numpy())
    def update(self,fragments):
        before=self.state(); before_grads=[None if p.grad is None else p.grad.clone() for p in self.net.parameters()]
        try:
            result=self._update(fragments)
            assert all(torch.isfinite(p).all() for p in self.net.parameters())
            return result
        except BaseException:
            self.load(before)
            for p,g in zip(self.net.parameters(),before_grads):p.grad=g
            raise
    def _update(self,fragments):
        prepared=[]
        for f in fragments:
            ts=f['transitions'];assert ts and all(t['version']==self.version for t in ts)
            assert all(np.isfinite(t['reward']) for t in ts)
            h=replay(self.net,f['prefix']+[t['x'] for t in ts])
            with torch.no_grad(): _,nv,_=self.net(torch.as_tensor(ts[-1]['next_x']),h)
            adv,ret=gae([t['reward'] for t in ts],[t['value'] for t in ts],float(nv),f['terminal'])
            prepared.append((f,adv,ret))
        aa=np.concatenate([p[1] for p in prepared]);scale=aa.std()+1e-8; center=aa.mean()
        steps=[]; params_before=copy.deepcopy(self.net.state_dict())
        for epoch in range(CONFIG['epochs']):
            logps=[];old=[];means=[];behavior=[];values=[];advantages=[];returns=[]
            for f,adv,ret in prepared:
                # No-grad burn-in is re-run with CURRENT weights each epoch; BPTT covers this fragment only.
                h=replay(self.net,f['prefix'])
                for i,t in enumerate(f['transitions']):
                    mean,v,h=self.net(torch.as_tensor(t['x']),h)
                    dist=torch.distributions.Normal(mean,CONFIG['sigma'])
                    logps.append(dist.log_prob(torch.as_tensor(t['action'])).sum());old.append(t['logprob'])
                    means.append(mean);behavior.append(torch.as_tensor(t['mean']));values.append(v)
                    advantages.append((adv[i]-center)/scale);returns.append(ret[i])
            lp=torch.stack(logps); ol=torch.tensor(old,dtype=torch.float64)
            kl=((torch.stack(means)-torch.stack(behavior)).square().sum(-1)/(2*CONFIG['sigma']**2)).mean()
            if float(kl.detach())>CONFIG['target_kl']:
                steps.append(dict(epoch=epoch,early_stop=True,behavior_kl=float(kl.detach())));break
            ratio=(lp-ol).exp(); a=torch.tensor(advantages,dtype=torch.float64)
            policy=-torch.minimum(ratio*a,ratio.clamp(1-CONFIG['clip'],1+CONFIG['clip'])*a).mean()
            vf=(torch.stack(values)-torch.tensor(returns,dtype=torch.float64)).square().mean()
            loss=policy+CONFIG['value_coef']*vf;assert torch.isfinite(loss)
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=nn.utils.clip_grad_norm_(self.net.parameters(),CONFIG['max_grad_norm'],error_if_nonfinite=True)
            self.optimizer.step()
            steps.append(dict(epoch=epoch,early_stop=False,behavior_kl=float(kl.detach()),
                loss=float(loss.detach()),policy_loss=float(policy.detach()),value_loss=float(vf.detach()),gradient_norm=float(norm)))
        self.version+=1
        delta=sum(float((self.net.state_dict()[k]-v).square().sum()) for k,v in params_before.items())**.5
        return dict(version=self.version,macro_samples=len(aa),epochs=steps,parameter_delta=delta,
             entropy=8*.5*math.log(2*math.pi*math.e*CONFIG['sigma']**2),rewards=[t['reward'] for f in fragments for t in f['transitions']])
