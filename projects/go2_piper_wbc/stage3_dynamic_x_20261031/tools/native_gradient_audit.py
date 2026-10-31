"""Read-only audit of real native GAE and gradients; never substitutes advantages."""
import pathlib,json,math
import torch

class NativeGradientAudit:
    def __init__(self,alg,out,manager):
        self.alg,self.out=alg,pathlib.Path(out);self.sample=None;self.rows=[];self.minibatches=[];self.rollout={}
        self.output=[(n,p) for n,p in alg.actor.named_parameters() if p.ndim==2 and p.shape[0]==18][-1]
        offset=0;self.command_slice=None
        for name,shape in zip(manager.active_terms['policy'],manager.group_obs_term_dim['policy']):
            width=math.prod(shape)
            if name=='pos_commands':self.command_slice=(offset,offset+width)
            offset+=width
        assert offset==210 and self.command_slice is not None

    def input(self,obs):
        if self.sample is None:self.sample={k:v[:64].detach().clone() for k,v in obs.items()}

    def rollout_ready(self,masks,physical_rows):
        a=self.alg.storage.advantages.detach().squeeze(-1);r=self.alg.storage.returns.detach().squeeze(-1);m=torch.stack(masks)
        assert a.shape==r.shape==m.shape and torch.isfinite(a).all() and torch.isfinite(r).all()
        result=dict(native_GAE_unmodified=True,per_minibatch_advantage_normalization=self.alg.normalize_advantage_per_mini_batch,cohorts={})
        for name,mask in [('dynamic',m),('fixed',~m)]:
            values=a[mask];result['cohorts'][name]=dict(samples=int(mask.sum()),native_GAE_mean=float(values.mean()) if len(values) else None,native_GAE_std=float(values.std()) if len(values)>1 else None,return_mean=float(r[mask].mean()) if len(values) else None)
        result['native_reward_rate_components']={k:float(torch.stack([v[k] for v in physical_rows]).mean()) for k in physical_rows[0] if k.startswith(('reward_rate/','cohort_reward_rate/'))}
        self.rollout=result

    def mean(self,sign=None):
        with torch.no_grad(),torch.random.fork_rng(devices=[self.alg.device]):
            sample=self.sample
            if sign is not None:
                sample={k:v.clone() for k,v in sample.items()};start,end=self.command_slice
                sample['policy'][:,start:end].reshape(len(sample['policy']),-1,7)[:,-1,0]=.45+sign*.02
            return self.alg.actor(sample,stochastic_output=False).detach().clone()

    def before_update(self):
        assert self.sample is not None
        self.before_mean=self.mean();self.before_sensitivity=(self.mean(1)-self.mean(-1))/.04;self.minibatches=[]

    def native_optimizer_gradient(self):
        name,p=self.output;g=p.grad
        assert g is not None and torch.isfinite(g).all()
        self.minibatches.append(dict(native_total_actor_mean_output_gradient_per_action_l2=g.detach().square().sum(-1).sqrt().cpu().tolist(),output_parameter=name,
            measurement='Native total-loss backward and native gradient clipping, immediately before native Adam.step; no per-reward PPO gradient attribution.'))

    def after_update(self,update,loss,KL):
        after=self.mean();delta=after-self.before_mean
        sensitivity=(self.mean(1)-self.mean(-1))/.04
        row=dict(update=update,native_losses=loss,native_KL=KL,cohorts_and_native_GAE=self.rollout,
            same_actual_observation_mean_action_delta_per_action_rms=delta.square().mean(0).sqrt().cpu().tolist(),
            position_only_newest_target_X_sensitivity_before_mean=self.before_sensitivity.mean(0).cpu().tolist(),
            position_only_newest_target_X_sensitivity_after_mean=sensitivity.mean(0).cpu().tolist(),
            native_minibatch_gradients=self.minibatches,advantage_replacement=False,optimizer_or_objective_reimplemented=False)
        self.rows.append(row)
        with (self.out/'native_gradient_audit.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.sample=None
