"""Versioned, zero-initialized low-rank arm mean inside native Gaussian PPO.

The geometric basis is a frozen P1 measurement, not an inference-time solver.
All deployed inputs come from the original 210D observation. The original MLP,
legs, arm and distribution remain trainable. No action is changed after sampling.
"""
import copy
import torch
from torch import nn
from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories

SCHEMA='original210_arm_synergy_mean_v1'

class MeanModule(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.obs_normalizer=copy.deepcopy(model.obs_normalizer)
        self.mlp=copy.deepcopy(model.mlp)
        self.synergy_head=copy.deepcopy(model.synergy_head)
        for n in ('synergy_basis','synergy_q_reference','synergy_Jx','synergy_x_reference'):
            self.register_buffer(n,getattr(model,n).detach().clone())
    def forward(self,x):
        x=self.obs_normalizer(x);h=x
        for index,layer in enumerate(self.mlp):
            if index<6:h=layer(h)
        base=self.mlp[-1](h)
        target=x[:,203:204]
        velocity=(target-x[:,189:190])/.04
        local_x=self.synergy_x_reference+((x[:,66:72]-self.synergy_q_reference)*self.synergy_Jx).sum(-1,keepdim=True)
        features=torch.cat((h,(target-.45)/.02,velocity/.02,(target-local_x)/.02),-1)
        scalar=torch.tanh(self.synergy_head(features))
        return base+scalar*self.synergy_basis
    @torch.jit.export
    def reset(self):pass

class SynergyModel(MLPModel):
    def __init__(self,original,recipe):
        nn.Module.__init__(self)
        assert recipe['schema']==SCHEMA and original.obs_dim==210
        assert isinstance(original.obs_normalizer,nn.Identity),'Feature ABI requires original unnormalized210D'
        self.obs_groups=original.obs_groups;self.obs_dim=original.obs_dim
        self.obs_normalization=original.obs_normalization
        self.obs_normalizer=original.obs_normalizer;self.distribution=original.distribution;self.mlp=original.mlp
        assert self.mlp[-1].in_features==128 and self.mlp[-1].out_features==18
        device=next(original.parameters()).device
        # Do not consume the physical experiment's CPU/CUDA RNG during attachment.
        with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):
            self.synergy_head=nn.Linear(131,1,device=device)
            nn.init.zeros_(self.synergy_head.weight);nn.init.zeros_(self.synergy_head.bias)
        basis=torch.zeros(18,device=device);basis[12:]=torch.tensor(recipe['arm_basis_raw_action'],device=device)
        assert basis.abs().max()<=.480001,'P1-derived .12rad diagnostic range'
        for n,value in [('synergy_basis',basis),('synergy_q_reference',recipe['observed_arm_q_reference']),('synergy_Jx',recipe['local_FK_Jx']),('synergy_x_reference',[recipe['local_FK_x_reference']])]:
            self.register_buffer(n,torch.as_tensor(value,device=device,dtype=torch.float32))
    def mean(self,x):
        h=x
        for index,layer in enumerate(self.mlp):
            if index<6:h=layer(h)
        base=self.mlp[-1](h);target=x[:,203:204];velocity=(target-x[:,189:190])/.04
        local_x=self.synergy_x_reference+((x[:,66:72]-self.synergy_q_reference)*self.synergy_Jx).sum(-1,keepdim=True)
        features=torch.cat((h,(target-.45)/.02,velocity/.02,(target-local_x)/.02),-1)
        return base+torch.tanh(self.synergy_head(features))*self.synergy_basis
    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None and not self.is_recurrent else obs
        result=self.mean(self.get_latent(obs,masks,hidden_state))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(result);return self.distribution.sample()
            return self.distribution.deterministic_output(result)
        return result
    def as_jit(self):return MeanModule(self)
    def as_onnx(self,verbose=False):
        raise NotImplementedError('Use validated full-mean JIT export; native MLP-only ONNX would omit synergy')

def attach(alg,recipe,optimizer=True):
    old=alg.actor;actor=SynergyModel(old,recipe)
    # Existing actor/critic parameters keep object identity and all parent Adam state.
    alg.actor=actor
    if optimizer:
        assert len(alg.optimizer.param_groups)==1
        alg.optimizer.param_groups[0]['params'].extend(list(actor.synergy_head.parameters()))
    return actor

def restore_recipe(runner,path,expected_updates):
    from recipe_aplus.state import equal_tree,restore_rng
    cp=torch.load(path,map_location='cpu',weights_only=False);runner.load(str(path),strict=True,map_location=runner.device)
    assert cp['infos']['completed_updates']==expected_updates and cp['iter']==expected_updates-1
    for name in ('actor','critic','optimizer'):assert equal_tree(getattr(runner.alg,name).state_dict(),cp[name+'_state_dict'])
    runner.current_learning_iteration=expected_updates;runner.alg.learning_rate=float(cp['infos']['learning_rate'])
    alg=runner.alg;head=set(alg.actor.synergy_head.parameters());steps={}
    for label,params in [('native',[p for g in alg.optimizer.param_groups for p in g['params'] if p not in head]),('new_head',list(head))]:
        actual=[float(alg.optimizer.state[p]['step']) for p in params if p in alg.optimizer.state and 'step' in alg.optimizer.state[p]]
        expected=(expected_updates if label=='native' else expected_updates-1800)*20
        assert actual and min(actual)==max(actual)==expected,(label,actual,expected)
        steps[label]=dict(actual=min(actual),expected=expected)
    assert all(g['lr']==alg.learning_rate for g in alg.optimizer.param_groups)
    restore_rng(cp['infos'])
    return cp,dict(actor_bitwise_equal=True,critic_bitwise_equal=True,adam_bitwise_equal=True,step_groups=steps,new_head_not_falsely_aged_as_parent=True,physx_state_restored=False)

def validate_zero_attach(original,actor,obs,action_term):
    """Real observations plus synthetic inputs, exact means, native Gaussian law and JIT."""
    report={}
    with torch.no_grad(),torch.random.fork_rng(devices=[next(actor.parameters()).device]):
        for label,sample in [('actual',obs),('synthetic',{actor.obs_groups[0]:torch.linspace(-.5,.5,64*210,device=next(actor.parameters()).device).reshape(64,210)})]:
            before=original(sample);after=actor(sample);assert torch.equal(before,after),label
            def fit(value):
                if value.ndim==1:return value
                return value[:len(before)]
            offset=fit(torch.as_tensor(action_term._offset,device=before.device))
            lower=fit(action_term.target_lower);upper=fit(action_term.target_upper)
            target_before=(before*.25+offset).clamp(lower,upper)
            target_after=(after*.25+offset).clamp(lower,upper)
            assert torch.equal(target_before,target_after)
            actor(sample,stochastic_output=True);draw=actor.output_mean+.1
            lp=actor.get_output_log_prob(draw)
            expected=torch.distributions.Normal(actor.output_mean,actor.output_std).log_prob(draw).sum(-1)
            assert torch.allclose(lp,expected,atol=2e-5),label
            jit=torch.jit.script(actor.as_jit());x=torch.cat([sample[g] for g in actor.obs_groups],-1)
            assert torch.equal(jit(x),after)
            report[label]=dict(mean_max_error=float((before-after).abs().max()),target_delta_max_error=float((target_after-target_before).abs().max()),native_logprob_max_error=float((lp-expected).abs().max()),full_mean_JIT_equal=True)
    return report
