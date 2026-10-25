"""Analytic CPU tests; synthetic data never count as robot gradient evidence."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_go2_piper_gradients import (FrozenMLP,component_gae,complete_mc,
    normalize_components,policy_loss,gradient,pair_stats,snapshot_transition,analyze,sha256,validate_bundle)


def state(inp,out):
    net=nn.Module(); net.mlp=nn.Sequential(nn.Linear(inp,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),
        nn.Linear(256,128),nn.ELU(),nn.Linear(128,out))
    net.distribution=nn.Module();net.distribution.log_std_param=nn.Parameter(torch.zeros(out))
    return net.state_dict()


def fixture(path):
    torch.manual_seed(12)
    torch.save(dict(actor_state_dict=state(210,18),critic_state_dict=state(102,1)),path)
    c=torch.load(path,weights_only=True); actor=FrozenMLP(c['actor_state_dict']); critic=FrozenMLP(c['critic_state_dict'])
    obs=torch.randn(12,210); co=torch.randn(12,102)
    with torch.no_grad():
        d=actor.normal(obs);act=d.sample()
        # Synthetic deliberately opposed return estimators, not physical rewards.
        x=torch.linspace(-1,1,12);adv=torch.stack((x,-x,x.square()),-1)
        b=dict(actor_obs=obs,critic_obs=co,actions=act,old_mu=d.mean.clone(),old_std=d.stddev.clone(),
            old_logp=d.log_prob(act).sum(-1),component_advantages=adv,task_id=torch.arange(12)%2,
            base_level=torch.zeros(12,dtype=torch.long),ee_level=torch.ones(12,dtype=torch.long),
            old_values=critic(co).squeeze(-1),total_returns=torch.randn(12))
    b['metadata']=dict(schema_version=1,behavior_checkpoint_sha256=sha256(path),rollout_id='SYNTHETIC',
        seed=12,estimator='complete_mc_finite_horizon',estimator_bias='SYNTHETIC TEST ONLY',on_policy_stochastic=True,
        control_structure='B0_18D',reward_group_definition='synthetic',reward_unit='weighted_dt_once',gamma=.99,
        normalization_scope='whole_rollout')
    return b


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def test_pair_geometry_and_zero(self):
        x=torch.tensor([1.,0.]);y=-x
        self.assertEqual(pair_stats(x,y)['cosine'],-1.)
        self.assertTrue(pair_stats(x,y)['conflict'])
        self.assertIsNone(pair_stats(x,torch.zeros(2))['cosine'])
        self.assertEqual(pair_stats(x,torch.tensor([0.,1.]))['dot_product'],0.)
    def test_mc_exact_and_incomplete_refusal(self):
        r=torch.tensor([[[1.]],[[2.]],[[3.]]]);valid=torch.ones(3,1,dtype=torch.bool)
        ends=torch.tensor([[False],[False],[True]])
        torch.testing.assert_close(complete_mc(r,ends,valid,.5)[:,0,0],torch.tensor([2.75,3.5,3.]))
        with self.assertRaises(ValueError):complete_mc(r,torch.zeros_like(ends),valid)
    def test_mc_reset_and_padding(self):
        r=torch.ones(4,1,1);v=torch.tensor([[True],[True],[True],[False]])
        ends=torch.tensor([[True],[False],[True],[False]])
        torch.testing.assert_close(complete_mc(r,ends,v,1.)[:,0,0],torch.tensor([1.,2.,1.,0.]))
    def test_gae_terminal_timeout_trace(self):
        r=torch.ones(2,1,2);v=torch.zeros_like(r);nv=torch.full_like(r,10.)
        term=torch.tensor([[False],[True]]);ends=torch.ones_like(term)
        a=component_gae(r,v,nv,term,ends,.9,.95)
        torch.testing.assert_close(a[0],torch.full((1,2),10.))
        torch.testing.assert_close(a[1],torch.ones(1,2))
        with self.assertRaises(ValueError):component_gae(r,v[:,:,:1],nv,term,ends)
    def test_gae_linearity(self):
        torch.manual_seed(5);r=torch.randn(5,2,3);v=torch.randn_like(r);nv=torch.randn_like(r)
        e=torch.zeros(5,2,dtype=torch.bool);e[-1]=True
        a=component_gae(r,v,nv,e,e)
        total=component_gae(r.sum(-1,keepdim=True),v.sum(-1,keepdim=True),nv.sum(-1,keepdim=True),e,e)
        torch.testing.assert_close(a.sum(-1),total.squeeze(-1))
    def test_normalization_additivity_and_positive_scaling(self):
        torch.manual_seed(6);a=torch.randn(10,3)
        torch.testing.assert_close(normalize_components(a,'shared_total_std').sum(-1),
            (a.sum(-1)-a.sum(-1).mean())/(a.sum(-1).std()+1e-8))
        torch.testing.assert_close(normalize_components(a*10,'per_objective_std'),normalize_components(a,'per_objective_std'))
    def test_clip_common_mask_additive_separate_not(self):
        lp=torch.tensor([.4],requires_grad=True);old=torch.zeros(1)
        a=torch.tensor([2.]);b=torch.tensor([-1.])
        aggregate=torch.autograd.grad(policy_loss(lp,old,a+b,'separate_clip'),lp)[0]
        sums=[]
        for mode in ['total_mask','separate_clip']:
            gs=[]
            for adv in [a,b]:
                gs.append(torch.autograd.grad(policy_loss(lp,old,adv,mode,a+b),lp)[0])
            sums.append(sum(gs))
        torch.testing.assert_close(aggregate,sums[0]);self.assertNotEqual(float(aggregate),float(sums[1]))
    def test_loss_finite_difference(self):
        x=torch.tensor([.1,-.1],dtype=torch.float64,requires_grad=True);old=torch.zeros(2,dtype=torch.float64)
        adv=torch.tensor([2.,-3.],dtype=torch.float64)
        g=torch.autograd.grad(policy_loss(x,old,adv),x)[0]
        for i in range(2):
            d=torch.zeros_like(x);d[i]=1e-6
            fd=(policy_loss(x+d,old,adv)-policy_loss(x-d,old,adv))/2e-6
            self.assertAlmostEqual(float(g[i]),float(fd),places=7)
    def test_unused_critic_std(self):
        c=FrozenMLP(state(102,1));g,unused=gradient(c(torch.zeros(2,102)).square().mean(),c)
        self.assertEqual(unused,['distribution.log_std_param'])
        self.assertEqual(float(g[unused[0]].norm()),0.)
    def test_snapshot_alias_and_rng(self):
        x=torch.randn(2,3);rng=torch.get_rng_state().clone()
        snap=snapshot_transition(*([x]*9));x.fill_(100)
        self.assertFalse(torch.equal(snap['actions'],x));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'cp.pt';b=fixture(cp)
            b['metadata']['on_policy_stochastic']=False
            with self.assertRaises(ValueError):validate_bundle(b,sha256(cp))
            b['metadata']['on_policy_stochastic']=True;b['metadata']['estimator']='reward_split_single_critic'
            with self.assertRaises(ValueError):validate_bundle(b,sha256(cp))
    def test_end_to_end_no_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'cp.pt';b=fixture(cp);before=sha256(cp);adv=b['component_advantages'].clone()
            result=analyze(cp,b,minibatch_size=4)
            self.assertEqual(sha256(cp),before);torch.testing.assert_close(b['component_advantages'],adv)
            self.assertEqual(result['optimizer_steps'],0);self.assertEqual(result['device'],'cpu')
            self.assertEqual(result['critic_unused_parameters'],['distribution.log_std_param'])
            rr=[r for r in result['actor_pairs'] if r['kind']=='actor' and r['normalization']=='raw' and
                r['scope']=='trunk' and r['pair']=='locomotion/manipulation']
            self.assertTrue(all(r['cosine'] < -.9999 for r in rr))
            self.assertTrue(all(x['clip_fraction']==0 for x in result['clipping']))
            json.dumps(result,allow_nan=False)
            b['actions'][0]+=1
            with self.assertRaises(ValueError):analyze(cp,b,minibatch_size=4)
    def test_raw_builder_additivity_and_mc(self):
        from build_go2_gradient_bundle import build
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'cp.pt'; b=fixture(cp)
            raw={k:v.reshape(3,4,*v.shape[1:]) for k,v in b.items() if isinstance(v,torch.Tensor) and k!='component_advantages'}
            raw['metadata']=dict(b['metadata'],time_limit_objective='finite_horizon_terminal')
            raw['component_rewards']=torch.ones(3,4,3)
            raw['total_reward']=torch.full((3,4),3.)
            raw['valid']=torch.ones(3,4,dtype=torch.bool)
            raw['episode_end']=torch.zeros(3,4,dtype=torch.bool);raw['episode_end'][-1]=True
            built=build(raw,cp)
            torch.testing.assert_close(built['component_advantages'][:4],torch.full((4,3),1+.99+.99**2))
            raw['total_reward'][0,0]=999
            with self.assertRaises(ValueError):build(raw,cp)
    def test_crossfit_leakage_refused(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'cp.pt';b=fixture(cp)
            b['metadata'].update(estimator='crossfit_component_gae',value_fit_rollout_ids=['SYNTHETIC'],
                heldout_value_metrics={},timeout_convention='terminal_next_value')
            with self.assertRaises(ValueError):validate_bundle(b,sha256(cp))
    def test_candidate_provenance_refused(self):
        with tempfile.TemporaryDirectory() as td:
            cp=Path(td)/'cp.pt';b=fixture(cp)
            with self.assertRaises(ValueError):analyze(cp,b,candidate=cp)
    def test_native_rsl_forward_and_gradient_parity(self):
        try:
            from rsl_rl.models import MLPModel
            from tensordict import TensorDict
        except ImportError: self.skipTest('Native rsl_rl not installed')
        torch.manual_seed(9)
        obs=torch.randn(7,210);td=TensorDict({'policy':obs},batch_size=[7])
        native=MLPModel(td,{'actor':['policy']},'actor',18,[512,256,128],'elu',False,
            dict(class_name='GaussianDistribution',init_std=1.,std_type='log'))
        adapter=FrozenMLP(native.state_dict())
        torch.testing.assert_close(native(td),adapter(obs))
        native(td,stochastic_output=True);actions=native.output_mean.detach()+.2
        lp=native.get_output_log_prob(actions);aa=torch.linspace(-2,2,7)
        old=lp.detach();gn,_=gradient(policy_loss(lp,old,aa),native)
        ga,_=gradient(policy_loss(adapter.normal(obs).log_prob(actions).sum(-1),old,aa),adapter)
        for k in gn:torch.testing.assert_close(gn[k],ga[k])
    def test_existing_checkpoint_native_cpu_parity(self):
        root=os.environ.get('B0_CHECKPOINT_ROOT')
        if not root:self.skipTest('Set B0_CHECKPOINT_ROOT for read-only real checkpoint parity')
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict
        for step in (0,100,3000):
            p=Path(root)/f'cp_{step}.pt';before=sha256(p)
            c=torch.load(p,map_location='cpu',weights_only=False)
            for role,inp,out,group in [('actor',210,18,'policy'),('critic',102,1,'critic')]:
                x=torch.zeros(4,inp);td=TensorDict({group:x},batch_size=[4])
                native=MLPModel(td,{role:[group]},role,out,[512,256,128],'elu',False,
                    dict(class_name='GaussianDistribution',init_std=1.,std_type='log'))
                native.load_state_dict(c[role+'_state_dict'])
                adapter=FrozenMLP(c[role+'_state_dict'])
                torch.testing.assert_close(native(td),adapter(x),rtol=0,atol=0)
            self.assertEqual(sha256(p),before)

if __name__=='__main__':unittest.main(verbosity=2)
