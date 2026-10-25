import sys
from pathlib import Path
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_go2_gradient_probe import ridge_fit,ridge_predict,coefficients,gram_statistics,bootstrap_counts,score_sums,pooled_moments


class SmallActor(nn.Module):
    def __init__(self):
        super().__init__();self.mean=nn.Linear(2,2);self.log_std=nn.Parameter(torch.zeros(2))
    def normal(self,x):return torch.distributions.Normal(self.mean(x),self.log_std.exp())


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_ridge_heldout_linear(self):
        torch.manual_seed(44);x=torch.randn(200,4);w=torch.randn(4,3);y=x@w+2
        f=ridge_fit(x,y,1e-8);z=torch.randn(40,4)
        torch.testing.assert_close(ridge_predict(f,z),z@w+2,rtol=1e-5,atol=1e-5)
    def test_score_sums_match_direct_loss_and_no_rng(self):
        torch.manual_seed(45);m=SmallActor();x=torch.randn(8,2);d=m.normal(x);a=d.sample();old=d.log_prob(a).sum(-1).detach()
        adv=torch.randn(8,3);rng=torch.get_rng_state().clone();got,_=score_sums(m,x,a,old,adv)
        for i in range(3):
            lp=m.normal(x).log_prob(a).sum(-1);loss=-(adv[:,i]*(lp-old).exp()).sum()
            g=torch.autograd.grad(loss,list(m.parameters()))
            torch.testing.assert_close(got[i],torch.cat([v.flatten() for v in g]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_gram_bootstrap_equals_explicit_resampled_gradient(self):
        torch.manual_seed(46)
        # independent block basis; returns/score correlation is irrelevant to identity.
        g=torch.randn(3,4,7,dtype=torch.float64)
        av=[torch.randn(n,3,dtype=torch.float64) for n in [4,5,6]]
        n=torch.tensor([len(x) for x in av]);s=torch.stack([x.sum(0) for x in av]);cr=torch.stack([x.T@x for x in av])
        counts=torch.tensor([[1.,2.,0.],[0.,1.,2.]])
        gram=g.flatten(0,1)@g.flatten(0,1).T
        for mode in ['raw','shared_total_std','per_objective_std']:
            c=coefficients(counts,n,s,cr,mode);dot,_,_=gram_statistics(gram,c)
            for row in range(2):
                sample=torch.cat([av[i].repeat(int(counts[row,i]),1) for i in range(3) if counts[row,i]>0])
                explicit=torch.einsum('b,bip->ip',counts[row].double(),g)/len(sample)
                expected=explicit[:3]
                if mode!='raw':
                    expected=expected-sample.mean(0)[:,None]*explicit[3]
                    scale=sample.sum(-1).std() if mode=='shared_total_std' else sample.std(0)[:,None]
                    expected=expected/(scale+1e-8)
                torch.testing.assert_close(dot[row],expected@expected.T)
    def test_subset_uses_fixed_global_normalizer(self):
        torch.manual_seed(49);g=torch.randn(2,4,7,dtype=torch.float64)
        av=[torch.randn(8,3,dtype=torch.float64)+i*3 for i in range(2)]
        n=torch.tensor([8,8]);ss=torch.stack([x.sum(0) for x in av]);cc=torch.stack([x.T@x for x in av])
        ref=pooled_moments(n,ss,cc);sample=torch.cat(av)
        for mode in ['shared_total_std','per_objective_std']:
            c=coefficients(torch.eye(2),n,ss,cc,mode,ref)
            actual=torch.einsum('rip,pd->rid',c,g.flatten(0,1))
            scale=sample.sum(-1).std() if mode=='shared_total_std' else sample.std(0)[:,None]
            for i in range(2):
                expected=(g[i,:3]-sample.mean(0)[:,None]*g[i,3])/8/(scale+1e-8)
                torch.testing.assert_close(actual[i],expected)

    def test_bootstrap_keeps_hierarchy_and_reproducible(self):
        seeds=[1]*4+[2]*4+[3]*4;reps=[0,1,2,3]*3
        c=bootstrap_counts(seeds,reps,20);torch.testing.assert_close(c,bootstrap_counts(seeds,reps,20))
        torch.testing.assert_close(c.sum(1),torch.full((20,),12.,dtype=torch.float64))
        self.assertTrue(torch.all(c.reshape(20,3,4).sum(-1).remainder(4)==0))

if __name__=='__main__':unittest.main(verbosity=2)
