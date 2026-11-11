import copy,unittest
import numpy as np
import torch
from cl1_core import Learner,CONFIG,gae,replay,observation
from cpu_hash import tree_hash
class Tests(unittest.TestCase):
 def fragments(self,l):
  fs=[]
  for seed in [3,9]:
   rng=np.random.default_rng(seed);hist=[];ts=[]
   for i in range(2):
    x=rng.normal(size=CONFIG['input_dim']);a=l.act(x,hist)
    ts.append(dict(**a,x=x,next_x=rng.normal(size=CONFIG['input_dim']),reward=float(.1*(i+1)*seed)))
    hist.append(x)
   fs.append(dict(prefix=[],transitions=ts,terminal=False))
  return fs
 def test_gradient_and_macro_samples(self):
  l=Learner();f=self.fragments(l);r=l.update(f)
  self.assertEqual(r['macro_samples'],4);self.assertGreater(r['parameter_delta'],0)
  self.assertTrue(any(e.get('gradient_norm',0)>0 for e in r['epochs']))
 def test_checkpoint_full_restore(self):
  l=Learner();l.update(self.fragments(l));s=l.state();b=Learner();b.load(s)
  f=self.fragments(l);r1=l.update(f);r2=b.update(copy.deepcopy(f))
  self.assertEqual(tree_hash(l.net.state_dict()),tree_hash(b.net.state_dict()))
  self.assertEqual(tree_hash(l.optimizer.state_dict()),tree_hash(b.optimizer.state_dict()))
  self.assertEqual(r1,r2)
 def test_load_does_not_alias_adam(self):
  l=Learner();l.update(self.fragments(l));s=l.state();h=tree_hash(s);b=Learner();b.load(s);b.update(self.fragments(b))
  self.assertEqual(h,tree_hash(s))
 def test_exception_restores_complete_state(self):
  l=Learner();l.update(self.fragments(l));f=self.fragments(l);s=l.state();original=l.optimizer.step
  def bad(): original();raise RuntimeError('injected post-step fault')
  l.optimizer.step=bad
  with self.assertRaises(RuntimeError):l.update(f)
  self.assertEqual(tree_hash(s),tree_hash(l.state()))
 def test_terminal_vs_bootstrap_and_isolation(self):
  a,_=gae([1,1],[0,0],5,False);b,_=gae([1,1],[0,0],5,True)
  self.assertGreater(a[-1],b[-1]);self.assertEqual(b[-1],1)
  l=Learner();f=self.fragments(l)
  h1=replay(l.net,[f[0]['transitions'][0]['x']]);h2=replay(l.net,[f[1]['transitions'][0]['x']])
  self.assertFalse(torch.equal(h1,h2));self.assertTrue(torch.equal(h1,replay(l.net,[f[0]['transitions'][0]['x']])))
 def test_behavior_logprob(self):
  l=Learner();x=np.ones(CONFIG['input_dim']);a=l.act(x,[])
  dist=torch.distributions.Normal(torch.as_tensor(a['mean']),CONFIG['sigma'])
  self.assertAlmostEqual(a['logprob'],float(dist.log_prob(torch.as_tensor(a['action'])).sum()),12)
 def test_no_update_when_behavior_kl_excessive(self):
  l=Learner();f=self.fragments(l)
  for q in f:
   for t in q['transitions']:t['mean']+=10
  s=tree_hash(l.optimizer.state_dict());r=l.update(f)
  self.assertTrue(r['epochs'][0]['early_stop']);self.assertEqual(r['parameter_delta'],0)
  self.assertEqual(s,tree_hash(l.optimizer.state_dict()))
 def test_fixed_parameters_state_response_and_rng_restore(self):
  l=Learner();l.update(self.fragments(l));snap=l.state();b=Learner();b.load(snap)
  x=np.zeros(CONFIG['input_dim']);a=l.act(x,[]);c=b.act(x,[])
  np.testing.assert_array_equal(a['action'],c['action'])
  with torch.no_grad():
   m1,_,h1=l.net(torch.as_tensor(x),l.net.zero())
   m2,_,h2=l.net(torch.ones(CONFIG['input_dim'],dtype=torch.float64),l.net.zero())
  self.assertFalse(torch.equal(h1,h2));self.assertFalse(torch.equal(m1,m2))
 def test_decoder_probability(self):
  from teacher_v02 import Teacher,FeatureEncoder
  rng=np.random.default_rng(8);e=FeatureEncoder(rng.normal(size=(64,20)))
  d=object.__new__(Teacher);d.encoder=e;d.config=e.config
  q=d.probability_from_preference(e.command_basis,np.ones(8))
  self.assertGreaterEqual(q.min(),.2/64);self.assertAlmostEqual(q.sum(),1,12)
  np.testing.assert_allclose(d.probability_from_preference(e.command_basis,np.zeros(8)),np.full(64,1/64))
if __name__=='__main__': unittest.main(verbosity=2)
