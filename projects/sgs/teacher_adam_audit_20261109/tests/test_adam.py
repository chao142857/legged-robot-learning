"""Independent analytic Adam oracle and integration regressions, synthetic only."""
import copy, io, sys, unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from teacher_fixed import Teacher
from teacher_v02 import Teacher as Legacy
from cpu_hash import tree_hash

class AdamTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1)
  self.x=np.random.default_rng(93).normal(size=(64,12));self.d=np.zeros((64,30));self.g=np.linspace(-.7,.8,10)
 def make(self,cls=Teacher):return cls(self.x,{'isolated_synthetic':True})
 def decision(self,t):return t.sample_pair(self.d,self.g,t.updates)
 def warm(self):
  t=self.make();t.config['max_preference_step_kl']=1e6
  t.learn_pair(self.decision(t),.3,.31,.34,.28)
  return t
 def test_checkpoint_input_is_not_aliased_by_load(self):
  t=self.warm();s=t.state();h=tree_hash(s);c=self.make();c.config=copy.deepcopy(t.config);c.load(s)
  c.learn_pair(self.decision(c),.3,.31,.35,.26)
  self.assertEqual(tree_hash(s),h)
  for k,v in c.optimizer.state_dict()['state'].items():
   for field in ('step','exp_avg','exp_avg_sq'):self.assertNotEqual(v[field].data_ptr(),s['optimizer']['state'][k][field].data_ptr())
 def test_legacy_checkpoint_alias_reproduces(self):
  t=self.warm();s=t.state();h=tree_hash(s);c=self.make(Legacy);c.config=copy.deepcopy(t.config);c.load(s)
  c.learn_pair(self.decision(c),.3,.31,.35,.26)
  # A normal old learn_pair first loads a separate deep snapshot: the input is
  # unsafe in general, but this path may detach it before stepping. Probe raw step.
  c.load(s);c.optimizer.step();self.assertNotEqual(tree_hash(s),h)
 def test_no_backtrack_identical_legacy_and_fixed(self):
  for warm in (False,True):
   a=self.warm() if warm else self.make();a.config['max_preference_step_kl']=1e6
   b=self.make(Legacy);b.config=copy.deepcopy(a.config);b.load(a.state());d=self.decision(a);b.load(d['teacher'])
   ra=a.learn_pair(copy.deepcopy(d),.3,.31,.35,.26);rb=b.learn_pair(copy.deepcopy(d),.3,.31,.35,.26)
   self.assertEqual(len(ra['step_backtracking']),1);self.assertEqual(tree_hash(a.state()),tree_hash(b.state()))
   np.testing.assert_array_equal(ra['gradient_vector'],rb['gradient_vector'])
 def test_every_retry_same_state_and_single_analytic_adam_step(self):
  t=self.warm();t.config['max_preference_step_kl']=.0001;d=self.decision(t);before=t.state();starts=[];grads=[]
  native=t.optimizer.step
  def traced(*a,**kw):
   starts.append(copy.deepcopy(t.optimizer.state_dict()));grads.append([p.grad.clone() for p in t.net.parameters()]);return native(*a,**kw)
  t.optimizer.step=traced;r=t.learn_pair(d,.3,.31,.35,.26)
  self.assertFalse(r['step_rejected']);self.assertGreater(len(starts),2)
  for s in starts:
   self.assertEqual(tree_hash(s['state']),tree_hash(before['optimizer']['state']))
   for a,b in zip(s['param_groups'],before['optimizer']['param_groups']):
    self.assertEqual({k:v for k,v in a.items() if k!='lr'},{k:v for k,v in b.items() if k!='lr'})
  after=t.optimizer.state_dict();group=before['optimizer']['param_groups'][0];b1,b2=group['betas'];lr=group['lr']*r['accepted_learning_rate_multiplier']
  for j,(key,old) in enumerate(before['optimizer']['state'].items()):
   g=grads[0][j];m=b1*old['exp_avg']+(1-b1)*g;v=b2*old['exp_avg_sq']+(1-b2)*g*g;n=int(old['step'])+1
   now=after['state'][key];self.assertEqual(int(now['step']),n)
   torch.testing.assert_close(now['exp_avg'],m,atol=1e-16,rtol=1e-14);torch.testing.assert_close(now['exp_avg_sq'],v,atol=1e-16,rtol=1e-14)
   pn=list(before['network'])[j];expected=before['network'][pn]-lr*(m/(1-b1**n))/(torch.sqrt(v/(1-b2**n))+group['eps'])
   torch.testing.assert_close(t.net.state_dict()[pn],expected,atol=1e-15,rtol=1e-13)
  self.assertEqual(after['param_groups'],before['optimizer']['param_groups'])
 def test_all_rejected_restores_all_state_and_gradients(self):
  for warm in (False,True):
   t=self.warm() if warm else self.make();t.config['max_preference_step_kl']=1e-30
   for p in t.net.parameters():p.grad=torch.ones_like(p)*.123
   d=self.decision(t);before=t.state();grads=[p.grad.clone() for p in t.net.parameters()]
   r=t.learn_pair(d,.3,.31,.35,.26);self.assertTrue(r['step_rejected']);self.assertEqual(len(r['step_backtracking']),9)
   after=t.state()
   for k in ('network','optimizer','encoder','rng','config','binding'):self.assertEqual(tree_hash(after[k]),tree_hash(before[k]))
   for p,g in zip(t.net.parameters(),grads):self.assertTrue(torch.equal(p.grad,g))
   self.assertEqual(t.updates,before['updates']+1);self.assertIn(str(d['version']),t.used_versions)
   with self.assertRaises(AssertionError):t.learn_pair(d,.3,.31,.35,.26)
 def test_exception_after_optimizer_step_is_atomic(self):
  t=self.warm();d=self.decision(t);before=t.state();grads=[p.grad.clone() for p in t.net.parameters()];native=t.optimizer.step
  def bad():native();raise RuntimeError('injected post-step failure')
  t.optimizer.step=bad
  with self.assertRaises(RuntimeError):t.learn_pair(d,.3,.31,.35,.26)
  self.assertEqual(tree_hash(t.state()),tree_hash(before))
  for p,g in zip(t.net.parameters(),grads):self.assertTrue(torch.equal(p.grad,g))
 def test_zero_signal_has_no_step(self):
  t=self.warm();d=self.decision(t);s=t.state();r=t.learn_pair(d,.3,.31,.32,.32)
  self.assertTrue(r['zero_signal']);self.assertEqual(tree_hash(s['optimizer']),tree_hash(t.state()['optimizer']))
 def test_save_resume_same_decision_and_next_result(self):
  a=self.warm();a.config['max_preference_step_kl']=.001;b=self.make();b.config=copy.deepcopy(a.config)
  buf=io.BytesIO();torch.save(a.state(),buf);buf.seek(0);s=torch.load(buf,weights_only=False);b.load(s);h=tree_hash(s)
  da=self.decision(a);db=self.decision(b);self.assertEqual(tree_hash(da),tree_hash(db))
  a.learn_pair(da,.3,.31,.35,.26);b.learn_pair(db,.3,.31,.35,.26)
  self.assertEqual(tree_hash(a.state()),tree_hash(b.state()));self.assertEqual(tree_hash(s),h)
 def test_historical_adam_step_not_rewritten(self):
  t=self.warm();s=t.state()
  for v in s['optimizer']['state'].values():v['step'].fill_(15)
  s['updates']=8;t.load(s);d=self.decision(t);t.learn_pair(d,.3,.31,.35,.26)
  self.assertEqual({int(v['step']) for v in t.state()['optimizer']['state'].values()},{16})
 def test_old_multiple_backtracks_demonstrably_differ(self):
  t=self.warm();t.config['max_preference_step_kl']=.0001;d=self.decision(t);s=t.state()
  b=self.make(Legacy);b.config=copy.deepcopy(t.config);b.load(copy.deepcopy(s));r=b.learn_pair(copy.deepcopy(d),.3,.31,.35,.26)
  t.learn_pair(d,.3,.31,.35,.26)
  self.assertGreater(len(r['step_backtracking']),1);self.assertNotEqual(tree_hash(t.state()['optimizer']),tree_hash(b.state()['optimizer']))
 def test_amsgrad_and_multiple_param_groups_rejection(self):
  t=self.make();ps=list(t.net.parameters());t.optimizer=torch.optim.Adam([{'params':ps[:2],'lr':.01},{'params':ps[2:],'lr':.02}],amsgrad=True)
  t.config['max_preference_step_kl']=1e6;t.learn_pair(self.decision(t),.3,.31,.35,.26)
  t.config['max_preference_step_kl']=1e-30;d=self.decision(t);s=t.state();r=t.learn_pair(d,.3,.31,.35,.26)
  self.assertTrue(r['step_rejected']);self.assertEqual(tree_hash(t.state()['optimizer']),tree_hash(s['optimizer']))
  for v in t.optimizer.state_dict()['state'].values():self.assertIn('max_exp_avg_sq',v)
 def test_nonfinite_gradient_leaves_full_state_unchanged(self):
  t=self.warm();d=self.decision(t);s=t.state();p=next(t.net.parameters());hook=p.register_hook(lambda g:g*float('nan'))
  with self.assertRaises(FloatingPointError):t.learn_pair(d,.3,.31,.35,.26)
  hook.remove();self.assertEqual(tree_hash(t.state()),tree_hash(s))
if __name__=='__main__':unittest.main(verbosity=2)
