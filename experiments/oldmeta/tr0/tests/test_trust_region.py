"""Synthetic CPU-only tests; these inputs never enter an experiment dataset."""
import copy,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
archive=ROOT.parent/'nsgs_v02_20261106/code'
if archive.exists():sys.path.append(str(archive))
from teacher_v02 import Teacher
from trust_region import construct_teachers,frozen_input,panel_probabilities,distribution_change,clone
from cpu_hash import tree_hash

class TrustRegionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  cls.teacher=Teacher(np.random.default_rng(51).normal(size=(64,12)),{'pool':'isolated-synthetic-not-training'})
  cls.context=dict(dynamic=np.zeros((64,30)),progress=np.zeros(64),progress_observed=np.zeros(64),partial=np.zeros((64,8)))
  cls.g=np.linspace(-.7,.8,10)
  for version in range(8):
   d=cls.teacher.sample_pair(cls.context['dynamic'],cls.g,version,cls.context['progress'],cls.context['progress_observed'],cls.context['partial'])
   cls.teacher.learn_pair(d,.3,.3,.34,.28)
 def case(self):
  t=clone(self.teacher)
  d=t.sample_pair(self.context['dynamic'],self.g,8,self.context['progress'],self.context['progress_observed'],self.context['partial'])
  panel=[frozen_input(t,'isolated-%d'%i,self.context,self.g+.05*i,{'synthetic':True}) for i in range(3)]
  return t,d,panel
 def test_complete_mixed_probability_valid_and_same_parameters_tv_zero(self):
  t,d,p=self.case();q=panel_probabilities(t,p)
  self.assertTrue(np.isfinite(q).all());self.assertTrue((q>=.2/64).all());np.testing.assert_allclose(q.sum(1),1,rtol=0,atol=1e-12)
  for x in q:self.assertEqual(distribution_change(x,x)['tv'],0.)
 def test_nonbinding_is_bit_identical_including_adam_rng(self):
  t,d,p=self.case();a,b,c,r=construct_teachers(t,d,[.3,.31,.35,.27],p,tv_limit=1.)
  self.assertFalse(r['constraint_intervention']);self.assertEqual(tree_hash(b.state()),tree_hash(c.state()))
  self.assertEqual(tree_hash(a.rng.bit_generator.state),tree_hash(b.rng.bit_generator.state))
  self.assertEqual(tree_hash(a.optimizer.state_dict()),tree_hash(d['teacher']['optimizer']))
 def test_binding_backtracks_same_adam_once_and_no_amplification(self):
  t,d,p=self.case();_,_,_,raw=construct_teachers(t,d,[.3,.31,.35,.27],p,tv_limit=1.)
  limit=raw['original_max_tv']*.49;self.assertGreater(limit,1e-12)
  _,b,c,r=construct_teachers(t,d,[.3,.31,.35,.27],p,tv_limit=limit)
  self.assertTrue(r['constraint_intervention']);self.assertTrue(r['accepted']);self.assertLessEqual(r['max_accepted_tv'],limit)
  self.assertLessEqual(r['accepted_learning_rate_multiplier'],raw['original_step']['accepted_learning_rate_multiplier'])
  self.assertEqual(tree_hash(b.optimizer.state_dict()['state']),tree_hash(c.optimizer.state_dict()['state']))
  self.assertEqual(len({x['starting_optimizer_sha256'] for x in r['attempts']}),1)
  self.assertLessEqual(r['attempts'][-1]['preference_kl'],.05)
  self.assertEqual([g['lr'] for g in c.optimizer.param_groups],[g['lr'] for g in t.optimizer.param_groups])
 def test_full_rejection_restores_parameters_adam_encoder_rng(self):
  t,d,p=self.case();_,_,c,r=construct_teachers(t,d,[.3,.31,.35,.27],p,tv_limit=1e-15)
  self.assertTrue(r['constraint_intervention']);self.assertFalse(r['accepted']);self.assertEqual(len(r['attempts']),9)
  state=c.state()
  for key in ('network','optimizer','encoder','rng'):self.assertEqual(tree_hash(state[key]),tree_hash(d['teacher'][key]))
  self.assertEqual(c.updates,9);self.assertIn('8',c.used_versions)
 def test_checkpoint_restore_and_private_rng_exact(self):
  t,d,p=self.case();_,_,c,_=construct_teachers(t,d,[.3,.31,.35,.27],p)
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'teacher.pt';torch.save({'teacher':c.state()},path)
   other=clone(t,torch.load(path,map_location='cpu',weights_only=False)['teacher'])
   self.assertEqual(tree_hash(other.state()),tree_hash(c.state()))
   np.testing.assert_array_equal(other.rng.normal(size=8),c.rng.normal(size=8))
 def test_input_change_is_not_a_parameter_update(self):
  t,d,p=self.case();q=panel_probabilities(t,p)
  for a,b in zip(q,panel_probabilities(clone(t),p)):self.assertEqual(distribution_change(a,b)['tv'],0)
  damaged=copy.deepcopy(p);damaged[0]['global_state'][0]+=.1
  with self.assertRaises(AssertionError):panel_probabilities(t,damaged)
 def test_zero_real_contrast_does_not_create_a_teacher_step(self):
  t,d,p=self.case();_,b,c,r=construct_teachers(t,d,[.3,.31,.32,.32],p)
  self.assertFalse(r['constraint_intervention']);self.assertEqual(r['original_max_tv'],0)
  self.assertEqual(tree_hash(b.state()),tree_hash(c.state()))
 def test_nominal_two_percent_limit_and_source_immutability(self):
  t,d,p=self.case();before=tree_hash(t.state());ids=[tree_hash(h) for h in p]
  with tempfile.TemporaryDirectory() as folder:
   _,_,_,r=construct_teachers(t,d,[.3,.31,.35,.27],p,evidence_dir=Path(folder)/'candidates')
   self.assertTrue((Path(folder)/'candidates/old_and_original_full_probabilities.npz').exists())
   self.assertTrue((Path(folder)/'candidates/accepted_full_probabilities.npz').exists())
  self.assertLessEqual(r['max_accepted_tv'],.02);self.assertEqual(tree_hash(t.state()),before);self.assertEqual(ids,[tree_hash(h) for h in p])
 def constrained_case(self,kl_limit):
  t=clone(self.teacher);t.config['max_preference_step_kl']=kl_limit
  d=t.sample_pair(self.context['dynamic'],self.g,8,self.context['progress'],self.context['progress_observed'],self.context['partial'])
  p=[frozen_input(t,'isolated-backtrack',self.context,self.g,{'synthetic':True})]
  return t,d,p
 def test_original_kl_backtracks_do_not_alias_or_advance_adam_twice(self):
  t,d,p=self.constrained_case(.001)
  _,b,c,r=construct_teachers(t,d,[.3,.31,.35,.27],p,tv_limit=1.)
  self.assertGreater(len(r['original_step']['step_backtracking']),1)
  self.assertFalse(r['original_step']['step_rejected'])
  old=d['teacher']['optimizer']['state'];new=b.optimizer.state_dict()['state']
  for k,v in old.items():self.assertEqual(int(new[k]['step']),int(v['step'])+1)
  self.assertEqual(tree_hash(b.state()),tree_hash(c.state()))
  self.assertNotIn('load_state_dict',b.optimizer.__dict__)
 def test_original_full_rejection_restores_the_original_adam_snapshot(self):
  t,d,p=self.constrained_case(1e-20)
  _,b,c,r=construct_teachers(t,d,[.3,.31,.35,.27],p,tv_limit=1.)
  self.assertTrue(r['original_step']['step_rejected'])
  self.assertEqual(tree_hash(b.optimizer.state_dict()),tree_hash(d['teacher']['optimizer']))
  self.assertEqual(tree_hash(b.net.state_dict()),tree_hash(d['teacher']['network']))
  self.assertEqual(tree_hash(b.state()),tree_hash(c.state()))

if __name__=='__main__':unittest.main()
