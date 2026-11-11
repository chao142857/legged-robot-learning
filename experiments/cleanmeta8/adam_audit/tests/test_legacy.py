import sys,copy,unittest
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from legacy_load_guard import isolated_teacher_class
from v0_core import Teacher as V0
from neural_v01 import Teacher as V01, DYNAMIC_FIELDS
from protocol import CONFIG
from cpu_hash import tree_hash
class LegacyTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def case(self,base):
  cls=isolated_teacher_class(base);t=cls(np.random.default_rng(4).normal(size=(64,12)),{'test_only':True},91)
  d=np.zeros((64,15 if base==V0 else len(DYNAMIC_FIELDS)));g=np.linspace(0,.7,10);counts=np.arange(1,65)
  t.learn(t.decide(d,g,0),counts,.2)
  return t,d,g,counts
 def test_load_no_shared_snapshot_for_both_versions(self):
  for base in (V0,V01):
   t,d,g,c=self.case(base);s=t.state();h=tree_hash(s);t.load(s);t.learn(t.decide(d,g,1),c,.1);self.assertEqual(tree_hash(s),h)
 def test_rejection_restores_optimizer_both_versions(self):
  bound=CONFIG['max_teacher_step_kl']
  try:
   for base in (V0,V01):
    t,d,g,c=self.case(base);s=t.state();CONFIG['max_teacher_step_kl']=-1
    r=t.learn(t.decide(d,g,1),c,.1);self.assertTrue(r['step_rejected']);self.assertEqual(tree_hash(t.state()['optimizer']),tree_hash(s['optimizer']));self.assertEqual(tree_hash(t.state()['network']),tree_hash(s['network']))
    CONFIG['max_teacher_step_kl']=bound
  finally:CONFIG['max_teacher_step_kl']=bound
 def test_save_and_load_keep_baseline_and_state_exact(self):
  for base in (V0,V01):
   t,d,g,c=self.case(base);s=t.state();r=isolated_teacher_class(base)(np.random.default_rng(4).normal(size=(64,12)),{'test_only':True},91);r.load(s)
   t.learn(t.decide(d,g,1),c,.1);r.learn(r.decide(d,g,1),c,.1);self.assertEqual(tree_hash(t.state()),tree_hash(r.state()))
