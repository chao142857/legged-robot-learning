import copy,unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from teacher_v02 import Teacher
from cpu_hash import tree_hash
class ContinualTests(unittest.TestCase):
 def test_inherited_optimizer_rng_versions_and_new_decision(self):
  x=np.random.default_rng(1).normal(size=(32,12));t=Teacher(x,{'pool':'synthetic'})
  d=np.zeros((32,30));g=np.zeros(10)
  for v in range(8):
   q=t.sample_pair(d,g,v);t.learn_pair(q,.3,.3,.31,.29)
  saved=copy.deepcopy(t.state());fresh=Teacher(x,{'pool':'synthetic'});fresh.load(saved)
  self.assertEqual(tree_hash(fresh.state()),tree_hash(saved))
  with self.assertRaises(AssertionError):fresh.sample_pair(d,g,0)
  q=fresh.sample_pair(d,g,8);self.assertEqual(q['teacher_updates'],8)
  fresh.learn_pair(q,.3,.3,.29,.31)
  self.assertEqual(fresh.updates,9);self.assertEqual(saved['updates'],8)
  self.assertIn('8',fresh.used_versions)
  with self.assertRaises(AssertionError):fresh.learn_pair(q,.3,.3,.29,.31)
if __name__=='__main__':unittest.main()
