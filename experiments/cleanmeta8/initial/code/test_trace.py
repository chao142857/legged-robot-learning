"""Test instrumentation equivalence; synthetic artifacts outside training experience."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np,torch
from cpu_hash import tree_hash
from teacher_v02 import Teacher
from common import ROOT
from traced_teacher_ops import traced
class TraceTests(unittest.TestCase):
 def test_trace_does_not_change_update_or_restore(self):
  torch.set_num_threads(1)
  for bound in (1e6,.0001,1e-30):
   x=np.random.default_rng(3).normal(size=(64,12));a=Teacher(x,{'synthetic_only':True});d=np.zeros((64,30));g=np.linspace(-.7,.8,10)
   a.learn_pair(a.sample_pair(d,g,0),.2,.21,.23,.19);a.config['max_preference_step_kl']=bound
   b=copy.deepcopy(a);decision=a.sample_pair(d,g,1);b.load(decision['teacher'])
   state=a.state();r=a.learn_pair(copy.deepcopy(decision),.2,.21,.23,.19)
   old=sys.argv
   try:
    with tempfile.TemporaryDirectory(prefix='clean-trace-test-') as folder:
     sys.argv=['test','--out',folder];s=traced(b,copy.deepcopy(decision),.2,.21,.23,.19)
     self.assertEqual(tree_hash(a.state()),tree_hash(b.state()));self.assertEqual(tree_hash(r),tree_hash(s))
     receipt=json.loads((Path(folder)/'adam_acceptance.json').read_text());self.assertTrue(receipt['passed'])
     saved=torch.load(Path(folder)/'adam_full_trace.pt',weights_only=False)
     self.assertEqual(tree_hash(saved['before']),tree_hash(state));self.assertEqual(tree_hash(saved['after']),tree_hash(a.state()))
   finally:sys.argv=old
 def test_trace_preserves_failure_evidence(self):
  t=Teacher(np.random.default_rng(3).normal(size=(64,12)),{'synthetic_only':True})
  decision=t.sample_pair(np.zeros((64,30)),np.linspace(-.7,.8,10),0);before=t.state();native=t.optimizer.step
  def fail():native();raise RuntimeError('injected failure after step')
  t.optimizer.step=fail;old=sys.argv
  try:
   with tempfile.TemporaryDirectory(prefix='clean-failure-test-') as folder:
    sys.argv=['test','--out',folder]
    with self.assertRaises(RuntimeError):traced(t,decision,.2,.21,.23,.19)
    trace=torch.load(Path(folder)/'adam_fault_trace.pt',weights_only=False)
    self.assertEqual(tree_hash(trace['after_failure']),tree_hash(before));self.assertEqual(len(trace['candidates']),1)
  finally:sys.argv=old
if __name__=='__main__':unittest.main(verbosity=2)
