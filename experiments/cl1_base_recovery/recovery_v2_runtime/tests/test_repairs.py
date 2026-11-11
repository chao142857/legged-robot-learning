import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from checkpoint_pair import pair,sha
from cpu_hash import tree_hash
from selection_fix import category
class Repairs(unittest.TestCase):
 def test_normal_excludes_single_indicator_decline(self):
  self.assertEqual(category(dict(delta_j=-.03945,delta_ee=.001668)),'ambiguous')
  self.assertEqual(category(dict(delta_j=.00749,delta_ee=-.000283)),'normal')
  self.assertEqual(category(dict(delta_j=-.045,delta_ee=.0035)),'degradation')
  self.assertEqual(category(dict(delta_j=0,delta_ee=.004)),'ambiguous')
 def fixture(self,d,pending=True):
  root=Path(d);run=root/'runs/u';(run/'checkpoints').mkdir(parents=True);(run/'teacher_requests').mkdir()
  net={'w':torch.tensor([1.])};q=np.full(32,1/32);x=np.zeros(34);x[0]=.14
  t=dict(macro_version=15,version=20,action=np.zeros(8),x=x)
  s=dict(pending=t if pending else None,history=[x]* (15 if pending else 14),transitions=[dict(completed_updates=1400)]*14)
  state=dict(learner=dict(net=net),students={'70158':s})
  tp=run/'teacher_requests'/('0015-decide.teacher_after.pt' if pending else '0015-complete.teacher_after.pt');torch.save(state,tp)
  info=dict(completed_updates=1400,student_seed=70158,optimizer_calls=28000,environment_snapshot_available=True,
    phase='ready' if pending else 'complete',transfer_kind='uniform',decision=dict(version=15,teacher_version=20,
      preference=np.zeros(8),teacher_sha256=tree_hash(net),probability=q))
  cp=run/'checkpoints/update_01400.pt';torch.save(dict(infos=info),cp)
  return cp,tp,state
 def test_pending_action_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   cp,tp,s=self.fixture(d);r=pair(cp);self.assertTrue(r['pending']);self.assertEqual(r['teacher_path'],str(tp))
 def test_block_end_has_no_pending_action(self):
  with tempfile.TemporaryDirectory() as d:
   cp,tp,s=self.fixture(d,False);self.assertFalse(pair(cp)['pending'])
 def test_future_teacher_feedback_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   cp,tp,s=self.fixture(d);s['students']['70158']['transitions'][-1]['completed_updates']=1500;torch.save(s,tp)
   with self.assertRaises(AssertionError):pair(cp)
 def test_wrong_teacher_weights_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   cp,tp,s=self.fixture(d);s['learner']['net']['w']+=1;torch.save(s,tp)
   with self.assertRaises(AssertionError):pair(cp)
 def test_missing_pending_memory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   cp,tp,s=self.fixture(d);s['students']['70158']['pending']=None;torch.save(s,tp)
   with self.assertRaises(AssertionError):pair(cp)
 def test_explicit_digest_is_checked(self):
  with tempfile.TemporaryDirectory() as d:
   cp,tp,s=self.fixture(d);a=torch.load(cp,weights_only=False);a['infos']['cl1_teacher_snapshot']=dict(path=str(tp),sha256=sha(tp));torch.save(a,cp)
   self.assertTrue(pair(cp)['passed']);s['learner']['net']['w']+=1;torch.save(s,tp)
   with self.assertRaises(AssertionError):pair(cp)
if __name__=='__main__':unittest.main(verbosity=2)
