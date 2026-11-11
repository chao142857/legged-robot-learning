import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import frozen_ops as op
from cl1_core import Learner
from cpu_hash import tree_hash
class Frozen(unittest.TestCase):
 def exercise(self,kind):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);previous_root=op.ROOT;argv=sys.argv.copy();op.ROOT=root
   try:
    (root/'prepared').mkdir();np.save(root/'prepared/command_features.npy',np.random.default_rng(1).normal(size=(64,16)))
    l=Learner()
    with torch.no_grad():l.net.mean.bias.fill_(.1)
    initial=dict(learner=l.state(),execution_kind=kind,students={'70158':dict(history=[],pending=None,previous=None,transitions=[],cursor=0)})
    stage=root/'teachers/test';stage.mkdir(parents=True)
    for name in ('initial.pt','current.pt'):op.save(stage/name,initial)
    dev=dict(continuous_joint_score=.2,pooled_rmse=dict(velocity_m_s=.1,yaw_rad_s=.1,ee_position_m=.1,ee_orientation_rad=.1))
    req=dict(kind='decide',seed=70158,version=1,dynamic=np.zeros((64,30)),global_state=np.zeros(10),dev=dev,progress=np.zeros(64),progress_observed=np.zeros(64),partial=np.zeros((64,8)))
    path=root/'0001-decide.pt';op.save(path,req);sys.argv=['frozen_ops.py','request','--stage','test','--request',str(path)];op.main()
    out=op.load(path.with_suffix('.done.pt'));self.assertIsNone(out['macro_action']['logprob'])
    self.assertFalse(out['macro_action']['eligible_for_on_policy_teacher_update'])
    if kind=='uniform':np.testing.assert_array_equal(out['probability'],np.full(64,1/64))
    else:self.assertGreater(out['distribution']['tv_uniform'],0)
    current=op.load(stage/'current.pt');self.assertEqual(tree_hash(current['learner']),tree_hash(initial['learner']))
    end=dict(kind='complete',seed=70158,version=1,checkpoint_update=100,dev=copy.deepcopy(dev),dynamic=req['dynamic'],global_state=np.r_[.01,np.zeros(9)])
    end['dev']['continuous_joint_score']=.3;p=root/'0001-complete.pt';op.save(p,end);sys.argv[-1]=str(p);op.main()
    current=op.load(stage/'current.pt');self.assertEqual(tree_hash(current['learner']),tree_hash(initial['learner']))
    self.assertIsNone(current['students']['70158']['pending']);self.assertAlmostEqual(current['students']['70158']['transitions'][0]['reward'],.1)
   finally:op.ROOT=previous_root;sys.argv=argv
 def test_uniform_is_real_uniform_not_shadow_proposal(self):self.exercise('uniform')
 def test_mean_is_deterministic_no_rng_or_adam_step(self):self.exercise('meta')
if __name__=='__main__':unittest.main(verbosity=2)
