"""Real owner modules composed on CPU, with explicit shape/measurement fixtures.

No mocked SGS/IK/mapper. Inputs are synthetic, so this is not physics or a test
of native Isaac action/reward/observation hooks.
"""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SGS_ROOT=Path(os.environ.get('B2_CPU_SGS_ROOT',str(ROOT.parent/'go2_piper_b1_sgs_20261025_work')))
IK_ROOT=Path(os.environ.get('B2_CPU_IK_ROOT',str(ROOT.parent/'go2_piper_b3_20261025_work')))
for path in (SGS_ROOT,IK_ROOT):sys.path.insert(0,str(path))
import torch
import numpy as np
from shared_sgs import TaskSpec,TaskPool,SamplerSession,T1_RULE_VERSION,t1_episode_labels,TerminalFeedback
from shared_ik import ArmOnlyIK,NativeActionMap,PolicyObservationDecoder,LEG_NAMES,ARM_NAMES
from b2.metrics import EpisodeMetrics
from b2.state import capture_module_state,restore_module_state
from b2.task_commands import phases_from_assignments

torch.set_num_threads(1)
SUPPORT=json.loads((ROOT/'protocol/task_support.json').read_text())
FIXTURE=json.loads((IK_ROOT/'frozen/native_binding_fixture.json').read_text())
from shared_ik.model import KINEMATICS

@unittest.skipUnless(KINEMATICS.is_file(), "Authorized model geometry excluded; set SHARED_IK_MODEL for asset-dependent tests")
class RealComposition(unittest.TestCase):
    def setUp(self):
        self.pool=TaskPool([TaskSpec.from_content(c) for c in SUPPORT['content']],SUPPORT['support_version'])
        self.s=SamplerSession(self.pool,run_id='B2_real_core_fixture',seed=9021,mode='sgs',feedback_rule_version=T1_RULE_VERSION)
        self.assign=self.s.assign(range(4),initial_task_ids={i:self.pool.ids[i] for i in range(4)})
        self.ik=ArmOnlyIK(4,dtype=torch.float64)
        self.map=NativeActionMap(FIXTURE['articulation_joint_names'],FIXTURE['action_joint_names'],FIXTURE['scale'],FIXTURE['action_offset'],dtype=torch.float64)
        widths=[3,3,18,18,18,3,7]
        self.terms=['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','velocity_commands','pos_commands']
        default=torch.tensor(FIXTURE['articulation_default'],dtype=torch.float64).repeat(4,1)
        self.decoder=PolicyObservationDecoder(self.terms,[(3*w,) for w in widths],FIXTURE['articulation_joint_names'],default)
        self.obs=torch.zeros(4,210,dtype=torch.float64)
        a,b=self.decoder.slices['projected_gravity'];self.obs[:,a:b].reshape(4,3,3)[:,-1]=torch.tensor([0.,0.,-1.])
        a,b=self.decoder.slices['pos_commands'];self.obs[:,a:b].reshape(4,3,7)[:,-1]=torch.tensor([.45,.03,.5,.9770240783691406,-.007009385,.2105149478,.03253136])
        self.raw12=torch.arange(48,dtype=torch.float64).reshape(4,12)*.001
        self.metrics=EpisodeMetrics(4,2)
    def command(self,controller=None):
        qref,diag=(controller or self.ik).command(**self.decoder.decode(self.obs),dt=.02)
        self.assertTrue(bool(diag['input_valid'].all()))
        return self.map.compose(self.raw12,qref),diag
    def test_real_modules_action18_and_observation_fixture(self):
        (a,p),diag=self.command()
        self.assertEqual(a.shape,(4,18));self.assertTrue(torch.equal(a[:,self.map.leg_slots],self.raw12))
        self.assertTrue(torch.allclose(p[:,self.map.arm_slots],diag['q_ref']))
        self.assertGreater(float(a[:,self.map.arm_slots].abs().max()),0.)
        a1,b1=self.decoder.slices['actions'];self.obs[:,a1:b1].reshape(4,3,18)[:,-1]=a
        torch.testing.assert_close(self.obs[:,a1:b1].reshape(4,3,18)[:,-1],a)
        before=a.clone();(after,_),_=self.command()
        full=(after-before).square().sum(-1);legs=(after[:,self.map.leg_slots]-before[:,self.map.leg_slots]).square().sum(-1)
        self.assertTrue(bool((full>legs).any()))
        scattered=self.map.articulation_targets(p)
        self.assertTrue(torch.allclose(scattered[:,[FIXTURE['articulation_joint_names'].index(n) for n in ARM_NAMES]],diag['q_ref']))
    def test_combined_sampler_ik_metrics_restore(self):
        self.command();self.s.advance_all()
        self.metrics.capture(range(4),base_command=np.zeros((4,3)),base_actual=np.zeros((4,3)),ee_position_error=np.full(4,.02),ee_orientation_error=np.full(4,.03),base_clearance=np.full(4,.28),joint_velocity_limit_ratio=np.full(4,.2),joint_position_limit_excess=np.zeros(4),nonfoot_max_force=np.zeros(4),failure=np.zeros(4))
        saved=capture_module_state(self.s,self.ik,self.metrics,protocol_sha256='cpu_fixture',dependency_lock_sha256='cpu_fixture')
        s=SamplerSession(self.pool,run_id='B2_real_core_fixture',seed=9021,mode='sgs',feedback_rule_version=T1_RULE_VERSION)
        ik=ArmOnlyIK(4,dtype=torch.float64);m=EpisodeMetrics(4,2)
        restore_module_state(saved,s,ik,m,protocol_sha256='cpu_fixture',dependency_lock_sha256='cpu_fixture')
        corrupt=copy.deepcopy(saved);corrupt['metrics']['horizon_steps']=3
        before=capture_module_state(s,ik,m,protocol_sha256='cpu_fixture',dependency_lock_sha256='cpu_fixture')
        with self.assertRaises(ValueError):restore_module_state(corrupt,s,ik,m,protocol_sha256='cpu_fixture',dependency_lock_sha256='cpu_fixture')
        self.assertEqual(s.state_dict(),before['sampler']);self.assertTrue(torch.equal(ik.q_ref,before['controller']['q_ref']));self.assertEqual(m.state_dict(),before['metrics'])
        (a,p),_=self.command();(b,q),_=self.command(ik)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(p,q));self.assertEqual(m.state_dict(),self.metrics.state_dict())
        for x in (self.s,s):x.censor_active('training_cutoff',[1,3])
        self.assertEqual(self.s.assign([1,3]),s.assign([1,3]));self.assertEqual(self.s.state_dict(),s.state_dict())
        before=ik.q_ref.clone();ik.reset([1,3]);self.assertTrue(torch.equal(ik.q_ref[[0,2]],before[[0,2]]))
        self.assertFalse(bool(ik.initialized[[1,3]].any()))
    def test_terminal_feedback_uses_same_episode_and_independent_history(self):
        self.command()
        for t in range(2):
            self.s.advance_all();self.metrics.capture(range(4),base_command=np.zeros((4,3)),base_actual=np.zeros((4,3)),ee_position_error=np.full(4,.02),ee_orientation_error=np.full(4,.03),base_clearance=np.full(4,.28),joint_velocity_limit_ratio=np.full(4,.2),joint_position_limit_excess=np.zeros(4),nonfoot_max_force=np.array([6.,0,0,0]) if t==1 else np.zeros(4),failure=np.array([True,False,False,False]) if t==1 else np.zeros(4))
        rows=[]
        for a in self.assign:
            m=self.metrics.finish(a.env_id);failed=a.env_id==0;m.update(t1_episode_labels(m,terminated=failed,timeout=not failed,max_steps=2))
            rows.append(TerminalFeedback(a,failed,not failed,m['joint_success_5cm'],2,T1_RULE_VERSION,m))
        self.assertEqual(self.s.observe_terminal(rows)['accepted'],4)
        self.assertEqual(self.s.engine.success_counts,[0,1,1,1])
        separate=SamplerSession(self.pool,run_id='B1_independent_fixture',seed=9021,mode='sgs',feedback_rule_version=T1_RULE_VERSION)
        self.assertEqual(separate.engine.completed_counts,[0,0,0,0])
        self.assertEqual(phases_from_assignments(self.assign,self.pool),[0,1,2,3])
if __name__=='__main__':unittest.main(verbosity=2)
