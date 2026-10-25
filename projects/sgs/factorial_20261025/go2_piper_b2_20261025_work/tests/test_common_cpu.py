import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from b2.dependencies import enable_owner_imports
sgs=enable_owner_imports(verify_lock=False)
from b2.metrics import EpisodeMetrics
from b2.task_commands import phase_groups, targets_numpy, phases_from_assignments
SUPPORT=json.loads((ROOT/'protocol/task_support.json').read_text())
def pool(): return sgs.TaskPool([sgs.TaskSpec.from_content(c) for c in SUPPORT['content']],SUPPORT['support_version'])
def session(run='B2_cpu_fixture',mode='sgs'):
    return sgs.SamplerSession(pool(),run_id=run,seed=1021,feedback_rule_version=sgs.T1_RULE_VERSION,mode=mode)
def sample(metrics, ids, failure=False, force=0):
    n=len(ids); metrics.capture(ids,base_command=np.zeros((n,3)),base_actual=np.zeros((n,3)),
      ee_position_error=np.full(n,.02),ee_orientation_error=np.full(n,.03),base_clearance=np.full(n,.28),
      joint_velocity_limit_ratio=np.full(n,.2),joint_position_limit_excess=np.zeros(n),nonfoot_max_force=np.full(n,force),failure=np.full(n,failure))
class CommonCPU(unittest.TestCase):
    def test_explicit_phase_preserves_existing_sequences(self):
        offsets=np.arange(64)%4
        for step in (0,1,299,300,301,599,600,899,900,999):
            age=np.full(64,step*.02)
            np.testing.assert_array_equal(phase_groups(offsets,age,6.),(np.arange(64)+(age/6.).astype(int))%4)
        before=offsets.copy(); offsets[[1,7]]=[2,0]
        np.testing.assert_array_equal(offsets[np.arange(64)%6==0],before[np.arange(64)%6==0])
        with self.assertRaises(ValueError): phase_groups([4],[0],6.)
    def test_real_pool_and_initial_assignments(self):
        s=session(); a=s.assign(range(16)); self.assertEqual(len(s.pool.ids),4)
        self.assertEqual(phases_from_assignments(a,s.pool),[s.pool.tasks[s.pool.index[x.task_id]].content['base_sequence_phase_offset'] for x in a])
        self.assertEqual(sum(s.engine.sample_counts),16)
        self.assertEqual(sum(s.summary()['tasks'][i]['transitions'] for i in range(4)),0)
    def test_metrics_real_feedback_and_old_task_ownership(self):
        s=session(); old=s.assign([0,1]); m=EpisodeMetrics(2,2)
        sample(m,[0,1]);s.advance_all()
        sample(m,[0],failure=True,force=6);sample(m,[1]);s.advance_all()
        rows=[]
        for a in old:
            metric=m.finish(a.env_id); terminated=a.env_id==0
            labels=sgs.t1_episode_labels(metric,terminated=terminated,timeout=not terminated,max_steps=2)
            metric.update(labels)
            rows.append(sgs.TerminalFeedback(a,terminated,not terminated,labels['joint_success_5cm'],2,sgs.T1_RULE_VERSION,metric))
        self.assertEqual(s.observe_terminal(reversed(rows))['accepted'],2)
        self.assertEqual(s.observe_terminal(rows)['duplicates'],2)
        self.assertFalse(rows[0].success);self.assertTrue(rows[1].success)
        self.assertEqual(rows[0].metrics['nonfoot_contact'],True)
        new=s.assign([0]);m.reset([0]);self.assertEqual(new[0].episode_id,1)
        self.assertEqual(m.finish(1)['steps'],2)
        # Old row cannot change a new episode, and duplicate is ignored.
        self.assertEqual(s.observe_terminal([rows[0]])['duplicates'],1)
        self.assertEqual(s.assignment(0),new[0])
    def test_censor_not_failure_and_rng_restore(self):
        s=session();s.assign(range(8));m=EpisodeMetrics(8)
        sample(m,list(range(8)));s.advance_all()
        state=json.loads(json.dumps(s.state_dict(),allow_nan=False)); ms=json.loads(json.dumps(m.state_dict(),allow_nan=False))
        replay=session();replay.load_state_dict(state); rm=EpisodeMetrics(8);rm.load_state_dict(ms)
        for x in (s,replay):x.censor_active('training_cutoff',[0,3])
        self.assertEqual(s.assign([0,3]),replay.assign([0,3]))
        self.assertEqual(s.state_dict(),replay.state_dict()); self.assertEqual(m.state_dict(),rm.state_dict())
        self.assertEqual(sum(s.engine.completed_counts),0)
        other=session('B1_cpu_fixture')
        with self.assertRaises(ValueError):other.load_state_dict(state)
    def test_empty_metric_state_and_nan_rejected_without_mutation(self):
        m=EpisodeMetrics(2);state=json.loads(json.dumps(m.state_dict(),allow_nan=False));r=EpisodeMetrics(2);r.load_state_dict(state)
        self.assertEqual(m.state_dict(),r.state_dict())
        before=copy.deepcopy(m.state_dict())
        with self.assertRaises(ValueError):
            m.capture([0],base_command=np.zeros((1,3)),base_actual=np.full((1,3),np.nan),
             ee_position_error=[0],ee_orientation_error=[0],base_clearance=[.2],joint_velocity_limit_ratio=[0],joint_position_limit_excess=[0],nonfoot_max_force=[0],failure=[False])
        self.assertEqual(m.state_dict(),before)
if __name__=='__main__':unittest.main(verbosity=2)
