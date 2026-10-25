import copy
import json
import math
from pathlib import Path
import unittest
import torch
from benchmarks.go2_piper.wbc_sgs.sampler import TaskSampler
from benchmarks.go2_piper.wbc_sgs.metrics import JointMetrics


class SamplerTests(unittest.TestCase):
    def test_exact_paper_kernel_and_softmax(self):
        s=TaskSampler(3,7,'sgs',history=10)
        s.record([0]*10+[1]*10+[2]*10,[False]*10+[True,False]*5+[True]*10)
        e=s.epsilon
        w=[(p+e)**.5*(1-p+e)**.5 for p in [0.,.5,1.]]
        raw=[(max(x,e)+e)**.5 for x in w]
        expected=[x/sum(raw) for x in raw]
        for a,b in zip(s.probabilities(),expected):self.assertAlmostEqual(a,b,places=14)
        self.assertGreater(expected[1],expected[0])
        self.assertGreater(min(expected),0.)
        counts=[0]*3
        for i in s.sample(30000):counts[i]+=1
        self.assertGreater(counts[1],counts[0]*5)

    def test_unseen_and_uniform(self):
        for mode in ['uniform','sgs']:
            s=TaskSampler(48,13,mode)
            self.assertEqual(s.probabilities(),[1/48]*48)
            self.assertEqual(sum(s.completed_counts),0)
            self.assertEqual(s.outcomes,[[] for _ in range(48)])
        u=TaskSampler(3,13,'uniform');u.record([0,1],[True,False])
        self.assertEqual(u.probabilities(),[1/3]*3)

    def test_fifo_same_task_multiple_completions(self):
        s=TaskSampler(2,7,'sgs',history=3)
        s.record([1,0,1,1,1],[True,False,False,True,False])
        self.assertEqual(s.outcomes,[[0],[0,1,0]])
        self.assertEqual(s.completed_counts,[1,4])
        self.assertEqual(s.success_counts,[0,2])

    def test_checkpoint_json_rng_exact(self):
        s=TaskSampler(7,13,'sgs');s.sample(51);s.record([0,1,1],[True,False,True])
        state=json.loads(json.dumps(s.state_dict()))
        t=TaskSampler(7,999,'sgs');t.load_state_dict(state)
        self.assertEqual(s.sample(100),t.sample(100))
        self.assertEqual(s.summary(),t.summary())

    def test_mass_moves_with_recent_success(self):
        s=TaskSampler(3,7,'sgs',history=10)
        s.record([0]*10+[1]*10,[1,0]*5+[0]*10)
        self.assertGreater(s.probabilities()[0],s.probabilities()[1])
        s.record([0]*10+[1]*10,[1]*10+[1,0]*5)
        self.assertGreater(s.probabilities()[1],s.probabilities()[0])


class MetricsTests(unittest.TestCase):
    def protocol(self):
        return json.loads((Path(__file__).parents[1]/'benchmarks/go2_piper/wbc_sgs/success_protocol.json').read_text())

    def test_marginal_is_not_joint_and_early_failure(self):
        p=self.protocol();p.update(settling_s=0,minimum_scored_samples=10,tracking_fraction=.5)
        m=JointMetrics(3,'cpu',p)
        for step in range(10):
            e=torch.zeros(3,4)
            if step<5:e[0,2]=1.
            else:e[0,0]=1.
            m.update(e,torch.tensor([True,True,True]),torch.zeros(3),.02)
        r=m.results([0,1,2],[True,False,True],[False,True,False])
        self.assertTrue(r[0]['locomotion'] and r[0]['manipulation'])
        self.assertFalse(r[0]['joint_success'])
        self.assertFalse(r[1]['joint_success'])
        self.assertTrue(r[2]['joint_success'])

    def test_safety_includes_startup_and_subset_reset(self):
        p=self.protocol();p.update(settling_s=.04,minimum_scored_samples=2)
        m=JointMetrics(2,'cpu',p)
        for i in range(5):
            m.update(torch.zeros(2,4),torch.tensor([i!=0,True]),torch.zeros(2),.02)
        r=m.results([0,1],[True,True],[False,False])
        self.assertFalse(r[0]['safety']);self.assertTrue(r[1]['joint_success'])
        m.reset([0]);self.assertEqual(int(m.steps[0]),0);self.assertEqual(int(m.steps[1]),5)

    def test_pool_crossing_and_no_rank(self):
        from tools.make_wbc_sgs_pool import generate
        a=generate(20261019);self.assertEqual(a,generate(20261019))
        tasks=a['tasks'];self.assertEqual(len(tasks),48)
        pairs={(t['factors']['velocity_family'],t['factors']['pose_family']) for t in tasks}
        self.assertEqual(len(pairs),48)
        for t in tasks:
            self.assertAlmostEqual(sum(x*x for x in t['ee_pose'][3:]),1.)


if __name__=='__main__':unittest.main(verbosity=2)
