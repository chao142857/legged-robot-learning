import copy,json,pathlib,pickle,random,sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'code'))
from grouped_sampling import GroupedSampler,GroupedOwnership
from vendor_sampler import TaskSampler
from common import digest

class SamplerTests(unittest.TestCase):
    def make(self,mapping=None,**kw):return GroupedSampler(np.repeat(np.arange(4),8) if mapping is None else np.array(mapping),'wbc:train:test',42,**kw)
    def test_empty_prior_is_not_a_fake_episode(self):
        s=self.make();np.testing.assert_array_equal(s.group_rates(),[.5]*4)
        self.assertEqual(sum(map(len,s.outcomes)),0);self.assertEqual(sum(s.completed_counts),0)
        np.testing.assert_allclose(s.probabilities(),np.full(32,1/32),rtol=0,atol=1e-16)
    def test_equal_weights_uniform_with_unequal_group_sizes(self):
        s=self.make([0,1,1,1,2,2]);np.testing.assert_allclose(s.probabilities(),np.full(6,1/6),rtol=0,atol=1e-16)
        np.testing.assert_allclose(s.group_probabilities(),[1/6,3/6,2/6])
        draws=np.bincount(s.sample(100000),minlength=6)/100000;self.assertLess(float(abs(draws-1/6).max()),.006)
    def test_native_kernel_equivalence_and_normalization(self):
        s=self.make();v=TaskSampler(4,42,mode='sgs')
        for g,ys in enumerate(([0]*100,[1]*100,[0,1]*50,[0]*75+[1]*25)):
            s.record([8*g]*100,ys);v.record([g]*100,ys)
        np.testing.assert_allclose(s.group_probabilities(),v.probabilities(),rtol=1e-13,atol=1e-15)
        self.assertAlmostEqual(sum(s.probabilities()),1);self.assertAlmostEqual(s.group_probabilities().sum(),1)
        self.assertGreater(s.group_probabilities()[2],s.group_probabilities()[0]);self.assertAlmostEqual(s.group_probabilities()[0],s.group_probabilities()[1])
    def test_group_window_is_last100_real_episodes(self):
        s=self.make();ids=[i%8 for i in range(130)];ys=[int(i>=30) for i in range(130)]
        for i,y in zip(ids,ys):s.record([i],[y])
        self.assertEqual(s.outcomes[0],[1]*100);self.assertEqual(s.group_rates()[0],1)
        self.assertEqual(sum(s.completed_counts),130);self.assertEqual(s.group_completed[0],130);self.assertEqual(len(set(s.window_members[0])),8)
    def test_every_member_shares_feedback(self):
        s=self.make();s.record([0,7],[1,0],control_step=24)
        np.testing.assert_array_equal(np.array(s.rates())[:8],[.5]*8);self.assertEqual(s.completed_counts[1],0)
        s.record([3],[1],control_step=48);np.testing.assert_allclose(np.array(s.rates())[:8],2/3)
    def test_invalid_feedback_is_atomic(self):
        s=self.make();before=pickle.dumps(s.state_dict())
        with self.assertRaises(ValueError):s.record([0,999],[1,0])
        self.assertEqual(before,pickle.dumps(s.state_dict()))
        with self.assertRaises(ValueError):s.record([0],[.5])
        with self.assertRaises(ValueError):s.record([0],[1],event_ids=[])
    def test_duplicate_and_stale_event_rejected(self):
        s=self.make();s.record([0],[1],control_step=10,event_ids=[(4,1)])
        for g in (1,0):
            with self.assertRaises(ValueError):s.record([1],[0],control_step=11,event_ids=[(4,g)])
        with self.assertRaises(ValueError):s.record([0],[0],control_step=9,event_ids=[(4,2)])
        s.record([1],[0],control_step=11,event_ids=[(4,2)]);self.assertEqual(sum(s.completed_counts),2)
    def test_checkpoint_rng_window_and_age_roundtrip(self):
        s=self.make();s.sample(25);s.record([0,8,16,24],[0,1,1,0],control_step=500,event_ids=[(i,1) for i in range(4)])
        state=pickle.loads(pickle.dumps(s.state_dict()));other=self.make();other.load_state_dict(state)
        self.assertEqual(s.sample(4096),other.sample(4096));self.assertEqual(s.diagnostics(700),other.diagnostics(700))
        self.assertEqual(digest(s.state_dict()),digest(other.state_dict()))
        s.record([1],[1],control_step=600,event_ids=[(0,2)]);self.assertNotEqual(s.outcomes,other.outcomes)
    def test_mapping_pool_and_parameter_mismatches_rejected(self):
        s=self.make();state=s.state_dict()
        for other in [self.make(np.tile(np.arange(4),8)),self.make(history=99),GroupedSampler(s.mapping,'foreign-pool')]:
            with self.assertRaises(ValueError):other.load_state_dict(state)
        bad=copy.deepcopy(state);bad['window_successes'][0]=5
        with self.assertRaises(ValueError):s.load_state_dict(bad)
    def test_native_ownership_foreign_duplicate_and_resume(self):
        class Pool:n=32;identity='wbc:train:test'
        a=GroupedOwnership(Pool(),4,np.repeat(np.arange(4),8));ids=np.arange(4)
        a.assign(ids);a.advance();token=a.token(ids);a.commit(token,[1,0,1,0]);a.check()
        with self.assertRaises(AssertionError):a.commit(token,[1,0,1,0])
        a.assign(ids);a.advance();bad=a.token(ids);bad['pool_identity']='wrong'
        with self.assertRaises(AssertionError):a.commit(bad,[0]*4)
        b=GroupedOwnership(Pool(),4,np.repeat(np.arange(4),8));b.load_state_dict(pickle.loads(pickle.dumps(a.state_dict())));b.check()
        self.assertEqual(a.engine.sample(40),b.engine.sample(40))
    def test_map_validation(self):
        for mapping in [[],[0,2],[-1,0],[0.,1.]]:
            with self.assertRaises(ValueError):self.make(mapping)
    def test_random_and_command_use_identical_code(self):
        a=self.make();b=self.make(np.tile(np.arange(4),8))
        self.assertEqual(type(a),type(b));self.assertEqual(a.history,b.history)
        for g in range(4):
            a.record([a.members[g][0]],[g%2]);b.record([b.members[g][0]],[g%2])
        np.testing.assert_array_equal(a.group_probabilities(),b.group_probabilities())
    def test_age_and_member_mixture_diagnostics(self):
        s=self.make();s.record([0],[1],control_step=10);s.record([1],[0],control_step=30)
        d=s.diagnostics(50);self.assertEqual(d['mean_feedback_age_control_steps'],30);self.assertEqual(d['feedback_total'],2)

if __name__=='__main__':unittest.main(verbosity=2)
