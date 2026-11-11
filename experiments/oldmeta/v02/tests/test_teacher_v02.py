"""Synthetic-only CPU checks; no real checkpoint or experience files modified."""
import copy
import io
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"code"))
from teacher_v02 import (Teacher, FeatureEncoder, V02_CONFIG, DYNAMIC_FIELDS,
                         GLOBAL_FIELDS, antithetic_loss, learning_progress_from_records,
                         state_hash)


class TeacherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.static=np.random.default_rng(9).normal(size=(64,17))
        self.dynamic=np.zeros((64,len(DYNAMIC_FIELDS)))
        self.global_state=np.zeros(len(GLOBAL_FIELDS))
        self.teacher=Teacher(self.static,{"pool":"synthetic-only"})

    def pair(self,teacher=None,version=1):
        return (teacher or self.teacher).sample_pair(self.dynamic,self.global_state,version)

    def test_initial_near_uniform_and_state_conditional(self):
        a=self.teacher.decide(self.dynamic,self.global_state,1)
        self.assertLess(a["distribution"]["tv_uniform"],.01)
        g=self.global_state.copy();g[0]=.5;g[8]=.4;g[9]=1
        b=self.teacher.decide(self.dynamic,g,2)
        self.assertFalse(np.array_equal(a["probability"],b["probability"]))
        self.assertEqual(self.teacher.updates,0)

    def test_floor_normalization_finite_and_different_arms(self):
        d=self.pair()
        for arm in ("plus","minus"):
            p=d[arm]["probability"]
            self.assertTrue(np.isfinite(p).all())
            self.assertGreaterEqual(p.min(),.2/len(p))
            self.assertAlmostEqual(float(p.sum()),1.,places=13)
        self.assertGreater(d["pair_tv"],1e-4)
        p=self.teacher.probability_from_preference(d["basis"],np.full(8,1e200))
        self.assertTrue(np.isfinite(p).all())
        self.assertGreaterEqual(p.min(),.2/len(p))

    def test_exact_antithetic_score_gradient_and_detachment(self):
        mean=torch.arange(8,dtype=torch.float64,requires_grad=True)/100
        epsilon=torch.tensor([.2,-.4,.8,1.,.3,-1.,.4,.7],dtype=torch.float64)
        plus=(mean.detach()+.5*epsilon).requires_grad_()
        minus=(mean.detach()-.5*epsilon).requires_grad_()
        reward=torch.tensor(.3,requires_grad=True)
        loss,_,_=antithetic_loss(mean,plus,minus,.5,reward,.1,.17)
        g=torch.autograd.grad(loss,(mean,plus,minus,reward),allow_unused=True)
        expected=-(float(reward.detach())-.1)*epsilon/(2*.5)
        torch.testing.assert_close(g[0],expected)
        self.assertIsNone(g[1]);self.assertIsNone(g[2]);self.assertIsNone(g[3])

    def test_baseline_cancels_and_sign_is_ascent(self):
        mean=torch.zeros(8,dtype=torch.float64,requires_grad=True)
        eps=np.arange(8,dtype=float)/10
        gradients=[]
        for baseline in (-.6,.8):
            loss,_,_=antithetic_loss(mean,.5*eps,-.5*eps,.5,.2,-.1,baseline)
            gradients.append(torch.autograd.grad(loss,mean)[0])
        torch.testing.assert_close(gradients[0],gradients[1],atol=1e-14,rtol=0)
        self.assertGreater(float((-gradients[0]*torch.tensor(eps)).sum()),0)

    def test_linear_objective_unbiased_antithetic_estimator(self):
        # Deterministic moment-exact quadrature verifies the analytic expectation
        # without a flaky finite Monte Carlo tolerance: E[epsilon epsilon^T]=I.
        eps=np.concatenate((np.eye(8),-np.eye(8)))*np.sqrt(8.)
        objective=np.linspace(-.3,.5,8)
        # R(z)=objective dot z, so E[(R+−R−)eps/(2sigma)]=objective.
        diff=2*.5*(eps@objective)
        gradient=(diff[:,None]*eps/(2*.5)).mean(0)
        np.testing.assert_allclose(gradient,objective,atol=1e-14,rtol=0)

    def test_real_update_and_replay_rejected(self):
        d=self.pair();before=state_hash(self.teacher.net.state_dict())
        result=self.teacher.learn_pair(d,.4,.42,.44,.41)
        self.assertGreater(result["gradient_norm_before_clip"],0)
        self.assertGreater(result["parameter_delta_l2"],0)
        self.assertNotEqual(before,state_hash(self.teacher.net.state_dict()))
        self.assertAlmostEqual(result["paired_reward_difference"],.03)
        self.assertAlmostEqual(result["uniform_gain_baseline"],.02)
        self.assertGreater(result['accepted_learning_rate_multiplier'],0)
        self.assertLessEqual(result['preference_step_kl'],self.teacher.config['max_preference_step_kl'])
        self.assertTrue(all(group['lr']==self.teacher.config['learning_rate'] for group in self.teacher.optimizer.param_groups))
        with self.assertRaises(AssertionError):self.teacher.learn_pair(d,.4,.42,.44,.41)

    def test_uniform_reward_does_not_change_teacher_gradient(self):
        a=Teacher(self.static,{"pool":"synthetic-only"});b=Teacher(self.static,{"pool":"synthetic-only"})
        da=self.pair(a);db=self.pair(b)
        ra=a.learn_pair(da,.2,.1,.3,.25);rb=b.learn_pair(db,.2,.9,.3,.25)
        np.testing.assert_allclose(ra["gradient_vector"],rb["gradient_vector"],atol=1e-14,rtol=0)
        for name,value in a.net.state_dict().items():
            torch.testing.assert_close(value,b.net.state_dict()[name],atol=1e-14,rtol=0)

    def test_zero_difference_skips_momentum_update(self):
        self.teacher.learn_pair(self.pair(),.4,.42,.44,.41)
        before=state_hash(self.teacher.net.state_dict())
        r=self.teacher.learn_pair(self.pair(version=2),.4,.42,.43,.43)
        self.assertTrue(r["zero_signal"])
        self.assertEqual(r["parameter_delta_l2"],0)
        self.assertEqual(before,state_hash(self.teacher.net.state_dict()))

    def test_checkpoint_restores_optimizer_private_rng_and_next_update(self):
        self.teacher.learn_pair(self.pair(),.4,.42,.44,.41)
        stream=io.BytesIO();torch.save(self.teacher.state(),stream);stream.seek(0)
        restored=Teacher(self.static,{"pool":"synthetic-only"})
        restored.load(torch.load(stream,weights_only=False))
        a=self.pair(version=2);b=self.pair(restored,version=2)
        for key in ("epsilon","mean","basis"):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a["plus"]["probability"],b["plus"]["probability"])
        ra=self.teacher.learn_pair(a,.4,.42,.41,.44)
        rb=restored.learn_pair(b,.4,.42,.41,.44)
        np.testing.assert_array_equal(ra["gradient_vector"],rb["gradient_vector"])
        self.assertEqual(state_hash(self.teacher.net.state_dict()),state_hash(restored.net.state_dict()))

    def test_student_random_stream_untouched(self):
        torch.manual_seed(72);before=torch.random.get_rng_state().clone()
        Teacher(self.static,{"pool":"synthetic-only"})
        self.pair();self.teacher.decide(self.dynamic,self.global_state,9)
        torch.testing.assert_close(torch.random.get_rng_state(),before)

    def test_nonfinite_inputs_outcomes_rejected_without_update(self):
        d=self.pair();before=state_hash(self.teacher.net.state_dict())
        with self.assertRaises(AssertionError):self.teacher.learn_pair(d,.4,.42,np.nan,.41)
        self.assertEqual(before,state_hash(self.teacher.net.state_dict()))
        g=self.global_state.copy();g[0]=np.inf
        with self.assertRaises(AssertionError):self.teacher.decide(self.dynamic,g,1)
        with self.assertRaises(AssertionError):self.teacher.probability_from_preference(d["basis"],np.full(8,np.nan))

    def test_policy_step_guard_rolls_back_network_and_optimizer(self):
        t=Teacher(self.static,{"pool":"synthetic-only"},config={"learning_rate":10.,"max_step_backtracks":0})
        d=self.pair(t);before=state_hash(t.net.state_dict())
        r=t.learn_pair(d,.2,.2,.8,.1)
        self.assertTrue(r["step_rejected"])
        self.assertEqual(before,state_hash(t.net.state_dict()))
        self.assertEqual(t.optimizer.state_dict()["state"],{})

    def test_tampered_and_stale_decision_rejected(self):
        d=self.pair();bad=copy.deepcopy(d);bad["basis"][0,0]+=.1
        with self.assertRaises(AssertionError):self.teacher.learn_pair(bad,.2,.2,.4,.3)
        bad=copy.deepcopy(d);bad["plus"]["preference"][0]+=.1
        with self.assertRaises(AssertionError):self.teacher.learn_pair(bad,.2,.2,.4,.3)
        self.teacher.learn_pair(d,.2,.2,.4,.3)
        stale=copy.deepcopy(d);stale["version"]=2
        with self.assertRaises(AssertionError):self.teacher.learn_pair(stale,.2,.2,.4,.3)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.static=np.random.default_rng(1).normal(size=(100,12))
        self.encoder=FeatureEncoder(self.static)
        self.d=np.zeros((100,len(DYNAMIC_FIELDS)))

    def test_missing_fields_are_explicit_and_no_fabricated_history(self):
        result=self.encoder.encode(self.d)
        self.assertEqual(result["dynamic"].sum(),0)
        self.assertEqual(result["progress_observed"].sum(),0)
        self.assertEqual(result["partial"].sum(),0)
        malformed=self.d.copy();malformed[:,7]=8.;malformed[:,21]=.8
        masked=self.encoder.encode(malformed)
        np.testing.assert_array_equal(result["basis"],masked["basis"])
        malformed[0,0]=.5
        with self.assertRaises(AssertionError):self.encoder.encode(malformed)

    def test_observed_real_feedback_changes_only_encoding_not_normalizer(self):
        before=self.encoder.encode(self.d);normalizer=self.encoder.command_scale.copy()
        self.d[0,[0,1,2,3,28,29]]=1
        self.d[0,7]=2.;self.d[0,9]=.4;self.d[0,4]=0
        after=self.encoder.encode(self.d)
        self.assertFalse(np.array_equal(before["basis"],after["basis"]))
        np.testing.assert_array_equal(normalizer,self.encoder.command_scale)
        self.assertEqual(after["dynamic"][0,4],0)  # real observed failure, not missing

    def test_partial_is_distinct_from_completed_feedback(self):
        partial=np.zeros((100,8));partial[0]=[1,1,.1,.4,.5,.6,1.2,.8]
        result=self.encoder.encode(self.d,partial=partial)
        self.assertEqual(result["dynamic"].sum(),0)
        np.testing.assert_array_equal(result["partial"],partial)
        self.assertNotEqual(result["basis_sha256"],self.encoder.encode(self.d)["basis_sha256"])

    def test_progress_both_halves_and_failure_zero(self):
        def row(good,fail=0,eligible=500):return np.array([eligible,eligible*good,.1,.1,.03,.1,0,fail,1.])
        records={0:[(100,row(.2)),(20000,row(.8))],1:[(15000,row(.2)),(20000,row(.8))],
                 2:[(100,row(.4)),(20000,row(.9,fail=1))],3:[(100,row(.3)),(25000,row(.8))]}
        p,m=learning_progress_from_records(records,4,24000)
        np.testing.assert_array_equal(m,[1,0,1,0])
        np.testing.assert_allclose(p,[.6,0,-.4,0])

    def test_projection_reproducible_and_full_pool_simplex(self):
        other=FeatureEncoder(self.static)
        np.testing.assert_array_equal(self.encoder.encode(self.d)["basis"],other.encode(self.d)["basis"])
        static=np.random.default_rng(55).normal(size=(32768,8))
        t=Teacher(static,{"pool":"synthetic32768"})
        p=t.sample_pair(np.zeros((32768,len(DYNAMIC_FIELDS))),np.zeros(len(GLOBAL_FIELDS)),1)["plus"]["probability"]
        self.assertEqual(len(p),32768);self.assertGreaterEqual(p.min(),.2/32768)
        self.assertAlmostEqual(float(p.sum()),1.,places=13)


if __name__=="__main__": unittest.main(verbosity=2)
