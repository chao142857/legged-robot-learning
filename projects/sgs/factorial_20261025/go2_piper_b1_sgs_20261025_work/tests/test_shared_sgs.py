"""CPU contract tests tied to the requested SGS failure modes."""
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared_sgs import (TaskSpec, TaskPool, SamplerSession, TerminalFeedback,
                        FixedEvaluation, T1_RULE_VERSION, t1_episode_labels)
from shared_sgs.session import PARAMETERS


def pool(n=4):
    content = json.loads((ROOT.parent/'go2_piper_b2_20261025_work/protocol/task_support.json').read_text())
    return TaskPool([TaskSpec.from_content(c) for c in content['content'][:n]], content['support_version'])


def session(n=4, run='B1-test', seed=9021, mode='sgs'):
    return SamplerSession(pool(n), run_id=run, seed=seed, mode=mode, feedback_rule_version=T1_RULE_VERSION)


def good_metrics(steps=1000):
    return dict(steps=steps, xy_rmse=.1, yaw_rmse=.1, ee_position_rmse=.03,
                ee_orientation_mean=.1, base_clearance_min=.2, nonfoot_contact=False,
                joint_velocity_limit_ratio_max=.9, joint_position_limit_excess_max_rad=0.)


def terminal(s, token, *, success=True, terminated=False, timeout=True):
    steps = s.episode_steps(token.env_id)
    return TerminalFeedback(token, terminated, timeout, success, steps, T1_RULE_VERSION, good_metrics(steps))


class SharedSGSTest(unittest.TestCase):
    def test_vendored_core_and_formula_are_preserved(self):
        data = (ROOT/'shared_sgs/vendor/sampler.py').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), '122b3e1a08e92d1ce815fa048fddfcf867bc1894123779c4d651d71d2d13df93')
        s = session()
        histories = [[0]*100, [1]*100, [0, 1]*50, []]
        for k, h in enumerate(histories):
            s.engine.record([k]*len(h), h)
        # Independent expression of the documented Eq1/2 + Appendix B.2.
        p = [0., 1., .5, .5]; e = PARAMETERS['epsilon']
        a = [(max(math.sqrt((x+e)*(1-x+e)), e)+e)**.5 for x in p]
        expected = [x/sum(a) for x in a]
        for actual, wanted in zip(s.probabilities(), expected):
            self.assertAlmostEqual(actual, wanted, places=14)
        self.assertTrue(all(math.isfinite(x) and x > 0 for x in s.probabilities()))
        self.assertAlmostEqual(sum(s.probabilities()), 1.)
        lower = min(a)/(min(a)+3*max(a))
        self.assertGreaterEqual(min(s.probabilities()), lower)

    def test_cold_start_all_failure_all_success_and_unvisited(self):
        for outcomes in ([], [False]*120, [True]*120):
            s = session()
            for k in range(4):
                s.engine.record([k]*len(outcomes), outcomes)
            self.assertEqual(s.probabilities(), [.25]*4)
            self.assertEqual(s.engine.completed_counts, [len(outcomes)]*4)
            self.assertTrue(all(len(h) <= 100 for h in s.engine.outcomes))
        s = session(); s.engine.record([0]*100, [False]*100)
        self.assertEqual(s.engine.rates()[1:], [.5]*3)
        self.assertGreater(s.probabilities()[1], s.probabilities()[0])

    def test_single_task_degenerates_without_fake_identities(self):
        for outcome in (True, False):
            s = session(n=1)
            for _ in range(4):
                a = s.assign([0])[0]; s.advance_all(7)
                s.observe_terminal([terminal(s, a, success=outcome)])
                self.assertEqual(s.probabilities(), [1.])
            self.assertEqual(s.engine.sample_counts, [4])
        p = pool()
        with self.assertRaisesRegex(ValueError, 'Duplicate semantic'):
            TaskPool([p.tasks[0], TaskSpec.from_content(p.tasks[0].content)], 'no-aliases')
        c = p.tasks[0].content; c['horizon_s'] = 20
        self.assertEqual(TaskSpec.from_content(c).task_id, p.tasks[0].task_id)
        c['env_id'] = 777
        with self.assertRaises(ValueError): TaskSpec.from_content(c)

    def test_terminal_old_task_ownership_dedup_and_partial_reset(self):
        s = session()
        initial = s.assign([3, 2, 1, 0], initial_task_ids={i:s.pool.ids[i] for i in range(4)})
        s.advance_all(5)
        old = [s.assignment(i) for i in (0, 2)]
        rows = [terminal(s, a, success=False, terminated=True, timeout=False) for a in old]
        result = s.observe_terminal(list(reversed(rows)))
        self.assertEqual(result, dict(accepted=2, duplicates=0))
        self.assertEqual(s.engine.completed_counts, [1, 0, 1, 0])
        with self.assertRaises(ValueError): s.assign([1])
        new = s.assign([2, 0])
        self.assertTrue(all(a.episode_id == 1 for a in new))
        self.assertEqual(s.assignment(1), initial[1]); self.assertEqual(s.assignment(3), initial[3])
        before = s.state_dict()
        self.assertEqual(s.observe_terminal(rows), dict(accepted=0, duplicates=2))
        self.assertEqual(s.state_dict(), before)
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            s.observe_terminal([replace(rows[0], metrics=dict(other='changed'))])
        self.assertEqual(s.state_dict(), before)
        s.advance_all(2)
        self.assertEqual(s.episode_steps(1), 7); self.assertEqual(s.episode_steps(0), 2)
        self.assertEqual(sum(s.transitions), 28)

    def test_invalid_terminal_batch_is_atomic_and_foreign_task_rejected(self):
        s = session(); a = s.assign([0, 1]); s.advance_all(3)
        valid = terminal(s, a[0], success=False)
        bad = terminal(s, replace(a[1], task_id=s.pool.ids[(s.pool.index[a[1].task_id]+1)%4]))
        before = s.state_dict()
        with self.assertRaises(ValueError): s.observe_terminal([valid, bad])
        self.assertEqual(s.state_dict(), before)
        with self.assertRaises(ValueError): s.observe_terminal([valid, valid])
        self.assertEqual(s.state_dict(), before)
        with self.assertRaises(ValueError): s.observe_terminal([replace(valid, terminated=True, success=True)])
        with self.assertRaises(ValueError): s.observe_terminal([replace(valid, steps=4)])
        self.assertEqual(s.state_dict(), before)

    def test_cutoff_is_censored_and_all_exposure_is_retained(self):
        s = session(); s.assign([0, 1]); s.advance_all(12); s.advance([1], 3)
        rows = s.censor_active('training_cutoff')
        self.assertEqual(sorted(x['steps'] for x in rows), [12, 15])
        self.assertEqual(sum(s.transitions), 27)
        self.assertEqual(sum(s.censored_transitions), 27)
        self.assertEqual(sum(s.engine.completed_counts), 0)
        self.assertEqual(sum(s.engine.success_counts), 0)
        self.assertEqual(s.engine.outcomes, [[] for _ in range(4)])
        restored = session(); restored.load_state_dict(json.loads(json.dumps(s.state_dict())))
        self.assertEqual(restored.state_dict(), s.state_dict())

    def test_fixed_eval_cannot_feed_training_or_draw_its_rng(self):
        s = session(); s.assign(range(8)); s.advance_all(4)
        before = s.state_dict()
        evaluation = FixedEvaluation(s.pool)
        self.assertEqual(evaluation.assignments(range(64)), {i:s.pool.ids[i%4] for i in range(64)})
        with self.assertRaises(RuntimeError): evaluation.observe_terminal([])
        self.assertEqual(s.state_dict(), before)

    def test_independent_global_rng_and_B1_B2_history(self):
        random.seed(123); before = random.getstate()
        b1 = session(); b2 = session(run='B2-test', seed=19021)
        b2_before = b2.state_dict()
        a = b1.assign([0])[0]; b1.advance_all(2)
        b1.observe_terminal([terminal(b1, a, success=False)])
        self.assertEqual(random.getstate(), before)
        self.assertEqual(b2.state_dict(), b2_before)
        with self.assertRaises(ValueError): b2.load_state_dict(b1.state_dict())

    def test_snapshot_is_deep_and_json_restore_replays_assignments(self):
        a = session(); a.assign(range(4)); a.advance_all(9)
        snapshot = a.state_dict(); encoded = json.dumps(snapshot)
        b = session(); b.load_state_dict(json.loads(encoded))
        for i in (0, 2):
            for s in (a, b):
                s.observe_terminal([terminal(s, s.assignment(i), success=False)])
        self.assertEqual(a.assign([0, 2]), b.assign([2, 0]))
        self.assertEqual(a.state_dict(), b.state_dict())
        self.assertEqual(json.dumps(snapshot), encoded)
        a.advance_all(11)
        self.assertEqual(json.dumps(snapshot), encoded)

    def test_corrupt_restore_rejected_without_partial_mutation(self):
        s = session(); s.assign(range(4)); s.advance_all(3)
        before = s.state_dict()
        mutations = [lambda x:x['transitions'].__setitem__(0, -1),
                     lambda x:x['active_counts'].__setitem__(0, 99),
                     lambda x:x['engine']['outcomes'].__setitem__(0, [2]),
                     lambda x:x.__setitem__('support_sha256', 'foreign'),
                     lambda x:x['engine'].__setitem__('rng_state', [3, [], None])]
        for mutate in mutations:
            corrupted = copy.deepcopy(before); mutate(corrupted)
            with self.assertRaises((ValueError, TypeError)): s.load_state_dict(corrupted)
            self.assertEqual(s.state_dict(), before)

    def test_latest_feedback_failure_timeout_cutoff_and_boundaries(self):
        m = good_metrics()
        good = t1_episode_labels(m, terminated=False, timeout=True)
        self.assertTrue(good['joint_success_5cm']); self.assertTrue(good['joint_success_10cm'])
        self.assertFalse(t1_episode_labels(m, terminated=True, timeout=True)['joint_success_5cm'])
        with self.assertRaises(ValueError): t1_episode_labels(m, terminated=False, timeout=False)
        for field, value in [('xy_rmse', .15), ('yaw_rmse', .2), ('ee_position_rmse', .05),
                             ('ee_orientation_mean', .2), ('nonfoot_contact', True),
                             ('base_clearance_min', .14999), ('joint_velocity_limit_ratio_max', 1.00011),
                             ('joint_position_limit_excess_max_rad', .00101), ('steps', 999)]:
            self.assertFalse(t1_episode_labels(dict(m, **{field:value}), terminated=False, timeout=True)['joint_success_5cm'], field)
        boundary = dict(m, base_clearance_min=.15, joint_velocity_limit_ratio_max=1.0001,
                        joint_position_limit_excess_max_rad=.001)
        self.assertTrue(t1_episode_labels(boundary, terminated=False, timeout=True)['joint_success_5cm'])
        self.assertTrue(t1_episode_labels(dict(m, ee_position_rmse=.05), terminated=False, timeout=True)['joint_success_10cm'])

    def test_uniform_and_sgs_start_with_paired_balanced_commands(self):
        u = session(mode='uniform'); s = session()
        global_rng = random.getstate()
        for x in (u, s):
            x.assign(range(4096), initial_task_ids={i:x.pool.ids[i%4] for i in range(4096)})
            self.assertEqual(x.engine.sample_counts, [1024]*4)
            self.assertEqual(x.engine.rng.getstate(), random.Random(9021).getstate())
        self.assertEqual([u.assignment(i).task_id for i in range(4096)],
                         [s.assignment(i).task_id for i in range(4096)])
        self.assertEqual(random.getstate(), global_rng)


if __name__ == '__main__':
    unittest.main(verbosity=2)
