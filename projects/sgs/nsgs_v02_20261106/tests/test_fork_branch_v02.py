import copy
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from fork_v02 import convert_adapter, partial_features, learning_progress, restored_owner
from neural_owner import NeuralOwner
from neural_v01 import NeuralSampler
def tree_hash(value):
    return json.dumps(value, sort_keys=True, default=lambda x: x.tolist())


class ForkTests(unittest.TestCase):
    def source(self):
        journal = np.array([[0, 1, 100, 0, 1], [1, 0, 200, 1, 1]], dtype=np.int32)
        return dict(owner=dict(engine=dict(episode_journal=journal,
            sample_counts=[2, 2, 0, 0], completed_counts=[1, 1, 0, 0],
            success_counts=[1, 0, 0, 0], rng_state=('old', 42)),
            task=np.array([0, 1]), generation=np.array([2, 2]), steps=np.array([50, 60]),
            transitions=np.array([200, 200, 0, 0]), control_steps=200,
            accepted=2, censored=np.zeros(4, dtype=np.int64),
            closed_generation=np.array([1, 1]), evaluation_ignored=0,
            pool_identity='synthetic'),
            metrics=dict(scored=torch.tensor([20, 30]), good=torch.tensor([10, 15]),
                sq=torch.tensor([[.8, 1.8, .05, .8], [1.2, 2.7, .075, 1.2]])),
            summary=dict(episodes=2, successes=1, native_failures=0,
                effective_commands=dict(command_draws={name: dict(count=4, minimum=[0.]*dim, maximum=[1.]*dim)
                    for name, dim in [('ee', 7), ('velocity', 3)]}, standing_draws=1, velocity_draws=4)))

    def row(self, step, seq, success, env):
        return dict(control_step=step, task_ids=[seq], env_ids=[env], generation=[1],
            rows=[[100, 90 if success else 50, .1, .2, .04, .1, success, 0, 1]])

    def test_real_feedback_prefix_and_restore(self):
        source = self.source()
        before = tree_hash(source)
        with tempfile.TemporaryDirectory() as name:
            p, q = Path(name)/'prefix.jsonl', Path(name)/'branch.jsonl'
            p.write_text(json.dumps(self.row(100, 0, 1, 0))+'\n')
            q.write_text(json.dumps(self.row(200, 1, 0, 1))+'\n'+json.dumps(self.row(201, 3, 1, 0))+'\n')
            converted, receipt = convert_adapter(source, [p, q], 123)
        self.assertEqual(receipt['committed_terminal_records'], 2)
        self.assertEqual(receipt['duplicate_completed_updates'], 0)
        self.assertEqual(before, tree_hash(source))
        engine = NeuralSampler(4, 0)
        engine.load_state_dict(converted['owner']['engine'])
        np.testing.assert_array_equal(engine.completed_counts, [1, 1, 0, 0])
        np.testing.assert_array_equal(engine.success_counts, [1, 0, 0, 0])
        np.testing.assert_array_equal(engine.last_feedback_step, [100, 200, -1, -1])
        self.assertEqual(engine.last_real_row[1, 1], 50)
        engine.install(np.full(4, .25), 0)
        clone = NeuralSampler(4, 2)
        clone.load_state_dict(engine.state_dict())
        np.testing.assert_array_equal(engine.sample(1000), clone.sample(1000))
        np.testing.assert_array_equal(converted['owner']['teacher_version'], [-1, -1])
        self.assertTrue((converted['owner']['decision_id'] < 0).all())

    def test_mismatched_feedback_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)/'feedback.jsonl'
            p.write_text(json.dumps(self.row(100, 0, 0, 0))+'\n')
            with self.assertRaises(AssertionError):
                convert_adapter(self.source(), p, 123)

    def test_bootstrap_only_fields_cannot_survive_source_restore(self):
        source = self.source()
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)/'feedback.jsonl'
            p.write_text(json.dumps(self.row(100, 0, 1, 0))+'\n'+json.dumps(self.row(200, 1, 0, 1))+'\n')
            converted, _ = convert_adapter(source, p, 123)
        state = converted['owner']
        pool = SimpleNamespace(n=4, identity=state['pool_identity'])
        bootstrap = NeuralOwner(pool, 2, 99)
        bootstrap.engine.install(np.full(4, .25), 0)
        bootstrap.assign(np.arange(2))
        self.assertIn('last_serials', bootstrap.engine.state_dict())
        self.assertNotIn('last_serials', state['engine'])
        # This is the observed failing GPU lifecycle, not a fabricated mismatch.
        bootstrap.load_state_dict(state)
        self.assertIn('last_serials', bootstrap.engine.state_dict())
        clean = restored_owner(pool, state)
        self.assertEqual(tree_hash(clean.state_dict()), tree_hash(state))

    def test_partial_data_is_separate_and_weighted(self):
        source = self.source()
        source['owner']['task'][:] = 0
        f, receipt = partial_features(source, 4)
        self.assertEqual(receipt['contribution_to_completed_history'], 0)
        self.assertEqual(f[0, 0], 1)
        self.assertEqual(f[1, 0], 0)
        self.assertAlmostEqual(f[0, 3], .5)
        self.assertAlmostEqual(f[0, 4], 1.)
        self.assertAlmostEqual(f[0, 5], 1.)
        self.assertAlmostEqual(f[0, 6], 1.)

    def test_learning_progress_uses_real_two_halves(self):
        engine = NeuralSampler(4, 0)
        success = np.array([100, 90, .1, .2, .04, .1, 1, 0, 1.])
        failed = success.copy(); failed[1] = 10; failed[6] = 0
        engine.bootstrap_record(0, failed, 1000)
        engine.bootstrap_record(0, success, 20000)
        engine.bootstrap_record(1, success, 20000)
        p, mask = learning_progress(engine, 24000)
        self.assertAlmostEqual(p[0], .8)
        self.assertEqual(mask[0], 1)
        self.assertEqual(mask[1], 0)


if __name__ == '__main__':
    unittest.main()
