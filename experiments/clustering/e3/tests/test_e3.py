import copy
import math
import pickle
import unittest

import numpy as np
from design import LOCO, MANIP, map_key, map_change, normalized_blocks, weighted_features
from grouped_sampling import GroupedSampler
from journal import JournalSampler


class JournalTest(unittest.TestCase):
    def setUp(self):
        self.old = np.repeat(np.arange(2), 8)
        self.new = np.tile(np.arange(2), 8)
        self.sampler = JournalSampler(self.old, "test-pool", history=3)
        self.data = [(1, 1), (0, 0), (2, 1), (4, 0), (6, 0), (8, 0), (10, 1), (12, 1), (14, 0)]
        for tick, (seq, label) in enumerate(self.data, 1):
            self.sampler.record([seq], [label], control_step=tick, event_ids=[(0, tick)])

    def test_switch_recovers_feedback_evicted_from_old_window(self):
        self.assertNotIn(1, self.sampler.window_members[0])
        rng = self.sampler.rng.getstate()
        journal = self.sampler.journal.copy()
        self.assertTrue(self.sampler.switch(self.new))
        self.assertEqual(self.sampler.window_members[1], [1])
        self.assertEqual(self.sampler.outcomes[1], [1])
        np.testing.assert_array_equal(self.sampler.journal, journal)
        self.assertEqual(self.sampler.rng.getstate(), rng)
        reference = GroupedSampler(self.new, "test-pool", history=3)
        for seq, label, tick, env, generation in journal:
            reference.record([seq], [label], control_step=tick, event_ids=[(env, generation)])
        np.testing.assert_array_equal(self.sampler.probabilities(), reference.probabilities())
        self.assertEqual(self.sampler.window_events, reference.window_events)
        self.assertFalse(self.sampler.switch(self.new))

    def test_ties_retain_native_terminal_order(self):
        sampler = JournalSampler(self.old, "p", history=1)
        sampler.record([3, 1], [0, 1], control_step=6, event_ids=[(0, 1), (1, 1)])
        sampler.switch(self.new)
        self.assertEqual(sampler.window_members[1], [1])
        self.assertEqual(sampler.window_events[1], [(1, 1)])

    def test_probability_formula(self):
        cold = JournalSampler(self.old, "p")
        np.testing.assert_allclose(cold.probabilities(), np.full(16, 1 / 16))
        self.sampler.switch(self.new)
        eps = 1e-4
        factors = [math.sqrt(max(math.sqrt((p + eps) * (1 - p + eps)), eps) + eps)
                   for p in self.sampler.group_rates()]
        expected = np.asarray(factors)[self.new]
        expected /= expected.sum()
        np.testing.assert_allclose(self.sampler.probabilities(), expected, atol=1e-15, rtol=0)
        self.assertAlmostEqual(sum(self.sampler.group_probabilities()), 1.0, places=15)

    def test_checkpoint_round_trip_rng_active_history_and_later_remap(self):
        self.sampler.sample(41)
        self.sampler.switch(self.new)
        saved = pickle.loads(pickle.dumps(self.sampler.state_dict()))
        restored = JournalSampler(self.new, "test-pool", seed=1, history=3)
        restored.load_state_dict(saved)
        self.assertEqual(restored.sample(4096), self.sampler.sample(4096))
        for instance in (restored, self.sampler):
            instance.record([15], [1], control_step=10, event_ids=[(0, 10)])
            instance.switch(self.old)
        self.assertEqual(restored.window_members, self.sampler.window_members)
        self.assertEqual(restored.window_events, self.sampler.window_events)
        self.assertEqual(restored.rng.getstate(), self.sampler.rng.getstate())
        np.testing.assert_array_equal(restored.journal, self.sampler.journal)

    def test_wrong_checkpoint_cannot_mutate_sampler(self):
        state = self.sampler.state_dict()
        bad = copy.deepcopy(state)
        bad["episode_journal"][0, 1] = 0
        with self.assertRaises(ValueError):
            self.sampler.load_state_dict(bad)
        np.testing.assert_array_equal(self.sampler.journal, state["episode_journal"])
        self.assertEqual(self.sampler.rng.getstate(), state["rng_state"])
        with self.assertRaises(ValueError):
            self.sampler.record([1], [1], control_step=10, event_ids=[(0, 1)])
        np.testing.assert_array_equal(self.sampler.journal, state["episode_journal"])


class DesignTest(unittest.TestCase):
    def test_schedule_boundaries(self):
        for update in (0, 1999, 2000, 2499):
            self.assertEqual(map_key("D", update), map_key("B", update))
        for i in range(1, 17):
            self.assertEqual(map_key("D", 2000 + 500 * i), f"m{i:02d}")
        self.assertEqual(map_key("D", 9999), "m15")

    def test_squared_metric_weights_and_normalization(self):
        dims = {name: 1 for name in sorted(LOCO | MANIP)}
        x = np.random.default_rng(42).normal(size=(100, len(dims)))
        blocks, _ = normalized_blocks(x, dims)
        for block in blocks.values():
            self.assertAlmostEqual(2 * block.var(axis=0).sum(), 1.0)
        for w in (.5, .8, 1):
            a, b = weighted_features(blocks, w)[[0, 1]]
            loco = np.sum((blocks["loco"][0] - blocks["loco"][1]) ** 2)
            manip = np.sum((blocks["manip"][0] - blocks["manip"][1]) ** 2)
            self.assertAlmostEqual(np.sum((a - b) ** 2), (1 - w) * loco + w * manip)

    def test_relabel_is_not_member_change(self):
        old = np.repeat(np.arange(2), 8)
        self.assertEqual(map_change(old, 1 - old)["sequences_with_changed_group_members"], 0)
        self.assertEqual(map_change(old, np.tile(np.arange(2), 8))["sequences_with_changed_group_members"], 16)


if __name__ == "__main__":
    unittest.main()
