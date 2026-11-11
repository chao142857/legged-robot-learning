"""E3 installs the episode journal behind the E2 terminal ownership checks."""
import numpy as np
import torch

from adapter import SequenceAdapter
from design import mapping_is_balanced
from grouped_sampling import GroupedOwnership
from journal import JournalSampler


class E3Adapter(SequenceAdapter):
    def __init__(self, env, pool, out, mapping):
        assert mapping_is_balanced(mapping)
        super().__init__(env, "wbc", "sgs", pool, out)
        self.owner = GroupedOwnership(pool, env.num_envs, mapping, seed=42)
        self.owner.engine = JournalSampler(mapping, pool.identity, seed=42)

    def restore(self, state):
        assert state["kind"] == "wbc"
        self.owner.load_state_dict(state["owner"])
        self.metrics.load_state_dict(state["metrics"])
        self.task_tensor.copy_(state["task_tensor"].to(self.env.device))
        for key, value in state["cursor"].items():
            self.cursor[key].copy_(value.to(self.env.device))
        summary = state["summary"]
        self.episodes = summary["episodes"]
        self.successes = summary["successes"]
        self.failures = summary["native_failures"]
        effective = summary["effective_commands"]
        for name, values in effective["command_draws"].items():
            for key, value in values.items():
                target = self.command_stats[name][key]
                target.copy_(torch.as_tensor(value, dtype=target.dtype, device=target.device))
        self.standing_draws.fill_(effective["standing_draws"])
        self.velocity_draws.fill_(effective["velocity_draws"])
        self.owner.check()
        np.testing.assert_array_equal(self.task_tensor.cpu().numpy(), self.owner.task)
