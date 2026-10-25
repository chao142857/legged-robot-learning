"""Both actual command sources read the same episode task; no interval resampling."""
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class EpisodeTaskCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.value = torch.zeros(self.num_envs, 3 if cfg.channel == 'velocity' else 7, device=self.device)
        if cfg.channel != 'velocity':
            self.value[:, 3] = 1.

    @property
    def command(self):
        return self.value

    def _resample_command(self, env_ids):
        task_ids = self._env.wbc_task_ids[env_ids]
        assert bool((task_ids >= 0).all()), 'Task assignment must precede command reset'
        table = self._env.task_velocities if self.cfg.channel == 'velocity' else self._env.task_poses
        self.value[env_ids] = table[task_ids]

    def compute(self, dt):
        # A genuine command source: no parent timer or random generation runs.
        # It remains fixed even when a caller forces time_left to zero.
        self._update_metrics()

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass


@configclass
class EpisodeTaskCommandCfg(CommandTermCfg):
    class_type: type = EpisodeTaskCommand
    channel: str = 'velocity'
    resampling_time_range: tuple = (1.e9, 1.e9)
    debug_vis: bool = False
