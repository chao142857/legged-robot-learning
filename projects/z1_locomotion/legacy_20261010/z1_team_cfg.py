"""Independent team collision tasks; v5 tasks and their evidence remain reproducible."""
from pathlib import Path
import os
import gymnasium as gym
from isaaclab.utils import configclass
from basic_locomotion_isaaclab.tasks.locomotion.agents.rsl_rl_ppo_cfg import RoughPPORunnerCfg
from z1_cfg import Z1BaseOnlyCfg, Z1Env

@configclass
class Z1TeamBaseOnlyCfg(Z1BaseOnlyCfg):
    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = str(Path(os.environ['Z1_USD_PATH']))
        # Supplied box ends at calfz=-.30, without the old30mm pad radius.
        self.desired_base_height = .373

@configclass
class Z1TeamLegSafeCfg(Z1TeamBaseOnlyCfg):
    z1_contact_profile: str = 'leg_safe'

for name,cfg in [('BaseOnly',Z1TeamBaseOnlyCfg),('LegSafe',Z1TeamLegSafeCfg)]:
    gym.register(f'Locomotion-Z1-Team-Rough-Blind-{name}',entry_point=Z1Env,
                 disable_env_checker=True,kwargs={'env_cfg_entry_point':cfg,'rsl_rl_cfg_entry_point':RoughPPORunnerCfg})
