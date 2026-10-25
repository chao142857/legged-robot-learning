"""Prepared A/B/C task registrations. This file never starts a training process."""
import json
from pathlib import Path
def register():
 from register_tasks import register as baseline_register
 baseline_register()
 import gymnasium as gym
 from isaaclab.utils import configclass
 from z1_team_cfg import Z1TeamLegSafeCfg
 from z1_cfg import Z1Env
 from basic_locomotion_isaaclab.tasks.locomotion.agents.rsl_rl_ppo_cfg import RoughPPORunnerCfg
 base=Path(__file__).resolve().parents[2]
 c=json.loads((base/'diagnostics/stance_height_20261016/candidates.json').read_text())
 @configclass
 class StanceA(Z1TeamLegSafeCfg):
  standing_env_fraction: float = .061
  standing_seed: int = 42
 @configclass
 class StanceB(StanceA):
  def __post_init__(self):
   super().__post_init__();self.desired_base_height=c['B']['desired_base_height']
 @configclass
 class StanceC(StanceA):
  def __post_init__(self):
   super().__post_init__();theta=c['C_theta_rad']
   self.robot.init_state.joint_pos={'.*_hip_joint':0.,'.*_thigh_joint':theta,'.*_calf_joint':-2*theta}
 @configclass
 class StanceB1(StanceA):
  def __post_init__(self):
   super().__post_init__();self.desired_base_height=.384
 for label,cls in [('A',StanceA),('B',StanceB),('B1',StanceB1),('C',StanceC)]:
  gym.register(f'Locomotion-Z1-Team-Stance-{label}-Rough-LegSafe',entry_point=Z1Env,disable_env_checker=True,
   kwargs={'env_cfg_entry_point':cls,'rsl_rl_cfg_entry_point':RoughPPORunnerCfg})
