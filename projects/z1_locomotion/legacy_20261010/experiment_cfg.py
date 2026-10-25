"""Separate stair curriculum experiment; leaves every upstream A2 source file unchanged."""
import gymnasium as gym
from isaaclab.utils import configclass
from basic_locomotion_isaaclab.tasks.locomotion.a2_env_cfg import A2RoughBlindEnvCfg
from basic_locomotion_isaaclab.tasks.locomotion.locomotion_env import LocomotionEnv
from basic_locomotion_isaaclab.tasks.locomotion.agents.rsl_rl_ppo_cfg import RoughPPORunnerCfg


@configclass
class A2StairsCurriculumCfg(A2RoughBlindEnvCfg):
    def __post_init__(self):
        gen = self.terrain.terrain_generator
        gen.curriculum = True
        self.terrain.max_init_terrain_level = 0
        weights = {'flat':0.10,'boxes':0.05,'star':0.0,'random_rough':0.05,
                   'hf_pyramid_slope':0.10,'hf_pyramid_slope_inv':0.10,
                   'pyramid_stairs':0.30,'pyramid_stairs_inv':0.30}
        for name,terrain in gen.sub_terrains.items():
            terrain.proportion = weights[name]
        for name in ('pyramid_stairs','pyramid_stairs_inv'):
            gen.sub_terrains[name].step_height_range = (0.025,0.16)


gym.register('Locomotion-A2-Rough-Stairs-Curriculum',entry_point=LocomotionEnv,
             disable_env_checker=True,kwargs={'env_cfg_entry_point':A2StairsCurriculumCfg,
                                               'rsl_rl_cfg_entry_point':RoughPPORunnerCfg})


@configclass
class A2Stairs20CurriculumCfg(A2StairsCurriculumCfg):
    """Separate high-riser profile motivated by measured 16/18cm boundary failures."""
    def __post_init__(self):
        super().__post_init__()
        for name in ('pyramid_stairs','pyramid_stairs_inv'):
            self.terrain.terrain_generator.sub_terrains[name].step_height_range = (0.025,0.20)


gym.register('Locomotion-A2-Rough-Stairs20-Curriculum',entry_point=LocomotionEnv,
             disable_env_checker=True,kwargs={'env_cfg_entry_point':A2Stairs20CurriculumCfg,
                                               'rsl_rl_cfg_entry_point':RoughPPORunnerCfg})


@configclass
class A2Stairs20DescentExposureCfg(A2Stairs20CurriculumCfg):
    """Bias the initial outward motion toward descent; retain native rewards, curriculum and actuators."""
    def __post_init__(self):
        super().__post_init__()
        # The positive pyramid spawns at its high central platform; outward motion descends.
        # Keep the aggregate stair share at 60% and every other terrain proportion unchanged.
        self.terrain.terrain_generator.sub_terrains['pyramid_stairs'].proportion = .45
        self.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv'].proportion = .15


gym.register('Locomotion-A2-Rough-Stairs20-DescentExposure',entry_point=LocomotionEnv,
             disable_env_checker=True,kwargs={'env_cfg_entry_point':A2Stairs20DescentExposureCfg,
                                               'rsl_rl_cfg_entry_point':RoughPPORunnerCfg})
