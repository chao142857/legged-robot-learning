"""Fixed-task derivative: original WBC reward, actions, observations, asset, PPO."""
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from ..leggedmanip_lab.config.go2_piper.wbc_env_cfg import Go2PiperWBCEnvCfg
from .commands import EpisodeTaskCommandCfg
from .env import apply_task_reset, observe_joint_metrics


@configclass
class EpisodeCommandsCfg:
    ee_pose = EpisodeTaskCommandCfg(channel='ee_pose')
    base_velocity = EpisodeTaskCommandCfg(channel='velocity')


@configclass
class FixedTaskWBCEnvCfg(Go2PiperWBCEnvCfg):
    commands: EpisodeCommandsCfg = EpisodeCommandsCfg()
    task_pool_path: str = ''
    success_protocol_path: str = ''
    sampler_mode: str = 'uniform'
    sampler_seed: int = 13
    episode_log_path: str = ''

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.
        self.scene.num_envs = 128
        self.scene.env_spacing = 4.
        # Fixed-task experiment: all reset variation is explicitly in the pool.
        for name in ['physics_material','add_mass','base_com','randomize_actuator_gains',
                     'randomize_rigid_body_inertia','reset_robot_joints','push_robot']:
            setattr(self.events, name, None)
        self.events.reset_base = EventTermCfg(func=apply_task_reset, mode='reset')
        # Disabled original curriculum FUNCTIONS still replace ranges with limits.
        # Remove those terms altogether in this derivative only.
        for name in ['lin_vel_cmd_levels','ang_vel_cmd_levels','pos_cmd_levels']:
            setattr(self.curriculum, name, None)
        self.terminations.joint_observer = TerminationTermCfg(func=observe_joint_metrics)
        self.observations.policy.enable_corruption = False
        # No cameras/render assets needed; reuse local physics asset unmodified.
        self.scene.terrain.visual_material = None
        self.scene.sky_light.spawn.texture_file = None
