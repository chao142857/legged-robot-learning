"""Only change the standing allocation; all inherited v6 parameters remain unchanged."""
def register():
    from register_tasks import register as baseline_register
    baseline_register()
    import gymnasium as gym
    from isaaclab.utils import configclass
    from z1_team_cfg import Z1TeamLegSafeCfg
    from z1_cfg import Z1Env
    from basic_locomotion_isaaclab.tasks.locomotion.agents.rsl_rl_ppo_cfg import RoughPPORunnerCfg
    @configclass
    class Z1ShortSamplingCfg(Z1TeamLegSafeCfg):
        standing_env_fraction: float = .061
        standing_seed: int = 42
    gym.register('Locomotion-Z1-Team-Rough-Blind-LegSafe-Short5060', entry_point=Z1Env,
                 disable_env_checker=True, kwargs={'env_cfg_entry_point': Z1ShortSamplingCfg,
                                                  'rsl_rl_cfg_entry_point': RoughPPORunnerCfg})
