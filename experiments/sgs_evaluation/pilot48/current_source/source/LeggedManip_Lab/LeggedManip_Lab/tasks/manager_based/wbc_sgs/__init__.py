"""Independent registration. The GO2-PIPER-WBC registration is untouched."""
import gymnasium as gym

gym.register(id="GO2-PIPER-WBC-SGS", entry_point=f"{__name__}.env:FixedTaskWBCEnv",
    disable_env_checker=True, kwargs={
        'env_cfg_entry_point': f'{__name__}.env_cfg:FixedTaskWBCEnvCfg',
        'rsl_rl_cfg_entry_point': 'LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.go2_piper.agents.rsl_rl_ppo_cfg:Go2PiperWBCPPORunnerCfg'})
