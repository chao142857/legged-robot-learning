"""Build Recipe V1 from native task data; PPO and robot are not reconstructed."""
import copy
import json
from pathlib import Path

from task_suite.config import build_cfg,profile_signature


def build_recipe(*,admission,num_envs,seed=21,evaluation=None,health_termination=True):
    from isaaclab.managers import RewardTermCfg,TerminationTermCfg
    from .control import SoftLimitJointPositionAction,SustainedLowHeight,low_height_penalty,raw_action_tail_penalty,ee_precision
    from .commands import StageAPoseCommand,StageAVelocityCommand,FrozenAPoseCommand,FrozenAVelocityCommand
    certificate=json.loads(Path(admission).read_text())
    if not certificate['passed']:raise RuntimeError('P0 has not passed')
    p=json.loads(Path(__file__).with_name('spec.json').read_text())
    p['admitted_ee_points']=certificate['admitted_ee_points']
    assert len(p['admitted_ee_points'])>=8,'Need an actually tested cloud, not one target'
    formal=evaluation in ('T1','T0')
    family='T0' if evaluation=='T0' else 'T1'
    overrides=None if formal else dict(base_linear_speed=.55 if evaluation is None else .45,
        base_yaw_rate=.55 if evaluation is None else .45,base_segment_s=10.,
        base_slew_linear=.4,base_slew_yaw=.6,ee_amplitude=[0.,0.,0.])
    cfg=build_cfg(family,num_envs=num_envs,seed=seed,ee_mode='smooth' if formal else 'fixed',
                  evaluation=evaluation is not None,profile_overrides=overrides)
    cfg.recipe_v1=copy.deepcopy(p)
    cfg.recipe_v1['stage']='A';cfg.recipe_v1['evaluation_protocol']=evaluation
    cfg.actions.joint_pos.class_type=SoftLimitJointPositionAction
    if not formal:
        cfg.commands.ee_pose.class_type=StageAPoseCommand if evaluation is None else FrozenAPoseCommand
        cfg.commands.base_velocity.class_type=StageAVelocityCommand if evaluation is None else FrozenAVelocityCommand
        cloud=p['admitted_ee_points']
        for ranges in (cfg.commands.ee_pose.ranges,cfg.commands.ee_pose.limit_ranges):
            ranges.pos_x=(min(x[0] for x in cloud),max(x[0] for x in cloud))
            ranges.pos_y=(min(x[1] for x in cloud),max(x[1] for x in cloud))
            ranges.pos_z=(min(x[2] for x in cloud),max(x[2] for x in cloud))
    cfg.rewards.track_lin_vel_xy_exp.params['std']=p['rewards']['linear_std_mps']
    cfg.rewards.track_ang_vel_z_exp.params['std']=p['rewards']['yaw_std_radps']
    cfg.rewards.track_base_height_exp.weight=p['rewards']['base_height_weight']
    cfg.rewards.low_height_l2=RewardTermCfg(func=low_height_penalty,weight=p['health']['penalty_weight'],
        params=dict(threshold=p['health']['penalty_height_m'],scale=p['health']['penalty_scale_m']))
    cfg.rewards.raw_action_tail_l2=RewardTermCfg(func=raw_action_tail_penalty,
        weight=p['rewards']['raw_tail_weight_candidate'],params=dict(threshold=p['rewards']['raw_tail_threshold']))
    cfg.rewards.ee_precision=RewardTermCfg(func=ee_precision,weight=0.,params=dict(std=.05))
    if health_termination:
        cfg.terminations.sustained_low_height=TerminationTermCfg(func=SustainedLowHeight,
            params=dict(threshold=p['health']['termination_height_m'],dwell_s=p['health']['termination_dwell_s'],grace_s=p['health']['reset_grace_s']))
    cfg.task_suite['recipe_v1']=dict(stage='A',evaluation_protocol=evaluation,admitted_ee_points=p['admitted_ee_points'],
        action_target_margin_rad=p['control']['target_margin_rad'],stage_A_sampling=p['stage_A'],health=p['health'])
    cfg.task_suite_signature=profile_signature(cfg.task_suite)
    return cfg
