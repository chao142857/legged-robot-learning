"""Additive configurations derived from the verified native WBC task.

This module never registers or changes GO2-PIPER-WBC. build_cfg is deliberately
available for quarantined smoke tests; the runner must verify admission before PPO.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

PROFILES = json.loads(Path(__file__).with_name('profiles.json').read_text())


def profile_signature(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_cfg(family, terrain='stairs_up', difficulty='easy', num_envs=64,
              seed=7001, ee_mode='smooth', evaluation=False, allow_unvalidated=False,
              profile_overrides=None, profile_json=None):
    from isaaclab.managers import EventTermCfg, TerminationTermCfg
    from isaaclab.sensors import RayCasterCfg, patterns
    from isaaclab.terrains import TerrainGeneratorCfg, MeshPyramidStairsTerrainCfg, HfSteppingStonesTerrainCfg
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.go2_piper.wbc_env_cfg import Go2PiperWBCEnvCfg
    from . import mdp

    if family not in ('T0', 'T1', 'T2') or ee_mode not in ('fixed', 'smooth'):
        raise ValueError('Unknown task family or EE mode')
    if difficulty != 'easy' and not (profile_overrides or profile_json):
        raise ValueError('Intermediate/advanced ranges are quarantined: provide exact profile overrides for physical validation first')
    if family == 'T2' and terrain not in ('stairs_up', 'stairs_down', 'stepping_stones'):
        raise ValueError('T2 requires a real supported mesh terrain')
    cfg = Go2PiperWBCEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    p = copy.deepcopy(PROFILES['easy'])
    if family == 'T2' and terrain == 'stepping_stones':
        p['route_distance'] = 3.0
    overrides = dict(profile_overrides or {})
    if profile_json:
        overrides.update(json.loads(Path(profile_json).read_text()))
    unknown = set(overrides)-set(p)
    if unknown:
        raise ValueError(f'Unknown profile parameters: {sorted(unknown)}')
    p.update(overrides)
    if not (0 < p['ee_period_s'] and 0 < p['base_segment_s'] and 0 < p['terrain_speed']):
        raise ValueError('Command periods and route speed must be positive')
    if not (0 < p['route_distance'] <= 3.5):
        raise ValueError('This 8m terrain tile admits route distances up to 3.5m; longer paths need a separately validated tile')
    if min(p['ee_amplitude']) < 0 or len(p['ee_center']) != 3 or len(p['ee_amplitude']) != 3:
        raise ValueError('EE center/amplitude must be xyz triples with nonnegative amplitude')
    meta = dict(schema_version=1, family=family, terrain=terrain if family == 'T2' else 'plane',
                difficulty=difficulty, ee_mode=ee_mode, params=p,
                terrain_seed=20261023 if family == 'T2' else None,
                ee_frame='xy in link0; z world with support-height offset; quaternion relative link0',
                policy_observation='native unchanged; scanner excluded from actor and critic',
                curriculum='disabled in both prospective sampling arms',
                route=None)
    # Explicitly prevent native curricula from extending the admitted parameter range.
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.curriculum.ang_vel_cmd_levels = None
    cfg.curriculum.pos_cmd_levels = None
    cfg.commands.ee_pose.class_type = mdp.SuitePoseCommand
    cfg.commands.ee_pose.debug_vis = False
    cfg.commands.ee_pose.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_velocity.class_type = mdp.SuiteVelocityCommand
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    if family in ('T0', 'T1'):
        # SuiteVelocityCommand consumes the explicit profile, not native ranges.
        # Make the serialized ranges truthful without changing emitted commands.
        linear = p['base_linear_speed'] if family == 'T1' else 0.0
        angular = p['base_yaw_rate'] if family == 'T1' else 0.0
        for target in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
            target.lin_vel_x = (-linear, linear)
            target.lin_vel_y = (0.0, 0.0)
            target.ang_vel_z = (-angular, angular)
    for target in (cfg.commands.ee_pose.ranges, cfg.commands.ee_pose.limit_ranges):
        target.pos_x = (p['ee_center'][0]-p['ee_amplitude'][0], p['ee_center'][0]+p['ee_amplitude'][0])
        target.pos_y = (-p['ee_amplitude'][1], p['ee_amplitude'][1])
        target.pos_z = (p['ee_center'][2]-p['ee_amplitude'][2], p['ee_center'][2]+p['ee_amplitude'][2])
        target.roll = target.pitch = target.yaw = (0., 0.)
    cfg.episode_length_s = p['episode_s_flat']
    cfg.events.reset_base = EventTermCfg(func=mdp.reset_suite_root, mode='reset')
    # Native broad reset scaling can plant randomized calves in stairs. Retain reset
    # event and actuators but use the nominal joint pose for this admission stage.
    cfg.events.reset_robot_joints.params['position_range'] = (1.0, 1.0)
    cfg.terminations.nonfoot_contact = TerminationTermCfg(func=mdp.nonfoot_contact,
                                                         params={'threshold': 5.0, 'grace_s': 0.5})
    if family == 'T2':
        cfg.episode_length_s = p['episode_s_terrain']
        if terrain in ('stairs_up', 'stairs_down'):
            sub = MeshPyramidStairsTerrainCfg(proportion=1., step_height_range=(p['stairs_height'],)*2,
                    step_width=p['stairs_tread'], platform_width=2., border_width=1., holes=False)
            start, goal = ((-p['route_distance'], 0.), (0., 0.)) if terrain == 'stairs_up' else ((0., 0.), (p['route_distance'], 0.))
        else:
            sub = HfSteppingStonesTerrainCfg(proportion=1., stone_height_max=p['stones_height'],
                    stone_width_range=(p['stones_width'],)*2, stone_distance_range=(p['stones_gap'],)*2,
                    holes_depth=-0.3, platform_width=2., border_width=0.)
            start, goal = (0., 0.), (p['route_distance'], 0.)
        meta['route'] = dict(start_offset=list(start), goal_offset=list(goal),
                             goal_tolerance_m=.25, nominal_speed_mps=p['terrain_speed'], settle_s=1.0)
        cfg.scene.terrain.terrain_type = 'generator'
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(seed=meta['terrain_seed'], size=(8., 8.),
                border_width=2., num_rows=4, num_cols=4, curriculum=False,
                difficulty_range=(0., 0.), horizontal_scale=.02, vertical_scale=.005,
                sub_terrains={terrain: sub}, use_cache=True, color_scheme='height')
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.scene.terrain.visual_material = None
        cfg.scene.height_scanner = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/base',
                offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 2.)), ray_alignment='yaw',
                pattern_cfg=patterns.GridPatternCfg(resolution=.1, size=(.8, .6)),
                mesh_prim_paths=['/World/ground'], update_period=.02, debug_vis=False)
        cfg.rewards.track_base_height_exp.func = mdp.terrain_relative_base_height
        cfg.terminations.route_boundary = TerminationTermCfg(func=mdp.route_boundary)
    cfg.task_suite = meta
    cfg.task_suite_signature = profile_signature(meta)
    return cfg
