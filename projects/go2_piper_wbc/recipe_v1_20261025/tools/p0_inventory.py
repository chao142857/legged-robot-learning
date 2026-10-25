"""Read actual action/actuator state and perform eight headless physics steps."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
out = Path(args.out)
if out.exists(): parser.error('Use a fresh output directory')
app = AppLauncher(args).app

import json
import torch
import gymnasium as gym
import LeggedManip_Lab.tasks
import task_suite
from task_suite.config import build_cfg
from telemetry import plain, write_json, sha


def main():
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    assert Path(task_suite.__file__).resolve().is_relative_to(source.parent/'task_suite_20261023')
    cfg = build_cfg('T1', num_envs=8, seed=8101, ee_mode='smooth', evaluation=True)
    cfg.sim.device = args.device or 'cuda:0'
    env = gym.make('GO2-PIPER-WBC', cfg=cfg)
    raw = env.unwrapped
    try:
        env.reset()
        robot = raw.scene['robot']; term = raw.action_manager.get_term('joint_pos')
        ids = term._joint_ids
        lower = robot.data.soft_joint_pos_limits[0, ids, 0]
        upper = robot.data.soft_joint_pos_limits[0, ids, 1]
        offset = robot.data.default_joint_pos[0, ids]
        table = []
        hard = robot.data.joint_pos_limits[0, ids]
        for j, name in enumerate(term._joint_names):
            table.append(dict(name=name, action_index=j, default_rad=float(offset[j]),
                hard_limits_rad=plain(hard[j]), soft_limits_rad=[float(lower[j]), float(upper[j])],
                margin_002_limits_rad=[float(lower[j]+.02), float(upper[j]-.02)],
                default_margin_rad=float(torch.minimum(offset[j]-lower[j], upper[j]-offset[j]))))
        synthetic = []
        for v in (-100., -40., 0., 40., 100.):
            actions = torch.full((8,18), v, device=raw.device)
            term.process_actions(actions)
            expected = (actions*.25 + term._offset).clamp(-10.,10.)
            error = float((term.processed_actions-expected).abs().max())
            assert error < 1e-6
            synthetic.append(dict(raw_value=v, processed_targets_rad=plain(term.processed_actions[0]), max_formula_error=error))
        term.process_actions(torch.zeros((8,18),device=raw.device))
        actuators = {}
        for name, actuator in robot.actuators.items():
            row = {'class':type(actuator).__name__, 'cfg':plain(actuator.cfg)}
            for attr in ('joint_names','joint_indices','effort_limit','velocity_limit','stiffness','damping','computed_effort','applied_effort'):
                if hasattr(actuator,attr): row[attr] = plain(getattr(actuator,attr))
            actuators[name] = row
        for _ in range(8):
            env.step(torch.zeros((8,18),device=raw.device))
            assert bool(torch.isfinite(robot.data.joint_pos).all())
        result = dict(passed=True, source=str(source), task_suite_import=str(task_suite.__file__),
            physics_steps=8, scope='Interface and finite-state smoke, not a capability evaluation',
            scale=plain(term._scale), clip=plain(term._clip[0]), joint_table=table,
            synthetic_pipeline=synthetic, actuators=actuators,
            final_joint_position=plain(robot.data.joint_pos), final_computed_torque=plain(robot.data.computed_torque),
            final_applied_torque=plain(robot.data.applied_torque),
            body_names=robot.body_names, body_position_world=plain(robot.data.body_pos_w),
            body_quaternion_world=plain(robot.data.body_quat_w),
            asset_sha256=sha(cfg.scene.robot.spawn.usd_path),
            observation_dims=plain(raw.observation_manager.group_obs_dim),
            action_dim=raw.action_manager.total_action_dim)
        write_json(out/'inventory.json', result)
        print('P0_INVENTORY_COMPLETE', str(out), flush=True)
    finally:
        env.close()


try:
    with torch.inference_mode(): main()
except BaseException as exc:
    import traceback
    print(traceback.format_exc(), flush=True)
    if out.exists(): write_json(out/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
    raise
finally: app.close()
