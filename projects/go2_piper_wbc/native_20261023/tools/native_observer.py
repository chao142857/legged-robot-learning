"""Read-only telemetry around unmodified upstream training calls.

No new random draws, environment resets, reward/command writes, policy forwards,
optimizer changes, or gradients are introduced. clip_grad_norm_ and KL wrappers
return the original result. Checkpoint serialization adds no optimizer updates.
"""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_error_magnitude, euler_xyz_from_quat


def plain(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "to_dict"):
        return plain(x.to_dict())
    if callable(x):
        return f"{x.__module__}:{getattr(x, '__qualname__', type(x).__name__)}"
    return str(x)


def write_json(path, data):
    Path(path).write_text(json.dumps(plain(data), indent=2, ensure_ascii=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def model_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def commands(env):
    result = {}
    for name in env.command_manager.active_terms:
        term = env.command_manager.get_term(name)
        cmd = term.command.detach()
        result[name] = dict(class_name=f"{type(term).__module__}.{type(term).__name__}",
                            cfg=plain(term.cfg), min=plain(cmd.amin(0)), max=plain(cmd.amax(0)),
                            mean=plain(cmd.mean(0)), sample=plain(cmd[:16]),
                            time_left=plain(term.time_left[:16]), command_counter=plain(term.command_counter[:16]))
    return result


def snapshot(env, runner=None):
    result = dict(common_step_counter=env.common_step_counter,
                  episode_length_s=env.max_episode_length_s, max_episode_length=env.max_episode_length,
                  step_dt=env.step_dt, num_envs=env.num_envs,
                  action_dim=env.action_manager.total_action_dim,
                  observation_dims=plain(env.observation_manager.group_obs_dim),
                  commands=commands(env), curriculum=plain(env.cfg.curriculum),
                  rewards=plain(env.cfg.rewards), terminations=plain(env.cfg.terminations),
                  reset_randomization=plain(env.cfg.events),
                  contact_sensor=plain(env.scene.sensors["contact_forces"].cfg),
                  joint_names=env.scene["robot"].joint_names,
                  contact_body_names=env.scene.sensors["contact_forces"].body_names,
                  forbidden_attributes=[k for k in vars(env) if any(s in k.lower() for s in ("task_id", "task_pool", "sgs", "trajectory"))],
                  imported_modules={k: v.__file__ for k, v in sys.modules.copy().items()
                                    if k.startswith("LeggedManip_Lab") and getattr(v, "__file__", None)})
    if runner is not None:
        alg = runner.alg
        result.update(runner_class=f"{type(runner).__module__}.{type(runner).__name__}",
                      runner_cfg=plain(runner.cfg), optimizer=type(alg.optimizer).__name__,
                      optimizer_state_entries=len(alg.optimizer.state),
                      optimizer_defaults=plain(alg.optimizer.defaults),
                      actor_normalizer=str(alg.actor.obs_normalizer), critic_normalizer=str(alg.critic.obs_normalizer),
                      actor_distribution=str(alg.actor.distribution), critic_distribution=str(alg.critic.distribution),
                      actor_sha256=model_sha(alg.actor), critic_sha256=model_sha(alg.critic),
                      algorithm_source=inspect.getfile(type(alg)), runner_source=inspect.getfile(type(runner)))
    return result


def physical(env):
    """Current pre-reset state and the commands that produced this step's reward."""
    robot = env.scene["robot"]
    d = robot.data
    term = env.command_manager.get_term("ee_pose")
    cmd = term.command
    velcmd = env.command_manager.get_command("base_velocity")
    ee = d.body_pos_w[:, term.body_idx]
    eeq = d.body_quat_w[:, term.body_idx]
    linkq = d.body_quat_w[:, term.link0_idx]
    ee_link = quat_apply_inverse(linkq, ee - d.body_pos_w[:, term.link0_idx])
    actual_mixed = torch.cat((ee_link[:, :2], ee[:, 2:3]), dim=-1)
    pos_err = torch.linalg.vector_norm(actual_mixed - cmd[:, :3], dim=-1)
    # Both conventions are retained: the native reward uses root; command metrics use link0.
    ori_err_reward = quat_error_magnitude(eeq, quat_mul(d.root_quat_w, cmd[:, 3:7]))
    ori_err_command = quat_error_magnitude(eeq, quat_mul(linkq, cmd[:, 3:7]))
    vel = torch.cat((d.root_lin_vel_b[:, :2], d.root_ang_vel_b[:, 2:3]), dim=-1)
    sensor = env.scene.sensors["contact_forces"]
    force_hist = sensor.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
    feet = [i for i, name in enumerate(sensor.body_names) if name.endswith("_foot")]
    nonfeet = [i for i in range(len(sensor.body_names)) if i not in feet]
    roll, pitch, yaw = euler_xyz_from_quat(d.root_quat_w)
    roll = torch.remainder(roll + math.pi, 2 * math.pi) - math.pi
    pitch = torch.remainder(pitch + math.pi, 2 * math.pi) - math.pi
    return dict(base_cmd=velcmd, base_actual=vel, ee_cmd=cmd, ee_actual_mixed=actual_mixed,
                ee_actual_quat_world=eeq, ee_target_quat_world=quat_mul(linkq, cmd[:, 3:7]),
                xy_error_sq=(vel[:, :2] - velcmd[:, :2]).square().sum(-1),
                yaw_error_sq=(vel[:, 2] - velcmd[:, 2]).square(),
                ee_position_error=pos_err, ee_orientation_error=ori_err_command,
                ee_orientation_reward_error=ori_err_reward,
                base_height=d.root_pos_w[:, 2], base_roll=roll, base_pitch=pitch,
                base_pos_world=d.root_pos_w, foot_forces=force_hist[:, feet],
                nonfoot_max_force=force_hist[:, nonfeet].amax(-1),
                nonfoot_support_z=sensor.data.net_forces_w[:, nonfeet, 2].clamp(min=0).sum(-1),
                total_support_z=sensor.data.net_forces_w[:, :, 2].clamp(min=0).sum(-1),
                action_abs_max=env.action_manager.action.abs().amax(-1),
                joint_speed_max=d.joint_vel.abs().amax(-1))


def install(runner):
    out = Path(os.environ["NATIVE_WBC_DIAGNOSTICS"])
    out.mkdir(parents=True, exist_ok=True)
    env = runner.env.unwrapped
    wt = Path(os.environ["NATIVE_WBC_WORKTREE"]).resolve()
    audit = snapshot(env, runner)
    write_json(out / "initial_runtime.json", audit)
    assert all(Path(p).is_relative_to(wt) for p in audit["imported_modules"].values()), "Foreign LeggedManip import"
    assert env.max_episode_length_s == 20.0 and env.max_episode_length == 1000
    assert env.action_manager.total_action_dim == 18
    assert not audit["forbidden_attributes"]
    assert len(runner.alg.optimizer.state) == 0 and not runner.cfg["resume"]
    assert runner.alg.schedule == "adaptive" and runner.alg.learning_rate == 5e-4
    assert runner.alg.entropy_coef == .005 and runner.alg.desired_kl == .01
    assert runner.current_learning_iteration == 0
    assert subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=wt) == b""
    state = dict(update=0, steps=[], kls=[], gradients=[], bad_kl=0, started=time.monotonic())
    log = open(out / "updates.jsonl", "a", buffering=1)

    def save_cp(name):
        path = out / f"{name}.pt"
        saved = runner.alg.save()
        saved.update(iter=state["update"] - 1, infos=dict(completed_updates=state["update"],
                     native_upstream_commit="4e12109d1eccb1d9da2e4a35f189fabea3f13677"))
        torch.save(saved, path)
        write_json(out / f"{name}.json", dict(completed_updates=state["update"], sha256=sha(path),
                   path=str(path), runtime=snapshot(env)))

    save_cp("cp_0")
    original_reward = env.reward_manager.compute
    def reward(*args, **kwargs):
        result = original_reward(*args, **kwargs)
        with torch.no_grad():
            p = physical(env)
            values = dict(xy_mse=p["xy_error_sq"].mean(), yaw_mse=p["yaw_error_sq"].mean(),
                          ee_position_mse=p["ee_position_error"].square().mean(),
                          ee_orientation_rad=p["ee_orientation_error"].mean(),
                          ee_orientation_reward_rad=p["ee_orientation_reward_error"].mean(),
                          base_height=p["base_height"].mean(),
                          nonfoot_fraction=(p["nonfoot_max_force"] > 1).float().mean(),
                          base_xy_speed=p["base_actual"][:, :2].norm(dim=-1).mean(),
                          commanded_xy_speed=p["base_cmd"][:, :2].norm(dim=-1).mean(),
                          reward_step=result.mean(),
                          done_count=env.reset_buf.float().sum(),
                          done_episode_length_sum=(env.episode_length_buf * env.reset_buf).float().sum())
            for name in env.termination_manager.active_terms:
                values["termination/" + name] = env.termination_manager.get_term(name).float().sum()
            for i, name in enumerate(env.reward_manager.active_terms):
                values["reward_rate/" + name] = env.reward_manager._step_reward[:, i].mean()
            state["steps"].append(values)
        return result
    env.reward_manager.compute = reward

    original_kl = runner.alg.actor.get_kl_divergence
    def kl(*args, **kwargs):
        result = original_kl(*args, **kwargs)
        state["kls"].append(result.detach().mean())
        return result
    runner.alg.actor.get_kl_divergence = kl

    original_clip = torch.nn.utils.clip_grad_norm_
    def clip(*args, **kwargs):
        result = original_clip(*args, **kwargs)
        state["gradients"].append(result.detach())
        if not torch.isfinite(result):
            save_cp("emergency_nonfinite_gradient")
            raise RuntimeError("Native WBC stop: nonfinite gradient; no parameter adjustment")
        return result
    torch.nn.utils.clip_grad_norm_ = clip

    original_update = runner.alg.update
    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        state["update"] += 1
        metrics = {k: float(torch.stack([s[k] for s in state["steps"]]).mean()) for k in state["steps"][0]}
        kls = torch.stack(state["kls"]).cpu()
        grads = torch.stack(state["gradients"]).cpu()
        record = dict(completed_updates=state["update"], env_steps=state["update"] * env.num_envs * runner.cfg["num_steps_per_env"],
                      elapsed_s=time.monotonic()-state["started"], losses=result,
                      learning_rate=runner.alg.learning_rate, action_std=plain(runner.alg.actor.output_std[0]),
                      kl_mean=float(kls.mean()), kl_max=float(kls.max()),
                      actor_grad_norm_mean=float(grads[0::2].mean()), actor_grad_norm_max=float(grads[0::2].max()),
                      critic_grad_norm_mean=float(grads[1::2].mean()), critic_grad_norm_max=float(grads[1::2].max()),
                      metrics=metrics)
        log.write(json.dumps(record) + "\n")
        write_json(out / "latest.json", record)
        state["bad_kl"] = state["bad_kl"] + 1 if record["kl_mean"] > .1 else 0
        if not all(math.isfinite(v) for v in list(result.values()) + list(metrics.values()) + [record["kl_mean"]]):
            save_cp("emergency_nonfinite")
            raise RuntimeError("Native WBC stop: nonfinite telemetry")
        if state["bad_kl"] >= 20:
            save_cp("emergency_sustained_kl")
            raise RuntimeError("Native WBC stop: mean minibatch KL > 0.1 for 20 consecutive updates")
        if state["update"] in (1000, 2000, 3000):
            save_cp(f"cp_{state['update']}")
        if state["update"] == 1 or state["update"] % 50 == 0:
            write_json(out / f"runtime_{state['update']:04d}.json", snapshot(env))
        state["steps"].clear()
        state["kls"].clear()
        state["gradients"].clear()
        return result
    runner.alg.update = update
    print("NATIVE_WBC_READONLY_AUDIT_PASS", out, flush=True)
