"""Checkpoint the WBC state exposed by Isaac Lab and the native PPO runner."""
from collections import deque
import hashlib

import numpy as np
import torch
from isaaclab.utils.buffers.timestamped_buffer import TimestampedBuffer

from common import tensor_hash


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(child) for child in value)
    return value


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: device_tree(child, device) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(device_tree(child, device) for child in value)
    return value


def tree_hash(value):
    """Deterministic typed hash, avoiding tensor-to-JSON expansion."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            result.update(f"array:{item.dtype}:{item.shape}:".encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            result.update(f"dict:{len(item)}:".encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        else:
            result.update(f"{type(item).__name__}:{item!r};".encode())

    visit(value)
    return result.hexdigest()


def tensors(obj):
    return {key: cpu_tree(value) for key, value in vars(obj).items() if isinstance(value, torch.Tensor)}


def restore_tensors(obj, state):
    for name, value in state.items():
        # E2 assigns a fresh tensor on each native resampling batch, whose
        # size at checkpoint time differs from a fresh all-env reset.
        if name == "_sequence_inputs":
            setattr(obj, name, value.to(obj.device))
            continue
        target = getattr(obj, name)
        assert target.shape == value.shape and target.dtype == value.dtype, name
        target.copy_(value.to(target.device))


def startup_hash(robot):
    view = robot.root_physx_view
    return tensor_hash({
        "mass": view.get_masses(), "inertia": view.get_inertias(),
        "com": view.get_coms(), "material": view.get_material_properties(),
    })


def history_buffers(raw):
    return {
        f"{group}/{term}": buffer
        for group, entries in raw.observation_manager._group_obs_term_history_buffer.items()
        for term, buffer in entries.items()
    }


def managers(raw):
    result = {
        "action": raw.action_manager, "termination": raw.termination_manager,
        "reward": raw.reward_manager, "event": raw.event_manager,
        "contact": raw.scene["contact_forces"], "contact_data": raw.scene["contact_forces"]._data,
    }
    result.update({f"action/{key}": value for key, value in raw.action_manager._terms.items()})
    result.update({f"command/{key}": value for key, value in raw.command_manager._terms.items()})
    return result


def capture(raw, logger):
    robot = raw.scene["robot"]
    return dict(
        scene=cpu_tree(raw.scene.get_state()), robot_buffers=tensors(robot.data),
        startup_sha256=startup_hash(robot), robot_time=robot.data._sim_timestamp,
        robot_cache={
            key: (value.timestamp, cpu_tree(value.data))
            for key, value in vars(robot.data).items() if isinstance(value, TimestampedBuffer)
        },
        actuators={name: tensors(actuator) for name, actuator in robot.actuators.items()},
        wrenches={
            name: dict(active=value.active, force=cpu_tree(value.composed_force_as_torch),
                       torque=cpu_tree(value.composed_torque_as_torch))
            for name, value in (("permanent", robot.permanent_wrench_composer),
                                ("instantaneous", robot.instantaneous_wrench_composer))
        },
        raw={
            name: cpu_tree(getattr(raw, name))
            for name in ("common_step_counter", "_sim_step_counter", "episode_length_buf", "reset_buf",
                         "reset_terminated", "reset_time_outs", "reward_buf", "obs_buf")
        },
        managers={name: tensors(value) for name, value in managers(raw).items()},
        command_metrics={
            name: cpu_tree(term.metrics) for name, term in raw.command_manager._terms.items()
        },
        episode_reward_sums=cpu_tree(raw.reward_manager._episode_sums),
        event_clocks={
            name: cpu_tree(getattr(raw.event_manager, name))
            for name in ("_interval_term_time_left", "_reset_term_last_triggered_step", "_reset_term_last_triggered")
            if hasattr(raw.event_manager, name)
        },
        history={
            name: dict(pointer=buf._pointer, num_pushes=cpu_tree(buf._num_pushes), buffer=cpu_tree(buf._buffer))
            for name, buf in history_buffers(raw).items()
        },
        logger={
            name: cpu_tree(list(getattr(logger, name)) if isinstance(getattr(logger, name), deque)
                           else getattr(logger, name))
            for name in ("rewbuffer", "lenbuffer", "cur_reward_sum", "cur_episode_length", "tot_timesteps", "tot_time")
        },
    )


def restore(raw, logger, state):
    device = raw.device
    robot = raw.scene["robot"]
    assert startup_hash(robot) == state["startup_sha256"], "Startup domain randomization differs at fork"
    raw.scene.reset_to(device_tree(state["scene"], device))
    robot.write_joint_stiffness_to_sim(state["robot_buffers"]["joint_stiffness"].to(device))
    robot.write_joint_damping_to_sim(state["robot_buffers"]["joint_damping"].to(device))
    restore_tensors(robot.data, state["robot_buffers"])
    robot.data._sim_timestamp = state["robot_time"]
    for name, (timestamp, data) in state["robot_cache"].items():
        cached = getattr(robot.data, name)
        cached.timestamp, cached.data = timestamp, device_tree(data, device)
    for name, actuator in robot.actuators.items():
        restore_tensors(actuator, state["actuators"][name])
    for name, composer in (("permanent", robot.permanent_wrench_composer),
                           ("instantaneous", robot.instantaneous_wrench_composer)):
        saved = state["wrenches"][name]
        composer._active = saved["active"]
        composer.composed_force_as_torch.copy_(saved["force"].to(device))
        composer.composed_torque_as_torch.copy_(saved["torque"].to(device))
    for name, value in state["raw"].items():
        setattr(raw, name, device_tree(value, device))
    for name, manager in managers(raw).items():
        restore_tensors(manager, state["managers"][name])
    for name, term in raw.command_manager._terms.items():
        term.metrics = device_tree(state["command_metrics"][name], device)
    raw.reward_manager._episode_sums = device_tree(state["episode_reward_sums"], device)
    for name, value in state["event_clocks"].items():
        setattr(raw.event_manager, name, device_tree(value, device))
    for name, buf in history_buffers(raw).items():
        saved = state["history"][name]
        buf._pointer = saved["pointer"]
        buf._num_pushes.copy_(saved["num_pushes"].to(device))
        buf._buffer = device_tree(saved["buffer"], device)
    for name, value in state["logger"].items():
        target = getattr(logger, name)
        if isinstance(target, deque):
            target.clear()
            target.extend(value)
        elif isinstance(target, torch.Tensor):
            target.copy_(value.to(target.device))
        else:
            setattr(logger, name, value)
