"""Exact T1 terminal gates from the latest scratch_evaluation_core.py.

Selecting the existing 5cm field is a common B1/B2 proposal pending review.
This adapter does not alter thresholds or infer success from reward/timeout.
"""
import math

T1_RULE_VERSION = "latest_joint_success_5cm_v1"


def t1_episode_labels(metrics, *, terminated, timeout, max_steps=1000):
    names = ("xy_rmse", "yaw_rmse", "ee_position_rmse", "ee_orientation_mean",
             "base_clearance_min", "joint_velocity_limit_ratio_max",
             "joint_position_limit_excess_max_rad")
    if type(terminated) is not bool or type(timeout) is not bool:
        raise ValueError("Termination and timeout masks must be Boolean")
    if type(metrics.get("steps")) is not int or not 1 <= metrics["steps"] <= max_steps:
        raise ValueError("Invalid episode step count")
    if type(metrics.get("nonfoot_contact")) is not bool:
        raise ValueError("Contact must be the observed any(force > 1N) Boolean")
    if any(type(metrics.get(k)) not in (int, float) or not math.isfinite(metrics[k]) for k in names):
        raise ValueError("Missing/nonfinite terminal physical metrics")
    if any(metrics[k] < 0 for k in names if k != "base_clearance_min"):
        raise ValueError("Negative error/limit metric")
    if not (terminated or timeout):
        raise ValueError("Partial episodes are censored, never terminal feedback")
    complete = bool(metrics["steps"] == max_steps and not terminated)
    contact_safe = bool(complete and not metrics["nonfoot_contact"] and metrics["base_clearance_min"] >= .15)
    overspeed = bool(metrics["joint_velocity_limit_ratio_max"] > 1.0001)
    joint_limit_violation = bool(metrics["joint_position_limit_excess_max_rad"] > .001)
    safe = bool(contact_safe and not overspeed and not joint_limit_violation)
    ee5 = bool(complete and metrics["ee_position_rmse"] < .05 and metrics["ee_orientation_mean"] < .2)
    ee10 = bool(complete and metrics["ee_position_rmse"] < .1 and metrics["ee_orientation_mean"] < .2)
    movement = bool(complete and metrics["xy_rmse"] < .15 and metrics["yaw_rmse"] < .2)
    return dict(complete=complete, failure=terminated, contact_safe=contact_safe, safe=safe,
                overspeed=overspeed, joint_limit_violation=joint_limit_violation,
                ee_success_5cm=ee5, ee_success_10cm=ee10, movement_success=movement,
                joint_success_5cm=bool(safe and ee5 and movement),
                joint_success_10cm=bool(safe and ee10 and movement))
