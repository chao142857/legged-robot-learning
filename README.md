# Legged Robot Learning

**Reinforcement Learning for Legged Locomotion & Whole-Body Control**

[![English](https://img.shields.io/badge/English-2563eb?style=for-the-badge)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-64748b?style=for-the-badge)](README_CN.md)

A collection of my robot learning projects, from quadruped locomotion to whole-body control and reinforcement learning optimization. I'm interested in **how robots learn coordinated movement and manipulation**, and how training objectives, task distributions, and PPO updates affect their performance.

## Research Interests

- **Legged Locomotion:** learning to traverse stairs and challenging terrain, with contact-aware control and reward design.
- **Whole-Body Control (WBC):** coordinating quadruped locomotion and robotic-arm manipulation within a learned policy.
- **Reinforcement Learning Methods:** task sampling, training stability, gradient analysis, and PPO optimization.

## Projects

| Project | What I've built | Selected progress |
|---|---|---|
| [**Z1 Locomotion**](projects/z1_locomotion/README.md) | Quadruped training tasks, stair curricula, contact/height reward studies, and evaluation tools | **32/32** simulated ascending-stair trials with the CP5700 policy (eight 10 cm risers); later studied downstairs contact behavior |
| [**Go2 + Piper WBC**](projects/go2_piper_wbc/README.md) | Native WBC baseline, leg–arm task suite, T1 scratch training, and reward/control recipes | Native CP3000: **768/768** deterministic 20 s survival. Recipe v1 T1: **192/192** task completions, **2.42 cm** mean EE position RMSE |
| [**Success Guided Sampling**](projects/sgs/README.md) | Local SGS implementation, episode-feedback handling, Uniform comparisons, and factorial-study preparation | Completed a fixed-task SGS/Uniform pilot; investigating when success feedback provides a useful sampling signal |
| [**PPO Gradient Lab**](projects/ppo_gradient_lab/README.md) | Tools for native GAE, KL, clipping, Adam updates, and frozen-policy gradient diagnostics | Analyzed **120** distinct single-minibatch update candidates to examine PPO update behavior |

*Robot results above are from simulation. Task completion and survival are not strict safety acceptance; detailed metrics, unsuccessful runs, and experimental protocols are in [RESULTS.md](RESULTS.md).*

## Current Focus

I'm currently investigating **dynamic end-effector tracking, stable locomotion–manipulation training, and controlled comparisons of task sampling strategies**. The goal is to understand which design choices improve learning—and why some seemingly reasonable ones do not.

## Code & Experiments

The four project directories contain selected implementations, experiment configurations, evaluation tools, and numerical results. Start with a project README, or use the CPU-only checks below:

~~~bash
python tools/audit_publication.py
python tools/cpu_smoke.py
~~~

Full simulator runs require their respective external frameworks and robot assets.

**Documentation:** [Results](RESULTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Validation](VALIDATION.md) · [Source Map](SOURCE_MAP.md)

## Credits

Built on [IIT DLS Lab's basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab), [LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab), [Isaac Lab](https://github.com/isaac-sim/IsaacLab), and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). SGS follows the method studied in [*A Balanced Data Diet*](https://arxiv.org/abs/2610.12465v1). The collaborating team provided the Z1 robot model; its assets are not included.

Original contributions are under **Apache-2.0**; see [third-party attribution and licenses](THIRD_PARTY.md).
