# Legged Robot Learning

**From quadruped locomotion to whole-body control and PPO training diagnostics.**

[English](README.md) · [简体中文](README_CN.md) · [Results](RESULTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

This repository presents a collection of independently developed robot-learning experiments by [chao142857](https://github.com/chao142857). The central question is **how task design, sampling, feedback, and policy updates shape reliable locomotion and mobile manipulation**. It brings together implemented training systems, quantitative evaluations, negative results, and diagnostic tools—not a single unified algorithm or benchmark.

> **Research snapshot — October 2026.** All robot results are from **simulation**. The public release contains selected source code and numerical evidence; it does **not** establish hardware performance, complete end-to-end reproducibility, or a stable improvement from SGS.

## Research trajectory

**Quadruped locomotion → Whole-body control (WBC) → Task sampling and PPO optimization**

- **Locomotion:** investigate stairs, contact behavior, body-height rewards, and controlled policy continuations.
- **Whole-body control:** train and evaluate coupled base-motion / end-effector tracking in Go2 + Piper.
- **Learning dynamics:** investigate whether adaptive task sampling helps, and diagnose gradient, KL, clipping, and Adam update behavior when learning fails.

These tracks share research questions but use **distinct environments and experimental protocols**. Results across tracks are not interchangeable.

## Selected experimental findings

| Track | Verified observation | Important qualification |
|---|---|---|
| **Z1 locomotion** | An October 10 CP5700 simulation logged **32/32** traversals up eight 10 cm risers. | In a separate October 16 downstairs study, strict safety was **0/48** for the original policy and each of two continuations; calf contact persisted. |
| **Go2 + Piper — native WBC** | At CP3000: **768/768** deterministic 20 s survival, **8.07 cm** EE-position RMSE; **294.9M** training transitions. | A separate precision diagnostic passed **0/768** episodes. Survival is not precise, safe control. |
| **Go2 + Piper — Recipe v1** | At CP1200: formal T1 completion **192/192**, mean EE RMSE **2.42 cm**; **118.0M** training transitions. | Full-episode strict safety **15/192**; right-turn consistency and dynamic-X tracking remained below admission requirements. **Formal T1 gate: failed.** |
| **SGS pilot** | Matched fixed-48-task Uniform and SGS runs both went from **40/48** joint success initially to **0/48** after training checkpoints. | No observed SGS gain; training produced no positive joint-success feedback. This is not proof of general ineffectiveness. |
| **PPO Gradient Lab** | **120** distinct frozen-state, single-minibatch candidate updates examined native GAE, KL, clipping, and Adam. | Smaller LR reduced held-out clipped-surrogate deviation in the main window—not a demonstrated gain in robot return. |

**Do not rank these numbers against one another.** Native WBC, Recipe T1, Z1, SGS, and frozen-gradient analysis use different task definitions, evaluation rules, and/or initializations. The robot learning studies reported here each use **one training seed**; multiple evaluation seeds are not training replications.

![Numerical summaries from distinct simulation protocols; not a cross-method leaderboard](docs/figures/simulation_results.svg)

See [**full experimental results, metrics, protocols, and evidence links**](RESULTS.md) for the denominators, failure conditions, and original numerical files.

## Projects

| Project | What's included | Current research question |
|---|---|---|
| [**Z1 Locomotion**](projects/z1_locomotion/README.md) | Stair curriculum, contact-aware task/reward configurations, standing sampling, height-reward interventions, evaluation and contact statistics | Why can traversal succeed while strict downstairs contact-safety still fails? |
| [**Go2 + Piper WBC**](projects/go2_piper_wbc/README.md) | Native PPO WBC baseline, task adapters, T1 scratch v2, Recipe v1, A+ preparation and guarded launchers | How do task, reward, control, and startup constraints affect coupled locomotion and EE tracking? |
| [**Success Guided Sampling (SGS)**](projects/sgs/README.md) | Local SGS implementation, feedback/reset lifecycle tests, Uniform comparison and factorial preparation | Under which feedback conditions can task sampling provide a discriminating learning signal? |
| [**PPO Gradient Lab**](projects/ppo_gradient_lab/README.md) | Frozen-gradient probes, native GAE calibration, KL/clipping analysis, Adam update microscope | Which measurable update effects accompany instability, and which are merely correlated diagnostics? |

**Method attribution:** SGS is studied as a method from [Zhang et al., *A Balanced Data Diet*](https://arxiv.org/abs/2610.12465v1), **not** claimed as an algorithm invented in this repository. Original contributions lie in local implementation, integration, experimental protocols, and analyses, as applicable.

## Quick start: CPU-only checks

From the repository root, run:

~~~bash
python tools/audit_publication.py
python tools/cpu_smoke.py
python projects/z1_locomotion/run.py train --check
python projects/go2_piper_wbc/run.py native-train --check
~~~

These commands **do not start GPU training or launch a simulator**. Optional ML tests require their dependencies; use `python tools/cpu_smoke.py --require-ml` to enforce them. The release validation recorded **74 passed tests and 1 skipped real-checkpoint parity test**, plus source/entry-point checks—not a fresh Isaac simulation run.

Actual Z1 and Go2 + Piper runs require **different simulator/framework versions**, independently obtained robot assets, and (for historical continuation/Recipe studies) source-bound checkpoints or runtime evidence. See [reproducibility and exact commands](docs/REPRODUCIBILITY.md) before attempting training.

## Current boundaries and open questions

- **Z1:** team-supplied URDF/meshes and derived visualizations are excluded. Actuator parameters are provisional; strict downhill contact-safety remains unresolved.
- **WBC:** native and Recipe evaluations are **not** a controlled head-to-head algorithm comparison. Recipe v1's T1 gate failed; A+ had **zero additional completed updates** in the inspected snapshot.
- **SGS:** the historical pilot shows no benefit. Prepared B1/B2/B3 factorial studies are **not** completed training results.
- **PPO diagnostics:** native total-advantage checks and frozen surrogate probes should not be mistaken for causal proof of training failure, gradient surgery gains, or improved policy return.
- **Reproduction:** private checkpoints, model assets, raw rollouts and some runtime receipts are not redistributed; published summaries do not reconstruct them.

The immediate research objective is **controlled, protocol-matched experimentation** on training stability, task sampling, dynamic end-effector tracking, and strict safety—not announcing a solved method.

## Documentation and provenance

| Document | Purpose |
|---|---|
| [RESULTS.md](RESULTS.md) | Per-experiment numbers, denominators, protocol differences and supporting files |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Environment versions, CPU checks, simulator prerequisites and commands |
| [VALIDATION.md](VALIDATION.md) | What the publication tests actually verified and skipped |
| [SOURCE_MAP.md](SOURCE_MAP.md) | Source snapshots and their research lineage |
| [PUBLICATION_AUDIT.md](PUBLICATION_AUDIT.md) | Inclusion/exclusion decisions and history isolation |
| [THIRD_PARTY.md](THIRD_PARTY.md) · [NOTICE](NOTICE) · [LICENSES/](LICENSES/) | Attribution and licensing scope |

## Acknowledgements and license

Experiments build on external research and software including [IIT DLS Lab basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab), [LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab), [Isaac Lab](https://github.com/isaac-sim/IsaacLab), and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). The collaborating team supplied the Z1 robot model, which is **not** redistributed here. No affiliation or endorsement is implied.

Original contributions are licensed under **Apache-2.0**, subject to retained upstream notices and [third-party terms](THIRD_PARTY.md). Source-code licensing does not grant access to excluded model assets, checkpoints, or policies.
