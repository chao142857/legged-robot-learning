# Legged Robot Learning

**Reinforcement Learning for Legged Locomotion and Whole-Body Control**

An independent research collection by [chao142857](https://github.com/chao142857), covering quadruped locomotion, mobile manipulation, adaptive task sampling, and PPO update diagnostics. The research implementations, experiment design, and analyses are organized here with explicit upstream attribution and evidence boundaries.

**All reported robot experiments are simulation results. This release establishes no real-robot validation and no stable SGS advantage over Uniform sampling.**

## Current research — corrected CleanMeta8 (10 November 2026)

[**Completed reproduction and checkpoint curves**](projects/sgs/cleanmeta8_20261109/README.md) · [Machine-readable evidence](projects/sgs/cleanmeta8_20261109/evidence/completed_curves.json) · [Optimizer-fix audit](https://github.com/chao142857/legged-robot-learning/blob/c43b36ea4cb3dbdaa3d0f72730c349188a6d5130/projects/sgs/teacher_adam_audit_20261109/REPORT.md)

A 9 November audit identified Adam snapshot aliasing in V0.2 Teacher backtracking: rejected trial steps could alter the optimizer state intended for restoration. Historical Meta8/Meta16 artifacts remain archived as **legacy implementation results**. The corrected reproduction starts from the original Fresh Teacher state and collects **eight new three-arm experiments × 100 PPO updates = 2,400 new Student updates**; it does not replay old outcomes as new on-policy feedback.

**Both corrected frozen-Teacher transfers completed on 9 November:** WBC, 4,096 environments, rollout 24, seeds 48/49 from zero to **10,000 PPO updates each**. Final strict joint success uses the same 1,024-sequence mean-action test, 5 cm EE tolerance and 80% simultaneous tracking criterion. Uniform and Fresh below are the explicitly reused baselines.

| Student seed | Uniform | Fresh | CleanMeta8 | vs Uniform | vs Fresh |
|---:|---:|---:|---:|---:|---:|
| 48 | 25.78% | 35.94% | **33.98%** | +8.20 pp | -1.95 pp |
| 49 | 35.94% | 29.98% | **34.47%** | -1.46 pp | +4.49 pp |

Seed 48 Uniform/CleanMeta8 ran on RTX 5060 Laptop; its Fresh baseline ran on RTX 3060. All seed 49 methods ran on RTX 3060. **The final-checkpoint advantage is mixed across the two Student seeds; neither a stable gain nor a causal performance benefit from the fix is established.** One corrected Teacher was shared by both transfers.

![Corrected CleanMeta8 and reused baselines, evaluated every 1,000 PPO updates](docs/figures/cleanmeta8_learning_curves.svg)

The [corrected implementation and executed queue revision](https://github.com/chao142857/legged-robot-learning/tree/b1b5cd5b092967678b9b5f315676da6f442bc9f4/projects/sgs/cleanmeta8_20261109) and [Adam regression tests](https://github.com/chao142857/legged-robot-learning/tree/c43b36ea4cb3dbdaa3d0f72730c349188a6d5130/projects/sgs/teacher_adam_audit_20261109/tests) remain on their research branches. The [6 November V0.2 snapshot](https://github.com/chao142857/legged-robot-learning/tree/da583b33b10351a8e5c9f19e61b75ba929ea026d/projects/sgs/nsgs_v02_20261106) is historical, not the current status. [CL1](https://github.com/chao142857/legged-robot-learning/tree/ab4532ea1af7ab7751136e95dd4187d386457ee5/projects/sgs/nsgs_cl1_20261109) is a separate follow-up protocol and is not pooled with these frozen-Teacher results.

## Research Overview

The work studies how task distributions, reward design, reset semantics, and policy updates affect learned control. It includes successful learning runs, failed baselines, controlled interventions, and tools for examining why an apparently reasonable PPO update can behave poorly.

The projects retain their original framework generations and experimental identities. A fixed-task warm-start pilot, a native-distribution scratch baseline, and a constrained recipe are different experiments; their scores are not combined into a single leaderboard.

## Projects

| Project | Research content | Current evidence |
|---|---|---|
| [Z1 Locomotion](projects/z1_locomotion/README.md) | Training configurations, stairs curriculum, contact-aware rewards, standing sampling, height-reward A/B study, evaluation | Simulated traversal and contact diagnostics; strict downstairs safety remains unresolved |
| [Go2 + Piper WBC](projects/go2_piper_wbc/README.md) | Native WBC baseline, fixed task suite, T1 scratch study, Recipe v1 and A+ preparation | Native learning demonstrated; formal Recipe v1 T1 gate remains false |
| [SGS / nSGS](projects/sgs/README.md) | Success-guided and neural task sampling, feedback lifecycle, optimizer integrity and controlled transfer studies | Corrected CleanMeta8: two completed 10k Student runs; mixed final results against reused Uniform/Fresh baselines |
| [PPO Gradient Lab](projects/ppo_gradient_lab/README.md) | Frozen-policy gradients, component value probes, native GAE, KL, clipping, Adam and held-out surrogate diagnostics | CPU diagnostic evidence; no demonstrated control-performance improvement |

## Methods

- **Locomotion and WBC:** PPO through the native RSL-RL/Isaac training stacks; independent task adapters and explicit reset, action, contact, and observation checks.
- **Controlled interventions:** matched initialization and budgets where the protocol supports them; reward-height intervention, command-sampling changes, and native-distribution references. Differences in task, reset, action dimension, or controller priors remain visible.
- **Adaptive sampling:** a local implementation of SGS from [Zhang et al., *A Balanced Data Diet*](https://arxiv.org/abs/2610.12465v1), with episode ownership, censored-outcome handling, isolated RNG, state validation, and evaluation separation. The unvisited-task prior of 0.5 is a documented local choice.
- **Update analysis:** native total-advantage calibration and isolated PPO/Adam steps, with frozen held-out surrogate measurements. Component probes are diagnostic approximations, not an exact decomposition of native PPO.

## Experimental Results

| Experiment | Selected observation | Interpretation |
|---|---|---|
| Z1, October 16 height/contact study | Downstairs strict safety: **0/48** for the original policy and both continuations; calf-contact fraction A 10.17%, B1 6.88% | Lower contact exposure did not establish safe downstairs control; early foot loading also increased |
| Go2 + Piper, native WBC CP3000 | Deterministic survival **768/768**; XY RMSE **0.1189 m/s**, EE position RMSE **8.07 cm** | Learned simulated locomotion and pose tracking; not high-precision or complete safety acceptance |
| Recipe v1 CP1200, formal T1 | Completion **192/192**, EE RMSE **2.42 cm**, original full-episode strict safety **15/192** | Different task distribution from native WBC; turning, startup overspeed and dynamic X tracking remain limitations |
| SGS fixed-48-task pilot | Both methods: **40/48** joint success at initialization, **0/48** after 32, 64 and 128 updates | No observed SGS joint-success gain; degenerate success feedback limits the comparison |
| PPO microscope | 120 distinct isolated candidates; smaller LR reduced held-out clipped-surrogate deviation in the main window | Surrogate evidence, not improved robot return or a proven cause of historical training failure |

Native WBC errors use valid first-episode samples after the first second. Recipe values use equal averaging over three evaluation seeds. These are different aggregation rules and protocols. The earlier studies in this table each have one training seed. The separate CleanMeta8 comparison above has two Student seeds sharing one corrected Teacher training run; evaluation resets do not add independent training replications. Full definitions and result links are in [RESULTS.md](RESULTS.md).

![Numerical simulation results from two separate protocols](docs/figures/simulation_results.svg)

## Reproducibility

Start with the [environment and entry-point guide](docs/REPRODUCIBILITY.md). CPU checks use synthetic inputs and published numerical summaries:

```bash
python tools/audit_publication.py
python tools/cpu_smoke.py
```

`cpu_smoke.py --require-ml` requires the optional Torch/RSL dependencies. Missing dependencies and the real-checkpoint parity test are reported explicitly. Simulator-free `--check` entry points are provided for both robot projects. No training starts during these checks.

Full simulation reproduction requires independently obtained upstream software, compatible Linux/GPU environments, and authorized robot assets. Model geometry, pretrained policies, checkpoints, and raw rollouts are excluded. Historical continuation studies additionally require the original source-bound inputs; public summaries cannot reconstruct them.

The [source map](SOURCE_MAP.md), [publication audit](PUBLICATION_AUDIT.md), [per-file provenance](provenance/files.json), and [validation report](VALIDATION.md) separate historical observations from checks executed for this release. This repository starts with new Git history; it does not expose the private source repository.

## Limitations

- No hardware validation, sim-to-real guarantee, or continuous-time safety claim.
- Z1 actuator parameters are provisional simulation settings. Team-supplied model assets and their visualizations are not redistributed.
- The native WBC baseline and Recipe v1 retain distinct task domains and safety statistics; their error values are not a matched algorithm comparison.
- SGS has no demonstrated stable gain here. B1/B2/B3 are preparation candidates, not trained or physically admitted baselines.
- Recipe A+ has **0 additional completed updates** in the inspected snapshot; its planned continuation is not a result.
- Release CPU smoke tests cover imports, numerical contracts, and entry-point checks. CleanMeta8 GPU runs were executed in the recorded source environments; publication checks do not rerun those training budgets.
- Selected numerical summaries are available; withheld raw trajectories, weights, and geometry limit independent end-to-end recomputation.

## Acknowledgements

The experiments build on [IIT DLS Lab's basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab), [LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab), [Isaac Lab](https://github.com/isaac-sim/IsaacLab), and [RSL-RL](https://github.com/leggedrobotics/rsl_rl). SGS is credited to its original authors. The collaborating team supplied the Z1 robot model; model ownership is distinct from the independent algorithm/training work presented here.

Original contributions use Apache-2.0. Applicable upstream notices are retained in [NOTICE](NOTICE), [LICENSES](LICENSES/), and [THIRD_PARTY.md](THIRD_PARTY.md). Source-code licenses do not confer redistribution rights to excluded assets or policies.

