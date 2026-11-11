# nSGS

**Neural Success-Guided Sampling for Whole-Body Control**

English · [简体中文](README.zh-CN.md)

nSGS explores how task sampling can adapt to the learning state of a robot policy. Rather than sampling training tasks uniformly or following a fixed curriculum, the goal is to learn which training experiences are most useful for improving downstream control.

Our experiments use a Go2 quadruped equipped with a Piper manipulator, trained with PPO for whole-body locomotion and manipulation in simulation.

## Results

The corrected CleanMeta8 Teacher achieves **33.50% strict joint success**, compared with **31.88% for Uniform sampling** and **33.30% for a Fresh Teacher**, averaged across four Student seeds after 10,000 PPO updates.

![Four-seed CleanMeta8 learning curves](assets/figures/cleanmeta8-four-seed.png)

*Strict joint success evaluated on 1,024 fixed test sequences, averaged across Student seeds 42, 47, 48, and 49. Shading indicates ±1 sample standard deviation across seeds.*

| Sampling method | Strict joint success |
| --- | ---: |
| Uniform | 31.88% |
| Fresh Teacher | 33.30% |
| **CleanMeta8** | **33.50%** |

The displayed means differ by **+1.62 percentage points** (the unrounded difference is +1.6113 points). Performance varies across individual Student seeds, and the four transfers share one meta-trained Teacher. These results are preliminary rather than evidence of a consistent advantage across independently trained Teachers. [Numerical results](results/cleanmeta8/curve_data.json) · [Full evaluation protocol](docs/reproduction.md#evaluation-protocol)

## Method

The central idea is to use the Student's learning state and previous training outcomes to guide the distribution of future training tasks.

- **Student:** A PPO policy controlling the Go2–Piper system, with 18-dimensional joint actions.
- **Task distribution:** A pool of training command sequences covering locomotion and end-effector tracking.
- **Teacher:** A state-conditioned neural sampler that assigns preferences to training sequences.
- **Feedback:** Changes in downstream control performance after additional Student training.

In the CleanMeta8 experiment, the Teacher is updated using counterfactual Student continuations from matched starting states. Uniform, positive-perturbation and negative-perturbation sampling distributions are each evaluated after 100 PPO updates.

The corrected implementation restores independent Adam optimizer states during Teacher backtracking. Earlier Meta8/Meta16 experiments are retained separately because their optimizer-state restoration was affected by aliasing.

## Experimental Development

| Stage | Focus |
| --- | --- |
| SGS | Success feedback, task sampling and evaluation protocol |
| Clustering | Structured task distributions and clustered sampling |
| OldMeta | Counterfactual neural Teacher and early meta-learning experiments |
| CleanMeta | Corrected Teacher optimization and cross-Student transfer |
| Base / Recovery | Recurrent Teacher learning and adaptation to Student degradation |

The current Base/Recovery study uses a recurrent Teacher with a GRU and an outer PPO learner. It is a separate learning algorithm from CleanMeta8, not a continuation of the same Teacher weights.

As checked on **12 November 2026**, Base source Students 43 and 44 completed 10k updates with 29.20% and 30.08% strict success. Recovery V2 completed 2k additional Student updates. The held-out **Student seed70158** comparison also completed all three 10k arms: Uniform **34.57%**, frozen Base **31.05%**, frozen Recovery V2 **35.55%**. This single-Student comparison is separate from the CleanMeta8 four-seed mean; it does not isolate recovery-specific adaptation from the extra training experience. The second held-out Student, seed70159, remains in progress.

See [Experiment History](docs/experiments.md) for protocols, results, and links to the corresponding research revisions.

## Reproduction

The Student experiments use Isaac Lab, LeggedManip_Lab and RSL-RL, with 4,096 parallel environments and a rollout length of 24.

The repository contains [shared components](src/nsgs), [experiment runtimes](experiments), [configurations](configs), [numerical results](results), and [CPU tests](tests). Original execution snapshots retain their external asset and mount contracts.

See the [Reproduction Guide](docs/reproduction.md) for environment setup, required external assets, experiment entry points, and evaluation procedures. [Migration and historical references](docs/migration.md)

## Acknowledgements

This research builds on:

- [A Balanced Data Diet (SGS)](https://arxiv.org/abs/2610.12465v1)
- [LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

The experiments reported here are conducted in simulation.

## License

Original code is released under [Apache-2.0](LICENSE). Upstream implementations and external robot assets remain subject to their respective licenses. See [THIRD_PARTY.md](THIRD_PARTY.md), [NOTICE](NOTICE), and [LICENSES](LICENSES).
