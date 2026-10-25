# Z1 Locomotion

Research on a team-model quadruped, distinct from the Unitree Z1 manipulator. The algorithm/task/analysis work is independently organized here; the team's URDF, meshes, converted USD, geometric reconstruction helpers and robot-model visualizations are excluded.

| Directory | Scope |
|---|---|
| `legacy_20261010/` | IIT/Newton task extensions, contact-aware control/reward telemetry, stairs curriculum and first-trial evaluators |
| `studies_20261014/` | Standing-sampling implementation and native-PPO continuation checks; exact original CP5700/Adam restoration |
| `studies_20261016/height_reward_ab/` | Separate A/B1 100-update height-reward study; original continuation constraints retained |
| `results/` | Selected numerical contact and height-intervention summaries with source hashes |

Use [run.py](run.py) for explicit fresh training and stairs/rough evaluation entry points. `python projects/z1_locomotion/run.py train --check` imports no simulator. Full commands, external asset requirements and environment pins are in the [reproducibility guide](../../docs/REPRODUCIBILITY.md).

The most recent inspected outcome is not a claim of safe downstairs locomotion: strict safety remains 0/48 for all three evaluated policies. Provisional motor settings, post-solver sampling limits, single-training-seed evidence and increased early foot loading remain documented in [RESULTS.md](../../RESULTS.md).
