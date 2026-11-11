# Reproduction

## CPU checks

From the repository root, in a Python environment with NumPy and PyTorch:

```bash
PYTHONPATH=src python -m nsgs.evaluation --protocol
python tools/cpu_tests.py
python tools/verify_repository.py
```

The migration checks used Python 3.11, NumPy 2.4.6 and PyTorch 2.13.0 on macOS, reusing an existing environment. No packages or GPU jobs were installed or started. The CPU suites exercise the real Adam/backtracking implementation against an analytic Adam oracle and the historical aliasing implementation, recurrent Teacher state, paired checkpoints/pending actions, frozen RPC state, sampling/evaluation isolation, strict success boundaries, and E2/E3 grouped sampling. They use synthetic state and do not recreate training results.

`src/nsgs/` exposes the corrected CleanMeta Teacher, CL1 learner, sampling, Student ownership/checkpoint adapters and evaluation components. Package imports were changed; simulator math imports in `evaluation/metrics.py` are lazy so the scoring accumulator can be tested without Isaac. Experiment runtimes remain separate, preserving their original flat imports and source hashes. OldMeta is deliberately not routed through the corrected Teacher.

## Simulation environment

The [observed environment](../configs/observed-simulation-environment.json) records Python 3.11.16, PyTorch 2.7.0+cu128, NumPy 1.26.0, RSL-RL 5.0.1, TensorDict 0.8.3, Gymnasium 1.2.1, Isaac Sim 5.1.0.0 and Isaac Lab package metadata 0.54.4. It is an observed historical environment, not a portable installation lock. CleanMeta8's [actual optimizer traces](../results/cleanmeta8/runtime_binding/CLEANMETA8_ACCEPTANCE_SOURCE.json) independently record PyTorch 2.7.0+cu128.

Source revisions: LeggedManip_Lab `4e12109d1eccb1d9da2e4a35f189fabea3f13677`; Isaac Lab `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`. Preserve the matching RSL-RL API, robot task configuration, 18-action interface, assets and simulator runtime. Student training uses 4,096 environments and 24-step rollouts; physics dt is 0.005 s with decimation 4 (50 Hz control).

Robot geometry, USD/mesh assets, original Student checkpoints, Teacher optimizer/memory states, native initial-state files and sequence pools remain external. Obtain them with the original permissions; the public numerical summaries do not substitute for these files. No training or Isaac-dependent check was run during migration.

## Entry points and external mount contract

| Stage | Actual entry points | Required execution context |
| --- | --- | --- |
| SGS/E1 | `experiments/sgs_evaluation/sgs/code/`; `experiments/sgs_evaluation/e1/code/train_e1.py` | Original native task, sequence pools and E1 configuration |
| E2/E3 | `experiments/clustering/e2/code/`; `experiments/clustering/e3/code/train.py`, `evaluate.py` | E2 groups/feature definition, E3 maps and matched prefix state |
| OldMeta | `experiments/oldmeta/v02/code/train_v02.py`, `teacher_ops.py` | Historical uncorrected runtime and original source states |
| CleanMeta8 | `experiments/cleanmeta8/runtime/code/train_v02.py`, `teacher_ops.py`, `traced_teacher_ops.py`, `evaluate.py` | Corrected runtime, prepared manifest, fixed Fresh identity and source seeds 43/44 |
| CL1 Base/V1 | `experiments/cl1_base_recovery/base_v1_runtime/code/train_cl1.py`, `cl1_ops.py` | Original Base/V1 root and recurrent state |
| CL1 V2 transfer | `experiments/cl1_base_recovery/transfer70158/code/train_cl1.py`, `frozen_ops.py`, `checkpoint_pair.py` | Immutable paired Student/Teacher snapshot, V2 lock and frozen arm identity |

These are execution sources, not self-contained launch commands. The recorded `python.sh` and controllers assume their original Linux paths, `/workspace`, `/native`, `/reference`, `/e1reference`, `/e2reference` and authorized asset mounts. Their source identities are preserved rather than silently rewritten to point at nonexistent public assets. Stage the relevant runtime's `code/` and its source-bound prepared/configuration inputs in an isolated new workspace before reproducing. Do not run a deployment/controller script against an existing experiment directory.

CLI help for the simulator entry points imports Isaac's AppLauncher, so it was not claimed as a CPU smoke test. The shared protocol CLI and CPU test runner are directly executable without Isaac. CleanMeta's source-point order and the three 100-update branches are in [configs/cleanmeta8.json](../configs/cleanmeta8.json); CL1 has a [separate configuration](../configs/cl1-base.json).

## Evaluation protocol

The authoritative implementation is [Metrics](../src/nsgs/evaluation/metrics.py) and [PROTOCOL](../src/nsgs/common.py), copied from the executed runtime. Strict joint success requires **all** of:

1. No native failure termination.
2. At least 100 eligible control steps. Both velocity and EE command ages must be at least 2 seconds; each command resample restarts its own age. The terminal transition is measured before autoreset.
3. All four errors simultaneously within tolerance on at least 80% of eligible steps: XY velocity norm ≤ 0.20 m/s, yaw-rate absolute error ≤ 0.30 rad/s, EE position norm ≤ 0.05 m, and quaternion orientation error ≤ π/12 rad (15°).
4. Each of the four eligible-step RMSEs also within its tolerance. A timeout alone is insufficient.

WBC EE position uses native mixed coordinates (XY in link0, Z in world); base velocity is measured in the native body frame. Flat-task pose is in link0. The evaluation uses mean policy actions on the fixed 1,024-sequence test pool, seed 922042; all test episodes remain in the denominator. The continuous joint score is a different metric, used for development feedback, not the strict success rate.

Training pool SHA256: `86f39ce6b71d17d84c47681fe5e44d0fd7efac605ef366c3d8e91916b3eb56db` (32,768 sequences).
Test pool SHA256: `d48b5850285e26b9cec447a0bb4bd6c84493afec5efa0c7725ddcbacbd714afa`.
Protocol SHA256: `93067793642c17e12618129548914f7b440edce267ad20d4df34b8a19c855fad`.

Development evaluation uses its separately prepared pool (256 sequences × 2 repeats). Official test results never feed the Teacher. Do not replace development feedback with the reported test score or average CleanMeta and CL1 experiments together.

## Assets and recoverability

E3's 19 original mapping arrays remain at `yanglab:/data/go2_piper_sgs_e3_20261104/maps/`, with sizes and hashes in the [external inventory](../results/clustering/e3/external-map-inventory.json). The arrays total 4,983,168 bytes. E2 feature/reference metadata and command pools are also needed. Source recovery and CPU verification are complete; public end-to-end E3 asset packaging remains separate.

CL1 original/V1, repaired V2 and held-out transfer roots remain under `/data/go2_piper_nsgs_cl1_20261109/`, `/data/go2_piper_cl1_repair_20261110/` and `/data/go2_piper_nsgs_cl1_gen_seed70158_fixed_20261110/`. No checkpoints, trajectories, Student/Teacher states or experiment logs were deleted or rewritten. PhysX hidden contact caches are not guaranteed bitwise-restorable by the saved-state mechanism.
