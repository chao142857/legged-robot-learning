# Reproducibility

This is a collection of independent research snapshots, not one installable simulator stack. Package observations, pinned source revisions, explicit command-line inputs, and source/result hashes are provided separately. A package list is not a complete environment lock.

## CPU validation

From the repository root, Python 3.10+ runs the standard-library audit and SGS lifecycle tests. ML diagnostics require a separate environment; the historical replay requirements are [here](../projects/ppo_gradient_lab/requirements.txt).

```bash
python tools/audit_publication.py
python tools/cpu_smoke.py
# In an environment containing the optional ML dependencies:
python tools/cpu_smoke.py --require-ml --report /tmp/legged-cpu-report.json
```

The checker never imports Isaac or launches a GPU simulator. Missing optional packages are reported as skipped/partial; `--require-ml` makes missing ML packages a failure. Asset-dependent IK and real-checkpoint parity remain separate from synthetic tests. The publication [validation report](../VALIDATION.md) identifies the interpreter, versions, passes and exclusions actually observed.

## Z1: Isaac Lab v3 / Newton

The October 14 restored runtime observed Python 3.12.14, Torch 2.11.0+cu128, RSL-RL 5.4.1, Newton 1.5.2, Warp 1.16.0 and MuJoCo 3.11.0. It did **not** install full Isaac Sim. The distribution reported `isaaclab=17.0.2`; this metadata is distinct from the v3.0.0-EA source generation and pinned Git revision. See [observed environment](../projects/z1_locomotion/observed_environment.json).

Obtain the IIT framework at `8bda7c7df8b223acf649e0b212635a92c8528900` and the documented historical Isaac stack independently. Supply a Z1 USD and dependencies only if you have permission to use them. No asset is fetched or reconstructed by the public launcher.

```bash
python projects/z1_locomotion/run.py train --check

# Inside the separate compatible Z1 environment:
python projects/z1_locomotion/run.py train \
  --iit-checkout /path/to/basic-locomotion-isaaclab \
  --asset /path/to/authorized/Z1_team.usda -- \
  --task Locomotion-Z1-Team-Rough-Blind-LegSafe --headless

python projects/z1_locomotion/run.py evaluate-stairs \
  --iit-checkout /path/to/basic-locomotion-isaaclab \
  --asset /path/to/authorized/Z1_team.usda -- --help
```

The final `--help` belongs to the historical evaluator in the simulator environment; it lists checkpoint, trial and terrain parameters. The wrapper's own `--help`/`--check` are simulator-free. `Z1_USD_PATH` changes only external asset resolution. The unpublished CAD task is not registered. Provisional actuator settings and strict contact metrics remain in the source.

October 14/16 continuation scripts preserve the original CP5700/Adam checks. They require `Z1_PRIVATE_RESTORE_ROOT`, `LEGGED_RUNTIME_ROOT`, and the original protected receipts/checkpoint; they are historical study entries, not asset-free reproduction commands. The release does not invent a replacement checkpoint or silently weaken those gates. The native command-schedule test also requires the external patched runtime; the published core sampler test explicitly does not cover that schedule.

## Go2 + Piper: Isaac Sim 5.1 / PhysX

Use a separate Python 3.11 environment. The observed stack was Torch 2.7.0+cu128, Isaac Sim 5.1.0.0, Isaac Lab checkout `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`, RSL-RL 5.0.1 and NumPy 1.26.0. Distribution metadata and Git revisions are recorded independently in [observed environment](../projects/go2_piper_wbc/observed_environment.json). The Z1 runtime is incompatible with this configuration.

Independently obtain LeggedManip_Lab at `4e12109d1eccb1d9da2e4a35f189fabea3f13677`, install its framework dependencies, and obtain the required Go2/Piper assets under their own terms. Public code licensing does not establish asset/weight redistribution rights.

```bash
python projects/go2_piper_wbc/run.py native-train --check

# Separate Go2 environment; explicit GPU training command, not run for publication:
python projects/go2_piper_wbc/run.py native-train \
  --upstream /path/to/LeggedManip_Lab \
  --diagnostics-out /new/external/native-observer-output -- \
  --num_envs 4096 --headless --max_iterations 3000 --seed 21

# Frozen evaluation; fails rather than overwriting an existing output:
python projects/go2_piper_wbc/run.py native-eval \
  --upstream /path/to/LeggedManip_Lab -- \
  --checkpoint /path/to/your/cp_3000.pt --out /new/evaluation \
  --seed 7001 --num_envs 256 --mode deterministic --headless
```

The optional observer produces runtime receipts and exact completed-update checkpoint copies. Its bootstrap is explicitly imported only when requested; it is not installed as a global startup hook. The launcher requires a clean, pinned upstream `source/` and `scripts/`. It neither installs packages nor changes GPU/network settings or accepts simulator terms for you.

## Recipe and scratch entries

`run.py` exposes `p0-inventory`, `scratch`, `scratch-eval`, `recipe-audit`, `recipe-train`, `recipe-eval`, `aplus-audit`, `aplus-train`, and `startup-eval`. Use `MODE --check` to inspect the wrapper without a simulator. Actual launches require the Go2 environment and explicit external inputs:

```bash
python projects/go2_piper_wbc/run.py recipe-audit \
  --upstream /path/to/LeggedManip_Lab \
  --runtime-root /path/to/your/external-experiment-root \
  --native-params /path/to/native/env-and-agent-params -- \
  --out /new/recipe-interface-audit --admission /path/to/fresh/p0_admission.json \
  --num_envs 32 --headless

# Only after a fresh source-bound P0 and interface/RNG audit:
python projects/go2_piper_wbc/run.py recipe-train \
  --upstream /path/to/LeggedManip_Lab \
  --runtime-root /path/to/your/external-experiment-root \
  --native-params /path/to/native/env-and-agent-params -- \
  --out /new/recipe-pilot --admission /path/to/fresh/p0_admission.json \
  --interface-audit /path/to/completed/interface-audit \
  --num_envs 4096 --updates 1200 --seed 21 --headless

python projects/go2_piper_wbc/run.py recipe-eval \
  --upstream /path/to/LeggedManip_Lab -- \
  --checkpoint /path/to/your/cp_1200.pt --out /new/recipe-evaluation \
  --recipe-protocol T1 --admission /path/to/fresh/p0_admission.json \
  --num_envs 64 --seed 8101 --headless
```

These are bounded, guarded research entries. They require external native-runtime receipts and freshly source-bound P0 admission; the new checkout does not fabricate those inputs or automatically certify a robot workspace. Historical certificates are not reused as authorization for changed source. Publication source-binding keys cover this new directory layout; original hashes remain available for comparison.

P0 source is included as `p0_inventory.py`, `p0_frozen.py` (paired policy/control and `--workspace` physical collection), `recover_workspace_screening.py` (named-joint/FK screening), and `finalize_p0.py` (CPU certificate assembly). The latter expects the original protocol's paired seeds and workspace evidence, including the negative-control checkpoint; those external inputs are not bundled. It refuses to manufacture a certificate from the published aggregate tables.

The task-suite, telemetry, evaluator and tracking support shared by T1-v2 and Recipe were verified against nine exact execution-host source files before release. Their identity was checked by SHA256, not inferred. See [the verification receipt](../provenance/shared_support_verification.json).

A+ further requires the original source-bound CP1200/Adam/RNG input named in its spec. Its planned 600-update continuation was not completed in the inspected snapshot. The factorial B1/B2/B3 entries retain **false** admission records and intentionally refuse formal execution before physical/common-protocol binding. Original dependency locks involving withheld geometry are not represented as valid new locks.

## SGS and gradient diagnostics

The SGS core needs only the Python standard library. The historical 48-task experiment and the factorial candidate use different task pools and feedback protocols. Do not substitute one for the other. B2 and B3 reference the single B1 sampler implementation; the optional IK model must be independently supplied through `SHARED_IK_MODEL`.

PPO Gradient Lab can replay synthetic tests without a robot model. Its analysis CLIs require authorized, trusted checkpoint/rollout inputs and retain frozen-batch checks. No raw training tensors or checkpoints are in this repository:

```bash
python projects/ppo_gradient_lab/tools/diagnose_go2_piper_gradients.py --help
python projects/ppo_gradient_lab/tools/ppo_update_microscope.py --help
```

To rebuild the scientific figure from the published numerical summaries, use Matplotlib in an analysis environment:

```bash
python tools/plot_results.py
```

This produces an SVG numerical plot, not a robot-model render. It does not recompute the original simulator trajectories.
