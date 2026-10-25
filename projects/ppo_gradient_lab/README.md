# PPO Gradient Lab

CPU tools for frozen-policy gradients, component value probes, native GAE calibration, KL, clipping, Adam update direction and held-out surrogate measurements. Original RSL-RL implementations remain external and explicitly versioned.

| Tool | Purpose |
|---|---|
| `diagnose_go2_piper_gradients.py` | Frozen network/batch validation, reward/return decomposition and gradient measurements |
| `build_go2_gradient_bundle.py` | Build checked diagnostic bundles from authorized raw rollout inputs |
| `analyze_go2_gradient_probe.py` | Independent component value probes and held-out analysis |
| `ppo_update_microscope.py` | Restore native actor/critic/Adam state per candidate; perform one isolated native minibatch step |
| `decompose_ppo_microscope_clip.py` | Separate unclipped surrogate change from clipping penalty |
| `summarize_ppo_microscope.py` | Aggregate the paired diagnostic results |

Use a dedicated Python 3.11 environment with [historical requirements](requirements.txt). From the repository root:

```bash
python -m unittest discover -s projects/ppo_gradient_lab/tests -v
python projects/ppo_gradient_lab/tools/ppo_update_microscope.py --help
```

The real-checkpoint parity test requires an external `B0_CHECKPOINT_ROOT`; it is skipped when unavailable. Synthetic tests neither require robot assets nor launch a simulator. The [release validation](../../VALIDATION.md) records the actual CPU test environment, which differs from historical CUDA training.

Published `results/microscope/` files contain numerical summaries and preregistration. They do not include original checkpoints, rollouts, or gradient-direction tensors. Smaller-step surrogate improvements do not establish robot return improvements or a causal explanation of historical failure. Component probes are not the native scalar critic's exact decomposition. See [RESULTS.md](../../RESULTS.md).
