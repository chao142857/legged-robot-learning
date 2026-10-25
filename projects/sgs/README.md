# Success Guided Sampling

Local implementation and controlled-study tooling for SGS from [Zhang et al., *A Balanced Data Diet*](https://arxiv.org/abs/2610.12465v1), §3.2 and Appendix B.2. This work implements and evaluates that method; it does not claim to introduce SGS.

## Implementations and experiments

- `historical_48task_20261019/` retains **both** the executed source and later working source, fixed task/success definitions, and the Uniform/SGS pilot results. The versions are not silently merged.
- `factorial_20261025/go2_piper_b1_sgs_20261025_work/shared_sgs/` is the sole shared sampler/lifecycle implementation for the prepared B1/B2 comparison. It adds stable task content identities, terminal ownership, feedback deduplication, censoring, isolated RNG/state restoration and fixed-evaluation isolation.
- The sibling B2/B3 folders retain common-protocol, leg-PPO/arm-IK composition and shared-IK preparation. They are **not formally admitted or trained results**. Model-derived `kinematics.json` is excluded; an authorized external model is required for IK/physical tests.

For each task, the frozen core uses recent Boolean outcomes, a beta-shaped kernel, and temperature-scaled log weights. Defaults are history 100, target 0.5, concentration 1, temperature 2 and epsilon 1e-4. The unvisited prior of 0.5 is local; epsilon floors the kernel and is not a uniform-mixture probability floor.

```bash
# Standard library only; run from repository root:
python -m unittest discover \
  -s projects/sgs/factorial_20261025/go2_piper_b1_sgs_20261025_work/tests -v
```

The historical pilot provides **no observed joint-success benefit**: both methods begin at 40/48 and finish at 0/48, with no positive joint-success training feedback. The newer implementation's CPU contract tests establish engineering behavior, not algorithmic superiority. Full evidence and comparison limits are in [RESULTS.md](../../RESULTS.md).

The fixed historical success protocol, latest factorial common protocol, and evaluation-only scheduler are distinct. Do not reuse old thresholds, initialization or task counts in the newer experiment without a new, explicit protocol binding.
