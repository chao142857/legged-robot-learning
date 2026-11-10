# Success Guided Sampling

Local implementation and controlled-study tooling for SGS from [Zhang et al., *A Balanced Data Diet*](https://arxiv.org/abs/2610.12465v1), §3.2 and Appendix B.2. This work implements and evaluates that method; it does not claim to introduce SGS.

## Current neural-sampler results — 10 November 2026

[CleanMeta8 completed reproduction](cleanmeta8_20261109/README.md) records the corrected V0.2 Teacher and both 10k Student transfers. At CP10000, strict joint success is **33.98% for seed 48** and **34.47% for seed 49**. The reused Uniform results are 25.78% / 35.94%, and Fresh results are 35.94% / 29.98%; the result does not establish a consistent advantage across both seeds.

The [Adam snapshot audit](https://github.com/chao142857/legged-robot-learning/blob/c43b36ea4cb3dbdaa3d0f72730c349188a6d5130/projects/sgs/teacher_adam_audit_20261109/REPORT.md) identifies the historical backtracking-state aliasing bug. Every corrected Teacher update starts its trial steps from an independent optimizer snapshot; eight accepted updates end at Adam step8. Old Meta8/16 weights and curves retain their legacy identity and must not be presented as corrected-algorithm evidence.

This neural sampler is a separate protocol from the fixed-target SGS implementation below: CleanMeta8 uses state-conditioned preferences, new counterfactual branch feedback and 20% Uniform mixing, then freezes Teacher parameters during Student transfer. See the [full checkpoint curves](cleanmeta8_20261109/evidence/completed_curves.json), [per-update Adam acceptance](cleanmeta8_20261109/evidence/adam_acceptance_summary.json), and [executed source](https://github.com/chao142857/legged-robot-learning/tree/b1b5cd5b092967678b9b5f315676da6f442bc9f4/projects/sgs/cleanmeta8_20261109). [CL1](https://github.com/chao142857/legged-robot-learning/tree/ab4532ea1af7ab7751136e95dd4187d386457ee5/projects/sgs/nsgs_cl1_20261109) remains a separate follow-up study.

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

