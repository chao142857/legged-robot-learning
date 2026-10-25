# Go2 + Piper Whole-Body Control

An 18-action leg/arm research stack based on the independently obtained LeggedManip_Lab framework. Native WBC, fixed tasks, scratch training and recipe interventions are separate experiments with separate result definitions.

| Snapshot | Status |
|---|---|
| `native_20261023/` | Native-distribution scratch baseline, 3000 updates, frozen evaluation and observer/analysis tools |
| `task_suite_20261023/` | Task adapters and offline metrics; selected warm-start guard code is retained for CPU checks, not a full published warm-start launcher |
| `scratch_t1_20261025/` | Actual executed T1 v2 source and negative-result evaluation; invalid observer v1 excluded from valid training counts |
| `recipe_v1_20261025/` | Bounded 1200-update stage A, fixed EE anchors, source-bound interface checks, full T1 failure/safety statistics |
| `recipe_aplus_20261025/` | Recovery and sampling preparation; **0 additional completed updates**, initialization blocker retained |

The public [launcher](run.py) resolves these independent module paths without merging their algorithms or environments. It requires a clean pinned external framework checkout. `--check` is CPU-only; execution requires independently obtained assets and runtime receipts. See the [reproduction commands](../../docs/REPRODUCIBILITY.md).

The native baseline learned simulated locomotion/pose tracking, but survival is not a precision gate. Recipe v1 changes the task distribution, reward/control constraints and reset policy, so its lower EE error is not a matched comparison of algorithms. Its formal T1 gate remains false. B1/B2/B3 preparation lives under [SGS](../sgs/README.md), with the shared sampler and controller ownership kept explicit.
