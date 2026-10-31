# Stage2: original Go2+Piper PPO + Uniform

This experiment restores the historical B0-X task, commit `ae1054046f792df8da6362c36ee168ded6b2ebcd`, on the user-approved existing YangLab RTX3060. The original primary host is offline; this is not a claim to have inspected its latest working tree. No SuperMimic learner, checkpoint, teacher or task is used.

| Contract | Frozen source / actual execution |
|---|---|
| Initialization | Own A+ CP1800, SHA256 `16333b869458ae62c371fd68b92f61e4740c250635f8b57906809ac067a77091`; actor, critic, Gaussian, Adam, adaptive LR and saved RNG restored |
| Actor / critic | Native 18D PPO; term-major 70D × 3 history = 210D actor, 102D critic |
| Action | Original affine scale .25, soft-limit margin .02rad, .5s startup ramp, original DelayedPD / DR |
| EE frame | Original `aim_pose`: link0-local XY, world height Z, original quaternion/reward conventions; preserved, not replaced with Stage8/9 goals |
| Training control | Original Uniform dynamic .30 / fixed .70 mixture, 27 fixed anchors, X ±.02m / 8s |
| Single-factor treatment | Uniform dynamic .70 / fixed .30; all other task, reward, network and PPO settings unchanged |
| Physics | Original A/T0/T1/X, full20s; native startup, failures, strict safety and independent phase bank retained |
| Learning gate | Original X gain ≥.35, correlation ≥.5, amplitude / error / retention gates from frozen B0 spec; native `joint_success_5cm` alone is insufficient |
| Budget | Approved SUM 14,400 GPU-container seconds, last1,800 reserved. Counts are accounting, not additional caps. |

`BUDGET_REQUEST.json` is the preserved request view. Approval and the explicit host amendment are in `AUTHORIZATION.json`; its initial zero-use field is not the live GPU clock.

The public archive does not contain robot assets, pretrained weights or an Isaac runtime. Reproduction requires the verified original source/assets/checkpoint and pinned existing Docker image. Each run freezes its `source/`, `config.json`, `launch.json` and SHA256. Runtime paths are explicit external configuration; historical outputs stay read-only.

```bash
# Existing authorized experiment root; do not reopen a completed ledger.
python3 tools/gpu_budget.py --root "$STAGE2_ROOT"
python3 tools/pipeline.py --root "$STAGE2_ROOT"
python3 tools/continue_pipeline.py --root "$STAGE2_ROOT" --source "$ORIGINAL_TASK_SOURCE"
python3 tools/finalize_stage2.py --root "$STAGE2_ROOT" --source "$ORIGINAL_TASK_SOURCE"
```

Run these as separate persistent processes, not competing GPU launchers. `CPU_RUNTIME.json` selects an already-installed NumPy-capable Python. The first pipeline performs paired200 windows; continuation checks real physics before matched400 and any justified later window. A plateau writes an analysis boundary rather than consuming the time allowance blindly. The finalizer freezes selection before the three independent final seeds, uses the original evaluator, verifies full learner restoration and closes the ledger.

`train_entry.py` calls installed native `OnPolicyRunner.learn` and native PPO/Adam; it does not implement a new optimizer. Its additional code observes finite values, all18 output-row changes, KL, true reward-time physics and Uniform coverage. A normal restart restores learner/RNG, then explicitly cold-resets PhysX; it does not claim internal PhysX-state restoration. Resume accounting is parent-relative. Independent time stop is honored at a complete rollout boundary.

An optional subsequent contrast tests native PPO entropy coefficient `.005` versus `.001` only after the declared weak-response diagnosis. It starts again from safe CP1800 with the original `.30/.70` Uniform sampler; it is not combined with the `.70` dynamic-coverage treatment. `entropy_trial.py` records its admission, activation and actual run separately. Engineering admission and the formal strict-safety gate are distinct; the latter remains95% per seed. The already executed same-parent, same-seed, same-window native control is reused, and its missing T0/T1 tests are completed before comparison.

`analyze_physics.py` reuses the original capability function and leaves raw scores unchanged. `response_breakdown.py` reuses the existing named-joint/FK harmonic diagnosis on new traces. Geometric projections and steady-window diagnosis are separate from native full20s acceptance. Model files and raw trajectories remain on the execution host.
