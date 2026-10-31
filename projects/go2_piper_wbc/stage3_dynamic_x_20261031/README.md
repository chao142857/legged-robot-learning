# Stage3: original PPO Uniform dynamic X

This is an independent 21,600 GPU-container-second authorization based on Stage2 commit `52e0bfa`. The original historical B0-X task `ae1054046f792df8da6362c36ee168ded6b2ebcd`, native18D /210D actor /102D critic, DelayedPD, DR, mixed EE frame, Uniform.30/.70 and full20s A/T0/T1/X gates are retained. The original primary-host latest tree is still unverified. Neither CP1800 nor the Stage2 CP2000 is a qualified Uniform baseline.

| Allocation | Maximum GPU-container SUM |
|---|---:|
| P1 causal physics / native-gradient diagnostics | 2,400s |
| P2 evidence-selected single mechanism | 6,600s |
| P3 warranted continuation | 9,000s |
| P4 final evaluation / restoration / close | 3,600s |

Counts are accounting only. `gpu_budget.py` enforces the total, phase allocations and final reserve independently of the research controller. Each start freezes source, explicit config and full checkpoint SHA. Stage2 history and its closed ledger are only read.

P1 is implemented as an in-memory diagnostic overlay on the original evaluator. It preserves the physical task and separates its data from formal evaluations. `response_tap.py` reuses the earlier actual-input and four-substep DelayBuffer taps. New tests alter only observable target history while keeping actual body/joint state fixed, compare all18 mean actions, distinguish affine/soft/ramped requests, and decompose effective X contributions. They do not modify actor weights.

The physical intervention is explicitly diagnostic: a bounded6D FK/Jacobian feedback correction added to the six arm requests after a shared4s cold-start action prefix. The leg policy and original target/delay/physics/safety paths are retained. Leg actions may respond to the altered physical state; the intervention directly changes only arm requests. It reuses original Piper FK, not a new kinematic model. It is never counted as learned PPO success. The CPU reduction verifies the actual prefix and measured initial state/DR, compares requested/delayed/actual joint responses, retains all20s failures and reports native27-term reward differences separately from partial steady-window diagnosis.

A separate50-update native PPO diagnostic window covers the full reference timespan and observes original GAE, cohort exposure, actual total-loss gradients before Adam, fixed-input mean-action changes and native KL. It never constructs alternate advantages or per-reward surrogate PPO gradients. Its weights do not become the P2 parent; any treatment starts from the same original CP1800. This window and failed starts consume P1's allocation.

At preparation time both SSH and the Tailscale data plane were unreachable. Local syntax/overlay checks are not PhysX evidence. Only remote `BOOTSTRAP.json`, `LEDGER.json`, actual Docker state and run receipts establish deployment. No new physical result or P1 PASS is claimed here. A bounded connection-recovery process uses the existing SSH only, with no chat reminders or network/configuration changes.

Runtime roots, original asset/chain locations and private native configs are supplied explicitly at deployment. The internal `/stage2-active` and `/stage2-config.json` container mount names are retained compatibility aliases; they bind the new Stage3 run's frozen files, never the old Stage2 run directory. New resource ledger files are not copied from Stage2.

```bash
# New authorized Stage3 root only; no completed ledger may be reopened.
python3 tools/gpu_budget.py --root "$STAGE3_ROOT"
python3 tools/p1_pipeline.py --root "$STAGE3_ROOT"
# CPU-only reduction of actual P1 evidence:
python3 tools/reduce_p1.py --root "$STAGE3_ROOT"
```

P2/P3 are contingent on actual causal evidence. They are not asserted implemented or running merely because P1 code exists. Formal seeds511031/521031/531031 remain closed until development selection and the full original acceptance checks are frozen.


2026-10-31实际执行：P1已完成，严格冷启动前缀逐项相等。三个物理对照均32/32安全产生至少8mm有效X响应，原27项总奖励均提高；请求目标的X幅度约3.3mm，经真实DelayedPD后仍约3.3mm。P2选择零初始化的Gaussian均值协同头，保持原210D观测、27项奖励、Uniform、LR/熵/执行器和原评测器。此时仅在集成smoke阶段，尚无新增策略收益或baseline PASS。失败启动及未配对诊断完整留账。证据见evidence/P1_CAUSAL_SUMMARY.json；后续运行状态以YangLab独立账本为准。
