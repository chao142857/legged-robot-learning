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

A separate50-update native PPO diagnostic window covers the full20s task timespan and observes original GAE, cohort exposure, actual total-loss gradients before Adam, fixed-input mean-action changes and native KL. It never constructs alternate advantages or per-reward surrogate PPO gradients. Its weights do not become the P2 parent; any treatment starts from the same original CP1800. This window and failed starts consume P1's allocation.

The initial network block is historical and preserved in `evidence/NETWORK_BLOCK_AT_DEPLOYMENT.json`. SSH recovered and the experiments below actually ran. Each remote start freezes source/config/checkpoint SHA; the persistent GPU watcher and CPU closer are independent of chat.

Runtime roots, original asset/chain locations and private native configs are supplied explicitly at deployment. The internal `/stage2-active` and `/stage2-config.json` container mount names are retained compatibility aliases; they bind the new Stage3 run's frozen files, never the old Stage2 run directory. New resource ledger files are not copied from Stage2.

```bash
# New authorized Stage3 root only; no completed ledger may be reopened.
python3 tools/gpu_budget.py --root "$STAGE3_ROOT"
python3 tools/p1_pipeline.py --root "$STAGE3_ROOT"
# CPU-only reduction of actual P1 evidence:
python3 tools/reduce_p1.py --root "$STAGE3_ROOT"
```

Actual results as of 2026-11-01: P1 completed with three identical4s cold prefixes and measured initial DR/state equality. All three diagnostic treatments produced >=8mm effective X response in32/32 strictly safe complete20s trials; original27-term reward rate increased. Requested3.3mm X motion remained3.3mm after DelayedPD, while target-conditioned shoulder/elbow/wrist coordination substantially cancelled. These are diagnostic actions, not PPO success.

P2 uses `original210_arm_synergy_mean_v1`: one zero-initialized131->1 bounded arm-synergy mean head, original128latent plus derived observable X/velocity/local linear-FK error, fixed original-FK geometric direction. It enters the native Gaussian before sampling/log-probability. All original leg/arm MLP parameters remain trainable. No inference IK, BC, external leg policy, extra privileged actor inputs, reward/LR/entropy/sampler/PD changes. The training distribution is original dynamic X +/-20mm only (.30), fixed27anchors (.70), with original A+ base-command sampling.

The true native control and treatment use the same CP1800, trainingseed21 and2/100/200 physical-reset boundaries; the old Stage2 uninterrupted200 is diagnostic only. Zero-output means/targets, Gaussian log-probability, full JIT and full native Adam/RNG resume have executed. At100 new updates, original20s X tests were:

| Development physics seed | Frozen parent gain / amp | Matched native100 | Synergy100 |
|---|---|---|---|
|6181031|.109 /3.64mm|.114 /2.99mm|.229 /4.94mm|
|6281031|.083 /3.35mm|.084 /2.61mm|.210 /4.57mm|

This is real partial learning, not baseline PASS. Seed21's A test had one actual FL_hip event at7.16s:1.047mrad hard-limit excess, with a valid requested target and saturated return torque. Neither frozen parent nor matched control had that hard event. The branch was stopped; its checkpoint is preserved as an unsafe diagnostic, never the next initialization. The preregistered independent continuationseed22 started from common CP1800 with the same mechanism and paired native control. These are shared-parent continuation seeds, not full from-scratch seeds.

Evidence: `evidence/DEVELOPMENT100_ALL_PROTOCOLS.json`, `evidence/SEED21_FL_HIP_COUPLING.json`, and `evidence/CPU_POLICY_SENSITIVITY_1900.json`. The CPU counterfactual/Jacobian projection is separate from real physical scores. Figures: `evidence/figures/P1_PHYSICS-v1.svg` (amplitude follows original after1s metric; safety covers all20s) and `evidence/figures/LEARNING-24rows.svg`.

Formal511031/521031/531031 seeds remain unopened until frozen development selection. Original gates, including continuous saved-node stability, remain unchanged. Large weights and full trajectories stay on YangLab. The current result is not a qualified Uniform baseline for SGS.
