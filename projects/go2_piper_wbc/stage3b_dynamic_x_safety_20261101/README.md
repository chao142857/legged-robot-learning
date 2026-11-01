# Stage3b: original Uniform PPO dynamic X and safety

**Final result: complete baseline FAIL; best single checkpoint passes12/12 formal cells, its frozen predecessor passes8/12. Not admitted as an SGS baseline.** Training and the independent28,800s GPU-container grant are closed. No gate, target amplitude or historical score was changed after formal evaluation.

The target is historical B0-X `ae1054046f792df8da6362c36ee168ded6b2ebcd`: original18D PPO,210D term-major actor history,102D critic,27rewards, Uniform30% dynamic X /70% fixed targets, native mixed-frame EE, DR, DelayedPD and20s A/T0/T1/X. The offline primary-host latest tree remains unverified. This is a modified **Uniform training recipe**, with an internal Gaussian arm-synergy mean head and explicit function-retention losses; it is not an unmodified native-PPO claim. SuperMimic, SGS, hardware and unrelated tasks were not run.

All learning uses the common own A+ CP1800 warm-start, SHA256 `16333b869458ae62c371fd68b92f61e4740c250635f8b57906809ac067a77091`. It is not a qualified teacher. Continuationseeds21/22 share its pretraining and are not from-scratch seeds.

The original weak response is localized to goal-to-joint coordination: the commanded-target FK response was already about3.29mm, the delayed target3.29mm and actual response4.09mm in the recorded diagnostic. Real bounded PhysX interventions produced at least8mm in three task classes. This establishes available physical authority; it does not establish an intervention-trained policy or uniquely rule out reward competition. Matched100-update native PPO yields only2.61/2.73mm versus4.57/4.73mm for the unregularized synergy actor on continuationseeds21/22; the latterseed21 has a real hard-limit failure.

The mean head is zero-initialized inside the native Gaussian before sampling/log-prob, using the original observable210D and a fixed measured six-joint geometric direction. No inference IK, external leg policy or post-sampling action is used. All18 original output rows remain trainable. Native GAE, reward, LR rule, entropy, sampler and physics stay unchanged.

Unconstrained shared-policy drift caused real leg safety failures. Parent-leg function substitution removed some matched failures, but safe200leg substitution and head-zero intervention did not remove the laterseed22 calf contact; restoring the complete parent function did. These are frozen diagnostic compositions, never policy successes. The final recipe keeps global leg Gaussian KL weight.02 and adds full18D mean-only parent preservation weight.00419 only when the original visible XYZ goal history is held. The parent anchor never moves; current std is not part of the new mean term. Both seeds have exact common-parent restoration and94 identical first-rollout physical/reward fields plus Uniform exposure against matched controls. Later on-policy data diverge.

| Final recipe / continuation seed | Additional updates | Worst development X gain / amplitude | Full8 development matrix |
|---|---:|---:|---|
| seed22 |300 |.38293 /8.00657mm |PASS;512 complete,505 strict-safe,0hard/contact |
| seed22 |400 |.44543 /9.13924mm |PASS;512 complete,506 strict-safe,0hard/contact |
| seed21 |300 |.37536 /7.82939mm |FAIL amplitude; retained unchanged |
| seed21 |400 |.44446 /9.16861mm |PASS;512 complete,505 strict-safe,0hard/contact |
| seed21 |500 |.48700 /10.07022mm |PASS;512 complete,502 strict-safe,0hard/contact |

Each cell uses two development seeds6181031/6281031 and all four original protocols. The same-seed .02-only500 had greater X10.776mm but failed original direction retention; the new recipe restores the full development contract instead of maximizing X alone. Separate learning curves retain every unsafe and retention-failed branch: [curve](figures/STAGE3B_PHYSICAL_LEARNING.svg), [numeric points](figures/STAGE3B_PHYSICAL_LEARNING.json).

Before opening new formal seeds611101/621101/631101, selection froze seed21 CP2300 and same-lineage CP2200. Selectorv4 prioritizes already-qualified adjacent pairs and their weaker X gain; all original final checks and the user's zero actual hard/nonfoot rule remain mandatory. Research closed at24,520.68s. An optional extra seed22window was skipped because it could not fit the protected final reserve, not because of an update quota. No tuning or reselection followed the formal results.

| Frozen policy | Formal protocol×seed checks | Complete episodes | Strict-safe episodes | Actual hard / nonfoot episodes |
|---|---:|---:|---:|---:|
| Common initialization |Not a qualified baseline |768/768 |749/768 |0 /0 |
| Selected CP2300 |12/12 PASS |768/768 |745/768 |0 /0 |
| Frozen adjacent CP2200 |8/12 PASS |767/768 |739/768 |1 /1 |

| Formal seed | Initialization X amplitude | CP2300 X gain | CP2300 X amplitude | CP2300 X correlation |
|---:|---:|---:|---:|---:|
|611101 |3.555mm |.50805 |10.588mm |.72024 |
|621101 |3.681mm |.50091 |10.471mm |.72238 |
|631101 |3.639mm |.52465 |10.874mm |.74824 |

CP2300 passes the original EE, body direction/yaw, dynamic-X, safety and survival checks on all three seeds. Its after1s EE position RMSE ranges1.41–2.94cm across protocols, orientation RMS.0196–.0563rad. Native joint-success5cm counts are A133/192, T0187/192, T1185/192 and X186/192; these counts are not substituted for full acceptance. Tracking retains the original after1s mask; strict safety and failure denominators include startup and all20s.

The continuous-node failures are concrete and remain FAIL:

- CP2200 A/621101:59/64 strict-safe, five native overspeed episodes, all hits within0–.44s; required fraction remains≥.95.
- CP2200 X/621101:60/64 strict-safe, four overspeed episodes within0–.32s.
- CP2200 A/631101: FL_hip actual excess2.041mrad, env20, control interval12.40–12.42s, body commandvx=-.45m/s; the native1mrad threshold is unchanged.
- CP2200 T1/631101: one real nonfoot terminal contact at18.82–18.84s, env20,2560N native scalar. The recorded command is yaw-rate-.102rad/s. Link/contact-pair identity was not recorded and is not inferred.

Thus X learning is real, while rare leg safety failures and checkpoint-to-checkpoint stability remain the blocking problem. A single passing checkpoint cannot satisfy the frozen baseline contract. See [full final audit](evidence/FINAL_RESULT.json), [exact native failure timing](evidence/FINAL_ADJACENT_FAILURE_TIMING.json) and [frozen selection](evidence/FROZEN_SELECTION.json). The [predeclared20s trace](figures/FINAL_PREDECLARED_NATIVE20S.svg) uses X/611101/env0 fixed before testing, keeps startup and actual body motion, and does not replace the full64-episode matrices.

Best single-checkpoint candidate, retained on YangLab:

```text
/data/go2_piper_wbc_stage3b_dynamic_x_20261101/runs/20261101T105851-steadyKL001-seed21-window-to500/ppo/checkpoints/cp_2300.pt
SHA256 29471d2e1536eef9df96853609a78b49dcf0886e59cef06030851dd0b637e5ef
```

Frozen predecessor SHA256: `046c1202b831b1e172aca2e310c29ea266d9fe1a16899444d82969adcadabe87`. It is a development pass with formal safety failures, not a safe continuation source for a new study.

Actual total cost: **26,061.5172/28,800 GPU-container seconds (7.2393h)**, including failures, diagnostics, training, formal tests and recovery.2,738.4828s are unused and not transferred. There are2,900 actual PPO updates,58,000 combined native actor/critic Adam calls,285,081,600 PPO-used transitions,13,568,000 measured evaluation slots,1,472,000 measured probe slots and98,304 final restore PhysX slots. Two failed probes have unmeasured partial-transition counts; their GPU time is included and their counts are not silently zero. No update or transition cap was imposed.

The selected lineage uses500 additional PPO updates,10,000 native Adam calls,49,152,000 training transitions and2,746.8866 GPU training-container seconds. These exclude its parent pretraining and study-wide diagnostics. Parent pretraining cost was not reconstructed or relabeled as free new learning. Original Stage3 remains independently closed at11,537.2300/21,600s; all protected closed-file SHAs are unchanged. The audit verifies265 frozen run snapshots and actual Docker SUM durations.

Stochastic training is separate from deterministic-policy acceptance: native failed exploration/reset records are retained. Across2,900 rollouts,1,659 have an all-joint maximum above1mrad,1,042 have positive nonfoot slots and1,364 positive-contact transition slots are recorded; these are not unique episode or joint counts. Maximum recorded all-joint excess is7.115mrad. Zero-event claims apply only to the stated policy-evaluation matrix, never to all exploration.

Full actor/critic/Adam/RNG restore was actually run from CP2300: zero PPO/optimizer updates,24 real PhysX controls over4096envs,98,304 slots, unchanged learner/Adam and exact native/head optimizer ages. It does not clone hidden PhysX state. The [reproduction receipts](evidence/REPRODUCTION_RECEIPTS.json) provide every actual training/evaluation/restore config and launch path with full SHA; all models and NPZ remain on YangLab.

To inspect the exact executed native restore command on YangLab without launching GPU:

```bash
TASK_STAGE3B_ROOT=/data/go2_piper_wbc_stage3b_dynamic_x_20261101
python3 - "$TASK_STAGE3B_ROOT" <<'READ_ONLY'
import json, pathlib, shlex, sys
root=pathlib.Path(sys.argv[1])
run=pathlib.Path(json.loads((root/'FINAL_RESTORE_RUN.json').read_text())['run'])
print((run/'config.json').read_text())
print(shlex.join(json.loads((run/'launch.json').read_text())['command']))
READ_ONLY
```

This prints the real Docker/native-entry invocation and exact config; it does not replay it. A fresh GPU reproduction requires a new independent authorization, metered root, container name and writable output mounts. Keep frozen checkpoint/source SHAs and native task inputs; do not reuse the closed container, overwrite its outputs or reopen this grant. Training-source reconstruction and one-time activation are documented by `SOURCE_REUSE.json`, `tools/build_payload.py` and the immutable authorization/receipt records. CPU contract tests cover Gaussian mean/log-prob attachment, fixed-anchor loss gradients, safe receipt-only coordinator restart, selection, and time-only budget protection. No more research was appended after opening formal seeds.
