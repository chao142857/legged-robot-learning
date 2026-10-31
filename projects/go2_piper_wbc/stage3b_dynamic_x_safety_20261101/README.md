# Stage3b: original Uniform PPO dynamic X and safety

Independent authorization:28,800 GPU-container SUM seconds, with the last3,600 unavailable to research. Counts are accounting, never lifetime quotas. Stage3 closed first at11,537.2300/21,600 seconds; its unused balance is not transferred. Its complete formal results and restoration are frozen on branch `research/stage3-dynamic-x-20261031`, commit `ded395e`.

This study uses the same historical B0-X `ae1054046f792df8da6362c36ee168ded6b2ebcd`,18D Gaussian PPO,210D term-major actor history,102D critic, original27rewards, Uniform.30dynamic-X/.70fixed, DR, DelayedPD and full20s A/T0/T1/X acceptance. The latest original primary-host tree remains unverified. SuperMimic, SGS, M3, hardware and video-demo work are excluded.

The original3mm response is no longer an untested actuator hypothesis: Stage3's real interventions established available X motion; the trained synergy head produced heldout10mm response. Its400-update candidate is strictly free of actual hard/contact events on768 formal episodes, but two T1 direction-retention cells fail. The600-update branch has an actual FL_hip event at1.98s and later nonfoot contacts and is never resumed for training. CP1800 and safe200 remain immutable own warm-starts, not qualified Uniform teachers.

New targeted frozen probes compare current functions, parent legs, parent arm and parent-all after an identical1s cold prefix and.25s blend. Their actual histories/DR, four-substep targets, applied efforts, contacts and original metrics are recorded.17cells completed; three further parent-arm cells were omitted after severe diagnostic instability. These compositions are **not trained PPO successes**.

| Actual paired case | Current | Parent legs only | Meaning |
|---|---|---|---|
| Native200, A6181031 | one hard event | no hard/contact;64 complete | Leg-function drift exists without the synergy head |
| Synergy100seed21, A6181031 | one hard event |64/64 strict-safe and complete | Independent continuation seed has the same leg-function effect |
| Synergy600seed22, T1 6281031 | one contact;63 complete | no hard/contact;64 complete; X13.39mm | Changing only legs can retain the learned arm response and remove this failure |
| Synergy600seed22, A6181031 | one hard event | hard removed, three contacts introduced | Hard frozen-leg substitution is not a joint solution |

Parent-arm replacement caused27/64 hard events in A and30/64 in T1, and was stopped. The evidence supports coupled control, not a unique claim that added arm motion or the actuator alone caused failure. `evidence/PHYSICAL_CAUSAL_PROBES.json` retains every completed cell, exact prefix/DR checks and unsafe omitted-cell reasons. Full arrays stay on YangLab.

The first new recipe is `arm_synergy_fixed_legKL_001`: one additive fixed-parent leg Gaussian KL of weight.01 inside the installed native total PPO loss. It leaves all18D actor parameters trainable and permits coadaptation. The anchor is the immutable common CP1800, SHA256 `16333b869458ae62c371fd68b92f61e4740c250635f8b57906809ac067a77091`; it never moves. Original rewards, GAE, Gaussian likelihood, optimizer, adaptive LR rule, entropy, sampler, limits and physics remain. No external leg policy or IK runs in trained-policy inference.

The first100-update seed22 matrix is real:512/512 complete,508/512 strict-safe, zero hard/nonfoot events and original retention passing. X gain .224/.199, amplitude4.95/4.45mm; this is slightly lower than same-parent/seed unconstrained100 (.242/.220,5.14/4.73mm) and below the final gate. No new reliable joint learning benefit is claimed from this early node. The next100-update continuous window actually started from its full learner checkpoint, SHA256 `12eedd7921ec698e7111431e173496caaf759a7da592a3c2e13049f699233525`. See `evidence/DEVELOPMENT100_SEED22.json`.

The200-update node completed512/512 episodes,506/512 strict-safe, with zero actual hard/nonfoot events and every original retention check passing. X gain is .344/.314, amplitude7.19/6.61mm and correlation .658/.621; the8mm gate remains unmet. Its full-state CP2000 SHA256 is `93f5fb99e6c3ec27af1ab41e46a81a41c86a794699611bf01cf35a7e57f3adc3`. The400-update window resumed this safe checkpoint and optimizer/RNG, not the unsafe old600. `evidence/DEVELOPMENT200_SEED22.json` contains all eight cells.

On all608 identical recorded P1 observations, leg mean-action drift versus the fixed parent is .0603 RMS at new200 versus .1276 at unconstrained200; arm drift is .2238 versus .2455. The128 backward-command states within that set have leg drift .0898 versus .1792 and arm drift .3808 versus .4057. At400, the all608 leg/arm drifts are .0778/.3916 versus unconstrained .2263/.4422. These are subgroup-specific function measurements, not physical tracking scores. In actual full20s A6181031, the minimum FL_hip mechanical margin at recorded50Hz control samples is36.66mrad versus1.60mrad, with identical bounds; applied-torque saturation is3.17% versus3.85%. This sampled margin does not resolve every5ms state. The separate hard-excess and saturation taps include physics substeps. Peak hip speed is higher (11.61 versus10.92rad/s), so this is not a claim that every stability measure improved. `evidence/FUNCTION_AND_LEG_MARGIN_AUDIT.json` distinguishes identical-input function diagnostics from closed-loop physical quantities. These are development results, not formal baseline PASS.

`figures/STAGE3B_PHYSICAL_LEARNING.svg` plots completed physical nodes, separate continuation and evaluation seeds, with original thresholds and hard/contact events. No diagnostic composition is plotted as learned capability, and no control result is extrapolated to unevaluated update counts.

At400 additional updates, both development seeds pass every unchanged A/T0/T1/X gate:512/512 complete,503/512 strict-safe, zero actual hard/nonfoot events. X gain is .499/.461, real amplitude10.33/9.57mm and correlation .723/.704. This is the first full development node satisfying the original gates, not yet the required consecutive-node or formal result. The actual CP2200 SHA256 is `60832742811af936192cc337d0d11714d7eccb0f97a317128c97cde09e36e1b8`; the next200-update window has resumed its complete learner/Adam/RNG state. See `evidence/DEVELOPMENT400_SEED22.json`.

The coefficient comes from a CPU scale audit: about25%-38% of observed native actor preclip-gradient norms at old200/400 fixed-state probes, not an exact decomposition of PPO gradients. The new GPU smoke performed2PPO updates/40Adam calls, restored the complete parent learner/RNG, proved exact real/synthetic mean/target/log-prob/JIT initialization and identical first-rollout94native physics/reward/cohort fields against the unconstrained same-parent/seed control. All18 output rows changed. This is engineering evidence; new physical learning is assessed separately at complete milestones.

`learning_pipeline.py` waits for actual causal completion, freezes the recipe, runs same-parent continuationseeds22/21 and full physical milestones, stops unsafe or twice-flat branches, and preserves every checkpoint. `finalize_stage3b.py` freezes one candidate and a same-recipe/seed adjacent node before new seeds611101/621101/631101. It applies every original gate plus the user's zero hard/contact branch-stop rule. Old Stage3 formal seeds are already consumed. No final PASS is claimed here.

The new root is `/data/go2_piper_wbc_stage3b_dynamic_x_20261101`. This single grant is pinned to that root and ID: another directory cannot receive a fresh8hours. The budget supervisor and CPU coordinators persist independently of chat; no chat reminders are created. Container startup, failures, paused occupation, collection, training and evaluations all enter the same sum. Closed studies reject re-entry.

Runtime source is assembled from hash-checked Stage3 templates plus the three small patches in `SOURCE_REUSE.json`, rather than publishing duplicate snapshots. The resulting full source is frozen in each actual run. Private native/asset configs are copied explicitly from closed Stage3; no implicit live dependency on an old port or current Mac directory is used.

```bash
# CPU-only source reconstruction; does not create a study or start GPU.
python3 tools/build_payload.py --base-tools "$CLOSED_STAGE3/active" --out /tmp/stage3b-source.json
# One-time activation requires the authorized root not to exist and Stage3
# final matrix, full learner restore and closed-file SHA proofs to exist.
python3 tools/bootstrap.py --old "$CLOSED_STAGE3" --new "$STAGE3B_ROOT" < /tmp/stage3b-source.json
nohup python3 "$STAGE3B_ROOT/active/causal_pipeline.py" --root "$STAGE3B_ROOT" > "$STAGE3B_ROOT/causal-launch.log" 2>&1 &
nohup python3 "$STAGE3B_ROOT/active/learning_pipeline.py" --root "$STAGE3B_ROOT" > "$STAGE3B_ROOT/learning-launch.log" 2>&1 &
nohup python3 "$STAGE3B_ROOT/active/finalize_stage3b.py" --root "$STAGE3B_ROOT" > "$STAGE3B_ROOT/final-launch.log" 2>&1 &
# CPU fixtures: no real Docker/GPU or historical file writes.
python3 tools/test_budget_cpu.py
```

The current root is already active; resume its existing coordinators/receipts rather than rerunning bootstrap. A new reproduction after this grant closes requires a new authorization. Large checkpoints, full trajectories and environment assets remain on the execution host.

`collect_delivery.py --root "$STAGE3B_ROOT" --wait` is a CPU-only final audit. It waits for actual completion, checks frozen-run and closed-Stage3 SHAs, real Docker SUM duration, unchanged formal checks, complete learner restoration and counter consistency. It does not select a model, open evaluation seeds, launch GPU or modify a closed ledger. Final audit results are stored in a new delivery subdirectory.
