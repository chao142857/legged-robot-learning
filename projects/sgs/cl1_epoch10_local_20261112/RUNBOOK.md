# CL1 Teacher epoch-budget local calibration

Only the Teacher PPO optimization budget changes: 2 versus at most 10 epochs, with the original KL stop. Architecture, inputs, sigma=0.2, learning rate, GAE, reward, gradient clipping, decoder and Student PPO remain fixed. The original Base is the zero-update control.

The fixed midpoint selection uses source seed43/44 at update5000 for two new500-update collection fragments. Both are collected under Base version20, then the same10 macro transitions are optimized from the same complete pre-update Teacher/Adam/RNG snapshot. Behavior log-probabilities, raw advantages and return targets are fixed across epochs. Historical fragments are never appended to this new on-policy batch; source history is used only for recurrent burn-in. Collection fragments bootstrap their continuation value (terminal=False), preserving the Base GAE convention.

Validation uses Uniform Student seed70158/70159 checkpoints at5000, each forked into frozen Base, epoch2 and epoch10 for200updates. All three arms share the exact Student source and initial real teaching history within each seed. Execution uses deterministic Teacher means, retaining the current deployment convention. Development evaluation occurs at+100 and+200; +200 continuous joint score is the primary endpoint, with EE errors, strict success and safety reported alongside. No formal-test results select settings. This is local calibration, not a claim of long-term generalization or significance.

The queue performs2200new Student updates total and stops. Its first100collection updates double as the real restore/feedback engineering check, within budget. CPU tests verify exact epoch2 equivalence, fixed targets, checkpoint roundtrip, KL stopping and transactional full Adam rollback using isolated synthetic data. Failures stop the queue and preserve evidence; there is no blind retry or automatic10k continuation.

YangLab directory: `/data/go2_piper_cl1_epoch10_local_20261112/`
Tmux session: `nsgs-cl1-epoch10-local`
Log: `controller.log`
Artifacts: `PROTOCOL.json`, `PREFLIGHT.json`, `CPU_TESTS.json`, `engineering/`, `fragments/on_policy_batch.pt`, `fragments/fixed_targets.json`, `teachers/update_before.pt`, `teachers/{base,epoch2,epoch10}.pt`, `UPDATE_COMPARISON.json`, `RESULTS.json`, `COSTS.json`.

The validated CL1 learner keeps its checkpoint schema compatible; `max_epochs` is an explicit per-update override, recorded as `epochs_requested` in the result. The stored common CONFIG is not silently relabeled as a different network or inference configuration.
