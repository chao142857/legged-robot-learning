# Experimental Results and Evidence Boundaries

All entries describe simulation or CPU diagnostics. No hardware result is included. Dates identify source snapshots, not new experiments performed during publication.

## Z1 locomotion

The October 10 CP5700 demonstration recorded 32/32 eight-riser, 10 cm upstairs traversals. Other scene demonstrations used CP2300, including downstairs trials with calf contact; these cannot be presented as one checkpoint passing every scene. Team-model videos remain private.

The more recent October 16 study resumed the original CP5700 independently in arms A and B1 for 100 PPO updates each: 9,830,400 transitions per arm. The height target changed from 0.373 to 0.384 m in B1. Here **B1 is a Z1 height-reward arm**, unrelated to the Go2 factorial method with the same short name.

| Downstairs metric | Original | A | B1 |
|---|---:|---:|---:|
| Trials | 48 | 48 | 48 |
| Strictly safe trials | 0 | 0 | 0 |
| Mean calf-contact fraction | 8.78% | 10.17% | 6.88% |
| Mean calf-contact events/trial | 11.92 | 14.38 | 9.35 |

Each policy completed the 48 reported traversals, but all strict failures included calf contact. The analysis associates B1 with fewer later contacts, while preserving the first-edge problem and increased early foot loading. It does not isolate a universal geometric mechanism or prove safe locomotion. One training seed, three evaluation seeds with 16 trials each, 50 Hz post-solver measurements, and protocol-level pairing limit inference.

Published measurements: [height-reward summary](projects/z1_locomotion/results/height_reward_ab_summary.json), [contact summary](projects/z1_locomotion/results/downstairs_contact_summary.json). Geometry-derived fields and raw model-dependent trajectories were omitted. Full original source hashes are retained in provenance.

## Go2 + Piper native WBC

The native baseline uses LeggedManip_Lab `4e12109d1eccb1d9da2e4a35f189fabea3f13677`, random actor/critic initialization, seed 21, 4096 environments, rollout 24, and 3000 updates: **294,912,000 transitions**. Evaluations use 256 environments × three seeds × two action modes × four checkpoints: 24 instances and 6144 first episodes.

| CP | Mode | XY RMSE (m/s) | yaw RMSE (rad/s) | EE position RMSE (m) | 20 s survival |
|---:|---|---:|---:|---:|---:|
| 0 | Deterministic | 0.64056 | 0.55028 | 0.49440 | 766/768 |
| 1000 | Deterministic | 0.14576 | 0.14060 | 0.11100 | 762/768 |
| 2000 | Deterministic | 0.12630 | 0.13020 | 0.08743 | 764/768 |
| 3000 | Deterministic | 0.11888 | 0.12945 | 0.08066 | 768/768 |
| 3000 | Stochastic | 0.14613 | 0.21624 | 0.08674 | 762/768 |

Errors pool all valid first-episode samples after one second, including later command-change transients. Failed episodes remain in episode denominators; pooled errors weight valid sample duration. Native EE position uses link0 XY/world Z, and reward/command orientation conventions differ. The additional precision diagnostic gate passed **0/768** CP3000 episodes in either action mode, so survival does not imply precision/safety acceptance.

Evidence: [pooled evaluation](projects/go2_piper_wbc/native_20261023/results/evaluation_pooled.json), [per-seed/group table](projects/go2_piper_wbc/native_20261023/results/evaluation_by_seed_group.csv), [diagnostic gates](projects/go2_piper_wbc/native_20261023/results/report_tables.json), [training summary](projects/go2_piper_wbc/native_20261023/results/training_summary.json).

## T1 scratch v2 and Recipe v1

The 5000-update T1 scratch v2 is a separate fixed low-speed/dynamic-EE study. Its earlier v1 observer consumed observation-noise RNG and stopped after 148 updates; those transitions are excluded from valid v2 training. The valid study preserved the native rewards and trained from scratch, but did not establish a healthy T1 baseline. Its final [frozen evaluation table](projects/go2_piper_wbc/scratch_t1_20261025/results/frozen_evaluation.csv) and [verification summary](projects/go2_piper_wbc/scratch_t1_20261025/results/verification_summary.json) are retained as negative evidence. It is not an SGS run.

Recipe v1 adds documented control/reward/task changes, including soft-limit target constraints, low-height penalties and admitted fixed EE anchors. A single seed completed 1200 updates with 4096 ×24 samples/update: **117,964,800 transitions**. Forty-two evaluation instances contain 2688 first episodes; three evaluation seeds are not three training replications.

At CP1200, formal T1 completed 192/192 episodes with mean EE position RMSE about 2.42 cm, but the original full-episode strict safety gate passed only **15/192**. Startup Piper joint-speed violations remain counted. Right-turn sign consistency stayed below the 90% gate; dynamic X response gain was approximately 0.017. The all-seed, consecutive-node A→B gate is **false**. Fixed-EE learning, survival and aggregate RMSE do not establish full dynamic tracking.

Evidence: [per-instance results](projects/go2_piper_wbc/recipe_v1_20261025/results/physical_results.json), [CSV](projects/go2_piper_wbc/recipe_v1_20261025/results/physical_results.csv), [gate decision](projects/go2_piper_wbc/recipe_v1_20261025/results/gate_decision.json), [safety attribution](projects/go2_piper_wbc/recipe_v1_20261025/results/safety_joint_attribution.json), [optimization summary](projects/go2_piper_wbc/recipe_v1_20261025/results/optimization_summary.json).

Recipe A+ restores this study's own CP1200 and changes command sampling. The inspected source reports successful recovery checks but a simulator-initialization block: **zero new completed updates**. Planned CP1400/1600/1800 are not achieved results. B1/B2/B3 likewise remain preparatory candidates with no new training evidence.

## SGS: negative/indeterminate pilot

The historical fixed-48-task experiment uses a common mature actor plus a newly initialized critic, not scratch training. Uniform and SGS share the task pool, success protocol, initialization and 128-update budget. Each has 393,216 transitions. Joint success falls from 40/48 to 0/48 at all later evaluation points. Training feedback contains zero joint successes (Uniform 0/642 completed episodes, SGS 0/641), causing SGS's empirical distribution to become uniform when every task's rate is zero.

This pilot offers no joint-success improvement and lacks a discriminating positive-success signal. It does not prove that SGS is ineffective in other WBC settings. The newer shared lifecycle implementation has stronger CPU contracts; those engineering checks do not change the historical outcome.

Evidence: [fixed learning curve](projects/sgs/historical_48task_20261019/results/fixed_learning_curve.json), [detailed results](projects/sgs/historical_48task_20261019/results/results.json).

## PPO update microscope

The microscope restores the same frozen CP100 actor, critic and Adam state for each candidate and performs one native minibatch update in memory. Five windows × three seed-role folds × four minibatches × two LR factors produce 120 unique candidates. Two numerical-reporting versions repeated the same candidates, totaling 240 isolated steps; this is not a training trajectory or an independent replication.

In the main window, reducing LR to 0.1× preserves direction and reduces held-out clipped-surrogate deviation. The full step still improves the average unclipped surrogate; clipping penalty explains the clipped-loss deterioration. Component-probe advantages differ substantially from native total GAE. Neither result proves a root cause of historical robot-training failure, an error in Adam, or a benefit from gradient surgery.

Evidence: [summary](projects/ppo_gradient_lab/results/microscope/SUMMARY.json), [native GAE calibration](projects/ppo_gradient_lab/results/microscope/CALIBRATION.json), [clip decomposition](projects/ppo_gradient_lab/results/microscope/CLIP_DECOMPOSITION.json), [paired table](projects/ppo_gradient_lab/results/microscope/PAIRED_TABLE.md). Raw direction tensors and original rollouts remain private; published summaries do not replace them.
