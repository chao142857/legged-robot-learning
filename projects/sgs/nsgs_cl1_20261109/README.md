# CL1 implementation and launch record

This is a new recurrent Teacher experiment, not a continuation of the CleanMeta8 MLP. Existing CleanMeta8 seed48/49 and legacy results stay in their original directories.

- Branch: `research/nsgs-cl1-20261109`.
- YangLab output: `/data/go2_piper_nsgs_cl1_20261109`.
- Shared 64-unit encoder, 64-unit GRU, 8-dimensional raw Gaussian macro-action, scalar value head. Fixed standard deviation 0.2; zero initial mean head; Teacher seed 20261109.
- Unchanged V0.2 FeatureEncoder and bounded-logit probability decoder, with 20% Uniform mixing. The Gaussian log-probability belongs to the raw preference, not to sequence draws or transformed logits.
- Source students seed43/44, native WBC 4096 environments and rollout24. Each starts from scratch and receives 10,000 PPO updates. Two five-macrostep trajectories are collected under one frozen Teacher, then updated jointly. Formal evaluation is never an input to the Teacher.
- Each student's GAE and memory are independent. Every epoch replays that student's prefix without gradients; BPTT covers the current fragment. Nonterminal 500-update boundaries bootstrap; 10k terminates. Full Teacher/Adam/RNG and prefix histories are saved.
- The outer learner uses PPO, with no V0.2 candidate/backtracking loop and no TR0 TV limit. Parameters are fixed in `PROTOCOL.json` and `cl1_core.CONFIG`.
- Single resident Student, process blocks of 500 updates, using the existing complete exposed-state restore. This incurs simulator startup overhead. Hidden PhysX contact caches are not serializable; no bitwise determinism claim is made. No deliberate boundary resets are introduced.
- Actual fixed runtime `teacher_v02.py` SHA256: `734721a9acb59651388b73b112fb0365dab8a593f00e107f4f79b715a3596f96`, byte-identical to the audited `teacher_fixed.py` and CleanMeta8 CPU acceptance record. The old source manifest's hash was stale; that mismatch stopped setup until verified against both sources.

Engineering acceptance uses eight new CPU tests plus twelve Adam regressions, then real physics smoke: each source executes two 100-update macrosteps, one outer update, followed by a third macrostep from a restored source checkpoint. The 500 smoke PPO updates and their development costs are separate; neither smoke Student nor smoke Teacher weights initialize the formal experiment. Engineering acceptance does not depend on success rate improvement.

After Base is preserved, recovery selection uses only past/current source development records: latest two J values minus the previous three <= -0.03, and EE position RMSE increase >= 0.003m. Starts must be <=9500 and separated by at least500 updates. Up to two events per source are selected chronologically; normal controls are nearest-stage eligible states, with preregistered random tie-breaking. The balanced budget is at most two batches of four 500-update fragments. If real events are insufficient, actual coverage and the reduced budget are reported; no artificial degradation or extra full training is created. Recovery continues the Base network and Adam, burns in each source prefix, and treats the end of each500-update recovery fragment as terminal.

The durable controller writes job launch/results, raw on-policy macro-transitions, full probability vectors, actual sequence draws/exposure, independent development/formal results, model checkpoints, Teacher states and COSTS.json. GPU job-wall sums include nested evaluation waits and are explicitly not disjoint device-time estimates. TR0, Meta24 and new three-arm10k tests are excluded.

At the initial code commit, CPU tests have passed and physical smoke is running. This document is not evidence that formal training has already started; consult STATUS.json and launch receipts.
