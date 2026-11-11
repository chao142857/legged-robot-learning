# nSGS-TR0: trust region on the actual mixed sequence distribution

This is an isolated, predeclared local experiment on **ubuntu-robot**, as explicitly requested by the user. It leaves historical E1–E3 and nSGS outputs and the active YangLab queue unchanged. The supplied execution sheet's old YangLab placement is superseded by that explicit Ubuntu instruction.

The preserved baseline was pushed on `research/nsgs-v02-matched-seed48-20261108` at `2ad082317e3511bec111fe16886b3577e316fa6c`. This independent worktree starts from that commit. No checkpoints, sequence pools, raw feedback, private training data or credentials belong in Git.

## Constraint and optimizer semantics

The inherited V0.2 guard is `||mu_new(g)-mu_old(g)||²/(2*sigma²) <= .05` at the current decision's global Student state. It is the equal-covariance Gaussian preference KL, symmetric in this case. It does not constrain the complete mixed sequence distribution or all reference states, and is unrelated to Student PPO desired KL.

`code/trust_region.py` constructs Original-step with the inherited `Teacher.learn_pair` gradient and KL rule, while isolating every Adam state load with a fresh deep copy. Sampling-TR uses that exact real gradient, clipping rule, old Adam state and direction, with the original LR schedule `1, 1/2, ..., 1/256`. Each candidate starts from the same old parameter/optimizer state and must pass the inherited preference bound and maximum full-distribution TV2%. Adam moments advance once only for the accepted candidate, or are restored on rejection; the next declared base LR remains0.01. Every original and TR candidate's complete float64 probabilities, KLs, ratios and fixed input identities are retained on the execution host.

An independent final real-state audit found that the historical CPU optimizer loader aliases snapshot tensors: the canonical Meta8 has logical updates8 but Adam step15, and the first intermediate new candidate advanced15→17 over two KL attempts. That intermediate is explicitly invalidated and archived, not used as the TR0 result. The historical Meta8/16 checkpoints and modules remain unchanged. TR0 preserves the exact inherited step15 asset, fixes snapshot isolation for **both** B/C, and reconstructs from the same immutable Meta8 and same new feedback (not continual replay). Regression tests force multiple KL backtracks and full rejection; the corrected real candidate must advance15→16. The gradient and first full Adam proposal must be bit-identical to the archived attempt. The earlier historical TV diagnostic describes the old executable behavior only.

The distribution is the actual `.2/N + .8*softmax(bounded_logits)` over all32,768 sequences. Every old/new comparison uses identical frozen command/feedback/missingness/global inputs and the inherited command-only normalization. The panel contains eight historical source snapshots, four actual validation starts, and the current new gradient decision (13 total). No formal-test feedback is used.

This is a backtracked bound on a sampler parameter update, not a TRPO solver or a monotonic-improvement theorem. It does not change exploration sigma, objective, gradient batching or native PPO/reward/action/observation configuration.

## Fresh direction and intervention gate

The fresh direction is fixed in advance: exact Meta8, seed43 CP1000, one new paired Uniform/q+/q− fork of100 PPO updates each. Historical feedback is allowed as frozen input evidence, not replayed as new on-policy gradient data. Existing historical Meta8→9 diagnostics are separately labeled read-only.

If the KL-protected Original-step already has maximum TV≤2%, B/C are bit-identical; the12k experiment is not launched. If a distinct three-policy comparison cannot be formed (for example rejection leaves C equal to Old), the corresponding operation stops with its reason recorded. No direction, source, sigma, radius or gradient is selected to force a difference.

## Predeclared local validation if the constraint binds

Four distinct Student seeds44/46/47/48 use locally available full checkpoints at3k/3k/8k/3k. These heterogeneous stages and parent sampling histories are reported; pairing applies within each fixed state. None contributes to the new seed43 gradient. Source selection precedes fresh outcomes and follows seed diversity/local availability, not performance.

For each source, three frozen Teachers × two paired repeats ×500 Student PPO updates =12,000 updates,1,179,648,000 new training transitions. The Student actor/critic/normalization/optimizer, physical and observation-manager state, genuine feedback and original RNG are fully restored. Repeat0 retains the source global RNG; repeat1 uses a declared new paired future RNG stream. All arms within a repeat have identical Student and sampler initial states. PhysX hidden caches are not serializable; repeated baseline and rollout-signature differences are retained.

Teacher parameters and Adam stay frozen throughout each500-update branch. The complete distribution refreshes every100 Student updates from current real state. Development is observed at+100/+300/+500; +500 continuous joint-score gain against the fixed source baseline is primary. Strict5cm/80%, four physical RMSEs, native failure and safety are secondary. There is no best-checkpoint endpoint or new10k Student run. Development evaluation and direction/smoke costs are counted separately.

The inherited native runner is copied to a new `train_tr0.py` via exact lifecycle replacements. Tests compare the ASTs of the PPO update, full-checkpoint Runner and feature/probability functions. The original runner stays untouched.

## Execution and evidence

`controller.py` first requires a verified pushed Git receipt, CPU/source acceptance and idle GPU. It runs a2-update GPU smoke, gathers the new300-update direction, records fixed source baselines and the13-input panel, audits real sampling/restore feedback, and applies the gate. Only then may it automatically run24 validation branches. Every completed branch saves full checkpoints and logs; failures stop the queue and preserve evidence.

Full execution evidence remains under `/home/alex/dog/go2_piper_nsgs_tr0_20261109/`, with large new artifacts on `/media/alex/Sharedata/go2_piper_nsgs_tr0_20261109/`. `CANDIDATE_RESULT.json` records the gate; `RESULTS.json`, `REPORT.md`, and `pair_audits/` distinguish completed validation from a legitimate non-intervention stop. A historical diagnostic or a queued job is never labeled completed PPO training.
