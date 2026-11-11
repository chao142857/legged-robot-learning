# Experiment history

This history follows the executed methods. Results from different Teachers, training seeds and evaluation domains are kept separate. The [source map](source-map.json) records original paths, revisions and file hashes; the [migration table](migration.md) identifies archived Git heads.

## SGS evaluation — October to 3 November

**Question:** Can success-guided task exposure improve control, and does the feedback describe the intended tracking task?

The [48-task pilot](../experiments/sgs_evaluation/pilot48) used a different fixed task domain: both methods fell from 40/48 joint successes at initialization to 0/48 after 32, 64 and 128 updates. This negative result is not a native 1,024-sequence benchmark result.

The [native SGS implementation](../experiments/sgs_evaluation/sgs/code) and [E1 implementation](../experiments/sgs_evaluation/e1/code) preserve sampling ownership, independent RNG, native terminal feedback and the strict joint protocol. E1 changed **training feedback** through three stages: 10 cm/50% (updates 1–4000), 7.5 cm/60% (4001–8000), and 5 cm/80% (8001–10000). It retained a shared window of real episodes with all three labels. Formal evaluation remained the fixed strict 5 cm/80% protocol.

At seed42 CP10000, WBC Uniform / SGS / SGS-E1 achieved **36.52% / 35.45% / 33.40%**; Flat achieved **31.84% / 22.07% / 29.79%**. These are single-training-seed observations, not evidence of a general SGS benefit. [Six completed curves and counts](../results/sgs_evaluation/e1_curves.json). Source revisions are the original local manifests under the SGS/E1 directories; recovery of those files does not invent a historical Git commit.

## Clustering — 4–6 November

**Question:** Can shared feedback over structured command groups overcome sparse sequence-level success feedback?

[E2](../experiments/clustering/e2) used 32,768 training sequences partitioned into 4,096 equal groups of eight, comparing command-based and random groupings. The recorded offline decision was negative: **zero formal PPO updates**, and the proposed large frozen-evaluation supplement was not executed. The [completion record](../results/clustering/e2/E2_completion_verified.json) preserves that distinction. Original local Git revision: `542b9e41668003822f0ea094aa57d0400d5020e9` (core `bc28815584adfe05c680908557a6f810ac4b351b`).

[E3](../experiments/clustering/e3/code) tested random grouping (A), fixed balanced command grouping (B), a fixed manipulation-focused grouping (C), and scheduled manipulation weighting (D). B/C/D shared a 2k prefix; D changed maps every 500 updates thereafter. The journal reconstructs group history after remapping without changing the RNG cursor. All 12 non-resource-fork source files in the original E3 manifest matched their archived hashes during migration.

Seed42 10k strict success was **A 34.47%, B 42.09%, C 30.18%, D 30.76%**. The larger fixed-B versus Uniform replication across seeds42/43/44 averaged **34.60% versus 36.17%**, so the initial B result did not establish a replicated advantage. [Original four-arm evidence](../results/clustering/e3/source_snapshot_E3_complete_20261105.json) · [Three-seed replication](../results/clustering/e3/wbc_strict_success_seed42_43_44_complete_20261106.json).

E3's original mapping assets were found and hashed on YangLab; they remain external. See [asset requirements](reproduction.md#assets-and-recoverability). No maps or historical results were regenerated.

## OldMeta — 6–9 November

**Question:** Can a neural Teacher use Student state and counterfactual PPO continuations to improve future task selection?

[V0.2](../experiments/oldmeta/v02) introduced low-dimensional state-conditioned preferences and matched Uniform/positive/negative 100-update branches. [Follow-up transfers and matched-source continuations](../experiments/oldmeta/followup) retain Meta8/Meta16 lineages and individual seed results, at original revisions `da583b33b10351a8e5c9f19e61b75ba929ea026d` and `2ad082317e3511bec111fe16886b3577e316fa6c`.

The historical optimizer loader aliased Adam snapshot tensors. Multiple backtracking attempts could advance moments more than once; Meta8 recorded eight logical updates but Adam step15. OldMeta8/16 results therefore remain historical, uncorrected results. They are not relabeled CleanMeta. The [audit and regression source](../experiments/cleanmeta8/adam_audit) retain the original finding and checks (`c43b36ea4cb3dbdaa3d0f72730c349188a6d5130`; fix introduced at `58a1e1a`).

[TR0](../experiments/oldmeta/tr0) tested an additional full-distribution TV limit. With corrected independent state loads, the recorded Original-step and TV-constrained candidate did not form a distinct intervention; the planned 12k comparison was not launched. Original revision `0cc1165929239e17f393d28080bc359cbe5538c9` is archived rather than retained as a dedicated long-term branch.

## CleanMeta8 — 9–11 November

**Question:** Does the corrected counterfactual Teacher transfer to new Student training runs?

CleanMeta8 restarted from the original Fresh identity with the Adam fix. Source Student seeds43/44 at updates1000/3000/6000/9000 supply eight matched points; each point runs Uniform/positive/negative branches for 100 updates, totaling 2,400 meta-training Student updates. The gradient, learning rate, exploration and KL rules were not retuned for the fix. [Executed protocol](../configs/cleanmeta8.json) · [Corrected execution runtime](../experiments/cleanmeta8/runtime/code) · [Trace evidence](../results/cleanmeta8/runtime_binding/CLEANMETA8_ACCEPTANCE_SOURCE.json).

The initial public reproduction revision is `b1b5cd5b092967678b9b5f315676da6f442bc9f4`. Later source-bound runtime and completed transfer records were recovered from the research hosts and local reports, not inferred from that launch snapshot.

Four Student seeds42/47/48/49 share **one** trained Teacher. Equal-weight CP10000 means are Uniform **31.88%**, Fresh **33.30%**, CleanMeta8 **33.50%** (displayed difference +1.62 percentage points; unrounded +1.611328125). The [curves](../results/cleanmeta8/curve_data.json), [supporting records](../results/cleanmeta8/evidence) and [original plotting source](../results/cleanmeta8/plot_curves.py) retain seed variation. Seeds were run on the recorded RTX3060/RTX5060 environments; seed47 CleanMeta resumed at update400. These are not four independently meta-trained Teachers.

## CL1 Base / Recovery — 9 November onward

**Question:** Can a recurrent Teacher learn across Student trajectories and adapt using degraded and normal source contexts?

CL1 uses a GRU Teacher and outer PPO. It has its own recurrent state, action provenance and optimizer; it does not continue CleanMeta8 weights. Original Base revision: `ab4532ea1af7ab7751136e95dd4187d386457ee5`; repair revision: `84bb25534928d2d229dc647180827410368edb23`.

Base source Students43/44 completed10k each, achieving **29.20% / 30.08%**. [Source endpoint evidence](../results/cl1/base_v1_runtime/evaluations). These Students participated in Teacher learning and are not held-out transfer results.

Recovery V1 actually collected four 500-update fragments (2k, not the planned4k), then performed one outer update. Collection used frozen Base; its before/after fragment gains are not measured effects of the updated Recovery Teacher. V1's normal-context selector admitted a one-indicator decline. V2 preserves the original degradation thresholds (ΔJ≤−0.03 and ΔEE≥0.003m), requires normal contexts to satisfy both ΔJ>−0.03 and ΔEE<0.003m, and excludes ambiguous cases. It replaces seed44's6800 normal start with7800, restarts from Base weights/Adam, and actually recollects four500-update fragments. [V2 completion](../results/cl1/recovery_v2_runtime/REPAIR_RECOVERY_ACCEPTANCE.json).

The repair pairs each Student checkpoint with its immutable Teacher memory snapshot, validates seed/arm/feedback boundary/hash, and preserves an already selected pending action. Frozen mean actions have null log-probability and cannot be reused as on-policy Teacher data. Uniform executes zero preference and uniform probabilities. [Repair source](../experiments/cl1_base_recovery/transfer70158/code) · [Historical repair report](../experiments/cl1_base_recovery/repair_v2/REPORT.md).

On 12 November, the original remote endpoint files confirmed **seed70158, all three arms at10k**:

| Arm | Strict joint success |
| --- | ---: |
| Uniform | 34.57% |
| Frozen Base | 31.05% |
| Frozen Recovery V2 | 35.55% |

[Endpoint files with checkpoint/pool/protocol hashes](../results/cl1/transfer70158/evaluations) · [Completed learning curves](../results/cl1/seed70158_complete_curves.json) · [Execution costs](../results/cl1/transfer70158/COSTS_FINAL.json).

This is one held-out Student, with no equal-budget ordinary-continuation control for Recovery's extra2k experience. It does not establish a recovery-specific causal improvement. The second held-out Student70159 was actively running on YangLab during collection; only its Uniform10k endpoint had completed (36.82%). Its three-arm comparison remains in progress. [Dated status and available evidence](../results/cl1/transfer70159). No running directory or checkpoint was modified.
