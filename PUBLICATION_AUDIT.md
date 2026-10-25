# Publication Audit

**Selected release: PASS for publication. No unresolved blocker remains among the selected files.** Unclear assets and operational artifacts are excluded individually and retain their original private copies. This audit does not grant rights to excluded material.

## Source isolation and ownership

The original `chao142857/z1-dog` was confirmed private. Its privacy status and branch SHA references are captured read-only for before/after comparison. Four original local Git source checkouts have baseline HEAD/status/source hashes recorded privately. No original branch, file, configuration, asset, checkpoint or result was intentionally rewritten, deleted or moved.

The publication directory is separate. `git init -b main` created an empty object database: zero loose objects, zero packs, no refs, no alternates. No `.git`, bundle, old branch, remote history, hook or worktree metadata was copied. The initial parentless commit is verified before the first push. No existing repository is overwritten or force-pushed.

The author's ownership of independent research implementations follows the user's explicit representation. Upstream frameworks, papers, APIs and copied/derived material are attributed separately. Exact source and publication hashes, changes and licenses are recorded in `provenance/files.json`; retained license texts and scope decisions are in `THIRD_PARTY.md` and `LICENSES/`.

## Included material

- Independent Z1 task/training configurations, curriculum, standing sampler, bounded continuation code and selected evaluation/contact/height measurements.
- Native Go2/Piper WBC observer/evaluation/analysis, the executed T1-v2 support, scratch-study evidence, Recipe v1 and A+ source and numerical/status evidence.
- Original SGS implementation and stronger lifecycle adapter, historical executed/current variants, common-protocol preparation and independent B1/B2/B3 modules.
- PPO frozen-gradient, probe, GAE/KL/clipping/Adam diagnostics, original tests and lightweight microscope result tables.
- Original English overview, evidence interpretation, independent-environment commands, validation, provenance and model-free scientific plots.

## Exclusions and reasons

| Material | Decision |
|---|---|
| Z1 URDF, meshes, collision/converted USD, CAD-derived geometric helpers | Excluded: team-owned model material; no migration authorization |
| Robot-model videos, screenshots, scenic renders | Excluded: team geometry or unverified media rights; no direct publication |
| Go2/Piper assets and model-derived kinematics JSON | Excluded: redistribution rights are not inferred from source-code licensing |
| Upstream pretrained policies, research checkpoints and actor exports | Excluded: independent weight/source rights not established for redistribution; preserve source-host copies and hashes |
| Rollout NPZ/tensors, full step logs and mixed archival payloads | Excluded: raw/private or large payloads, mixed content and minimal-transfer policy; selected numerical results retained |
| Credentials, internal hardware/deployment files, machine configs, SSH/install/graphics/recovery automation | Excluded: outside research release scope; no credential or private host route is published |
| Bundles, `.git`, partial archives and historical branch objects | Excluded: fresh history is mandatory; partial files cannot serve as evidence |
| Separate PiPER-X single-arm work, ROS sensor workspaces, presentation/literature material | Indexed in secondary discovery; not conflated with Go2 + Piper WBC or the four requested modules |

`provenance/exclusions.csv` provides public category counts. The private audit holds per-file paths/sizes/reasons for nonselected files and SHA256 for all 302 enumerated video/model/rollout files (1,013,232,641 logical bytes). These totals include duplicate snapshots and do not estimate unique storage. No such media/model payload is in the release. SHA inventory was computed in place; media was not moved or transferred.

## Claims audit

- Every robot result is labeled simulation; no hardware experiment is implied.
- The older CP5700/CP2300 demos are not combined into a single-policy acceptance claim. The newer Z1 strict-safety failures remain visible.
- Native WBC, fixed-task pilot, T1 scratch, and Recipe protocols remain distinct. Lower error across different domains is not claimed as an algorithmic comparison.
- SGS's historical joint-success results are negative/indeterminate. Sampler CPU correctness is not represented as a stable learning gain or invention of SGS.
- Recipe v1's formal T1 gate is false, startup safety regression is retained, and A+ has zero new completed updates in the inspected status. B1/B2/B3 are preparation only.
- Component probes and held-out surrogate changes are diagnostic; they are not true-return gains or proof of the historical failure mechanism.

## Verification and publication gate

The asset-free CPU suite completed 75 unittest cases (74 pass, 1 explicit private-checkpoint skip), additional Z1 sampler assertions and launcher checks. All candidate text/source files passed syntax, provenance, size and credential-pattern checks. Numerical plots were regenerated and visually inspected. Tests and residual runtime limitations are specified in `VALIDATION.md`.

Before push: review the exact staged file list and aggregate bytes; verify one new parentless root, no alternate/shallow history, no forbidden object/file; run `python tools/audit_publication.py --history`; verify the new remote is public and matches the local commit after push; recheck the old repository remains private and unchanged. These checks do not start training or alter the old repository.
