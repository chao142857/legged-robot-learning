# Migration to nSGS

The 12 November 2026 migration preserves the existing repository and commit history. Main advances from `a1fed2cee74b330e3d11af2149c588bf0dd7f002`; no force push or replacement repository is used.

## Branches and original identities

| Original branch | Original HEAD | New home | Immutable archive tag |
| --- | --- | --- | --- |
| `docs/bilingual-readme-20261025` | `ab14b4e08d4c8c9a6e936c33e1f4e1216b9b2e75` | `main (documentation)` | [archive/20261112/docs/bilingual-readme-20261025](https://github.com/chao142857/nsgs/tree/archive/20261112/docs/bilingual-readme-20261025) |
| `docs/cleanmeta-progress-20261110` | `fe22c92de7fb348739aee9b52c09dcd34aff7aaa` | `main (documentation)` | [archive/20261112/docs/cleanmeta-progress-20261110](https://github.com/chao142857/nsgs/tree/archive/20261112/docs/cleanmeta-progress-20261110) |
| `fix/cl1-recovery-and-resume-20261110` | `84bb25534928d2d229dc647180827410368edb23` | `exp/cl1-base-recovery` | [archive/20261112/fix/cl1-recovery-and-resume-20261110](https://github.com/chao142857/nsgs/tree/archive/20261112/fix/cl1-recovery-and-resume-20261110) |
| `fix/teacher-adam-audit-20261109` | `c43b36ea4cb3dbdaa3d0f72730c349188a6d5130` | `main + exp/cleanmeta8` | [archive/20261112/fix/teacher-adam-audit-20261109](https://github.com/chao142857/nsgs/tree/archive/20261112/fix/teacher-adam-audit-20261109) |
| `main` | `a1fed2cee74b330e3d11af2149c588bf0dd7f002` | `main` | [archive/20261112/main](https://github.com/chao142857/nsgs/tree/archive/20261112/main) |
| `research/cleanmeta8-reproduction-20261109` | `b1b5cd5b092967678b9b5f315676da6f442bc9f4` | `exp/cleanmeta8` | [archive/20261112/research/cleanmeta8-reproduction-20261109](https://github.com/chao142857/nsgs/tree/archive/20261112/research/cleanmeta8-reproduction-20261109) |
| `research/nsgs-cl1-20261109` | `ab4532ea1af7ab7751136e95dd4187d386457ee5` | `exp/cl1-base-recovery` | [archive/20261112/research/nsgs-cl1-20261109](https://github.com/chao142857/nsgs/tree/archive/20261112/research/nsgs-cl1-20261109) |
| `research/nsgs-tr0-20261109` | `0cc1165929239e17f393d28080bc359cbe5538c9` | `archive only; TR0 note in OldMeta` | [archive/20261112/research/nsgs-tr0-20261109](https://github.com/chao142857/nsgs/tree/archive/20261112/research/nsgs-tr0-20261109) |
| `research/nsgs-v02-20261106` | `da583b33b10351a8e5c9f19e61b75ba929ea026d` | `exp/oldmeta` | [archive/20261112/research/nsgs-v02-20261106](https://github.com/chao142857/nsgs/tree/archive/20261112/research/nsgs-v02-20261106) |
| `research/nsgs-v02-matched-seed48-20261108` | `2ad082317e3511bec111fe16886b3577e316fa6c` | `exp/oldmeta` | [archive/20261112/research/nsgs-v02-matched-seed48-20261108](https://github.com/chao142857/nsgs/tree/archive/20261112/research/nsgs-v02-matched-seed48-20261108) |
| `research/stage2-uniform-wbc-20261031` | `52e0bfa959bd129930ce577c46c56d0cc1564831` | `archive only (unrelated WBC)` | [archive/20261112/research/stage2-uniform-wbc-20261031](https://github.com/chao142857/nsgs/tree/archive/20261112/research/stage2-uniform-wbc-20261031) |
| `research/stage3-dynamic-x-20261031` | `ded395e5e5dbc94321c682805baad8755a03b63f` | `archive only (unrelated WBC)` | [archive/20261112/research/stage3-dynamic-x-20261031](https://github.com/chao142857/nsgs/tree/archive/20261112/research/stage3-dynamic-x-20261031) |
| `research/stage3b-dynamic-x-safety-20261101` | `e67f5b000513f9f059afeac110179560a4d23af6` | `archive only (unrelated WBC)` | [archive/20261112/research/stage3b-dynamic-x-safety-20261101](https://github.com/chao142857/nsgs/tree/archive/20261112/research/stage3b-dynamic-x-safety-20261101) |

Tags retain their original target SHA and are never moved by this migration. New experimental branches have scoped trees and their corresponding historical source commit as parent. E2 local Git history is additionally retained at `archive/20261112/e2-local-source`. [New experimental heads](experimental-branches.json) record scoped branch commits. SGS/E1 and clustering are recovered from actual local execution files and manifests, not presented as previously existing remote snapshots.

The research line is linear: V0.2 → matched follow-up → TR0 → Adam audit → CleanMeta → CL1 → recovery repair. Main's documentation updates diverged from the original collection; they did not contain the full research implementation. The three WBC branches form their own chain. See [ancestry and commit differences](branch-audit.json).

Long-term branches: `main`, `exp/sgs-evaluation`, `exp/clustering`, `exp/oldmeta`, `exp/cleanmeta8`, `exp/cl1-base-recovery`.

Three original branches are temporarily retained because existing local worktrees bind them: `research/stage3b-dynamic-x-safety-20261101`, `research/nsgs-v02-matched-seed48-20261108`, and `fix/cl1-recovery-and-resume-20261110`. They are recovery/compatibility refs, not additional nSGS methods. Their worktrees were not detached, reset or edited. The other nine original branches were removed after all archive tags and scoped branches were verified remotely.

## Source layout

| Former location | New location |
| --- | --- |
| `projects/sgs/historical_48task_20261019` and local SGS/E1 execution sources | `experiments/sgs_evaluation/` |
| Local E2/E3 working sources | `experiments/clustering/` |
| `projects/sgs/nsgs_v02*`, TR0 | `experiments/oldmeta/` |
| `projects/sgs/teacher_adam_audit_20261109`, CleanMeta reproduction and actual runtime | `experiments/cleanmeta8/` |
| `projects/sgs/nsgs_cl1_20261109`, repair and remote completed runtimes | `experiments/cl1_base_recovery/` |
| Shared corrected CPU components | `src/nsgs/{sampling,teacher,student,evaluation}/` |
| Selected numerical reports and figures | `results/`, `assets/figures/` |

[Per-file mapping](source-map.json) retains original paths, Git revisions where available, original SHA256 and migrated SHA256. Unchanged execution files retain exact bytes; package import changes and documentation link repairs are identified separately. Identical files across runtime snapshots are intentionally preserved when required by a stage's original flat import contract. [Duplicate-content inventory](duplicate-files.json) makes this explicit. Unrelated Z1, generic WBC, PPO diagnostics and their publication tests leave main; their history and license texts remain recoverable.

## Backup and research preservation

A complete local Git bundle was created before restructuring, verified with `git bundle verify`, cloned into an independent bare repository and checked with `git fsck --full`. Its SHA256 is `61930745a2671df4349d3b732e456fc04c2d5eb8df3e2470eee11eafe5e3a8e6`. The bundle, original remote refs and local worktree inventory are delivered outside the repository. E2's separate source repository has its own bundle; untracked E2 audit/supplement files are recorded by path/hash and left untouched.

The three original local worktrees and the latest public-update checkout were clean when inspected. Research directories outside Git were read only; migrated source/evidence files retain hashes. Both research hosts were inspected via their existing SSH routes. YangLab's seed70159 job was active. No GPU training, historical result regeneration, checkpoint transfer, experiment-directory modification, or network configuration change was performed.

Raw data, checkpoints, trajectories, Student/Teacher state and logs remain in their original host directories. The bundles preserve committed Git objects, not external robot assets. E3 mapping arrays remain a separately documented external dependency.
