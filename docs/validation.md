# Migration validation — 12 November 2026

- 70 CPU tests passed: shared corrected Adam/CL1/checkpoint/scoring tests (31), E2 (22), E3 (8), and CL1 repair/frozen RPC (9). Original historical Adam aliasing is reproduced by the regression oracle; the corrected implementation passes independent analytic checks.
- Shared CPU modules import and the evaluation protocol CLI/help run without Isaac. Python syntax is checked across all migrated source and tests.
- `tools/verify_repository.py` checks per-file migrated hashes, all Markdown file targets, CleanMeta8 source hashes and seed curves, displayed means, and the original CL1 seed70158 three-arm endpoint identities.
- The four-seed PNG was copied unchanged from the personal website checkout and matches the local research plot byte-for-byte. No historical results or plots were regenerated.
- Original remote heads were recorded; the pre-migration Git bundle passed verification, independent clone and full object checking. Separate E2 Git history and untracked-file inventory were preserved.

No Isaac/GPU training, native simulator import, real-checkpoint replay or hidden PhysX state equivalence test was run. Those checks require the original Linux simulation stack and external assets. CPU tests use synthetic state; remote result files are observed completed experiments, not newly rerun experiments.

The first CPU run exposed one stale test import after packaging; it was corrected and the complete suite rerun. No failing test was deleted or weakened.
