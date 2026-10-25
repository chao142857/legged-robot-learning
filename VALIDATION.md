# Validation Report

Publication audit date: 2026-10-25 (Asia/Shanghai). No GPU training, Isaac application, PhysX/Newton simulation, hardware deployment or model rendering was started for publication.

## Executed CPU checks

The complete optional-ML CPU harness passed. **75 unittest cases were discovered: 74 passed, 1 skipped.** The skipped case is the real B0 checkpoint/native parity test; no private checkpoint was supplied. Additional Z1 standing-sampler assertions and simulator-free launcher checks passed.

| Check group | Harness status | unittest cases | Skipped |
|---|---|---:|---:|
| publication | passed | assertions / entry check | 0 |
| sgs_lifecycle | passed | 12 | 0 |
| z1_standing_sampler | passed | assertions / entry check | 0 |
| task_metrics | passed | 18 | 0 |
| tracking_metrics | passed | 4 | 0 |
| b2_common | passed | 5 | 0 |
| b3_mapping | passed | 4 | 0 |
| gradient | passed | 17 | 1 |
| component_probe | passed | 5 | 0 |
| native_ppo_microscope | passed | 10 | 0 |
| z1_entry | passed | assertions / entry check | 0 |
| go2_entry | passed | assertions / entry check | 0 |

`task_metrics` includes offline analysis and mocked subprocess/preflight tests; no simulator is launched by those tests. B3 coverage here is the four mapping/observation tests only. Its model-dependent controller tests and B2 real IK-composition tests were not run because model-derived kinematic geometry is excluded.

## Actual test environment

- macOS CPU, Python 3.11.16, Torch 2.13.0, NumPy 2.4.6, TensorDict 0.8.3.
- RSL-RL 5.0.1 came from an existing local read-only source snapshot. All **29 Python source hashes** matched the previously recorded installed-package inventory. Native GAE/PPO/Adam synthetic replay tests use this real implementation, not a replacement PPO.
- The existing Torch environment was reused. Only TensorDict and its small missing dependencies were installed into a temporary test directory; no existing environment, simulator, driver, network or authentication configuration was changed.
- This CPU environment differs from historical robot training (Torch 2.7.0+cu128). Passing synthetic API/numerical checks does not demonstrate bitwise reproduction of historical CUDA rollouts.

## Static, provenance and presentation checks

All selected Python source files were parsed, all JSON was decoded, and all selected text files were checked for forbidden payload types, file size, private machine paths and known credential patterns. Original-to-publication hashes were verified. Selected local sources did not change during construction. A final rescan found an updated A+ status record; it still reported zero new completed updates and no started B stage.

The existing execution-host SSH path was used for a bounded read-only verification of nine common Recipe support files (74,805 bytes). All matched the local selected support byte for byte. No Git history, assets, checkpoints or rollouts were copied. The initial new Git object database and ref set were confirmed empty, with no alternates. The first commit/root and clean state are checked before the first push.

Local Markdown links were checked. The scientific SVG was regenerated from published numerical summaries and its rendered preview inspected. It contains plots only, with separate-protocol and single-training-seed labels; no team model is depicted.

## Remaining runtime limitations

- Full simulator imports, asset loading, GPU training, full evaluation rollouts and hardware behavior were not rerun.
- Z1 assets and the original CP5700/Adam/evidence are external requirements for historical continuation. Native command-schedule integration remains untested in this release environment.
- Go2/Recipe reproduction requires the exact external framework, assets, native runtime receipts, parameters and fresh source-bound P0/interface evidence. Aggregate tables cannot replace them.
- Model-dependent IK and real checkpoint parity are not validated by the asset-free suite. B1/B2/B3 formal admission stays false.
- The publication includes selected numerical summaries; excluded raw tensors and policies limit end-to-end independent recomputation. No stable SGS benefit or cross-training-seed reproducibility has been established.
