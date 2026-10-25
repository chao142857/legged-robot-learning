# Third-Party Sources and Licensing

The repository license applies to original research contributions. Retained source notices and the texts below apply to upstream material and derivatives. No endorsement by upstream authors or the Z1 model provider is implied.

| Source | Version checked | License and treatment |
|---|---|---|
| [IIT DLS Lab basic-locomotion-isaaclab](https://github.com/iit-DLSLab/basic-locomotion-isaaclab/tree/8bda7c7df8b223acf649e0b212635a92c8528900) | `8bda7c7df8b223acf649e0b212635a92c8528900` | BSD-3-Clause; external framework, inherited task/configuration concepts acknowledged; [text](LICENSES/IIT-DLS-BSD-3-Clause.txt) retained |
| [LeggedManip_Lab](https://github.com/zzzJie-Robot/LeggedManip_Lab/tree/4e12109d1eccb1d9da2e4a35f189fabea3f13677) | `4e12109d1eccb1d9da2e4a35f189fabea3f13677` | Apache-2.0; external framework, WBC/task adapters attributed; [original license/copyright text](LICENSES/LeggedManip_Lab-Apache-2.0.txt) retained; fixed tree has no root NOTICE |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8) | Separate Z1 and Go2 pins in SOURCE_MAP | BSD-3-Clause; external simulator framework; retained native-action test fixture is explicitly third-party, with original header and [license](LICENSES/IsaacLab-BSD-3-Clause.txt) |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl/tree/v5.0.1) | Go2/gradient 5.0.1; Z1 5.4.1 | BSD-3-Clause; external algorithms/models, not represented as original PPO; [license](LICENSES/RSL-RL-BSD-3-Clause.txt) retained |
| [SGS paper](https://arxiv.org/abs/2610.12465v1) | v1, §3.2 and Appendix B.2 | Method attribution to Zhang et al.; local Python implementation and lifecycle adaptations. No upstream implementation, paper figures or paper prose copied |

The `vendor/sampler.py` name denotes a frozen **local** SGS implementation reused across the author's snapshots; it is not a claim that the underlying method was invented here. Its SHA256 is tested against the original local implementation. The 0.5 unvisited prior is a local policy choice, distinct from the cited formula.

`provenance/files.json` identifies original source paths, original and publication hashes, and portability changes. Paths to execution machines were replaced by explicit environment inputs; source-binding keys were adapted to this new layout. These changes are not new algorithmic results.

PyTorch, NumPy, TensorDict, SciPy, Matplotlib, PyYAML, simulator packages, and other package dependencies are obtained separately under their own terms. Their binaries are not redistributed. NVIDIA simulator access/terms are not automatically accepted by the public launchers.

Robot descriptions, meshes, USD layers, model-derived kinematics JSON, pretrained policies and checkpoints are excluded. A source-code license was not treated as blanket authorization for those artifacts. Team Z1 assets, CAD-derived geometric helpers and model visualizations remain outside this release.
