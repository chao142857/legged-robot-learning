# nSGS — clustering

This scoped experimental branch preserves the clustering stage. The source parent is `542b9e41668003822f0ea094aa57d0400d5020e9`; original paths and hashes are retained in the [main source map](https://github.com/chao142857/nsgs/blob/main/docs/source-map.json).

- [Experiment source](experiments/clustering/)
- [Research history and results](https://github.com/chao142857/nsgs/blob/main/docs/experiments.md)
- [Environment, external assets and CPU checks](https://github.com/chao142857/nsgs/blob/main/docs/reproduction.md)
- [Current corrected implementation](https://github.com/chao142857/nsgs/tree/main/src/nsgs)

Runtime snapshots keep their original import and external mount contracts. They do not launch training on checkout. Historical OldMeta optimizer behavior is retained as historical; CleanMeta and CL1 remain distinct methods. This branch is a migration of real sources, not a newly executed experiment.
