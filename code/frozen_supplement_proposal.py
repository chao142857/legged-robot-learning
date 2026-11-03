"""Prepare cluster-complete frozen evaluation selection; this file runs no simulator."""
import pathlib,json
import numpy as np
from common import ROOT,write,file_sha
G=ROOT/'artifacts/groups';maps={p.stem.removeprefix('mapping_'):np.load(p) for p in sorted(G.glob('mapping_*.npy'))}
rng=np.random.default_rng(20261107);used={n:set() for n in maps};anchors=[]
for i in rng.permutation(32768):
    if all(int(m[i]) not in used[n] for n,m in maps.items()):
        anchors.append(int(i))
        for n,m in maps.items():used[n].add(int(m[i]))
        if len(anchors)==64:break
assert len(anchors)==64
selected=set();groups={}
for n,m in maps.items():
    groups[n]=sorted(used[n]);assert len(groups[n])==64
    for g in groups[n]:selected.update(np.flatnonzero(m==g).tolist())
ids=sorted(selected);assert len(ids)<=1856
for n,m in maps.items():
    for g in groups[n]:assert set(np.flatnonzero(m==g)).issubset(selected)
write(ROOT/'artifacts/minimal_frozen_supplement_proposal.json',dict(executed=False,training_updates=0,selection_seed=20261107,anchors=anchors,complete_groups=groups,
    sequence_ids=ids,unique_training_sequences=len(ids),repetitions_per_sequence_minimum=2,episodes_per_app=len(ids)*2,total_apps=4,total_episode_trials=len(ids)*2*4,
    checkpoints=['/reference/runs/formal-wbc-uniform/checkpoints/update_10000.pt','/continue_reference/runs/continue-wbc-uniform/checkpoints/update_15000.pt'],
    modes=['deterministic mean','native stochastic alpha1'],environment_seed=51002042,action_seed=51003039,
    pool_sha256=file_sha(ROOT/'pools/wbc_train.npz'),evaluation_pool_used=False,all_mappings_use_same_sequence_reset_panel=True,
    selection_rule='Uniform random anchor permutation with collision rejection so all4 mappings have64 distinct complete8-member groups. No outcome or checkpoint metric input.',
    metrics='Unchanged5cm/80% strict J plus EE position RMSE, continuous joint-good-time fraction and native failure; paired reset seeds; no history updates.',
    precision_note='2 resets is an engineering/continuous-consistency minimum. If binary within-sequence uncertainty dominates, a preregistered additional2 resets on the identical panel is preferable to changing thresholds or grouping.',
    authorization_note='Proposal only. Existing authorization covers formal PPO only after GO; no large extra GPU evaluation is launched automatically.'))
print('MINIMAL_FROZEN_PROPOSAL '+str(len(ids))+' sequences, '+str(len(ids)*2*4)+' total trials; not executed',flush=True)
