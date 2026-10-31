"""Frozen mean-function comparison on identical actual P1 states; CPU only.

Leg-function drift and arm-induced closed-loop coupling are separate hypotheses.
This audit can measure the former and cannot establish the latter's cause.
"""
import argparse,hashlib,pathlib
import numpy as np
import torch
from cpu_policy_sensitivity import mean
from gpu_budget import read,write

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main(root,labels):
    p=pathlib.Path(root);torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    r=pathlib.Path(read(p/'PROBE_INDEX.json')['P1-parent-X-s0.0'])
    trajectory=r/'evaluation/trajectories.npz'
    with np.load(trajectory,allow_pickle=False) as data:
        # A compressed NPZ lookup decompresses the entire member. Load once,
        # then sample the in-memory arrays, never once per timestep/env.
        valid=data['valid'];observations=data['diag_actor_obs'];recorded=data['diag_policy_mean']
        pairs=[(t,e) for t in range(50,min(1000,len(valid)),25) for e in range(valid.shape[1]) if valid[t,e]]
        times,envs=np.array(pairs).T
        x=torch.tensor(observations[times,envs].copy())
        recorded=torch.tensor(recorded[times,envs].copy())
    del observations,valid
    parent=pathlib.Path(read(p/'CHECKPOINTS.json')['parent']['checkpoint'])
    ps=torch.load(parent,map_location='cpu',weights_only=False)['actor_state_dict']
    with torch.no_grad():pm,_=mean(ps,x)
    assert float((pm-recorded).abs().max())<1e-4
    paths={'parent':parent}
    for label in labels:
        run=pathlib.Path(read(p/(label+'_RUN.json'))['run']);ck=read(run/'ppo/result.json')['checkpoint']
        path=run/'ppo/checkpoints'/pathlib.Path(ck['path']).name
        assert sha(path)==ck['sha256'];paths[label]=path
    rows=[]
    with torch.no_grad():
        for label,path in paths.items():
            state=torch.load(path,map_location='cpu',weights_only=False)['actor_state_dict']
            m,res=mean(state,x);groups={}
            masks={'all':torch.ones(len(x),dtype=torch.bool),'forward':x[:,186]>.05,
                   'backward':x[:,186]<-.05,'left':x[:,187]>.05,'right':x[:,187]<-.05}
            for tag,mask in masks.items():
                if not mask.any():continue
                diff=m[mask]-pm[mask]
                groups[tag]=dict(samples=int(mask.sum()),raw_mean_leg_RMS=float(diff[:,:12].square().mean().sqrt()),
                    raw_mean_arm_RMS=float(diff[:,12:].square().mean().sqrt()),
                    nominal_target_delta_RMS_rad_per_action=(.25*diff.square().mean(0).sqrt()).tolist(),
                    nominal_target_signed_delta_mean_rad_per_action=(.25*diff.mean(0)).tolist(),
                    head_raw_action_RMS_per_action=res[mask].square().mean(0).sqrt().tolist())
            rows.append(dict(label=label,checkpoint_sha256=sha(path),groups=groups))
    assert not torch.cuda.is_initialized()
    out=dict(CPU_only=True,parent_matches_actual_PhysX_recorded_mean=True,same_actual_state_history_samples=len(x),
        source_physical_trajectory_sha256=sha(trajectory),source_sha256=sha(__file__),rows=rows,
        scope='Frozen functions on identical real P1 physical states. Nominal target deltas are .25*mean differences before limits/ramp/DelayedPD, NOT closed-loop response or success. Leg drift is identified in the learned backbone; physical arm-induced coupling requires a separate intervention.',
        no_unique_physical_cause_claim=True)
    write(p/'CPU_LEG_FUNCTION_DRIFT.json',out)
    for row in rows:
        v=row['groups'].get('backward',row['groups']['all'])
        print(row['label'],'backward raw legRMS',v['raw_mean_leg_RMS'],'armRMS',v['raw_mean_arm_RMS'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--labels',nargs='+',required=True)
    a=p.parse_args();main(a.root,a.labels)
