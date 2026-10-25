"""Convert a locally exported raw rollout into a diagnostic bundle, CPU only.

No inferred component value functions and no scalar critic splitting.
The exporter must supply truthful metadata and pre-reset snapshots.
"""
import argparse
import json
from pathlib import Path
import torch
from diagnose_go2_piper_gradients import complete_mc, component_gae, validate_bundle, sha256


def build(raw, checkpoint):
    m=dict(raw['metadata'])
    r=raw['component_rewards']
    valid=raw['valid'].bool()
    if r.shape[-1] != 3 or r.shape[:2] != valid.shape or not torch.isfinite(r).all():
        raise ValueError('Need finite weighted per-step rewards [T,N,3] and valid [T,N]')
    if not torch.allclose(r.sum(-1)[valid],raw['total_reward'][valid],rtol=1e-5,atol=1e-6):
        raise ValueError('Reward partition not exhaustive/additive, or dt applied twice')
    if m['estimator']=='complete_mc_finite_horizon':
        if m.get('time_limit_objective') != 'finite_horizon_terminal':
            raise ValueError('Must explicitly declare finite-horizon timeout semantics')
        a=complete_mc(r,raw['episode_end'].bool(),valid,float(m['gamma']))
    elif m['estimator']=='crossfit_component_gae':
        if not valid.all():
            raise ValueError('GAE path currently requires dense rollout without padding')
        a=component_gae(r,raw['component_values'],raw['next_component_values'],
            raw['terminated'].bool(),raw['episode_end'].bool(),float(m['gamma']),float(m['lambda']))
    else:
        raise ValueError('Unsupported estimator')
    b={k:raw[k][valid].detach().cpu().clone() for k in
       ['actor_obs','critic_obs','actions','old_logp','old_mu','old_std','task_id','base_level','ee_level']}
    for k in ('total_returns','old_values'):
        if k in raw:b[k]=raw[k][valid].detach().cpu().clone()
    b['component_advantages']=a[valid].cpu()
    b['metadata']=m
    validate_bundle(b,sha256(checkpoint))
    return b


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(1)
    raw=torch.load(a.raw,map_location='cpu',weights_only=True)
    b=build(raw,a.checkpoint)
    b['metadata']['raw_bundle_sha256']=sha256(a.raw)
    with open(a.out,'xb') as f:torch.save(b,f)
    print(json.dumps({'output':a.out,'samples':len(b['actions']),'estimator':b['metadata']['estimator'],'optimizer_steps':0}))

if __name__=='__main__':main()
