"""Independent CPU checks on saved trajectories, checkpoint indices, and hashes."""
from publication_paths import research_path
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
ROOT=Path(research_path('go2_piper_upstream_wbc_20261023'))
WT=Path(research_path('LeggedManip_Lab_upstream_wbc_20261023'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def close(a,b,label):
    if not np.allclose(a,b,atol=2e-5,rtol=2e-5):
        raise AssertionError(f'{label}: mismatch, max error {np.max(np.abs(np.asarray(a)-np.asarray(b)))}')


paths=[ROOT/'eval_smoke16'] if args.smoke else sorted((ROOT/'evaluation').glob('cp_*/*/seed_*'))
if not args.smoke: assert len(paths)==24, f'Expected 24 evaluations including CP0, found {len(paths)}'
checks=[]
for path in paths:
    summary=json.loads((path/'summary.json').read_text())
    protocol=json.loads((path/'protocol.json').read_text())
    initial=json.loads((path/'initial_runtime.json').read_text())
    assert protocol['task']=='GO2-PIPER-WBC' and protocol['observation_noise']
    assert initial['episode_length_s']==20 and initial['action_dim']==18
    assert not initial['forbidden_attributes']
    assert all(str(WT) in p for p in initial['imported_modules'].values())
    assert sha(Path(protocol['checkpoint']))==protocol['checkpoint_sha256']
    with (path/'episodes.csv').open() as f: episodes=list(csv.DictReader(f))
    with np.load(path/'trajectories.npz') as z:
        valid=z['valid'];t,n=valid.shape
        assert n==protocol['num_envs']==len(episodes)
        assert valid[0].all() and not np.any(valid[1:]&~valid[:-1])
        assert valid.sum(0).min()>0 and valid.sum(0).max()<=1000
        close(z['age'][valid],np.broadcast_to(np.arange(1,t+1)[:,None],(t,n))[valid],'episode age')
        diff=z['base_actual']-z['base_cmd']
        xy=np.sum(diff[:,:,:2]**2,axis=-1)
        yaw=diff[:,:,2]**2
        ee=np.sqrt(np.sum((z['ee_actual_mixed']-z['ee_cmd'][:,:,:3])**2,axis=-1))
        close(xy[valid],z['xy_error_sq'][valid],'XY instantaneous squared error')
        close(yaw[valid],z['yaw_error_sq'][valid],'yaw instantaneous squared error')
        close(ee[valid],z['ee_position_error'][valid],'EE position norm')
        q1=z['ee_actual_quat_world'];q2=z['ee_target_quat_world']
        dots=np.abs(np.sum(q1*q2,axis=-1)/(np.linalg.norm(q1,axis=-1)*np.linalg.norm(q2,axis=-1)))
        angles=2*np.arccos(np.clip(dots,0,1))
        # arccos near zero loses float32 precision, use a 2 milliradian bound there.
        assert np.max(np.abs(angles[valid]-z['ee_orientation_error'][valid]))<.002
        close(z['reward'][valid],.02*z['reward_terms'].sum(-1)[valid],'sum of weighted reward terms')
        if 'all_body_contact_forces' in z:
            names=summary['native_contact_body_names'];nonfeet=[i for i,name in enumerate(names) if not name.endswith('_foot')]
            close(z['all_body_contact_forces'][:,:,nonfeet].max(-1)[valid],z['nonfoot_max_force'][valid],'nonfoot body mapping')
        for window,mask in [('all',valid),('after_1s',valid&(z['age']>50))]:
            group=summary['groups'][window]['all']
            assert group['samples']==int(mask.sum())
            if not mask.any(): continue
            close(np.sqrt(xy[mask].mean()),group['xy_rmse'],'pooled XY RMSE')
            close(np.sqrt(yaw[mask].mean()),group['yaw_rmse'],'pooled yaw RMSE')
            close(np.sqrt((ee[mask]**2).mean()),group['ee_position_rmse'],'pooled EE RMSE')
        for i,row in enumerate(episodes):
            mask=valid[:,i];length=int(mask.sum())
            assert int(row['steps'])==length
            close(np.sqrt(xy[:,i][mask].mean()),float(row['xy_rmse']),'episode XY RMSE')
            close(np.sqrt(yaw[:,i][mask].mean()),float(row['yaw_rmse']),'episode yaw RMSE')
            close(np.sqrt((ee[:,i][mask]**2).mean()),float(row['ee_position_rmse']),'episode EE RMSE')
            assert (row['nonfoot_contact']=='True')==bool((z['nonfoot_max_force'][:,i][mask]>1).any())
            failures=z['termination_base_contact'][:,i]|z['termination_bad_orientation'][:,i]
            assert (row['native_failure']=='True')==bool(failures[mask].any())
            if length<1000: assert failures[length-1]
            assert not np.any((failures|z['termination_time_out'][:,i])[:length-1])
            close(z['reward'][:,i][mask].sum(),float(row['return_sum']),'episode return')
        # Native targets are piecewise constant within each episode, no continuous task trajectory override.
        base_changes=((np.abs(np.diff(z['base_cmd'],axis=0)).max(-1)>1e-6)&valid[1:]&valid[:-1]).sum(0)
        ee_changes=((np.abs(np.diff(z['ee_cmd'],axis=0)).max(-1)>1e-6)&valid[1:]&valid[:-1]).sum(0)
        assert base_changes.max()<=2 and ee_changes.max()<=3
        checks.append(dict(path=str(path),episodes=n,valid_steps=int(valid.sum()),
                           max_base_command_changes=int(base_changes.max()),max_ee_command_changes=int(ee_changes.max()),
                           trajectory_sha256=sha(path/'trajectories.npz'),passed=True))

source=json.loads((ROOT/'audit/source_asset_dependencies.json').read_text())
for rel,entry in source['tracked_files'].items(): assert sha(WT/rel)==entry['sha256'],rel
for name,entry in source['dependency_sources'].items(): assert sha(Path(name))==entry['sha256'],name
assert subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=WT)==b''
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip()==source['upstream_commit']
report=dict(passed=True,scope='smoke' if args.smoke else 'complete',evaluations=checks,
            upstream_source_and_assets_unchanged=True,dependency_sources_unchanged=True)
if not args.smoke:
    train=[json.loads(l) for l in (ROOT/'formal_seed21/updates.jsonl').read_text().splitlines()]
    assert [r['completed_updates'] for r in train]==list(range(1,3001))
    assert train[-1]['env_steps']==294912000
    cps={}
    for cp in [0,1000,2000,3000]:
        metadata=json.loads((ROOT/f'formal_seed21/cp_{cp}.json').read_text())
        assert metadata['completed_updates']==cp
        assert metadata['sha256']==sha(ROOT/f'formal_seed21/cp_{cp}.pt')
        cps[str(cp)]=dict(path=str(ROOT/f'formal_seed21/cp_{cp}.pt'),sha256=metadata['sha256'])
    report['checkpoints']=cps
    report['completed_updates']=3000
    report['environment_steps']=294912000
    import torch
    def model_hash(state):
        h=hashlib.sha256()
        for key,value in sorted(state.items()):
            h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial=json.loads((ROOT/'formal_seed21/initial_runtime.json').read_text())
    cp0=torch.load(ROOT/'formal_seed21/cp_0.pt',map_location='cpu',weights_only=False)
    assert cp0['optimizer_state_dict']['state']=={}
    assert model_hash(cp0['actor_state_dict'])==initial['actor_sha256']
    assert model_hash(cp0['critic_state_dict'])==initial['critic_sha256']
    for cp in [0,1000,2000,3000]:
        blob=torch.load(ROOT/f'formal_seed21/cp_{cp}.pt',map_location='cpu',weights_only=False)
        assert blob['iter']==cp-1 and blob['infos']['completed_updates']==cp
        for name in ['actor_state_dict','critic_state_dict']:
            assert all(torch.isfinite(t).all() for t in blob[name].values())
    run=next(p for p in (ROOT/'native_logs/rsl_rl/go2_piper_wbc').iterdir() if 'smoke' not in p.name and p.is_dir())
    original_final=torch.load(run/'model_2999.pt',map_location='cpu',weights_only=False)
    exact_final=torch.load(ROOT/'formal_seed21/cp_3000.pt',map_location='cpu',weights_only=False)
    for name in ['actor_state_dict','critic_state_dict']:
        assert all(torch.equal(t,exact_final[name][key]) for key,t in original_final[name].items())
    report['cp0_fresh_optimizer_and_model_hash_verified']=True
    report['cp3000_native_final_actor_critic_bitwise_equal']=True
target=ROOT/'audit'/('evaluation_smoke_cpu_audit.json' if args.smoke else 'completion_cpu_audit.json')
target.write_text(json.dumps(report,indent=2)+'\n')
print('CPU_AUDIT_PASS',target)
