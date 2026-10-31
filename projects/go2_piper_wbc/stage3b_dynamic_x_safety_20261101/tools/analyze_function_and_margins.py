"""CPU function-preservation and actual FL_hip diagnostics; no policy updates."""
import argparse,hashlib,pathlib
import numpy as np
import torch
from cpu_policy_sensitivity import mean
from gpu_budget import read,write

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()

def main(root):
    p=pathlib.Path(root);old=pathlib.Path(read(p/'AUTHORIZATION.json')['prior_closed_stage3_root'])
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    probe=pathlib.Path(read(old/'PROBE_INDEX.json')['P1-parent-X-s0.0'])
    with np.load(probe/'evaluation/trajectories.npz',allow_pickle=False) as z:
        obs=z['diag_actor_obs'];valid=z['valid'];recorded=z['diag_policy_mean']
        pairs=[(t,e) for t in range(50,1000,50) for e in range(valid.shape[1]) if valid[t,e]]
        times,envs=np.array(pairs).T;x=torch.tensor(obs[times,envs].copy());recorded=torch.tensor(recorded[times,envs].copy())
    parent=read(p/'CHECKPOINTS.json')['parent'];state=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict']
    with torch.no_grad():baseline,_=mean(state,x)
    assert float((baseline-recorded).abs().max())<1e-4
    models=[dict(label='parent',checkpoint=parent)]
    for f in ('CANDIDATE_NODES.json','SEED22_NODES.json'):
        if (old/f).exists():
            for n in read(old/f):models.append(dict(label='old/'+n['label'],checkpoint=n['checkpoint']))
    if (p/'CANDIDATE_NODES.json').exists():
        for n in read(p/'CANDIDATE_NODES.json'):models.append(dict(label='new/'+n['label'],checkpoint=n['checkpoint']))
    functions=[]
    with torch.no_grad():
        for item in models:
            cp=item['checkpoint'];assert sha(cp['checkpoint'])==cp['checkpoint_sha256']
            s=torch.load(cp['checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict'];output,_=mean(s,x);groups={}
            for label,mask in {'all':torch.ones(len(x),dtype=torch.bool),'backward':x[:,186]<-.05}.items():
                if not mask.any():continue
                delta=output[mask]-baseline[mask]
                groups[label]=dict(n=int(mask.sum()),leg_raw_mean_RMS=float(delta[:,:12].square().mean().sqrt()),
                    arm_raw_mean_RMS=float(delta[:,12:].square().mean().sqrt()),nominal_target_delta_RMS_rad_per_action=(.25*delta.square().mean(0).sqrt()).tolist())
            functions.append(dict(label=item['label'],checkpoint_sha256=cp['checkpoint_sha256'],groups=groups))
    # Derive common mechanical limits only after checking all17actual measured
    # realizations. Never assume an old asset's bounds for a randomized limit.
    indices=read(p/'COMPONENT_PROBE_INDEX.json');limits=[];names=None
    for run in indices.values():
        meta=pathlib.Path(run)/'evaluation/component_realized_start.json'
        if meta.exists():
            d=read(meta);values=np.asarray(d['hard_joint_limits']);assert np.array_equal(values,np.broadcast_to(values[:1],values.shape));limits.append(values[0]);names=d['action_joint_names']
    assert limits and all(np.array_equal(v,limits[0]) for v in limits)
    hip=names.index('FL_hip_joint');low,high=limits[0][hip];physical=[]
    for origin in (old,p):
        for label,run in read(origin/'EVAL_INDEX.json').items():
            cfg=read(pathlib.Path(run)/'config.json')
            if cfg['protocol'] not in ('A','T1') or label.startswith('final-'):continue
            directory=pathlib.Path(run)/'evaluation'
            if not (directory/'manifest.json').exists():continue
            with np.load(directory/'trajectories.npz',allow_pickle=False) as z:
                mask=z['valid'].astype(bool);q=z['control_joint_position'][:,:,hip];qd=z['control_joint_velocity'][:,:,hip]
                excess=z['control_hard_limit_excess'][:,:,hip];sat=z['control_actuator_saturation_substep_fraction']
                saturation=float(sat[:,:,hip][mask].mean()) if sat.ndim==3 and sat.shape[-1]==18 else None
                margin=np.minimum(q-low,high-q)
                physical.append(dict(origin='old' if origin==old else 'new',label=label,protocol=cfg['protocol'],seed=cfg['seed'],checkpoint_sha256=cfg['checkpoint_sha256'],
                    actual_FL_hip_q_max=float(q[mask].max()),actual_FL_hip_abs_qd_peak=float(abs(qd[mask]).max()),
                    actual_FL_hip_nominal_limit_margin_min_rad=float(margin[mask].min()),actual_FL_hip_hard_excess_max_rad=float(excess[mask].max()),
                    FL_hip_applied_torque_saturation_fraction=saturation,actual_mechanical_bounds_rad=[float(low),float(high)]))
    assert not torch.cuda.is_initialized()
    write(p/'FUNCTION_AND_LEG_MARGIN_AUDIT.json',dict(CPU_only=True,functions=functions,physical=physical,
        common_mechanical_limits_verified_across_measured_DR_realizations=True,source_sha256=sha(__file__),
        scope='Frozen functions on identical actual P1 inputs are distinct from native closed-loop q/qd/torque measurements. Nominal target deltas precede soft/ramp/delay; not actual EE predictions. Native1mrad hard-event tolerance is unchanged. No final seeds or PPO updates.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
