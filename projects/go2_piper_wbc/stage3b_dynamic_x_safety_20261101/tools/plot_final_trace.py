"""CPU-only predefined native20s trace; neither video nor policy selection."""
import argparse, hashlib, json, pathlib, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gpu_budget import read


def main(p):
    assert (p/'COMPLETED.json').exists(), 'Formal result must be closed before plotting'
    plan=read(p/'FINAL_TRACE_PLAN.json'); assert plan['formal_seed']==611101 and plan['environment_index']==0 and plan['protocol']=='X'
    index=read(p/'EVAL_INDEX.json'); selected=read(p/'FROZEN_SELECTION.json'); env=plan['environment_index']
    # Reuse measured native mechanical bounds and action order; no new FK or
    # actuator model. Their constancy across measured DR was separately checked.
    meta=next(pathlib.Path(run)/'evaluation/component_realized_start.json' for run in read(p/'COMPONENT_PROBE_INDEX.json').values()
        if (pathlib.Path(run)/'evaluation/component_realized_start.json').exists())
    realized=read(meta); hip=realized['action_joint_names'].index('FL_hip_joint')
    bounds=np.asarray(realized['hard_joint_limits']); assert np.array_equal(bounds,np.broadcast_to(bounds[:1],bounds.shape))
    lower,upper=bounds[0,hip]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained'); colors={'initial':'#7d858b','candidate':'#188b78','adjacent':'#9a6b9f'}; records=[]
    for model in plan['models']:
        key='final-'+model+'-X-'+str(plan['formal_seed'])
        if key not in index:
            assert model=='adjacent' and selected['adjacent'] is None; continue
        run=pathlib.Path(index[key]); cfg=read(run/'config.json'); path=run/'evaluation/trajectories.npz'
        with np.load(path,allow_pickle=False) as z:
            mask=z['valid'][:,env].astype(bool); t=.02*np.arange(1,len(mask)+1); valid=t[mask]
            if not mask.any():
                records.append(dict(model=model,checkpoint_sha256=cfg['checkpoint_sha256'],valid_control_samples=0,no_line_drawn=True));continue
            cmd=z['ee_cmd'][:,env,:3][mask]; actual=z['ee_actual_mixed'][:,env][mask]
            base_cmd=z['base_cmd'][:,env][mask]; base=z['base_actual'][:,env][mask]
            error=z['ee_position_error'][:,env][mask]; q=z['control_joint_position'][:,env,hip][mask]
            world=z['base_pos_world'][:,env][mask]
        color=colors[model]
        axes[0,0].plot(valid,1000*actual[:,0],c=color,label=model);axes[0,0].plot(valid,1000*cmd[:,0],c=color,ls='--',alpha=.65)
        axes[0,1].plot(valid,100*error,c=color,label=model)
        for ax,dimension in [(axes[0,2],0),(axes[1,0],1),(axes[1,1],2)]:
            ax.plot(valid,base[:,dimension],c=color,label=model);ax.plot(valid,base_cmd[:,dimension],c=color,ls='--',alpha=.65)
        axes[1,2].plot(valid,q,c=color,label=model)
        records.append(dict(model=model,checkpoint_sha256=cfg['checkpoint_sha256'],valid_control_samples=int(mask.sum()),
            native_trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            body_command_abs_max_by_axis=np.abs(base_cmd).max(0).tolist(),
            net_world_base_xy_displacement_m=float(np.linalg.norm(world[-1,:2]-world[0,:2])),
            target_X_range_m=[float(cmd[:,0].min()),float(cmd[:,0].max())],actual_X_range_m=[float(actual[:,0].min()),float(actual[:,0].max())]))
    axes[0,1].axhline(5,c='#ba754b',ls=':',label='5cm reference; not standalone PASS')
    for limit in (lower,upper):axes[1,2].axhline(limit,c='#ba754b',ls=':')
    labels=['EE X, native mixed-frame (mm)','EE position error (cm)','Body vx (m/s)','Body vy (m/s)','Yaw rate (rad/s)','FL_hip q,50Hz control sample (rad)']
    for ax,label in zip(axes.flat,labels):ax.set(xlim=(0,20),xlabel='Physical control time (s)',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=7)
    fig.suptitle('Predeclared X / seed611101 / environment0. Solid=actual, dashed=native command. Startup retained; failed lines stop at valid boundary. Full64-episode matrix remains authoritative.',fontsize=9)
    directory=p/'figures/final_predeclared_trace';directory.mkdir(parents=True,exist_ok=False)
    target=directory/'FINAL_PREDECLARED_NATIVE20S.svg';fig.savefig(target,metadata=dict(Date=None));plt.close(fig)
    result=dict(plan=plan,plan_sha256=hashlib.sha256((p/'FINAL_TRACE_PLAN.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),file=target.name,
        bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),records=records,
        scope='Native logged arrays only; no smoothing, rescoring, FK reconstruction, best-episode selection or new GPU. One fixed episode is not full-task evidence.')
    target.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(file=str(target),bytes=result['bytes'],sha256=result['sha256'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=pathlib.Path,required=True);parser.add_argument('--wait',action='store_true');a=parser.parse_args()
    if a.wait:
        while not (a.root/'COMPLETED.json').exists():
            if (a.root/'FINALIZER_ERROR.json').exists():raise RuntimeError('Repair actual final evaluation before plotting')
            time.sleep(5)
    main(a.root)
