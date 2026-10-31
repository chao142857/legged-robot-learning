"""CPU reduction of unchanged native frozen evaluations; no training or rescoring."""
import argparse,ast,csv,hashlib,json,pathlib
import numpy as np
from gpu_budget import read,write

def native_capability(source):
    # Reuse the actual original report function, without its CLI/main/path imports.
    path=pathlib.Path(source)/'tools/report_recipe_a.py'
    module=ast.parse(path.read_text())
    functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('stat','digest','capability')]
    assert {n.name for n in functions}=={'stat','digest','capability'}
    namespace=dict(np=np,Path=pathlib.Path,hashlib=hashlib,read=read,DIRECTIONS=('forward','backward','left','right'))
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['capability'],hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(root,source):
    p=pathlib.Path(root);source=pathlib.Path(source);spec=read(source/'recipe_b0/spec.json')
    health=read(source/'recipe_v1/spec.json')['A_to_B_gate'];capability,reference_sha=native_capability(source)
    idx=read(p/'EVAL_INDEX.json');rows=[]
    for label,where in idx.items():
        directory=pathlib.Path(where)/'evaluation'
        if not (directory/'manifest.json').exists():continue
        cfg=read(pathlib.Path(where)/'config.json');s=read(directory/'summary.json')
        native=capability(dict(label=label,protocol=cfg['protocol'],seed=cfg['seed'],out=str(directory)),health)
        with (directory/'episodes.csv').open() as f:episodes=list(csv.DictReader(f))
        with np.load(directory/'trajectories.npz',allow_pickle=False) as trajectory:
            executed_steps=int(trajectory['valid'].shape[0])
            valid=trajectory['valid'].astype(bool);after=valid&(trajectory['age_steps']*.02>1.)
            orientation_RMS={}
            for name in ('ee_orientation_error','ee_orientation_reward_error'):
                if name in trajectory:
                    angles=trajectory[name]
                    orientation_RMS[name]={label:float(np.sqrt(np.mean(angles[mask]**2))) for label,mask in [('all20s',valid),('after1s',after)]}
        dynamic=read(directory/'dynamic_metrics.json');x=dynamic['windows']['after1s']['axes']['x'];allx=dynamic['windows']['all20s']['axes']['x']
        phase=[v['phase_lag_s'] for v in x['per_episode'] if v.get('phase_lag_is_interpretable')]
        r=dict(label=label,protocol=cfg['protocol'],seed=cfg['seed'],checkpoint=cfg['checkpoint'],checkpoint_sha256=cfg['checkpoint_sha256'],run=where,
            n=s['num_episodes'],complete=s['complete_count'],strict_safe=s['safe_count'],joint_5cm=s['joint_success_5cm_count'],
            nonfoot=s['nonfoot_contact_count'],overspeed=s['overspeed_count'],joint_limit=s['joint_limit_violation_count'],terminations=s['terminations'],
            full20=s['groups']['all']['all'],after1s=s['groups']['after_1s']['all'],
            X_after1s={k:v for k,v in x.items() if k!='per_episode'},X_all20={k:v for k,v in allx.items() if k!='per_episode'},
            axes_after1s={axis:{k:v for k,v in values.items() if k!='per_episode'} for axis,values in dynamic['windows']['after1s']['axes'].items()},
            X_phase_lag_qualified_mean_s=float(np.mean(phase)) if phase else None,X_phase_lag_qualified_fraction=len(phase)/len(x['per_episode']),
            directions=native['directions'],native_health={k:v for k,v in native.items() if k not in ('label','protocol','seed','out','ee_axes','directions','reward_rate_mean','reward_return_mean')},
            EE_after1s=dynamic['windows']['after1s']['EE_absolute_rmse_m'],base_drift=dynamic['base_final_xy_drift_m'],
            raw_trajectory_sha256=native['trajectory_sha256'])
        r.update(executed_vector_steps=executed_steps,simulator_transition_slots=executed_steps*s['num_episodes'],
            first_episode_valid_samples=s['groups']['all']['all']['samples'],
            return_mean=sum(float(e['return_sum']) for e in episodes)/s['num_episodes'],
            return_per_full20s_step=sum(float(e['return_sum']) for e in episodes)/s['num_episodes']/s['requested_steps'])
        r['orientation_RMS_rad']=orientation_RMS
        g=spec['learning_trend_gate']['CP2200']
        r['absolute_X_gate']=dict(gain=g['X_sync_gain_min']<=x['synchronous_gain']<=g['X_sync_gain_max'],
            correlation=x['correlation'] is not None and x['correlation']>=g['X_correlation_min'],
            amplitude=x.get('actual_fitted_amplitude_m') is not None and g['X_fitted_amplitude_min_m']<=x['actual_fitted_amplitude_m']<=g['X_fitted_amplitude_max_m'],
            EE=r['EE_after1s']<=g['EE_absolute_RMSE_max_m']) if cfg['protocol']=='X' else None
        rows.append(r)
    by_label={r['label']:r for r in rows}
    for r in rows:
        b=by_label.get('frozen1800-'+r['protocol']+'-'+str(r['seed']))
        if b and r is not b:
            h=spec['retention_gate'];x=r['X_after1s'];old=b['X_after1s'];g=spec['learning_trend_gate']['CP2200']
            r['relative_to_original']=dict(X_gain_delta=x['synchronous_gain']-old['synchronous_gain'] if x['synchronous_gain'] is not None and old['synchronous_gain'] is not None else None,
                X_demeaned_RMSE_delta_m=x['demeaned_rmse_m']-old['demeaned_rmse_m'],EE_after1s_delta_m=r['EE_after1s']-b['EE_after1s'],
                strict_safe_delta=r['strict_safe']-b['strict_safe'])
            r['retention_checks']=dict(complete=r['complete']/r['n']>=h['completion_fraction_min'],
                strict_safe=r['strict_safe']/r['n']>=h['strict_safe_fraction_min'] and r['strict_safe']/r['n']>=b['strict_safe']/b['n']-h['strict_safe_fraction_drop_vs_parent_max'],
                EE=r['EE_after1s']<=b['EE_after1s']+h['A_T0_T1_EE_RMSE_increase_max_m'])
            checks=r['native_health']['checks']
            r['retention_checks']['native_health']=r['native_health']['gate_pass'] if r['protocol']!='T0' else all(v for k,v in checks.items() if not k.startswith(('forward/','backward/','left/','right/')))
            if r['protocol'] in ('A','T1'):
                for direction in ('forward','backward','left','right'):
                    current=r['directions'][direction]['gain'];old_gain=b['directions'][direction]['gain']
                    r['retention_checks'][direction]=current is not None and old_gain is not None and abs(current-old_gain)<=h['A_T1_direction_gain_change_max']
            if r['protocol']=='T1':
                for axis in ('y','z'):
                    current=r['axes_after1s'][axis]['synchronous_gain'];old_gain=b['axes_after1s'][axis]['synchronous_gain']
                    r['retention_checks'][axis+'_retained']=current is not None and old_gain is not None and current>=old_gain-h['T1_YZ_sync_gain_drop_max']
            if r['protocol']=='T0':
                for field,limit in [('mean','T0_base_drift_mean_increase_max_m'),('p95','T0_base_drift_p95_increase_max_m')]:
                    r['retention_checks']['base_drift_'+field]=r['base_drift'][field]<=b['base_drift'][field]+h[limit]
            if r['protocol']=='X':
                r['relative_X_gate']=dict(gain=x['synchronous_gain']>=old['synchronous_gain']+g['X_gain_improvement_vs_CP1800_min'],
                    correlation=x['correlation'] is not None and old['correlation'] is not None and x['correlation']>=old['correlation']+g['X_correlation_improvement_min'],
                    demeaned_RMSE=x['demeaned_rmse_m']<=old['demeaned_rmse_m']*g['X_demeaned_RMSE_fraction_of_parent_max'],
                    absolute_RMSE=x['absolute_rmse_m']<=old['absolute_rmse_m']+g['X_absolute_RMSE_increase_max_m'])
    return dict(kind='Derived original native full20s physics, strict/startup unchanged',native_capability_source_sha256=reference_sha,
        spec_sha256=hashlib.sha256((source/'recipe_b0/spec.json').read_bytes()).hexdigest(),rows=rows,
        warning='Native joint_5cm is not dynamic X success. Truncated RMSE retains native valid mask; complete/safety and all failure denominators remain separate. One training seed does not certify cross-training-seed repeatability.')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--source',required=True);a.add_argument('--out',required=True);c=a.parse_args()
    result=summarize(c.root,c.source);write(c.out,result)
    print(json.dumps(dict(evaluations=len(result['rows']),out=c.out)))
