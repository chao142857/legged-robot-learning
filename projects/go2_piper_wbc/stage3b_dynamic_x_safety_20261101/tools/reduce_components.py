"""CPU reduction of real frozen-component interventions; never policy PASS."""
import argparse,collections,pathlib
import numpy as np
from gpu_budget import read,write
from analyze_physics import native_capability

PREFIX_FIELDS=('diag_actor_obs','component_current_mean','component_parent_mean','control_raw_action',
               'control_processed_target','diag_delay_position','diag_delay_delay','control_joint_position',
               'control_joint_velocity','base_actual','base_pos_world','ee_cmd')

def summarize(root):
    p=pathlib.Path(root);index=read(p/'COMPONENT_PROBE_INDEX.json');rows=[];prefixes={};realizations={}
    source=pathlib.Path(read(p/'CPU_RESPONSE_CONFIG.json')['kinematics_source']).parents[1]
    capability,source_sha=native_capability(source);health=read(source/'recipe_v1/spec.json')['A_to_B_gate']
    for label,where in index.items():
        run=pathlib.Path(where);out=run/'evaluation'
        if not (out/'component_result.json').exists():continue
        cfg=read(run/'config.json');s=read(out/'summary.json');contract=read(out/'component_contract.json');r=read(out/'component_realized_start.json')
        assert contract['optimizer_updates']==0 and not contract['policy_success_claim']
        assert read(out/'component_result.json')['parent_unchanged'] and s['actor_unchanged'] and s['critic_unchanged']
        mode=cfg['component_probe']['mode'];group=cfg['diagnostic_group'];key=(group,cfg['protocol'],cfg['seed'])
        native=capability(dict(label=label,protocol=cfg['protocol'],seed=cfg['seed'],out=str(out)),health)
        d=read(out/'dynamic_metrics.json');x=d['windows']['after1s']['axes']['x'];prefix_steps=int(round(cfg['component_probe']['start_s']/.02))
        with np.load(out/'trajectories.npz',allow_pickle=False) as data:
            valid=data['valid'].astype(bool);q=data['control_joint_position'];qd=data['control_joint_velocity']
            limits=np.asarray(r['hard_joint_limits']);margin=np.minimum(q-limits[None,:,:,0],limits[None,:,:,1]-q)
            efforts=data['component_applied_torque_substeps'];effort_limits=data['component_effort_limits']
            saturation=abs(efforts)>=effort_limits[:,:,None,:]*(1-1e-5)
            saturation_fraction=(saturation*valid[:,:,None,None]).sum((0,1,2))/(valid.sum()*4)
            current=data['component_current_mean'];prior=data['component_parent_mean'];diff=current-prior
            hard=(data['control_hard_limit_excess']>.001)&valid[:,:,None];events=np.argwhere(hard);first=None
            if len(events):
                t,e,j=map(int,events[0]);first=dict(time_s=float(data['age_steps'][t,e]*.02),env=e,joint=r['action_joint_names'][j],q=float(q[t,e,j]),qd=float(qd[t,e,j]),
                    target=float(data['control_processed_target'][t,e,j]),applied_torque4=efforts[t,e,:,j].tolist(),
                    angular_velocity_b=data['component_base_angular_velocity_b'][t,e].tolist(),foot_forces=data['foot_forces'][t,e].tolist())
            nonfoot=(data['nonfoot_max_force']>1)&valid;contacts=np.argwhere(nonfoot);first_contact=None
            if len(contacts):
                t,e=map(int,contacts[0]);forces=data['component_contacts_history_w'][t,e]
                peaks=np.linalg.norm(forces,axis=-1).max(0);names=r['contact_sensor_body_names']
                if isinstance(names,list):
                    candidates=[i for i,name in enumerate(names) if not name.endswith('_foot')];body=max(candidates,key=lambda i:peaks[i])
                    first_contact=dict(time_s=float(data['age_steps'][t,e]*.02),env=e,body=names[body],force_history_peak_N=float(peaks[body]))
                else:first_contact=dict(time_s=float(data['age_steps'][t,e]*.02),env=e,body='NOT_MEASURED')
            prefixes[(key,mode)]={name:data[name][:prefix_steps].copy() for name in PREFIX_FIELDS}
            realizations[(key,mode)]=r
            row=dict(label=label,group=group,protocol=cfg['protocol'],seed=cfg['seed'],mode=mode,run=str(run),checkpoint_sha256=cfg['checkpoint_sha256'],
                n=s['num_episodes'],strict_safe=s['safe_count'],complete=s['complete_count'],hard=s['joint_limit_violation_count'],nonfoot=s['nonfoot_contact_count'],
                overspeed=s['overspeed_count'],X={name:x.get(name) for name in ('synchronous_gain','correlation','actual_fitted_amplitude_m','demeaned_rmse_m','absolute_rmse_m')},
                EE_RMSE_m=d['windows']['after1s']['EE_absolute_rmse_m'],native_health=native['gate_pass'],native_health_checks=native['checks'],directions=native['directions'],
                minimum_actual_joint_margin_rad=np.where(valid[:,:,None],margin,np.inf).min((0,1)).tolist(),
                actual_joint_peak_abs_velocity=np.where(valid[:,:,None],abs(qd),0).max((0,1)).tolist(),
                actual_applied_torque_saturation_fraction_per_joint=saturation_fraction.tolist(),first_hard_event=first,first_nonfoot_event=first_contact,
                same_state_current_parent_leg_mean_RMS=float(np.sqrt((diff[:,:,:12]**2*valid[:,:,None]).sum()/(valid.sum()*12))),
                trained_policy_success=False)
            rows.append(row)
    comparisons=[]
    for key in sorted({key for key,mode in prefixes}):
        if (key,'current') not in prefixes:continue
        base=prefixes[(key,'current')];initial=realizations[(key,'current')]
        for mode in ('parent_legs','parent_arm','parent_all'):
            if (key,mode) not in prefixes:continue
            other=prefixes[(key,mode)];real=realizations[(key,mode)]
            errors={name:float(np.max(abs(base[name].astype(float)-other[name].astype(float)))) for name in PREFIX_FIELDS}
            assert max(errors.values())<2e-5,(key,mode,errors)
            measured={name:initial[name]==real[name] for name in ('root_state','joint_q','joint_qd','hard_joint_limits','actuators')}
            for name in ('get_masses','get_coms','get_inertias','get_material_properties'):
                if not isinstance(initial[name],str) and not isinstance(real[name],str):measured[name]=initial[name]==real[name]
            assert all(measured.values()),(key,mode,measured)
            comparisons.append(dict(group=key[0],protocol=key[1],seed=key[2],mode=mode,identical_actual_prefix_max_errors=errors,measured_initial_DR_state_equal=measured,
                hidden_PhysX_state_recovery_claim=False))
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['mode']].append(row)
    assessments={}
    for mode,values in grouped.items():
        xs=[r for r in values if r['protocol']=='X']
        assessments[mode]=dict(complete_cells=len(values),all_actual_hard_nonfoot_zero=all(r['hard']==r['nonfoot']==0 for r in values),
            strict_and_complete_at_least95pct_each=all(r['strict_safe']/r['n']>=.95 and r['complete']/r['n']>=.95 for r in values),
            native_health_all=all(r['native_health'] for r in values),
            active_X_preserved=len(xs)>0 and all(r['X']['synchronous_gain']>=.35 and r['X']['correlation']>=.5 and r['X']['actual_fitted_amplitude_m']>=.008 for r in xs))
    result=dict(rows=rows,comparisons=comparisons,mode_assessments=assessments,original_native_capability_sha256=source_sha,
        scope='Frozen action-function interventions after identical measured cold prefix, not a PPO policy or hidden-state clone. Each mode receives its actual changed observation history. All native20s safety/failure denominators remain.',
        final_policy_success=False,mechanism_choice_requires_complete_preregistered_cells=True)
    write(p/'COMPONENT_PHYSICS.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();d=summarize(a.root)
    print(d['mode_assessments'])
