"""Local geometric incentive audit on saved states; not a policy gradient or physical return."""
import argparse,json,pathlib,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from gpu_budget import read,write
from response_breakdown import functions

def audit(directory,ns,chain):
    directory=pathlib.Path(directory);z=np.load(directory/'trajectories.npz',allow_pickle=False)
    summary=read(directory/'summary.json');runtime=read(directory/'initial_runtime.json');rewards=runtime['rewards'];names=[x['name'] for x in read(directory/'control_summary.json')['per_joint']]
    arm=[names.index('joint'+str(j)) for j in range(1,7)];idx={n:i for i,n in enumerate(summary['reward_names'])}
    assert 'joint_deviation_l1' in str(rewards['arm_deviation']['func']),'Only audited original L1 arm term supported'
    weight=float(rewards['arm_deviation']['weight']);valid=z['valid'].astype(bool);records=[];contract=[];rejected=0
    for t in range(50,len(valid),50):
        for e in range(min(8,valid.shape[1])):
            if not valid[t,e]:continue
            p=z['ee_actual_mixed'][t,e].astype(float);goal=z['ee_cmd'][t,e,:3].astype(float);error=p-goal
            measured=float(z['ee_position_error'][t,e]);rates=z['reward_terms'][t,e]
            contract.append(abs(float(4.5*np.exp(-measured**2/.1))-float(rates[idx['end_effector_position_tracking_exp']])))
            contract.append(abs(float(1.5*np.exp(-measured**2/.05**2))-float(rates[idx['ee_precision']])))
            if abs(error[0])<.002 or z['control_soft_margin'][t,e,arm].min()<.05:rejected+=1;continue
            q=z['control_joint_position'][t,e,arm].astype(float);base=ns['fk'](chain,q);J=ns['jac'](chain,q)
            local_R=Rotation.from_quat(z['kin_link0_quat'][t,e,[1,2,3,0]]).as_matrix()
            h=1e-5;orient=np.column_stack([(Rotation.from_matrix(base[:3,:3]).inv()*Rotation.from_matrix(ns['fk'](chain,q+np.eye(6)[j]*h)[:3,:3])).as_rotvec()/h for j in range(6)])
            constraints=np.vstack((J[0],J[1],local_R[2]@J,orient))
            desired=np.array([-np.sign(error[0])*.001,0,0,0,0,0]);dq=np.linalg.lstsq(constraints,desired,rcond=None)[0]
            dq*=min(1.,.003/max(abs(dq).max(),1e-12));changed=ns['fk'](chain,q+dq)
            delta=changed[:3,3]-base[:3,3];mixed_delta=np.array([delta[0],delta[1],local_R[2]@delta]);angle=float((Rotation.from_matrix(base[:3,:3]).inv()*Rotation.from_matrix(changed[:3,:3])).magnitude())
            if mixed_delta[0]*desired[0]<=0 or np.max(abs(mixed_delta[1:]))>1e-5 or angle>1e-4:rejected+=1;continue
            def position_rate(err):return 4.5*np.exp(-np.sum(err**2)/.1)+1.5*np.exp(-np.sum(err**2)/.05**2)
            offset=(z['control_affine_target'][t,e]-z['control_raw_action'][t,e]*.25)[arm]
            old_arm=weight*np.abs(q-offset).sum();new_arm=weight*np.abs(q+dq-offset).sum()
            arm_validation=float(abs(old_arm-float(rates[idx['arm_deviation']])))
            dr=float(position_rate(error+mixed_delta)-position_rate(error));da=float(new_arm-old_arm)
            condition=float(np.linalg.cond(constraints))
            records.append(dict(env=e,time_s=float(z['age_steps'][t,e]*.02),X_error_m=float(error[0]),X_correction_m=float(mixed_delta[0]),requested_X_correction_m=float(desired[0]),matrix_condition_number=condition if np.isfinite(condition) else None,max_joint_delta_rad=float(abs(dq).max()),held_YZ_residual_m=float(np.max(abs(mixed_delta[1:]))),orientation_delta_rad=angle,
                position_reward_rate_delta=dr,arm_posture_reward_rate_delta=da,position_plus_arm_rate_delta=dr+da,arm_formula_error=arm_validation))
    assert contract and max(contract)<1e-5,'Original position reward formula not numerically matched'
    assert records,'No qualified bounded local directions; do not infer feasibility from empty samples'
    arm_valid=bool(max(r['arm_formula_error'] for r in records)<1e-5)
    return dict(samples=len(records),excluded_near_limits_or_near_zero_X_error_or_bad_direction=rejected,
        position_formula_max_error=max(contract),arm_formula_validated=arm_valid,arm_formula_max_error=max(r['arm_formula_error'] for r in records),arm_weight=weight,
        positive_position_fraction=float(np.mean([r['position_reward_rate_delta']>0 for r in records])),
        positive_position_plus_arm_fraction=float(np.mean([r['position_plus_arm_rate_delta']>0 for r in records])) if arm_valid else None,
        mean_position_rate_delta=float(np.mean([r['position_reward_rate_delta'] for r in records])),
        mean_arm_posture_rate_delta=float(np.mean([r['arm_posture_reward_rate_delta'] for r in records])) if arm_valid else None,
        mean_abs_X_correction_m=float(np.mean([abs(r['X_correction_m']) for r in records])),
        max_orientation_change_rad=max(r['orientation_delta_rad'] for r in records),max_held_YZ_residual_m=max(r['held_YZ_residual_m'] for r in records),records=records)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--config',required=True);a.add_argument('--labels',nargs='+',required=True);a.add_argument('--out',required=True);c=a.parse_args();p=pathlib.Path(c.root);cfg=read(c.config)
    ns=dict(np=np,Rotation=Rotation,_MODEL_CACHE={});fk_sha=functions(cfg['kinematics_source'],['fk'],ns);jac_sha=functions(cfg['original_response_source'],['jac'],ns);chain=read(cfg['chain_file'])['chain'];index=read(p/'EVAL_INDEX.json')
    results=[dict(label=label,**audit(pathlib.Path(index[label])/'evaluation',ns,chain)) for label in c.labels]
    write(c.out,dict(kind='Saved-state local geometry only; no simulated rollout, policy gradient or full-return claim',sources=dict(original_fk=fk_sha,original_jacobian=jac_sha),results=results,
        limitations='Bounds from recorded native soft margins. Native position kernels and arm L1 checked numerically. Other dynamic penalties and future base/arm coupling not evaluated; these virtual corrections are never executed by the actor.'))
    print(json.dumps([{k:v for k,v in r.items() if k!='records'} for r in results]))
