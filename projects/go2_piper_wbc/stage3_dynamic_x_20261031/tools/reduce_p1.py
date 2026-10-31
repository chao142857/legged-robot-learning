"""CPU causal reduction of actual P1 pairs; diagnostics are never policy PASS."""
import argparse,csv,hashlib,importlib.util,json,pathlib
import numpy as np
from scipy.spatial.transform import Rotation
from gpu_budget import read,write
from response_breakdown import functions

def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main(p):
    cfg=read(p/'CPU_RESPONSE_CONFIG.json');kin=module(cfg['kinematics_source'],'p1_original_FK')
    metrics_path=pathlib.Path(cfg['kinematics_source']).parents[1]/'recipe_b0/metrics.py';metrics=module(metrics_path,'p1_original_metrics')
    chain=read(cfg['chain_file'])['chain'];ns=dict(np=np,Rotation=Rotation,fk=kin.fk,W=2*np.pi/8)
    functions(cfg['original_response_source'],['jac','fit'],ns)
    index=read(p/'PROBE_INDEX.json');cases=[];pairs=[];loaded={}
    for label,where in index.items():
        run=pathlib.Path(where);d=run/'evaluation'
        if not (d/'manifest.json').exists():continue
        with np.load(d/'trajectories.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files if k!='body_pos_world_first16'}
        loaded[label]=(d,data);summary=read(d/'summary.json');sensitivity=read(d/'target_sensitivity.json');native=read(d/'dynamic_metrics.json')
        valid=data['valid'].astype(bool);steady=valid&(data['age_steps']*.02>=5.)
        after5=dict(data,valid=steady);steady_metrics=metrics.dynamic_metrics(after5)['windows']['all20s']
        counter=[]
        for mode in ('position_only','newest','coherent_history'):
            rows=[r for r in sensitivity['counterfactuals'] if r['mode']==mode and r['step']*.02>=1.]
            if rows:
                gains=np.array([r['local_X_gain'] for r in rows]);ratios=np.array([r['cancellation_ratio'] for r in rows])
                contribution=np.array([r['X_joint_contribution_m'] for r in rows])
                counter.append(dict(mode=mode,samples=int(gains.size),signed_local_gain_mean=float(gains.mean()),signed_local_gain_median=float(np.median(gains)),
                    positive_direction_fraction=float((gains>0).mean()),cancellation_ratio_median=float(np.median(ratios)),mean_abs_individual_joint_contribution_m=np.mean(abs(contribution),axis=(0,1)).tolist(),
                    effective_sum_abs_m=float(np.mean(abs(contribution.sum(-1)))),sum_of_individual_abs_m=float(np.mean(abs(contribution).sum(-1)))))
        transfers=[]
        for e in range(valid.shape[1]):
            mask=steady[:,e]
            if mask.sum()<100:continue
            t=data['age_steps'][mask,e]*.02;ti=t-.02
            q=data['control_joint_position'][mask,e,12:];qt=data['control_processed_target'][mask,e,12:]
            _,hc=ns['fit'](data['ee_cmd'][mask,e,0],ti);_,ha=ns['fit'](data['ee_actual_mixed'][mask,e,0],t)
            _,hq=ns['fit'](q,t);_,ht=ns['fit'](qt,ti);j=ns['jac'](chain,q.mean(0))
            delayed=[]
            for sub in range(4):delayed.append(ns['fit'](data['diag_delay_position'][mask,e,sub,12:],ti+sub*.005)[1])
            hd=np.mean(delayed,axis=0)
            transfers.append(dict(env=e,requested_local_J_X_amplitude_m=float(abs(j[0]@ht)),delayed_local_J_X_amplitude_m=float(abs(j[0]@hd)),actual_local_J_X_amplitude_m=float(abs(j[0]@hq)),
                real_EE_X_amplitude_m=float(abs(ha)),requested_to_delayed_complex_norm_ratio=float(np.linalg.norm(hd)/max(np.linalg.norm(ht),1e-12)),
                requested_to_actual_joint_complex_norm_ratio=float(np.linalg.norm(hq)/max(np.linalg.norm(ht),1e-12)),
                requested_joint_signed_X_contribution=[dict(real=float(v.real),imag=float(v.imag)) for v in j[0]*ht]))
        cases.append(dict(label=label,summary={k:summary[k] for k in ['num_episodes','complete_count','safe_count','nonfoot_contact_count','overspeed_count','joint_limit_violation_count','terminations']},
            native_full20=summary['groups']['all']['all'],native_X_after1s={k:v for k,v in native['windows']['after1s']['axes']['x'].items() if k!='per_episode'},
            diagnostic_after5s={k:v for k,v in steady_metrics.items() if k!='axes'},diagnostic_X_after5s={k:v for k,v in steady_metrics['axes']['x'].items() if k!='per_episode'},
            counterfactuals=counter,physical_transfer=transfers,PD_torque_equation_max_error=sensitivity['measured_PD_torque_equation_max_error'],
            raw_trajectory_sha256=hashlib.sha256((d/'trajectories.npz').read_bytes()).hexdigest()))
    for label,(directory,data) in loaded.items():
        config=read(directory.parent/'config.json');strength=config['probe']['strength']
        if strength<=0:continue
        baseline=label.rsplit('-s',1)[0]+'-s0.0';parent,parent_data=loaded[baseline]
        prefix={}
        for name in ['diag_actor_obs','diag_policy_mean','diag_delay_position','diag_delay_delay','control_joint_position','control_joint_velocity','kin_link0_pos','ee_cmd']:
            prefix[name]=float(np.max(abs(data[name][:200].astype(float)-parent_data[name][:200].astype(float))))
        assert max(prefix.values())<2e-5,dict(label=label,prefix=prefix)
        dr_equal=read(directory/'realized_physics_start.json')==read(parent/'realized_physics_start.json')
        assert dr_equal,'Measured initial DR/state mismatch; pair is not causal'
        common=data['valid'].astype(bool)&parent_data['valid'].astype(bool)&(data['age_steps']*.02>=5.)
        ss=read(directory/'summary.json');bs=read(parent/'summary.json');names=ss['reward_names'];assert names==bs['reward_names']
        delta=(data['reward_terms'][common]-parent_data['reward_terms'][common]).mean(0)
        consistency=max(float(abs(data['reward_terms'].sum(-1)*.02-data['reward']).max()),float(abs(parent_data['reward_terms'].sum(-1)*.02-parent_data['reward']).max()))
        assert consistency<2e-5,consistency
        dynamic=read(directory/'dynamic_metrics.json')['windows']['after1s']['axes']['x']['per_episode']
        with (directory/'episodes.csv').open() as f:episodes=list(csv.DictReader(f))
        eligible=[r['env'] for r in dynamic if episodes[r['env']]['safe']=='True' and r.get('actual_fitted_amplitude_m',0)>=.008 and r.get('synchronous_gain',-1)>=.35 and r.get('correlation',-1)>=.5]
        pairs.append(dict(treatment=label,baseline=baseline,identical_cold_prefix_max_errors=prefix,initial_realized_physics_equal=dr_equal,
            strict_safe_large_X_response_envs=eligible,strict_safe_large_X_response_count=len(eligible),
            native_reward_rate_delta_by_term=dict(zip(names,map(float,delta))),native_reward_rate_delta_sum=float(delta.sum()),
            native_reward_rate_sum_consistency_max_error=consistency,
            limitations='After5s common-live comparison is causal diagnosis, not full20s acceptance or exact PPO gradient. All native first-episode scores and failures remain unchanged.'))
    gradient=[];receipt=p/'P1_NATIVE_GRADIENT_RUN.json'
    if receipt.exists():
        run=pathlib.Path(read(receipt)['run']);path=run/'ppo/native_gradient_audit.jsonl'
        if path.exists():
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            gradient=dict(run=str(run),completed_audited_updates=len(rows),first_update=rows[0],last_update=rows[-1],raw_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                native_GAE_and_optimizer_unmodified=True,per_reward_exact_PPO_gradient_not_claimed=True)
    write(p/'P1_CAUSAL_EVIDENCE.json',dict(cases=cases,pairs=pairs,native_gradient_audit=gradient,
        original_FK_source_sha256=hashlib.sha256(pathlib.Path(cfg['kinematics_source']).read_bytes()).hexdigest(),
        original_metric_source_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        policy_success_claim=False,physical_feasibility_is_not_learned_policy_success=True))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);c=a.parse_args();main(pathlib.Path(c.root))
