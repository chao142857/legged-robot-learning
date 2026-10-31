"""CPU construction of a measured, fixed arm synergy; activation requires P1 evidence."""
import argparse,hashlib,importlib.util,json,pathlib
import numpy as np
from scipy.spatial.transform import Rotation
from gpu_budget import read,write

def main(p):
    evidence=read(p/'P1_CAUSAL_EVIDENCE.json')
    pairs=evidence['pairs'];assert len(pairs)>=3 and all(r['initial_realized_physics_equal'] and max(r['identical_cold_prefix_max_errors'].values())<2e-5 for r in pairs)
    assert all(r['strict_safe_large_X_response_count']>=24 for r in pairs),'Physical existence not established'
    assert all(r['native_reward_rate_delta_sum']>0 for r in pairs),'Do not attribute a negative original reward preference to expression alone'
    c=read(p/'CPU_RESPONSE_CONFIG.json');s=importlib.util.spec_from_file_location('nativeFK',c['kinematics_source']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    chain=read(c['chain_file'])['chain'];idx=read(p/'PROBE_INDEX.json');run=pathlib.Path(idx['P1-parent-X-s0.0'])
    with np.load(run/'evaluation/trajectories.npz',allow_pickle=False) as d:
        mask=d['valid'].astype(bool)&(d['age_steps']*.02>=5)
        q=d['control_joint_position'][mask][:,12:].mean(0)
        observed=d['diag_actor_obs'][mask][:,66:72].mean(0)
    t=m.fk(chain,q);cols=[];h=1e-5
    for i in range(6):
        plus=m.fk(chain,q+np.eye(6)[i]*h)
        cols.append(np.r_[(plus[:3,3]-t[:3,3])/h,(Rotation.from_matrix(t[:3,:3]).inv()*Rotation.from_matrix(plus[:3,:3])).as_rotvec()/h])
    j=np.column_stack(cols);delta=np.linalg.lstsq(j,np.array([.02,0,0,0,0,0]),rcond=1e-5)[0]
    delta*=min(1.,.12/max(abs(delta)))
    recipe=dict(schema='original210_arm_synergy_mean_v1',arm_basis_raw_action=(delta/.25).tolist(),observed_arm_q_reference=observed.tolist(),local_FK_Jx=j[0].tolist(),local_FK_x_reference=float(t[0,3]),
        basis_measured_actual_q=q.tolist(),basis_predicted_position_delta_m=(j[:3]@delta).tolist(),basis_predicted_orientation_delta_rad=(j[3:]@delta).tolist(),
        origin='Original FK at actual safe P1 locomotion states; fixed six-joint geometric direction, no inference IK or teacher supervision',
        P1_evidence_sha256=hashlib.sha256((p/'P1_CAUSAL_EVIDENCE.json').read_bytes()).hexdigest(),
        observation_ABI='unchanged210, term-major3x70: newest q_rel arm66:72, oldest targetX189, newest203',
        feature_ABI='original128latent + centered observable targetX/.02 + finite-difference observable targetXrate/.02 + locally linearized observable q-to-X error/.02',
        residual='zero131->1 head, tanh times fixed18D basis (zero legs), added before native Gaussian update/sample/logprob; original18D MLP fully trainable',
        unchanged=['reward27','Uniform.3/.7','learning_rate','entropy.005','nativeGAE_PPO','18Dphysics','PD_DR','safety','formal_evaluator'])
    write(p/'SYNERGY_RECIPE.json',recipe)
    write(p/'MECHANISM_DECISION.json',dict(chosen='A: Uniform-P1ArmSynergyMean-v1',hypothesis='Weak/compensating target-to-mean joint coordination; safe real control and higher original total reward exist. Test expression pathway first, not confirmed unique cause.',
        why_not_actuator='Use measured requested/delayed/actual harmonic transfer and torque equation in P1 causal evidence',
        teacher_and_BC_used=False,reward_changed=False,source=evidence['original_FK_source_sha256'],recipe=recipe))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);v=a.parse_args();main(pathlib.Path(v.root))
