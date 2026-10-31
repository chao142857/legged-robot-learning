"""CPU-only frozen-weight audit on real P1 states; never a physical success score."""
import argparse,hashlib,importlib.util,pathlib
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F
from gpu_budget import read,write

def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def mean(s,x):
    h=x
    for i in (0,2,4):h=F.elu(F.linear(h,s['mlp.'+str(i)+'.weight'],s['mlp.'+str(i)+'.bias']))
    base=F.linear(h,s['mlp.6.weight'],s['mlp.6.bias'])
    if 'synergy_head.weight' not in s:return base,torch.zeros_like(base)
    target=x[:,203:204];rate=(target-x[:,189:190])/.04
    local=s['synergy_x_reference']+((x[:,66:72]-s['synergy_q_reference'])*s['synergy_Jx']).sum(-1,keepdim=True)
    features=torch.cat((h,(target-.45)/.02,rate/.02,(target-local)/.02),-1)
    residual=torch.tanh(F.linear(features,s['synergy_head.weight'],s['synergy_head.bias']))*s['synergy_basis']
    return base+residual,residual

def main(p,additional):
    torch.set_num_threads(2);assert not torch.cuda.is_initialized(),'CPU diagnostic must not initialize CUDA'
    c=read(p/'CPU_RESPONSE_CONFIG.json');sp=importlib.util.spec_from_file_location('originalFK',c['kinematics_source']);kin=importlib.util.module_from_spec(sp);sp.loader.exec_module(kin)
    chain=read(c['chain_file'])['chain'];r=pathlib.Path(read(p/'PROBE_INDEX.json')['P1-parent-X-s0.0'])
    with np.load(r/'evaluation/trajectories.npz',allow_pickle=False) as d:
        indices=[(t,e) for t in range(50,1000,50) for e in range(0,32,8)]
        obs=np.array([d['diag_actor_obs'][t,e] for t,e in indices]);q=np.array([d['diag_q_pre'][t,e,12:] for t,e in indices]);z=np.array([d['kin_link0_pos'][t-1,e,2] for t,e in indices]);recorded=np.array([d['diag_policy_mean'][t,e] for t,e in indices])
    jac=[]
    for value in q:
        t=kin.fk(chain,value);jac.append([(kin.fk(chain,value+np.eye(6)[i]*1e-5)[0,3]-t[0,3])/1e-5 for i in range(6)])
    jac=np.array(jac);x=torch.tensor(obs);models=read(p/'CHECKPOINTS.json');paths={'parent':pathlib.Path(models['parent']['checkpoint'])}
    for name in ('synergy','matchedNative'):
        f=p/(name+str(additional)+'-seed21_RUN.json')
        if f.exists():
            run=pathlib.Path(read(f)['run']);result=run/'ppo/result.json'
            if result.exists():paths[name]=run/'ppo/checkpoints'/pathlib.Path(read(result)['checkpoint']['path']).name
    rows=[]
    with torch.no_grad():
        for name,path in paths.items():
            s=load(path)['actor_state_dict'];actual,residual=mean(s,x)
            if name=='parent':assert float(abs(actual-torch.tensor(recorded)).max())<1e-4
            item=dict(label=name,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),same_real_state_samples=len(indices),counterfactuals=[])
            if 'synergy_head.weight' in s:item['head_coefficients']=dict(latent_l2=float(torch.linalg.vector_norm(s['synergy_head.weight'][0,:128])),targetX=float(s['synergy_head.weight'][0,128]),targetXrate=float(s['synergy_head.weight'][0,129]),local_X_error=float(s['synergy_head.weight'][0,130]),bias=float(s['synergy_head.bias'][0]),actual_residual_raw_action_RMS_per_action=residual.square().mean(0).sqrt().tolist())
            for mode in ('position_only','coherent_aim_pose'):
                means=[];residuals=[]
                for sign in (-1,1):
                    changed=x.clone();view=changed[:,189:210].reshape(len(x),3,7)
                    for h in ([-1] if mode=='position_only' else range(3)):
                        view[:,h,0]=.45+sign*.02
                        if mode!='position_only':
                            v=view[:,h,:3].numpy();pitch=-np.arctan2(v[:,2]-z,np.linalg.norm(v[:,:2],axis=-1))+.6;yaw=np.arctan2(v[:,1],v[:,0]);quat=Rotation.from_euler('xyz',np.column_stack((np.zeros_like(pitch),pitch,yaw))).as_quat()[:,[3,0,1,2]];view[:,h,3:]=torch.tensor(quat,dtype=torch.float32)
                    m,r0=mean(s,changed);means.append(m.numpy());residuals.append(r0.numpy())
                contribution=jac*.25*(means[1]-means[0])[:,12:];head=jac*.25*(residuals[1]-residuals[0])[:,12:]
                item['counterfactuals'].append(dict(mode=mode,affine_local_J_X_signed_gain_mean=float(contribution.sum(-1).mean()/.04),affine_local_J_X_head_gain_mean=float(head.sum(-1).mean()/.04),cancellation_ratio_median=float(np.median(abs(contribution.sum(-1))/(abs(contribution).sum(-1)+1e-12))),positive_X_fraction=float((contribution.sum(-1)>0).mean()),signed_contribution_m_mean_per_arm_joint=contribution.mean(0).tolist()))
            rows.append(item)
    assert not torch.cuda.is_initialized()
    write(p/('CPU_POLICY_SENSITIVITY_'+str(1800+additional)+'.json'),dict(rows=rows,cpu_only=True,old_parent_forward_matches_actual_GPU_tap=True,physical_response_or_success_claim=False,scope='Same real P1 state/history and observable target changes. Fixed original FK Jacobian projection of requested affine joint target, not actual closed-loop performance. No training or heldout seeds used.'))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--additional',type=int,required=True);v=a.parse_args();main(pathlib.Path(v.root),v.additional)
