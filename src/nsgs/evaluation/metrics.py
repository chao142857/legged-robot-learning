import torch,numpy as np
from nsgs.common import PROTOCOL

def physical_errors(env,kind):
    from isaaclab.utils.math import quat_apply,quat_apply_inverse,quat_mul,quat_error_magnitude
    ee=env.command_manager.get_term('ee_pose');v=env.command_manager.get_term('base_velocity');robot=ee.robot
    ee_pos=robot.data.body_pos_w[:,ee.body_idx];ee_quat=robot.data.body_quat_w[:,ee.body_idx]
    idx=ee.root_idx if kind=='flat' else ee.link0_idx
    root_pos=robot.data.body_pos_w[:,idx];root_quat=robot.data.body_quat_w[:,idx]
    cmd=ee.command
    if kind=='flat':pos_error=torch.linalg.vector_norm(root_pos+quat_apply(root_quat,cmd[:,:3])-ee_pos,dim=-1)
    else:
        current=quat_apply_inverse(root_quat,ee_pos-root_pos)
        delta=torch.cat((current[:,:2]-cmd[:,:2],ee_pos[:,2:3]-cmd[:,2:3]),dim=-1)
        pos_error=torch.linalg.vector_norm(delta,dim=-1)
    ori_error=quat_error_magnitude(quat_mul(root_quat,cmd[:,3:]),ee_quat)
    vel_error=torch.linalg.vector_norm(v.command[:,:2]-robot.data.root_lin_vel_b[:,:2],dim=-1)
    yaw_error=torch.abs(v.command[:,2]-robot.data.root_ang_vel_b[:,2])
    return torch.stack((vel_error,yaw_error,pos_error,ori_error),dim=1)

class Metrics:
    def __init__(self,env,kind):
        self.env,self.kind=env,kind;n=env.num_envs;d=env.device
        self.age=torch.zeros(n,2,device=d,dtype=torch.long)
        self.scored=torch.zeros(n,device=d,dtype=torch.long);self.good=torch.zeros_like(self.scored)
        self.sq=torch.zeros(n,4,device=d,dtype=torch.float64)
        self.tol=torch.tensor(list(PROTOCOL['thresholds'].values()),device=d,dtype=torch.float64)
        self.settle_steps=round(PROTOCOL['settle_seconds']/env.step_dt)
        self.error_min=torch.full((4,),float('inf'),device=d);self.error_max=torch.zeros(4,device=d)
    def reset(self,ids):self.age[ids]=0;self.scored[ids]=0;self.good[ids]=0;self.sq[ids]=0
    def command(self,kind,ids):self.age[ids,0 if kind=='ee' else 1]=0
    def capture(self):
        self.age+=1;e=physical_errors(self.env,self.kind)
        if not bool(torch.isfinite(e).all()):raise FloatingPointError('Nonfinite physical tracking metric')
        eligible=(self.age>=self.settle_steps).all(1)
        self.scored+=eligible;self.good+=eligible&(e<=self.tol).all(1)
        self.sq+=e.double().square()*eligible[:,None]
        self.error_min=torch.minimum(self.error_min,e.min(0).values);self.error_max=torch.maximum(self.error_max,e.max(0).values)
    def finish(self,ids):
        n=self.scored[ids];good=self.good[ids];rmse=torch.sqrt(self.sq[ids]/n.clamp(min=1)[:,None])
        failure=self.env.reset_terminated[ids];timeout=self.env.reset_time_outs[ids]
        success=(~failure)&(n>=PROTOCOL['min_scored_steps'])&(good.double()>=PROTOCOL['joint_fraction']*n)&(rmse<=self.tol).all(1)
        data=torch.cat((n[:,None].double(),good[:,None].double(),rmse,success[:,None].double(),failure[:,None].double(),timeout[:,None].double()),dim=1)
        return data.cpu().numpy()
    def state_dict(self):return {k:getattr(self,k).clone().cpu() for k in ('age','scored','good','sq','error_min','error_max')}
    def load_state_dict(self,s):
        for k,v in s.items():getattr(self,k).copy_(v.to(self.env.device))

def unit_checks(device):
    from isaaclab.utils.math import quat_from_euler_xyz, quat_error_magnitude
    z=torch.zeros(1,device=device);q=quat_from_euler_xyz(z,z,z);pi=torch.full_like(z,np.pi)
    assert torch.allclose(quat_error_magnitude(q,q),z,atol=1e-6)
    assert abs(float(quat_error_magnitude(q,quat_from_euler_xyz(pi,z,z)))-np.pi)<1e-5
    assert abs(float(torch.linalg.vector_norm(torch.tensor([[.03,.04,0.]],device=device),dim=-1))-.05)<1e-7
    return dict(passed=True,position_3_4_cm_vector_norm_m=.05,quaternion_pi_rotation_rad=float(quat_error_magnitude(q,quat_from_euler_xyz(pi,z,z))),velocity_frame='native root body frame m/s, rad/s',scoring_tolerances=PROTOCOL['thresholds'])
