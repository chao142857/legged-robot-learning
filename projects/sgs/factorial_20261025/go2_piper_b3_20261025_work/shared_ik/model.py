"""Batched arm-only damped least squares. CPU-testable; no simulator imports.

Inputs: encoders, IMU gravity, mixed EE command. No task_id/q0/q1, contact forces,
simulator EE pose/Jacobian, base world pose, mass or terrain truth is accepted.
Flat-ground height is estimated from the lowest foot's encoder FK. This explicit
stance assumption must be validated dynamically; it is not a clean state oracle.
"""
import json
import os
import torch
from pathlib import Path
from .settings import IK, LEG_NAMES, ARM_NAMES
KINEMATICS = Path(os.environ.get('SHARED_IK_MODEL', str(Path(__file__).with_name('kinematics.json'))))

def quat_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)

def rotation_error(target,current):
    # Quaternion extraction remains well conditioned at pi; w>=0 selects shortest arc.
    r=target@current.transpose(-1,-2)
    trace=r.diagonal(dim1=-2,dim2=-1).sum(-1)
    candidates=torch.stack([1+trace,1+r[:,0,0]-r[:,1,1]-r[:,2,2],
                            1-r[:,0,0]+r[:,1,1]-r[:,2,2],1-r[:,0,0]-r[:,1,1]+r[:,2,2]],-1).clamp_min(0).sqrt()
    q=torch.stack([
        torch.stack([candidates[:,0]**2,r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]],-1),
        torch.stack([r[:,2,1]-r[:,1,2],candidates[:,1]**2,r[:,1,0]+r[:,0,1],r[:,0,2]+r[:,2,0]],-1),
        torch.stack([r[:,0,2]-r[:,2,0],r[:,1,0]+r[:,0,1],candidates[:,2]**2,r[:,2,1]+r[:,1,2]],-1),
        torch.stack([r[:,1,0]-r[:,0,1],r[:,0,2]+r[:,2,0],r[:,2,1]+r[:,1,2],candidates[:,3]**2],-1)],1)
    q=q/(2*candidates[:,:,None].clamp_min(1e-8))
    q=q[torch.arange(len(r),device=r.device),candidates.argmax(-1)]
    q=torch.where(q[:,:1]<0,-q,q);q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    v=q[:,1:];s=v.norm(dim=-1,keepdim=True)
    return v*(2*torch.atan2(s,q[:,:1].clamp_min(0))/s.clamp_min(1e-9))

class KinematicModel:
    def __init__(self,device='cpu',dtype=torch.float32,settings=None):
        self.settings=dict(IK if settings is None else settings)
        self.device=device;self.dtype=dtype
        if not KINEMATICS.is_file():
            raise FileNotFoundError('Set SHARED_IK_MODEL to an independently authorized kinematics JSON; geometry is excluded from this release')
        kin=json.loads(KINEMATICS.read_text());self.joints={j['name']:j for j in kin['joints']}
        self.frames={n:(self.tensor(j['T0']),torch.linalg.inv(self.tensor(j['T1']))) for n,j in self.joints.items()}
        self.lower=self.tensor([self.joints[n]['lower'] for n in ARM_NAMES])
        self.upper=self.tensor([self.joints[n]['upper'] for n in ARM_NAMES])
        self.mount=self.frames['connector_to_arm_joint'][0]@self.frames['connector_to_arm_joint'][1]

    def tensor(self,x):return torch.as_tensor(x,device=self.device,dtype=self.dtype)

    def identity(self,n):return torch.eye(4,device=self.device,dtype=self.dtype).repeat(n,1,1)

    def motion(self,q,axis):
        a=torch.eye(3,device=q.device,dtype=q.dtype)['XYZ'.index(axis)]
        x,y,z=a;K=torch.stack([a[0]*0,-z,y,z,a[0]*0,-x,-y,x,a[0]*0]).reshape(3,3)
        R=torch.eye(3,device=q.device,dtype=q.dtype)+q.sin()[:,None,None]*K+(1-q.cos())[:,None,None]*(K@K)
        t=self.identity(len(q));t[:,:3,:3]=R;return t

    def fk(self,q,jacobian=False):
        t=self.identity(len(q));origins=[];axes=[]
        for i,n in enumerate(ARM_NAMES):
            a,b=self.frames[n];t=t@a
            origins.append(t[:,:3,3]);axes.append(t[:,:3,'XYZ'.index(self.joints[n]['axis'])])
            t=t@self.motion(q[:,i],self.joints[n]['axis'])@b
        a,b=self.frames['ee_gripper'];t=t@a@b
        if not jacobian:return t
        origin=torch.stack(origins,-1);axis=torch.stack(axes,-1)
        jv=torch.linalg.cross(axis.transpose(1,2),(t[:,:3,3,None]-origin).transpose(1,2)).transpose(1,2)
        return t,torch.cat([jv,axis],1)

    def foot_positions(self,q_leg):
        feet=[]
        for i,leg in enumerate(('FR','FL','RR','RL')):
            t=self.identity(len(q_leg))
            for k,joint in enumerate(('hip','thigh','calf')):
                n=f'{leg}_{joint}_joint';a,b=self.frames[n]
                t=t@a@self.motion(q_leg[:,3*i+k],self.joints[n]['axis'])@b
            a,b=self.frames[f'{leg}_foot_joint'];t=t@a@b
            feet.append(t[:,:3,3]+t[:,:3,:3]@self.tensor(self.settings['foot_center_body_m']))
        return torch.stack(feet,1)

    def estimate_link0_height(self,q_leg,gravity_b):
        up=-gravity_b/gravity_b.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        feet=self.foot_positions(q_leg)
        base_z=self.settings['ground_z_m']+self.settings['foot_radius_m']-(feet*up[:,None]).sum(-1).amin(-1)
        link0_z=base_z+(up*self.mount[:3,3]).sum(-1)
        row=up@self.mount[:3,:3]
        return link0_z,row,base_z

    def mixed_to_local(self,command,link0_z,world_vertical_in_link0):
        row=world_vertical_in_link0
        valid=row[:,2].abs()>=self.settings['minimum_vertical_projection']
        z=(command[:,2]-link0_z-(row[:,:2]*command[:,:2]).sum(-1))/torch.where(valid,row[:,2],torch.ones_like(link0_z))
        return torch.cat([command[:,:2],z[:,None]],-1),quat_matrix(command[:,3:]),valid

    @torch.no_grad()
    def solve(self,q_feedback,position,rotation,lower=None,upper=None,iterations=None):
        lower=self.lower if lower is None else torch.maximum(self.lower,lower)
        upper=self.upper if upper is None else torch.minimum(self.upper,upper)
        q=q_feedback.clone().clamp(lower,upper)
        w=self.settings['orientation_length_m'];damp=self.settings['damping']
        for _ in range(iterations or self.settings['iterations']):
            t,J=self.fk(q,True)
            error=torch.cat([position-t[:,:3,3],w*rotation_error(rotation,t[:,:3,:3])],-1)
            J=J.clone();J[:,3:]*=w
            step=J.transpose(1,2)@torch.linalg.solve(J@J.transpose(1,2)+damp*damp*torch.eye(6,device=q.device,dtype=q.dtype),error[:,:,None])
            step=step.squeeze(-1).clamp(-self.settings['max_iteration_step_rad'],self.settings['max_iteration_step_rad'])
            q=(q+step).clamp(lower,upper)
        t=self.fk(q)
        pos=(t[:,:3,3]-position).norm(dim=-1)
        ori=rotation_error(rotation,t[:,:3,:3]).norm(dim=-1)
        return q,dict(position_residual_m=pos,orientation_residual_rad=ori,
                      converged=(pos<self.settings['position_tolerance_m'])&(ori<self.settings['orientation_tolerance_rad']))

