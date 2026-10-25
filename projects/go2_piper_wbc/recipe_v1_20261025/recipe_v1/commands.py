"""Independent training sampling and frozen first-episode evaluation commands."""
import torch
from isaaclab.utils.math import quat_from_euler_xyz
from task_suite.mdp import SuitePoseCommand
from task_suite.mdp import SuiteVelocityCommand


class CloudPoseCommand(SuitePoseCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        cloud=torch.tensor(env.cfg.recipe_v1['workspace_cloud'],device=env.device)
        self.cloud_targets=cloud[(torch.arange(env.num_envs,device=env.device)//4)%len(cloud)]

    def _update_command(self,env_ids=None,reset_phase=False):
        ids=slice(None) if env_ids is None else env_ids
        p=self.cloud_targets[ids]
        self.pose_command[ids,:3]=p
        z=self.robot.data.body_pos_w[ids,self.link0_idx,2]
        pitch=-torch.atan2(p[:,2]-z,p[:,:2].norm(dim=-1))+.6
        yaw=torch.atan2(p[:,1],p[:,0])
        self.pose_command[ids,3:]=quat_from_euler_xyz(torch.zeros_like(pitch),pitch,yaw)


def aim_pose(term,ids,positions):
    term.pose_command[ids,:3]=positions
    z=term.robot.data.body_pos_w[ids,term.link0_idx,2]
    pitch=-torch.atan2(positions[:,2]-z,positions[:,:2].norm(dim=-1))+.6
    yaw=torch.atan2(positions[:,1],positions[:,0])
    term.pose_command[ids,3:]=quat_from_euler_xyz(torch.zeros_like(pitch),pitch,yaw)


class StageAVelocityCommand(SuiteVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.parameters=env.cfg.recipe_v1['stage_A']
        self.desired=torch.zeros_like(self.vel_command_b)
        self.remaining=torch.zeros(env.num_envs,device=env.device)
        self.direction=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)

    def pick(self,ids,random_initial_phase=False):
        n=len(ids);p=self.parameters
        group=torch.randint(0,4,(n,),device=self.device)
        speed=torch.empty(n,device=self.device).uniform_(*p['linear_speed_abs_range_mps'])
        yaw=torch.empty(n,device=self.device).uniform_(*p['yaw_rate_abs_range_radps'])
        desired=torch.zeros((n,3),device=self.device)
        desired[:,0]=torch.where(group==0,speed,torch.where(group==1,-speed,0.))
        desired[:,2]=torch.where(group==2,yaw,torch.where(group==3,-yaw,0.))
        zero=torch.rand(n,device=self.device)<p['zero_fraction']
        desired[zero]=0.
        self.desired[ids]=desired;self.direction[ids]=group
        duration=torch.empty(n,device=self.device).uniform_(*p['segment_s'])
        if random_initial_phase:duration*=torch.rand(n,device=self.device)
        self.remaining[ids]=duration.clamp_min(self.env.step_dt)
        self.is_standing_env[ids]=zero

    def _resample_command(self,env_ids):
        self.vel_command_b[env_ids]=0.
        self.pick(env_ids,random_initial_phase=True)

    def _update_command(self):
        self.remaining-=self.env.step_dt
        ids=(self.remaining<=0).nonzero(as_tuple=False).flatten()
        if len(ids):self.pick(ids)
        p=self.parameters
        slew=torch.tensor([p['linear_slew_mps2'],p['linear_slew_mps2'],p['yaw_slew_radps2']],device=self.device)*self.env.step_dt
        self.vel_command_b+=(self.desired-self.vel_command_b).clamp(-slew,slew)


class StageAPoseCommand(SuitePoseCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.pool=torch.tensor(env.cfg.recipe_v1['admitted_ee_points'],device=env.device)
        self.remaining=torch.zeros(env.num_envs,device=env.device)
        self.fixed_target=torch.zeros((env.num_envs,3),device=env.device)
        self.hold_range=env.cfg.recipe_v1['stage_A']['fixed_ee_hold_s']

    def pick(self,ids):
        self.fixed_target[ids]=self.pool[torch.randint(len(self.pool),(len(ids),),device=self.device)]
        self.remaining[ids]=torch.empty(len(ids),device=self.device).uniform_(*self.hold_range)

    def _resample_command(self,env_ids):
        self.pick(env_ids)
        aim_pose(self,env_ids,self.fixed_target[env_ids])

    def _update_command(self,env_ids=None,reset_phase=False):
        self.remaining-=self.env.step_dt
        ids=(self.remaining<=0).nonzero(as_tuple=False).flatten()
        if len(ids):self.pick(ids)
        aim_pose(self,slice(None),self.fixed_target)


class FrozenAPoseCommand(SuitePoseCommand):
    """Two fixed goals per env, generated once using a separate CPU generator."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        pool=torch.tensor(env.cfg.recipe_v1['admitted_ee_points'],device=env.device)
        generator=torch.Generator(device='cpu').manual_seed(env.cfg.seed+101)
        indices=torch.randint(len(pool),(2,env.num_envs),generator=generator).to(env.device)
        self.bank=pool[indices]

    def _update_command(self,env_ids=None,reset_phase=False):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else torch.as_tensor(env_ids,device=self.device)
        age=self.env.episode_length_buf[ids]*self.env.step_dt
        if reset_phase:age=torch.zeros_like(age)
        phase=(age/10.).long().clamp_max(1)
        aim_pose(self,ids,self.bank[phase,ids])


class FrozenAVelocityCommand(SuiteVelocityCommand):
    def _update_command(self):
        p=self._suite['params'];desired=torch.zeros_like(self.vel_command_b)
        ids=torch.arange(self.num_envs,device=self.device)
        group=(ids+(self.env.episode_length_buf*self.env.step_dt/10.).long())%4
        desired[:,0]=torch.where(group==0,.45,torch.where(group==1,-.45,0.))
        desired[:,2]=torch.where(group==2,.45,torch.where(group==3,-.45,0.))
        desired[ids%10==9]=0.
        self.desired=desired
        slew=torch.tensor([p['base_slew_linear'],p['base_slew_linear'],p['base_slew_yaw']],device=self.device)*self.env.step_dt
        self.vel_command_b+=(desired-self.vel_command_b).clamp(-slew,slew)
