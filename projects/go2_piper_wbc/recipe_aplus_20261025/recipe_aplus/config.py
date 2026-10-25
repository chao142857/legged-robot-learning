import json
from pathlib import Path
import torch
from recipe_v1.commands import StageAVelocityCommand
from recipe_v1.config import build_recipe
from task_suite.config import profile_signature


class APlusVelocityCommand(StageAVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.aplus=env.cfg.recipe_aplus['sampling']
        self.kind=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self.sample_counts=torch.zeros(12,dtype=torch.long,device=env.device)

    def pick(self,ids,random_initial_phase=False):
        n=len(ids);p=self.aplus
        # Balance large reset batches without biasing n=1 resamples toward forward.
        rotation=torch.randint(0,4,(),device=self.device)
        group=(torch.arange(n,device=self.device)+rotation)%4
        group=group[torch.randperm(n,device=self.device)]
        low=torch.rand(n,device=self.device)<p['low_probability_given_nonzero']
        zero=torch.rand(n,device=self.device)<p['zero_probability']
        speed=torch.empty(n,device=self.device).uniform_(*p['medium_linear_abs_mps'])
        yaw=torch.empty(n,device=self.device).uniform_(*p['medium_yaw_abs_radps'])
        speed=torch.where(low,p['low_linear_abs_mps'],speed)
        yaw=torch.where(low,p['low_yaw_abs_radps'],yaw)
        desired=torch.zeros((n,3),device=self.device)
        desired[:,0]=torch.where(group==0,speed,torch.where(group==1,-speed,0.))
        desired[:,2]=torch.where(group==2,yaw,torch.where(group==3,-yaw,0.))
        desired[zero]=0.
        kind=torch.where(zero,0,torch.where(low,1,2))
        self.desired[ids]=desired;self.direction[ids]=group;self.kind[ids]=kind
        self.sample_counts+=torch.bincount(kind*4+group,minlength=12)
        duration=torch.empty(n,device=self.device).uniform_(*p['segment_s'])
        if random_initial_phase:duration*=torch.rand(n,device=self.device)
        self.remaining[ids]=duration.clamp_min(self.env.step_dt);self.is_standing_env[ids]=zero


def build_aplus(*,admission,num_envs,seed=21):
    cfg=build_recipe(admission=admission,num_envs=num_envs,seed=seed)
    cfg.recipe_aplus=json.loads(Path(__file__).with_name('spec.json').read_text())
    cfg.commands.base_velocity.class_type=APlusVelocityCommand
    cfg.task_suite['recipe_aplus']=cfg.recipe_aplus
    cfg.task_suite_signature=profile_signature(cfg.task_suite)
    return cfg
