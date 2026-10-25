"""Focused 5cm/eight-riser PPO demo; unchanged validated v6 robot and controls."""
import gymnasium as gym
import torch
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
from z1_team_cfg import Z1TeamLegSafeCfg
from z1_cfg import Z1Env
from stair_terrain import StaircaseCfg
from basic_locomotion_isaaclab.tasks.locomotion.agents.rsl_rl_ppo_cfg import RoughPPORunnerCfg
from basic_locomotion_isaaclab.tasks import custom_events

TASK='Locomotion-Z1-Team-Stairs-Demo-5cm'
def tensor(x):return x if torch.is_tensor(x) else x.torch

def demo_commands(env,env_ids=None):
    # Native observation construction resamples commands too. Override it in this
    # demo process so none of the first500 environments receive zero commands.
    env._commands[:]=0.;env._commands[:,0]=env.cfg.demo_forward_speed

custom_events._get_new_random_commands=demo_commands

@configclass
class Z1StairsDemoCfg(Z1TeamLegSafeCfg):
    demo_step_height:float=.05
    demo_forward_speed:float=.35
    demo_completion_bonus:float=10.
    demo_progress_reward_per_m:float=2.
    demo_lane_penalty_per_s:float=.5
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s=18.
        self.terrain.terrain_generator=TerrainGeneratorCfg(size=(8,3),num_rows=1,num_cols=1,
            border_width=2,seed=42,use_cache=False,curriculum=False,
            sub_terrains={'continuous_stairs':StaircaseCfg(step_height=self.demo_step_height)})
        self.terrain.max_init_terrain_level=0
        # Start with the nominal acceptance domain; motor/action dynamics unchanged.
        for field in vars(self.events):
            if not field.startswith('_'):setattr(self.events,field,None)
        self.observation_noise_model=None

class Z1StairsDemoEnv(Z1Env):
    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        self._demo_best_x=torch.zeros(self.num_envs,device=self.device)
        self._demo_won=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
    def _commands_forward(self):
        self._commands[:]=0.;self._commands[:,0]=self.cfg.demo_forward_speed
    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:ids=torch.arange(self.num_envs,device=self.device)
        else:ids=tensor(env_ids).long()
        self.episode_length_buf[ids]=0
        state=tensor(self._robot.data.default_root_state)[ids].clone()
        state[:,:3]+=self._terrain.env_origins[ids]
        state[:,3:7]=torch.tensor([0.,0.,0.,1.],device=self.device);state[:,7:]=0.
        q=tensor(self._robot.data.default_joint_pos)[ids].clone()+.03*torch.randn((len(ids),12),device=self.device)
        self._robot.write_root_pose_to_sim(state[:,:7],env_ids=ids)
        self._robot.write_root_velocity_to_sim(state[:,7:],env_ids=ids)
        self._robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)
        if hasattr(self,'_demo_best_x'):self._demo_best_x[ids]=0.;self._demo_won[ids]=False
        self._commands_forward()
    def _pre_physics_step(self,actions):
        self._commands_forward();super()._pre_physics_step(actions)
    def _get_dones(self):
        died,timeout=super()._get_dones()
        pos=tensor(self._robot.data.root_pos_w)-self._terrain.env_origins
        feet=tensor(self._robot.data.body_pos_w)[:,self._feet_ids_robot]-self._terrain.env_origins[:,None,:]
        forces=tensor(self._contact_sensor.data.net_forces_w)[:,self._feet_contact_sensor_ids]
        upright=tensor(self._robot.data.projected_gravity_b)[:,2]<-.75
        supported=(forces.norm(dim=-1)>5).any(dim=1)
        won=(pos[:,0]>=3.7)&(feet[:,:,0].amin(dim=1)>3.3)&upright&supported&(pos[:,1].abs()<.8)&~died
        self._demo_won=won
        self.extras.setdefault('log',{})['Demo/current_full_segment_completion_fraction']=won.float().mean()
        return died|(pos[:,1].abs()>.8),timeout|won
    def _get_rewards(self):
        reward=super()._get_rewards()
        pos=tensor(self._robot.data.root_pos_w)-self._terrain.env_origins
        x=pos[:,0].clamp(0.,3.7)
        progress=(x-self._demo_best_x).clamp(min=0.)
        self._demo_best_x=torch.maximum(self._demo_best_x,x)
        reward=reward+self.cfg.demo_progress_reward_per_m*progress+self.cfg.demo_completion_bonus*self._demo_won
        reward=reward-self.cfg.demo_lane_penalty_per_s*pos[:,1].square()*self.step_dt
        return reward

gym.register(TASK,entry_point=Z1StairsDemoEnv,disable_env_checker=True,
    kwargs={'env_cfg_entry_point':Z1StairsDemoCfg,'rsl_rl_cfg_entry_point':RoughPPORunnerCfg})

def register():
    """Native launcher callback; module import registers only this demo task."""
    return None
