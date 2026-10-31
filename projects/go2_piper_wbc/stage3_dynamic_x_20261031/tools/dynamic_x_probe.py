"""Actor target counterfactuals and bounded physical intervention, diagnostic only.

Reuses the original Piper FK and real delayed actuator taps. No replay restore,
sampling-after-the-fact policy claim, reward substitution or learned-weight edit.
"""
import inspect,json,math,pathlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import quat_from_euler_xyz
from recipe_v1.kinematics import fk
from telemetry import physical,write_json,plain
from response_tap import ResponseTap

class Probe:
    def __init__(self,env,runner,observer,core,out):
        self.env,self.raw,self.runner,self.out=env,env.unwrapped,runner,out
        self.config=json.loads(pathlib.Path('/stage2-config.json').read_text())['probe']
        self.chain=json.loads(pathlib.Path(self.config['chain_file']).read_text())['chain']
        self.tap=ResponseTap(self.raw,runner,observer,out)
        self.robot=self.raw.scene['robot'];self.term=self.raw.action_manager.get_term('joint_pos');self.pose=self.raw.command_manager.get_term('ee_pose')
        self.ids=self.term._joint_ids;self.arm=[self.term._joint_names.index('joint'+str(i)) for i in range(1,7)]
        assert self.arm==list(range(12,18)),self.term._joint_names
        self.policy=runner.get_inference_policy(device=self.raw.device);self.n=self.raw.num_envs;self.step=0
        self.correction=np.zeros((self.n,6));self.last={};self.counter_rows=[];self.rejected=0
        self.actual_step=env.step;env.step=self.step_env
        self.original_physical=core.physical_suite
        def observed(raw):
            p=self.original_physical(raw);p.update(self.tap.physical());p.update(self.last)
            p['diag_contact_forces_history_w']=raw.scene['contact_forces'].data.net_forces_w_history.clone()
            return p
        core.physical_suite=observed
        offset=0;self.command_slice=None
        for name,shape in zip(self.raw.observation_manager.active_terms['policy'],self.raw.observation_manager.group_obs_term_dim['policy']):
            width=math.prod(shape)
            if name=='pos_commands':self.command_slice=(offset,offset+width)
            offset+=width
        assert offset==210 and self.command_slice is not None
        write_json(out/'probe_contract.json',dict(config=self.config,policy_observation_dim=210,critic_dim=102,command_slice=self.command_slice,
            arm_columns=self.arm,native_joint_names=self.term._joint_names,physics_recovery='Independent cold starts and actual action prefix, not PhysX internal-state restoration',
            intervention='Diagnostic arm action addition only; leg actions unchanged. Not a PPO policy success.',
            source_reuse='Original recipe_v1.kinematics.fk and prior read-only ResponseTap',
            mean_policy=type(runner.alg.actor).__module__+'.'+type(runner.alg.actor).__name__))
        (out/'installed_actor_source.py').write_text(inspect.getsource(type(runner.alg.actor)))
        write_json(out/'actor_module_layout.json',dict(repr=str(runner.alg.actor),parameters=[dict(name=n,shape=list(p.shape)) for n,p in runner.alg.actor.named_parameters()]))
        realized=dict(root_state=plain(self.robot.data.root_state_w),joint_q=plain(self.robot.data.joint_pos),joint_qd=plain(self.robot.data.joint_vel),actuators={})
        for name,act in self.robot.actuators.items():
            realized['actuators'][name]={key:plain(getattr(act,key)) for key in ('stiffness','damping','effort_limit','velocity_limit') if hasattr(act,key)}
            realized['actuators'][name]['position_delay_lags']=plain(act.positions_delay_buffer.time_lags)
        for name in ('get_masses','get_coms','get_inertias','get_material_properties'):
            if hasattr(self.robot.root_physx_view,name):
                try:realized[name]=plain(getattr(self.robot.root_physx_view,name)())
                except (AttributeError,NotImplementedError) as error:realized[name]='NOT_MEASURED:'+type(error).__name__
            else:realized[name]='API_UNAVAILABLE_NOT_MEASURED'
        write_json(out/'realized_physics_start.json',realized)

    def jacobian(self,q):
        t=fk(self.chain,q);h=1e-5;columns=[]
        for j in range(6):
            changed=fk(self.chain,q+np.eye(6)[j]*h)
            columns.append(np.r_[(changed[:3,3]-t[:3,3])/h,(Rotation.from_matrix(t[:3,:3]).inv()*Rotation.from_matrix(changed[:3,:3])).as_rotvec()/h])
        return t,np.column_stack(columns)

    def capture_input(self,obs):
        self.tap.capture_input(obs)
        self.last=dict(diag_sensitivity_sample=torch.zeros(self.n,device=self.raw.device,dtype=torch.bool),
            diag_intervention_dq=torch.zeros((self.n,6),device=self.raw.device),diag_policy_mean=torch.zeros((self.n,18),device=self.raw.device))
        measured=physical(self.raw)
        self.last.update(diag_target_pre=measured['ee_cmd'].clone(),diag_actual_pre=measured['ee_actual_mixed'].clone(),
            diag_EE_error_pre=(measured['ee_cmd'][:,:3]-measured['ee_actual_mixed']).clone(),
            diag_input_time_s=(self.raw.episode_length_buf*self.raw.step_dt).clone())
        if hasattr(self.pose,'phase0'):
            self.last['diag_command_phase_rad']=self.raw.episode_length_buf*self.raw.step_dt*(2*math.pi/8)+self.pose.phase0
            self.last['diag_command_phase_known']=torch.ones(self.n,device=self.raw.device,dtype=torch.bool)
        elif getattr(self.pose,'_suite',{}).get('ee_mode')=='smooth':
            period=float(self.pose._suite['params']['ee_period_s'])
            self.last['diag_command_phase_rad']=self.raw.episode_length_buf*self.raw.step_dt*(2*math.pi/period)
            self.last['diag_command_phase_known']=torch.ones(self.n,device=self.raw.device,dtype=torch.bool)
        else:
            self.last['diag_command_phase_rad']=torch.zeros(self.n,device=self.raw.device)
            self.last['diag_command_phase_known']=torch.zeros(self.n,device=self.raw.device,dtype=torch.bool)
        for mode in ('position_only','newest','coherent_history'):
            for sign in ('plus','minus'):self.last['diag_'+mode+'_mu_'+sign]=torch.zeros((self.n,18),device=self.raw.device)
        if self.step%25:return
        results={}
        with torch.random.fork_rng(devices=[self.raw.device]):
            for mode in ('position_only','newest','coherent_history'):
                for sign in (-1,1):
                    changed=obs.clone() if hasattr(obs,'clone') else {k:v.clone() for k,v in obs.items()}
                    start,end=self.command_slice;view=changed['policy'][:,start:end].reshape(self.n,-1,7)
                    histories=range(view.shape[1]) if mode=='coherent_history' else [-1]
                    for h in histories:
                        view[:,h,0]=.45+sign*.02
                        pos=view[:,h,:3];z=self.robot.data.body_pos_w[:,self.pose.link0_idx,2]
                        pitch=-torch.atan2(pos[:,2]-z,pos[:,:2].norm(dim=-1))+.6;yaw=torch.atan2(pos[:,1],pos[:,0])
                        if mode!='position_only':view[:,h,3:]=quat_from_euler_xyz(torch.zeros_like(pitch),pitch,yaw)
                    mu=self.policy(changed,stochastic_output=False);assert torch.isfinite(mu).all()
                    results[(mode,sign)]=mu.detach().clone()
        q=self.robot.data.joint_pos[:,self.ids][:,self.arm].detach().cpu().numpy()
        for mode in ('position_only','newest','coherent_history'):
            plus=results[(mode,1)];minus=results[(mode,-1)]
            self.last['diag_'+mode+'_mu_plus']=plus;self.last['diag_'+mode+'_mu_minus']=minus
            offset=torch.as_tensor(self.term._offset,device=plus.device,dtype=plus.dtype).expand_as(plus)
            raw_delta=((plus-minus)[:,self.arm]*.25).cpu().numpy()
            qplus=(plus*.25+offset).clamp(self.term.target_lower,self.term.target_upper)
            qminus=(minus*.25+offset).clamp(self.term.target_lower,self.term.target_upper)
            soft_delta=(qplus-qminus)[:,self.arm]
            tau=float(self.term.tau);alpha=(self.term.elapsed/tau).clamp(0,1) if tau>0 else torch.ones(self.n,device=plus.device)
            alpha=alpha.square()*(3-2*alpha)
            delta=(soft_delta*alpha[:,None]).cpu().numpy();contrib=[]
            for e in range(self.n):
                t,j=self.jacobian(q[e]);contrib.append(j[0]*delta[e])
            contribution=np.asarray(contrib);self.counter_rows.append(dict(step=self.step,mode=mode,arm_joint_delta_rad=delta.tolist(),X_joint_contribution_m=contribution.tolist(),
                affine_joint_delta_rad=raw_delta.tolist(),soft_target_joint_delta_rad=soft_delta.cpu().numpy().tolist(),policy_18D_mean_delta=(plus-minus).cpu().numpy().tolist(),
                local_X_gain=(contribution.sum(-1)/.04).tolist(),cancellation_ratio=(abs(contribution.sum(-1))/(abs(contribution).sum(-1)+1e-12)).tolist()))
        self.last['diag_sensitivity_sample'][:]=True

    def step_env(self,actions):
        mean=actions.detach().clone();self.last['diag_policy_mean']=mean
        # Baseline and treatment share an exact cold-start prefix; no intervention before4s.
        if self.config['strength']>0:
            q=self.robot.data.joint_pos[:,self.ids][:,self.arm].detach().cpu().numpy()
            cmd=self.pose.command.detach().cpu().numpy();linkq=self.robot.data.body_quat_w[:,self.pose.link0_idx].cpu().numpy();linkp=self.robot.data.body_pos_w[:,self.pose.link0_idx].cpu().numpy()
            age=(self.raw.episode_length_buf*self.raw.step_dt).cpu().numpy()
            for e in range(self.n):
                if age[e]<self.config['start_s']:self.correction[e]=0;continue
                t,j=self.jacobian(q[e]);rw=Rotation.from_quat(linkq[e,[1,2,3,0]]).as_matrix()
                if abs(rw[2,2])<.7:self.rejected+=1;continue
                # Original mixed frame: link0 XY, world Z; original local quaternion goal.
                goal=np.array([cmd[e,0],cmd[e,1],(cmd[e,2]-linkp[e,2]-rw[2,0]*cmd[e,0]-rw[2,1]*cmd[e,1])/rw[2,2]])
                error=np.r_[np.clip(goal-t[:3,3],-.015,.015),np.clip((Rotation.from_matrix(t[:3,:3]).inv()*Rotation.from_quat(cmd[e,[4,5,6,3]])).as_rotvec(),-.05,.05)]
                weights=np.array([1/.03]*3+[1/.15]*3);a=j*weights[:,None];b=error*weights
                dq=a.T@np.linalg.solve(a@a.T+np.eye(6)*.01**2,b)
                if not np.isfinite(dq).all():raise RuntimeError('Nonfinite diagnostic Jacobian direction')
                self.correction[e]+=np.clip(dq*self.config['feedback_gain'],-self.config['increment_rad'],self.config['increment_rad'])
                self.correction[e]=np.clip(self.correction[e],-self.config['max_correction_rad'],self.config['max_correction_rad'])
            alpha=np.clip((age-self.config['start_s'])/.5,0,1)*self.config['strength']
            correction=torch.as_tensor(self.correction*alpha[:,None],device=actions.device,dtype=actions.dtype)
            offset=torch.as_tensor(self.term._offset,device=actions.device,dtype=actions.dtype).expand_as(mean)
            target=mean[:,self.arm]*.25+offset[:,self.arm]
            bounded=(target+correction).clamp(self.term.target_lower[:,self.arm],self.term.target_upper[:,self.arm])
            actual_delta=bounded-target;actions=mean.clone();actions[:,self.arm]+=actual_delta/.25
            self.last['diag_intervention_dq']=actual_delta
        assert torch.equal(actions[:,:12],mean[:,:12])
        result=self.actual_step(actions);self.step+=1;return result

    def finish(self,data,summary):
        write_json(self.out/'target_sensitivity.json',dict(counterfactuals=self.counter_rows,rejected_unstable_geometry=self.rejected,actor_unchanged=summary['actor_unchanged'],
            measured_PD_torque_equation_max_error=self.tap.max_pd_error,
            scope='Same actual body/joint history, altered observable target only; local Jacobian projections are not physical success. Real intervention trajectories and original metrics are separate.'))
        self.tap.close()

def install(env,runner,observer,core,out):return Probe(env,runner,observer,core,out)
