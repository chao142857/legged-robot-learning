"""Native PPO with bounded updates, exact original CP restore and read-only diagnostics."""
import argparse, hashlib, inspect, json, math, statistics, time
from pathlib import Path
START=time.perf_counter()
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--audit-output',type=Path,required=True)
p.add_argument('--arm',choices=['A','B1'],required=True)
args,native=p.parse_known_args()
from local_gate import validate
gate=validate()
assert Path(native[native.index('--checkpoint')+1]).resolve()==Path(gate['checkpoint']).resolve()
import torch
from rsl_rl.runners import OnPolicyRunner
from sampling_short import exposure_snapshot
from sampler_audit import terrain_groups
original_learn=OnPolicyRunner.learn

class BoundedStop(Exception): pass
def finite(tree):
    if torch.is_tensor(tree): return bool(torch.isfinite(tree).all())
    if isinstance(tree,dict): return all(finite(v) for v in tree.values())
    if isinstance(tree,(tuple,list)): return all(finite(v) for v in tree)
    return True
def exact(a,b):
    if torch.is_tensor(a): return torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict): return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b
def tensor(v): return v if torch.is_tensor(v) else v.torch
def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def learn(self,num_learning_iterations,init_at_random_ep_len=False):
    required=100
    assert num_learning_iterations==required and self.env.num_envs==4096
    assert self.cfg['num_steps_per_env']==24
    e=self.env.unwrapped
    assert e.cfg.standing_env_fraction==.061 and e.cfg.standing_seed==42
    assert e.cfg.z1_contact_profile=='leg_safe'
    expected_height=.373 if args.arm=='A' else .384
    assert abs(e.cfg.desired_base_height-expected_height)<1e-12
    assert e.cfg.robot.init_state.joint_pos=={'.*_hip_joint':0.,'.*_thigh_joint':.9,'.*_calf_joint':-1.8}
    assert e.cfg.sim.dt==.005 and e.cfg.decimation==4 and e.cfg.sim.physics.num_substeps==2
    assert str(e.device)=='cuda:0'
    obs=self.env.get_observations()
    assert obs['policy'].shape[-1]==260 and obs['critic'].shape[-1]==321 and self.env.num_actions==12
    saved=torch.load(gate['checkpoint'],map_location='cpu',weights_only=False)
    opt=self.alg.optimizer
    assert self.current_learning_iteration==5700
    assert exact(self.alg.actor.state_dict(),saved['actor_state_dict'])
    assert exact(self.alg.critic.state_dict(),saved['critic_state_dict'])
    assert exact(opt.state_dict(),saved['optimizer_state_dict']), 'Adam moments/steps/LR not exactly restored'
    restored_lr=opt.param_groups[0]['lr']
    assert abs(restored_lr-.0002562890625)<1e-15
    assert abs(self.alg.learning_rate-restored_lr)<1e-15, 'Adaptive LR scalar must agree with restored Adam LR'
    from basic_locomotion_isaaclab.tasks import custom_events
    fnsource=inspect.getsource(custom_events._get_new_random_commands)
    assert 'from sampling_short import apply_standing' in fnsource
    result={'scope':'height_reward_ab_100_bounded','arm':args.arm,'desired_base_height':expected_height,'gate':gate,'passed':False,'num_envs':4096,'rollout':24,
        'restore_exact_actor_critic_optimizer':True,'initial_adam_lr':restored_lr,
        'initial_adaptive_lr':self.alg.learning_rate,'initial_adam_steps':sorted(set(float(s['step']) for s in opt.state.values())),
        'initialization_s':time.perf_counter()-START,'log_dir':self.logger.log_dir,
        'task':f'Locomotion-Z1-Team-Stance-{args.arm}-Rough-LegSafe',
        'sampler_source':fnsource,'initial_exposure':exposure_snapshot(e),
        'standing_definition':'Bernoulli .061 by explicit seed42/envID/reset ordinal; held per episode; no global RNG consumed',
        'scheduled_stop_restart_preserved':True,
        'limits_note':'Provisional DC120Nm/20rad/s envelope; Newton does not enforce hard joint velocity limit. Motor samples at50Hz, not every physical substep.',
        'max_additional_updates':required,'wall_bound_s':900,
        'baseline_asset_reward_gait_control_unchanged':True,'updates':[]}
    before_actor={k:v.detach().clone() for k,v in self.alg.actor.state_dict().items()}
    before_critic={k:v.detach().clone() for k,v in self.alg.critic.state_dict().items()}
    names,labels=terrain_groups(e)
    exposure=torch.zeros(len(names),device=e.device,dtype=torch.long)
    standing_exposure=exposure.clone()
    counts=torch.zeros(7,device=e.device,dtype=torch.long)
    peaks=torch.zeros(3,device=e.device)
    rows=result['updates']; state={'steps':0,'kl':[],'lr_used':[],'grad_finite':[],'index':0,'slow':0,'height_frames':[]}
    act=self.alg.act; returns=self.alg.compute_returns; update=self.alg.update; log=self.logger.log
    envstep=self.env.step; getkl=self.alg.actor.get_kl_divergence; optstep=opt.step
    def act_wrapper(*a,**kw):
        if state['steps']==0:
            torch.cuda.synchronize();state['start']=time.perf_counter()
            counts.zero_();peaks.zero_();state['kl']=[];state['lr_used']=[];state['grad_finite']=[];state['height_frames']=[]
        state['steps']+=1
        exposure.add_(torch.bincount(labels[e._commands.norm(dim=1)>.01],minlength=len(names)))
        standing_exposure.add_(torch.bincount(labels[e._standing_short],minlength=len(names)))
        return act(*a,**kw)
    def step_wrapper(*a,**kw):
        value=envstep(*a,**kw)
        tau=tensor(e._robot.data.applied_torque).abs();vel=tensor(e._robot.data.joint_vel).abs()
        q=tensor(e._robot.data.joint_pos);limits=tensor(e._robot.data.joint_pos_limits)
        violation=torch.maximum(limits[:,:,0]-q,q-limits[:,:,1]).clamp_min(0.)
        counts[0]+=tau.numel();counts[1]+=(tau>120.001).sum();counts[2]+=(vel>20.01).sum()
        counts[3]+=sum((~torch.isfinite(v)).sum() for v in (tau,vel,q,value[0]['policy'],value[0]['critic'],value[1]))
        forces=tensor(e._contact_sensor.data.net_forces_w).norm(dim=-1)
        counts[4]+=(forces[:,e._calf_contact_sensor_ids]>5.).sum()
        counts[5]+=4096*len(e._calf_contact_sensor_ids)
        counts[6]+=(violation>.01).sum()
        peaks.copy_(torch.maximum(peaks,torch.stack([tau.max(),vel.max(),violation.max()])))
        # Read-only actual height, using exactly the finite/clamped ray mean of track_height_exp.
        hits=tensor(e._pose_height_scanner.data.ray_hits_w)[...,2]
        z=torch.nan_to_num(hits,nan=0.,posinf=1.,neginf=-1.).clamp(-5,5)
        h=tensor(e._robot.data.root_pos_w)[:,2]-z.mean(1)
        # Exclude reset frames and first5 episode steps; no effect on rollout or PPO.
        valid=(~value[2].bool()) & (e.episode_length_buf>5)
        state['height_frames'].append(torch.stack([h,z.amax(1)-z.amin(1),
            e._commands[:,:3].norm(dim=1),valid.to(h.dtype)],dim=1).detach().clone())
        return value
    def kl_wrapper(*a,**kw):
        value=getkl(*a,**kw);state['kl'].append(value.mean().detach());return value
    def optstep_wrapper(*a,**kw):
        state['lr_used'].append([g['lr'] for g in opt.param_groups])
        state['grad_finite'].append(torch.stack([torch.isfinite(p.grad).all() for g in opt.param_groups for p in g['params'] if p.grad is not None]).all())
        return optstep(*a,**kw)
    def returns_wrapper(*a,**kw):
        torch.cuda.synchronize();state['rollout_end']=time.perf_counter()
        value=returns(*a,**kw);torch.cuda.synchronize();state['returns_end']=time.perf_counter();return value
    def update_wrapper(*a,**kw):
        value=update(*a,**kw);torch.cuda.synchronize();state['update_end']=time.perf_counter()
        assert finite(value) and finite(self.alg.actor.state_dict()) and finite(self.alg.critic.state_dict()) and finite(opt.state_dict()), 'Nonfinite PPO state'
        assert all(bool(v) for v in state['grad_finite']), 'Nonfinite gradient'
        return value
    def log_wrapper(*a,**kw):
        log(*a,**kw);torch.cuda.synchronize();end=time.perf_counter()
        c=counts.cpu().tolist();peak=peaks.cpu().tolist();kl=torch.stack(state['kl']).cpu().tolist()
        row={'additional_update':len(rows)+1,'iteration':kw['it'],'rollout_s':state['rollout_end']-state['start'],
             'ppo_s':state['update_end']-state['returns_end'],'full_update_s':end-state['start'],
             'losses':{k:float(v) for k,v in kw['loss_dict'].items()},'kl_minibatch_means':kl,
             'kl_mean':statistics.mean(kl),'kl_max':max(kl),'adam_lr_used_each_minibatch':state['lr_used'],
             'final_adam_lr':[g['lr'] for g in opt.param_groups],'final_adaptive_lr':self.alg.learning_rate,
             'motor_joint_samples':c[0],'torque_over_120_count':c[1],'speed_over_20_count':c[2],
             'nonfinite_sample_count':c[3],'calf_contact_samples_5N':c[4],'calf_samples':c[5],
             'joint_position_over_limit_01_count':c[6],'peak_torque_Nm':peak[0],
             'peak_speed_rad_s':peak[1],'peak_joint_limit_excess_rad':peak[2],
             'gradients_finite':True,'exposure':exposure_snapshot(e)}
        assert finite(row['losses']) and all(math.isfinite(v) for v in row['losses'].values())
        hf=torch.stack(state['height_frames'])
        row['measured_height_by_terrain']={}
        for ii,name in enumerate(names):
            vv=hf[:,labels==ii,:].reshape(-1,4)
            vv=vv[vv[:,3]>.5]
            if not len(vv): continue
            row['measured_height_by_terrain'][name]={
                'valid_control_samples':len(vv),
                'height_mean_m':float(vv[:,0].mean()),
                'height_p10_p50_p90_m':torch.quantile(vv[:,0],torch.tensor([.1,.5,.9],device=vv.device)).cpu().tolist(),
                'scanner_span_ge_005_fraction':float((vv[:,1]>=.05).float().mean()),
                'moving_fraction':float((vv[:,2]>.01).float().mean())}
        vv=hf.reshape(-1,4);vv=vv[vv[:,3]>.5]
        row['actual_height_mean_m']=float(vv[:,0].mean()) if len(vv) else None
        row['height_sampling_note']='RootZ minus same clamp/nan processed base ray mean; post-step, excluding reset/first5steps; statistics do not alter reward.'
        rows.append(row);state['steps']=0
        write(args.audit_output.with_suffix('.progress.json'),{'log_dir':self.logger.log_dir,'updates':rows})
        # Additional-update cadence; native absolute-iteration saves remain available too.
        if len(rows)%25==0: self.save(str(Path(self.logger.log_dir)/f'model_{self.current_learning_iteration}.pt'))
        reason=None
        if c[3] or c[1]: reason='nonfinite simulation or torque beyond120.001Nm'
        if peak[1]>30 or c[2]/c[0]>.005 or peak[2]>.5: reason='gross/sustained motor or joint-limit anomaly'
        if len(rows)>=3 and all(r['kl_mean']>.2 for r in rows[-3:]): reason='sustained severe KL excursion'
        if len(rows)>20:
            reference=statistics.median(r['full_update_s'] for r in rows[2:15])
            slow=statistics.median(r['full_update_s'] for r in rows[-10:])>2*reference
            state['slow']=state['slow']+1 if slow else 0
            if state['slow']>=3: reason='persistent throughput below half warm smoke baseline'
        if time.perf_counter()-START>=result['wall_bound_s']: reason='15minute wall budget'
        if reason:
            result['stop_reason']=reason
            self.save(str(Path(self.logger.log_dir)/f'model_{self.current_learning_iteration}.pt'))
            raise BoundedStop(reason)
    self.alg.act=act_wrapper;self.alg.compute_returns=returns_wrapper;self.alg.update=update_wrapper
    self.logger.log=log_wrapper;self.env.step=step_wrapper;self.alg.actor.get_kl_divergence=kl_wrapper;opt.step=optstep_wrapper
    try:
        original_learn(self,num_learning_iterations,init_at_random_ep_len)
        result['passed']=True
    except BoundedStop:
        result['passed']=result.get('stop_reason')=='15minute wall budget'
    except BaseException as ex:
        result['error']=repr(ex);raise
    finally:
        result.update(actual_updates=len(rows),actual_transitions=len(rows)*4096*24,
            final_iteration=self.current_learning_iteration,wall_s=time.perf_counter()-START,
            finite_model_optimizer=finite(self.alg.actor.state_dict()) and finite(self.alg.critic.state_dict()) and finite(opt.state_dict()),
            final_adam_steps=sorted(set(float(s['step']) for s in opt.state.values())),
            changed_actor_tensors=[k for k,v in self.alg.actor.state_dict().items() if not torch.equal(v,before_actor[k])],
            changed_critic_tensors=[k for k,v in self.alg.critic.state_dict().items() if not torch.equal(v,before_critic[k])],
            final_checkpoint=str(Path(self.logger.log_dir)/f'model_{self.current_learning_iteration}.pt'),
            nonzero_command_env_steps={n:int(exposure[i]) for i,n in enumerate(names)},
            standing_env_steps={n:int(standing_exposure[i]) for i,n in enumerate(names)})
        measured=rows[2:]
        if measured:
            result.update(warmup_updates_excluded=2,steady_updates=len(measured),
                mean_full_update_s=statistics.mean(r['full_update_s'] for r in measured),
                steady_transitions_per_s=len(measured)*4096*24/sum(r['full_update_s'] for r in measured))
        write(args.audit_output,result)
    assert result['passed'],result.get('stop_reason',result.get('error'))

OnPolicyRunner.learn=learn
from isaaclab_rl.entrypoints.backends.train_rsl_rl import run
run(['--external_callback','register_stance.register']+native)
