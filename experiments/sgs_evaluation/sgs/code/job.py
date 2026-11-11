import argparse,hashlib,importlib.metadata,json,os,pathlib,pickle,random,sys,time,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--role',choices=['train','evaluate','baseline','video'],required=True)
p.add_argument('--kind',choices=['flat','wbc'],required=True)
p.add_argument('--mode',choices=['uniform','sgs'],default='uniform')
p.add_argument('--stage',choices=['short','formal'],default='short')
p.add_argument('--updates',type=int,default=200)
p.add_argument('--num_envs',type=int,default=4096)
p.add_argument('--out',required=True);p.add_argument('--checkpoint')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.role=='video':a.enable_cameras=True
launcher=AppLauncher(a);app=launcher.app
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
env=None;adapter=None;runner=None;completed=0;optimizer_calls=0;paused_evaluation_seconds=0.;start=time.time()
try:
    import copy
    import gymnasium as gym,numpy as np,torch
    import LeggedManip_Lab.tasks
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.io import dump_yaml
    from common import ROOT,NATIVE,Pool,PROTOCOL,PARAMETERS,plain,digest,write,emit,rng_state,restore_rng,tensor_hash,file_sha,model_cfg
    from commands import SequenceFlat,SequenceWBC,SequenceVelocity,wbc_equivalence
    from adapter import SequenceAdapter
    from metrics import unit_checks
    task='GO2-PIPER-'+('Flat' if a.kind=='flat' else 'WBC')
    training=a.role=='train'
    if training:
        assert a.num_envs==4096 and a.checkpoint is None
        assert a.updates==(200 if a.stage=='short' else 10000)
        frozen=json.loads((ROOT/'protocol_frozen.json').read_text());assert frozen['definition_sha256']==digest(PROTOCOL)
    cfg,agent=model_cfg(task)
    cfg.scene.num_envs=a.num_envs;cfg.seed=42 if training else (921042 if a.kind=='flat' else 922042)
    cfg.sim.device='cuda:0';cfg.log_dir=str(out/'policy')
    native=plain(cfg.to_dict());agent_native=plain(agent.to_dict())
    cfg.commands.ee_pose.class_type=SequenceFlat if a.kind=='flat' else SequenceWBC
    cfg.commands.base_velocity.class_type=SequenceVelocity
    adapted=plain(cfg.to_dict())
    def differences(x,y,path=''):
        if isinstance(x,dict) and isinstance(y,dict):
            return sum((differences(x[k],y[k],path+'.'+k) for k in x),[])
        return [] if x==y else [dict(path=path,native=x,adapted=y)]
    changes=differences(native,adapted)
    assert sorted(x['path'] for x in changes)==['.commands.base_velocity.class_type','.commands.ee_pose.class_type'],changes
    write(out/'configuration.json',dict(native_environment=native,agent=agent_native,allowed_changes=changes,kind=a.kind,mode=a.mode,stage=a.stage,role=a.role,seed=cfg.seed,success_definition=PROTOCOL,sgs_parameters=PARAMETERS,versions={k:importlib.metadata.version(k) for k in ('isaacsim','isaaclab','rsl-rl-lib','torch')}))
    if a.role=='video':
        cfg.viewer.origin_type='asset_root';cfg.viewer.asset_name='robot';cfg.viewer.eye=(2.5,2.5,1.7);cfg.viewer.lookat=(0.,0.,.4)
    env=gym.make(task,cfg=cfg,render_mode='rgb_array' if a.role=='video' else None)
    pool=Pool(a.kind,'train' if training else 'evaluation')
    adapter=SequenceAdapter(env.unwrapped,a.kind,a.mode if training else 'evaluation',pool,out)
    if a.role=='video':
        env=gym.wrappers.RecordVideo(env,video_folder=str(out/'videos'),step_trigger=lambda i:i==0,video_length=1500,disable_logger=True)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    assert wrapped.num_actions==18
    write(out/'effective_after_reset.json',adapter.effective())
    # Observe the ranges AFTER native curriculum_manager.compute/reset.
    for name in ('ee_pose','base_velocity'):
        term=env.unwrapped.command_manager.get_term(name)
        assert not term.cfg.curriculum_enabled and plain(term.cfg.ranges)==plain(term.cfg.limit_ranges)
    write(out/'unit_checks.json',unit_checks(env.unwrapped.device))
    write(out/'wbc_native_formula_equivalence.json',wbc_equivalence(env.unwrapped.command_manager.get_term('ee_pose')))
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
    class Runner(OnPolicyRunner):
        def save(self,path,infos=None):
            state=dict(completed_updates=completed,environment_transitions=completed*4096*24,adapter=adapter.state_dict(),rng=rng_state(),definition_sha256=digest(PROTOCOL),pool_sha256=pool.sha)
            return super().save(path,infos=state)
    runner=Runner(wrapped,agent.to_dict(),log_dir=str(out/'policy') if training else None,device=agent.device)
    if training:
        assert not runner.alg.optimizer.state and completed==0 and not any(adapter.owner.engine.completed_counts) and not any(adapter.owner.engine.outcomes)
        initial=dict(actor_sha256=tensor_hash(runner.alg.actor.state_dict()),critic_sha256=tensor_hash(runner.alg.critic.state_dict()),optimizer_state_empty=True,history_empty=True,seed=42,pool_sha256=pool.sha,observation_shapes=plain(env.unwrapped.observation_manager.group_obs_dim),actions=wrapped.num_actions)
        write(out/'initialization.json',initial)
        paired=ROOT/'runs'/(a.stage+'-'+a.kind+'-uniform')/'initialization.json'
        if a.mode=='sgs':
            pair=json.loads(paired.read_text())
            assert initial['actor_sha256']==pair['actor_sha256'] and initial['critic_sha256']==pair['critic_sha256'] and initial['pool_sha256']==pair['pool_sha256'],'Uniform/SGS initialization mismatch'
        dump_yaml(str(out/'policy/params/env.yaml'),cfg);dump_yaml(str(out/'policy/params/agent.yaml'),agent)
        # Retain the native optimizer, native learn loop and native save interval.
        original_update=runner.alg.update;original_step=runner.alg.optimizer.step;original_log=runner.logger.log
        def counted_step(*args,**kwargs):
            global optimizer_calls
            optimizer_calls+=1;return original_step(*args,**kwargs)
        runner.alg.optimizer.step=counted_step
        def observed_update():
            global completed
            before=optimizer_calls;t=time.perf_counter();result=original_update();completed+=1
            assert optimizer_calls-before==20,'Native PPO optimizer-call count changed'
            tensors=list(runner.alg.actor.parameters())+list(runner.alg.critic.parameters())
            tensors += [v for s in runner.alg.optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor) and v.is_floating_point()]
            groups={}
            for t in tensors:groups.setdefault(str(t.device),[]).append(torch.isfinite(t).all())
            if not all(bool(torch.stack(v).all()) for v in groups.values()):raise FloatingPointError('Nonfinite model/Adam state after PPO update')
            if not all(np.isfinite(float(v)) for v in result.values()):raise FloatingPointError('Nonfinite PPO loss')
            row=dict(completed_updates=completed,environment_transitions=completed*4096*24,optimizer_calls=optimizer_calls,ppo_update_seconds=time.perf_counter()-t,losses=plain(result),sampler=adapter.summary(),unix=time.time())
            emit(out/'updates.jsonl',row);write(out/'latest.json',dict(completed_updates=completed,environment_transitions=completed*4096*24,unix=time.time(),summary=row['sampler']))
            print('COMPARISON_UPDATE '+json.dumps(dict(stage=a.stage,task=task,mode=a.mode,completed_updates=completed,environment_transitions=row['environment_transitions'],optimizer_calls=optimizer_calls)),flush=True)
            return result
        runner.alg.update=observed_update
        def distribution(update):
            e=adapter.owner.engine
            write(out/'sampling'/('update_'+str(update).zfill(5)+'.json'),dict(update=update,pool_sha256=pool.sha,probabilities=e.probabilities(),rates=e.rates(),sampled=e.sample_counts,completed=e.completed_counts,successes=e.success_counts,history_lengths=[len(h) for h in e.outcomes],summary=adapter.summary()))
        def evaluated_log(*args,**kwargs):
            global paused_evaluation_seconds
            original_log(*args,**kwargs)
            if (a.stage=='formal' and completed%1000==0) or (a.stage=='short' and completed==200):
                cp=out/'checkpoints'/('update_'+str(completed).zfill(5)+'.pt');cp.parent.mkdir(exist_ok=True)
                runner.save(str(cp));distribution(completed)
                before=digest(adapter.owner.state_dict());before_rng=rng_state();steps=adapter.owner.control_steps
                request=out/'evaluation_requests'/(str(completed).zfill(5)+'.json')
                write(request,dict(update=completed,kind=a.kind,mode=a.mode,checkpoint=str(cp),checkpoint_sha256=file_sha(cp),definition_sha256=digest(PROTOCOL),eval_pool_sha256=Pool(a.kind,'evaluation').sha))
                ack=request.with_suffix('.done.json')
                print('EVALUATION_WAIT '+str(request),flush=True)
                wait_started=time.time()
                while not ack.exists():time.sleep(.25)
                paused_evaluation_seconds+=time.time()-wait_started
                answer=json.loads(ack.read_text());assert answer['exit_code']==0 and answer['checkpoint_sha256']==file_sha(cp),answer
                assert before==digest(adapter.owner.state_dict()) and steps==adapter.owner.control_steps,'Evaluation contaminated training sampler/steps'
                after_rng=rng_state();assert torch.equal(before_rng['torch_cpu'],after_rng['torch_cpu']) and all(torch.equal(x,y) for x,y in zip(before_rng['torch_cuda'],after_rng['torch_cuda']))
                emit(out/'evaluation_isolation.jsonl',dict(update=completed,passed=True,sampler_unchanged=True,training_control_steps_unchanged=True,torch_rng_unchanged=True))
                print('EVALUATION_CONTINUE '+str(completed),flush=True)
        runner.logger.log=evaluated_log
        distribution(0);loop_start=time.time()
        runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=True)
        loop_seconds=time.time()-loop_start
        assert completed==a.updates and adapter.owner.control_steps==a.updates*24 and optimizer_calls==completed*20
        final=out/'policy'/('model_'+str(completed-1)+'.pt');saved=torch.load(final,map_location='cpu',weights_only=False)
        assert saved['iter']==completed-1 and saved['infos']['completed_updates']==completed
        assert tensor_hash(saved['actor_state_dict'])==tensor_hash(runner.alg.actor.state_dict())
        assert tensor_hash(saved['critic_state_dict'])==tensor_hash(runner.alg.critic.state_dict())
        adam=[int(v['step']) for v in saved['optimizer_state_dict']['state'].values() if 'step' in v]
        assert adam and min(adam)==max(adam)==completed*20
        restore_check=adapter.restoration_check(saved['infos']['adapter'])
        before=rng_state();restore_rng(saved['infos']['rng'])
        draws=(random.random(),np.random.random(8),torch.rand(8),torch.rand(8,device='cuda:0'))
        restore_rng(saved['infos']['rng']);again=(random.random(),np.random.random(8),torch.rand(8),torch.rand(8,device='cuda:0'))
        assert draws[0]==again[0] and np.array_equal(draws[1],again[1]) and torch.equal(draws[2],again[2]) and torch.equal(draws[3],again[3]);restore_rng(before)
        distribution(completed);write(out/'effective_final.json',adapter.effective())
        result=dict(passed=True,engineering_only=a.stage=='short',stage=a.stage,task=task,mode=a.mode,completed_updates=completed,environment_transitions=completed*4096*24,optimizer_calls=optimizer_calls,training_loop_seconds=loop_seconds,paused_evaluation_seconds=paused_evaluation_seconds,transitions_per_second=completed*4096*24/loop_seconds,training_transitions_per_second_excluding_eval_wait=completed*4096*24/(loop_seconds-paused_evaluation_seconds),checkpoint=str(final),checkpoint_sha256=file_sha(final),adam_steps=[min(adam),max(adam)],restoration=restore_check,rng_roundtrip=True,summary=adapter.summary(),initialization=initial,finished_unix=time.time())
        write(out/'result.json',result);print('COMPARISON_TRAIN_PASS '+json.dumps(plain(result)),flush=True)
    else:
        assert a.checkpoint
        runner.load(a.checkpoint)
        policy=runner.get_inference_policy(device=env.unwrapped.device)
        obs=wrapped.get_observations();limit=1500 if a.role=='video' else env.unwrapped.max_episode_length+2
        with torch.inference_mode():
            for step in range(limit):
                actions=policy(obs);obs,_,dones,_=wrapped.step(actions);policy.reset(dones)
                if a.role!='video' and len(adapter.first_episode_rows)==a.num_envs:break
        adapter.owner.check()
        if a.role=='video':
            result=dict(passed=True,role='video',steps=limit,checkpoint=a.checkpoint,videos=[str(p) for p in (out/'videos').glob('*.mp4')],effective_commands=adapter.effective())
        else:
            rows=adapter.first_episode_rows
            assert len(rows)==a.num_envs==1024 and len({r['sequence_id'] for r in rows})==1024
            data=np.asarray([r['values'] for r in rows]);n=data[:,0]
            errors=np.sqrt((data[:,2:6]**2*n[:,None]).sum(0)/n.sum()) if n.sum() else None
            result=dict(passed=True,role=a.role,task=task,evaluation_seed=cfg.seed,pool_sha256=pool.sha,definition_sha256=digest(PROTOCOL),checkpoint=a.checkpoint,checkpoint_sha256=file_sha(a.checkpoint),episodes=len(rows),joint_success_rate=float(data[:,6].mean()),native_failure_rate=float(data[:,7].mean()),timeout_rate=float(data[:,8].mean()),insufficient_window_rate=float((n<PROTOCOL['min_scored_steps']).mean()),pooled_eligible_rmse=errors.tolist() if errors is not None else None,rmse_columns=['velocity_xy_m_s','yaw_rate_rad_s','ee_position_m','ee_orientation_rad'],scored_control_samples=int(n.sum()),evaluation_environment_transitions=adapter.owner.control_steps*a.num_envs,train_history_updates=sum(adapter.owner.engine.completed_counts),effective_commands=adapter.effective(),metric_min=plain(adapter.metrics.error_min),metric_max=plain(adapter.metrics.error_max))
            assert result['train_history_updates']==0
            write(out/'episode_results.json',rows)
        write(out/'result.json',result);print('COMPARISON_EVALUATION_PASS '+json.dumps(plain(result)),flush=True)
except BaseException as exc:
    (out/'fault.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),completed_updates=completed,optimizer_calls=optimizer_calls,control_steps=adapter.owner.control_steps if adapter else None,unix=time.time()),indent=2)+'\n')
    print(traceback.format_exc(),flush=True);raise
finally:
    if env is not None:env.close()
    app.close()
