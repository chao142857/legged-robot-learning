"""Stage3 native PPO: isolated Uniform-coverage or entropy-only contrasts.
Native loss implementation, network, task rewards, LR, PD and evaluator retained.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
CONFIG=json.loads(Path("/stage2-config.json").read_text())

p=argparse.ArgumentParser()
p.add_argument('--num-envs',type=int,choices=[4096],required=True)
p.add_argument('--updates',type=int,required=True)
p.add_argument('--resume-sha')
a=p.parse_args()
assert a.updates>=0
OUT=Path('/runs/ppo');OUT.mkdir(exist_ok=False)
IMAGE='sha256:8ee9b2bb8e6aae08dace71ee1e2141a4d62bcc63e4cf0eac28b347111a486a87'
CP1800_SHA='16333b869458ae62c371fd68b92f61e4740c250635f8b57906809ac067a77091'
app=None
class PlannedTimeStop(Exception):pass
physical_control_steps=0
optimizer_calls=0
try:
    from isaacsim import SimulationApp
    SimulationApp.DEFAULT_LAUNCHER_CONFIG=dict(SimulationApp.DEFAULT_LAUNCHER_CONFIG,disable_viewport_updates=True,enable_crashreporter=False)
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(dict(headless=True,enable_cameras=False,livestream=0,device='cuda:0',kit_args='--/persistent/isaac/asset_root/cloud=/node-assets --/persistent/isaac/asset_root/default=/node-assets --/persistent/isaac/asset_root/nvidia=/node-assets --/app/extensions/registryEnabled=false'))
    app=launcher.app
    import importlib.metadata
    import numpy as np
    import torch
    import gymnasium as gym
    import yaml
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner
    import LeggedManip_Lab.tasks
    from recipe_b0.config import build_b0
    from recipe_b0.state import restore_b0,restore_rng,state_hash
    from recipe_aplus.state import restore_native,equal_tree,tensor_hash
    from telemetry import plain,write_json,sha

    def finite(tree,label):
        tensors=[]
        def collect(x):
            if isinstance(x,torch.Tensor):
                if x.is_floating_point() or x.is_complex():tensors.append(torch.isfinite(x).all())
            elif hasattr(x,'values'):
                for y in x.values():collect(y)
            elif isinstance(x,(list,tuple)):
                for y in x:collect(y)
            elif isinstance(x,float):assert math.isfinite(x),label
        collect(tree)
        # All checked training tensors are CUDA or all CPU in any one call.
        groups={}
        for t in tensors:groups.setdefault(str(t.device),[]).append(t)
        assert all(bool(torch.stack(items).all()) for items in groups.values()),label+' NaN/Inf'

    def rng_proof(cp):
        infos=cp['infos'];restore_rng(infos)
        assert tensor_hash(torch.get_rng_state())==tensor_hash(infos['rng_cpu'])
        assert [tensor_hash(x) for x in torch.cuda.get_rng_state_all()]==[tensor_hash(x) for x in infos['rng_cuda']]
        assert state_hash(random.getstate())==state_hash(infos['rng_python'])
        assert state_hash(np.random.get_state())==state_hash(infos['rng_numpy'])
        py=[random.random() for _ in range(8)];np_values=np.random.random(8)
        restore_rng(infos)
        assert py==[random.random() for _ in range(8)] and np.array_equal(np_values,np.random.random(8))
        restore_rng(infos)
        return dict(torch_cpu_equal=True,torch_cuda_equal=True,python_equal=True,numpy_equal=True,python_numpy_replay=True,restored_after_probe=True)

    def adam_steps(alg):
        head=set(alg.actor.synergy_head.parameters()) if hasattr(alg.actor,'synergy_head') else set()
        values=[float(s['step']) for p,s in alg.optimizer.state.items() if p not in head and 'step' in s]
        assert values and min(values)==max(values)
        return int(values[0])

    def physx_proof(cfg):
        import omni.usd
        from pxr import PhysxSchema,PhysicsSchemaTools,UsdUtils
        from omni.physx import get_physx_interface,get_physx_statistics_interface
        from omni.physx.bindings._physx import PhysicsSceneStats
        stage=omni.usd.get_context().get_stage();path=cfg.sim.physics_prim_path
        scene=PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath(path));stats=PhysicsSceneStats()
        stage_id=UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        assert get_physx_statistics_interface().get_physx_scene_statistics(stage_id,PhysicsSchemaTools.sdfPathToInt(path),stats)
        values={name:int(getattr(stats,name)) for name in dir(stats) if name.startswith(('gpu_mem_','nb_'))}
        enabled=scene.GetEnableGPUDynamicsAttr().Get();broadphase=scene.GetBroadphaseTypeAttr().Get();override=get_physx_interface().get_overwrite_gpu_setting()
        assert enabled and broadphase=='GPU' and override!=0 and values['gpu_mem_heap_solver']>0 and values['gpu_mem_heap']>0,'GPU PhysX fallback'
        return dict(gpu_dynamics_enabled=bool(enabled),broadphase=broadphase,gpu_override=override,statistics=values)

    source=Path(CONFIG['native_source'])
    spec=json.loads((source/'recipe_b0/spec.json').read_text());original_cp=Path(spec['parent_checkpoint'])
    assert sha(original_cp)==CP1800_SHA==spec['parent_sha256']
    training_seed=int(CONFIG.get('training_seed',21));assert training_seed in (21,22)
    recipe=CONFIG.get('actor_recipe')
    cfg=build_b0(admission=CONFIG['admission'],num_envs=a.num_envs,seed=training_seed)
    cfg.sim.device='cuda:0';cfg.log_dir=str(OUT)
    fraction=float(CONFIG['dynamic_fraction']);assert fraction in (.3,.7)
    cfg.recipe_b0['sampling'].update(dynamic_fraction=fraction,fixed_fraction=1-fraction)
    cfg.task_suite['recipe_b0']['sampling']=cfg.recipe_b0['sampling'].copy()
    from task_suite.config import profile_signature
    cfg.task_suite_signature=profile_signature(cfg.task_suite)

    agent=handle_deprecated_rsl_rl_cfg(load_cfg_from_registry('GO2-PIPER-WBC','rsl_rl_cfg_entry_point'),importlib.metadata.version('rsl-rl-lib'))
    agent.seed=training_seed;agent.device='cuda:0';agent.max_iterations=1000000;agent.save_interval=1200;agent.resume=False
    parent_agent=yaml.safe_load((original_cp.parent/'agent.yaml').read_text())
    entropy_trial=CONFIG.get('entropy_coef')
    if entropy_trial is not None:
        assert fraction==.3 and float(entropy_trial) in (.005,.001),'Declared original-Uniform entropy-only contrast'
        assert parent_agent['algorithm']['entropy_coef']==.005
        agent.algorithm.entropy_coef=float(entropy_trial)
    resolved=plain(agent)
    for field in ['actor','critic','algorithm','num_steps_per_env','clip_actions','check_for_nan']:
        expected=parent_agent[field]
        if field=='algorithm' and entropy_trial is not None:expected=dict(expected,entropy_coef=float(entropy_trial))
        assert resolved[field]==expected,field
    assert agent.num_steps_per_env==24 and agent.algorithm.num_learning_epochs==5 and agent.algorithm.num_mini_batches==4
    canonical=lambda x:hashlib.sha256(json.dumps(plain(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    identity=dict(node_id='yanglab3060',purpose='stage3_original_uniform_dynamicX_PPO',dynamic_fraction=fraction,image_id=IMAGE,
                  b0_commit='ae1054046f792df8da6362c36ee168ded6b2ebcd',lab_commit='b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8',
                  method=spec['method'],root_parent_sha256=CP1800_SHA,num_envs=a.num_envs,seed=training_seed,
                  env_config_sha256=canonical(cfg),agent_config_sha256=canonical(agent),physics_config_sha256=canonical(cfg.sim))
    if recipe is not None:identity['actor_recipe']=recipe
    dump_yaml(str(OUT/'env.yaml'),cfg);dump_yaml(str(OUT/'agent.yaml'),agent)
    write_json(OUT/'identity.json',identity);write_json(OUT/'contact-buffer-settings.json',plain(cfg.sim.physx))
    env=RslRlVecEnvWrapper(gym.make('GO2-PIPER-WBC',cfg=cfg),clip_actions=agent.clip_actions)
    try:
        runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=agent.device);alg=runner.alg
        from telemetry import physical_suite
        original_reward=env.unwrapped.reward_manager.compute
        physical_rows=[];dynamic_masks=[]
        def observed_physical_reward(*args,**kwargs):
            global physical_control_steps
            result=original_reward(*args,**kwargs)
            physical_control_steps+=1
            raw=env.unwrapped;p=physical_suite(raw)
            with torch.no_grad():
                row=dict(xy_mse=p['xy_error_sq'].mean(),yaw_mse=p['yaw_error_sq'].mean(),ee_mse=p['ee_position_error'].square().mean(),ee_angle=p['ee_orientation_error'].mean(),nonfoot_fraction=(p['nonfoot_max_force']>1).float().mean(),overspeed_fraction=(p['joint_velocity_limit_ratio']>1.0001).float().mean(),hard_excess_max=p['joint_position_limit_excess'].max())
                row.update({'reward_rate/'+name:raw.reward_manager._step_reward[:,i].mean() for i,name in enumerate(raw.reward_manager.active_terms)})
                pose=raw.command_manager.get_term('ee_pose');dynamic=pose.is_dynamic.detach().clone();dynamic_masks.append(dynamic)
                for label,mask in [('dynamic',dynamic),('fixed',~dynamic)]:
                    row[label+'_samples']=mask.float().sum()
                    row[label+'_EE_squared_error_sum']=p['ee_position_error'].square()[mask].sum()
                    row[label+'_reward_sum']=result[mask].sum()
                    cohort_rates=raw.reward_manager._step_reward[mask].sum(0)/mask.sum().clamp_min(1)
                    row.update({'cohort_reward_rate/'+label+'/'+name:cohort_rates[i] for i,name in enumerate(raw.reward_manager.active_terms)})
                physical_rows.append({k:v.detach() for k,v in row.items()})
                if float(row['hard_excess_max'])>=.02:
                    torch.save(dict(q=raw.scene['robot'].data.joint_pos,qd=raw.scene['robot'].data.joint_vel,limits=raw.scene['robot'].data.joint_pos_limits,action=raw.action_manager.action,episode_step=raw.episode_length_buf),OUT/'severe_actual_joint_event.pt')
                    raise RuntimeError('Original severe actual joint excess>=.02rad; current branch stopped')
            return result
        env.unwrapped.reward_manager.compute=observed_physical_reward

        if a.resume_sha:
            cp_path=Path('/resume/checkpoint.pt');assert sha(cp_path)==a.resume_sha
            saved=torch.load(cp_path,map_location='cpu',weights_only=False)
            assert saved['infos']['stage2_identity']==identity,'Resume identity mismatch'
            parent_updates=saved['infos']['completed_updates']
            if recipe is not None:
                from synergy_actor import attach,restore_recipe
                attach(alg,recipe);cp,evidence=restore_recipe(runner,cp_path,parent_updates)
            else:cp,evidence=restore_native(runner,cp_path,expected_updates=parent_updates)
        else:
            cp_path=original_cp;cp,evidence=restore_b0(runner,cp_path);parent_updates=1800
            if recipe is not None:
                from synergy_actor import attach
                original_actor=alg.actor;attach(alg,recipe)
        evidence['all_rng_streams']=rng_proof(cp)
        evidence.update(loaded_checkpoint_sha256=sha(cp_path),new_container_process=True,environment_reset_required=True,physx_state_restored=False)
        assert env.unwrapped.action_manager.total_action_dim==18
        assert plain(env.unwrapped.observation_manager.group_obs_dim)=={'policy':[210],'critic':[102]}
        finite(alg.actor.state_dict(),'initial actor');finite(alg.critic.state_dict(),'initial critic');finite(alg.optimizer.state_dict(),'initial Adam')
        initial_adam_step=adam_steps(alg);assert initial_adam_step==parent_updates*20
        assert float(alg.entropy_coef)==float(agent.algorithm.entropy_coef),'Runtime entropy coefficient overridden during restore'
        write_json(OUT/'runtime-algorithm.json',dict(entropy_coef=float(alg.entropy_coef),new_entropy_trial=entropy_trial is not None,
            task_reward_unchanged=True,network_LR_KL_unchanged=(recipe is None),actor_recipe=recipe,dynamic_fraction=fraction,
            sampler_matches_original_parent=(fraction==.3),optimizer_fully_restored=True))
        write_json(OUT/'restore-evidence.json',evidence)
        # The saved learner/RNG is checked before an explicit new physical reset.
        restore_rng(cp['infos'])
        if training_seed!=21 and not a.resume_sha:
            random.seed(training_seed);np.random.seed(training_seed);torch.manual_seed(training_seed);torch.cuda.manual_seed_all(training_seed)
            write_json(OUT/'continuation-seed.json',dict(seed=training_seed,parent_weights_shared=True,independent_full_scratch_seed=False,parent_RNG_verified_before_explicit_reseed=True))
        reset_observations,_=env.reset()
        if recipe is not None and not a.resume_sha:
            from synergy_actor import validate_zero_attach
            # Reuse reset's actual observation: a diagnostic get_observations()
            # would draw fresh observation noise and potentially change history.
            write_json(OUT/'zero-mean-attachment.json',validate_zero_attach(original_actor,alg.actor,reset_observations,env.unwrapped.action_manager.get_term('joint_pos')))
        reset_control_steps=physical_control_steps
        physical_rows.clear();dynamic_masks.clear()
        if a.updates==0:
            runner.alg.eval_mode();obs=env.get_observations()
            with torch.inference_mode():
                for _ in range(24):
                    obs,rewards,dones,extras=env.step(runner.alg.act(obs));finite(obs,'restored observations');finite(rewards,'restored rewards')
            assert equal_tree(alg.actor.state_dict(),cp['actor_state_dict']) and equal_tree(alg.critic.state_dict(),cp['critic_state_dict'])
            assert equal_tree(alg.optimizer.state_dict(),cp['optimizer_state_dict']) and adam_steps(alg)==parent_updates*20
            assert runner.current_learning_iteration==parent_updates
            write_json(OUT/'result.json',dict(passed=True,kind='final_checkpoint_restore_zero_updates',identity=identity,updates=0,optimizer_steps=0,
                       completed_updates=parent_updates,loaded_native_iter=cp['iter'],next_runner_iteration=runner.current_learning_iteration,
                       loaded_checkpoint_sha256=sha(cp_path),adam_step=adam_steps(alg),learner_adam_unchanged=True,restore_evidence=evidence,
                       inference_steps=24,restore_inference_transition_slots=24*a.num_envs,
                       physical_control_steps=physical_control_steps,reset_control_steps=reset_control_steps,
                       reset_transition_slots=reset_control_steps*a.num_envs,
                       physical_transition_slots_including_reset=physical_control_steps*a.num_envs,
                       physx=physx_proof(cfg),physx_state_restored=False,new_training_transitions=0))
            print('STAGE2_RESTORE_PASS',flush=True);time.sleep(2);raise SystemExit(0)
        rows=[];kl_values=[];grad_norms=[];optimizer_calls=0;rollout_steps=0;rollout_start=None;rollout_seconds=None;gae_seconds=None
        reward_means=[];done_counts=[];checkpoint_records=[];phase_start=None;train_start_unix=None
        original_act=alg.act;original_process=alg.process_env_step;original_returns=alg.compute_returns;original_update=alg.update
        original_step=alg.optimizer.step;original_kl=alg.actor.get_kl_divergence
        from native_gradient_audit import NativeGradientAudit
        gradient_audit=NativeGradientAudit(alg,OUT,env.unwrapped.observation_manager) if CONFIG.get("gradient_audit",False) else None

        def observed_act(obs):
            global rollout_start
            if rollout_steps==0:
                if Path('/runs/TIME_STOP').exists():raise PlannedTimeStop('Independent GPU time reserve reached at rollout boundary')
                torch.cuda.synchronize();rollout_start=time.perf_counter()
            if gradient_audit is not None:gradient_audit.input(obs)
            actions=original_act(obs);finite(actions,'actions');return actions

        def observed_process(obs,rewards,dones,extras):
            global rollout_steps
            reward_means.append(rewards.detach().mean());done_counts.append(dones.detach().sum())
            finite(obs,'observations');finite(rewards,'rewards')
            result=original_process(obs,rewards,dones,extras);rollout_steps+=1
            return result

        def observed_returns(obs):
            global rollout_seconds,gae_seconds
            assert rollout_steps==24 and alg.storage.step==24,'Incomplete real rollout'
            torch.cuda.synchronize();now=time.perf_counter();rollout_seconds=now-rollout_start
            result=original_returns(obs);torch.cuda.synchronize();gae_seconds=time.perf_counter()-now
            finite(alg.storage.returns,'returns');finite(alg.storage.advantages,'GAE advantages')
            if gradient_audit is not None:gradient_audit.rollout_ready(dynamic_masks,physical_rows)
            return result

        def observed_kl(*args,**kwargs):
            result=original_kl(*args,**kwargs);kl_values.append(result.detach().mean());return result

        def observed_step(*args,**kwargs):
            global optimizer_calls
            assert optimizer_calls<a.updates*20,'Optimizer budget exhausted'
            gradients=[p.grad for group in alg.optimizer.param_groups for p in group['params'] if p.grad is not None]
            assert gradients,'Backward produced no gradients';finite(gradients,'gradients before Adam')
            if gradient_audit is not None:gradient_audit.native_optimizer_gradient()
            result=original_step(*args,**kwargs);optimizer_calls+=1;return result

        def parameter_delta(model,before):
            deltas=[(param.detach()-before[name]).flatten() for name,param in model.named_parameters()]
            flat=torch.cat(deltas)
            value=dict(l2=float(torch.linalg.vector_norm(flat)),max_abs=float(flat.abs().max()),changed_elements=int(torch.count_nonzero(flat)))
            if model is alg.actor:
                output=[(n,p) for n,p in model.named_parameters() if p.ndim==2 and p.shape[0]==18][-1]
                value['per_action_output_weight_delta_l2']=(output[1].detach()-before[output[0]]).square().sum(-1).sqrt().cpu().tolist()
            assert value['changed_elements']>0,'Model parameters did not change'
            return value

        original_clip=torch.nn.utils.clip_grad_norm_
        def observed_clip(*args,**kwargs):
            value=original_clip(*args,**kwargs);finite(value,'pre-clip gradient norm');grad_norms.append(value.detach());return value
        torch.nn.utils.clip_grad_norm_=observed_clip

        def save_periodic(completed):
            cpdir=OUT/'checkpoints';cpdir.mkdir(exist_ok=True);target=cpdir/('cp_'+str(completed)+'.pt');assert not target.exists()
            infos=dict(completed_updates=completed,method=spec['method'],initialization='Stage3 warm-start own original A+CP1800; full native learner restore',
                       parent_checkpoint_sha256=sha(cp_path),root_parent_sha256=CP1800_SHA,additional_completed_updates=completed-1800,
                       environment_steps=int(cp['infos']['environment_steps'])+(completed-parent_updates)*4096*24,
                       stage2_new_transitions=(completed-1800)*4096*24,learning_rate=alg.learning_rate,
                       accounting_version=2,parent_completed_updates=parent_updates,
                       rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                       stage2_identity=identity,physx_state_saved=False,environment_state_saved=False,rollout_boundary=True)
            if recipe is not None:infos['stage3_actor_recipe']=recipe
            state=alg.save();state.update(iter=completed-1,infos=infos)
            temporary=target.with_suffix('.partial');begin=time.perf_counter();torch.save(state,temporary);os.replace(temporary,target)
            elapsed=time.perf_counter()-begin
            record=dict(completed_updates=completed,path=str(target),bytes=target.stat().st_size,sha256=sha(target),save_seconds=elapsed,adam_step=adam_steps(alg))
            write_json(target.with_suffix('.json'),record);checkpoint_records.append(record)
            return record

        def observed_update():
            global rollout_steps,kl_values,grad_norms,reward_means,done_counts,phase_start
            assert len(rows)<a.updates and rollout_steps==24
            before_actor={n:p.detach().clone() for n,p in alg.actor.named_parameters()}
            before_critic={n:p.detach().clone() for n,p in alg.critic.named_parameters()}
            before_step=adam_steps(alg);before_calls=optimizer_calls
            if gradient_audit is not None:gradient_audit.before_update()
            torch.cuda.synchronize();start=time.perf_counter();loss=original_update();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            finite(loss,'losses');finite(alg.actor.state_dict(),'actor');finite(alg.critic.state_dict(),'critic');finite(alg.optimizer.state_dict(),'Adam')
            assert optimizer_calls-before_calls==20 and adam_steps(alg)==before_step+20
            assert alg.storage.step==0,'Native PPO did not clear rollout storage'
            assert len(kl_values)==20,'KL observer did not see each native minibatch'
            assert len(grad_norms)==40,'Expected actor/critic gradient norm for 20 minibatches'
            norms=torch.stack(grad_norms);finite(norms,'gradient norms')
            kl=torch.stack(kl_values);finite(kl,'KL');assert float(kl.min())>=-1e-5,'Invalid KL'
            row=dict(local_update=len(rows)+1,completed_updates=parent_updates+len(rows)+1,transitions=a.num_envs*24,
                     rollout_seconds=rollout_seconds,gae_seconds=gae_seconds,ppo_update_seconds=elapsed,
                     losses=loss,kl_mean=float(kl.mean()),kl_max=float(kl.max()),learning_rate=alg.learning_rate,
                     entropy_coef=float(alg.entropy_coef),
                     actor_parameter_delta=parameter_delta(alg.actor,before_actor),critic_parameter_delta=parameter_delta(alg.critic,before_critic),
                     adam_step=adam_steps(alg),optimizer_steps_this_update=20,finite_checks_passed=True,physx=physx_proof(cfg),
                     torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            row.update(unix=time.time(),env_steps=row['completed_updates']*4096*24,
                       reward_mean=float(torch.stack(reward_means).mean()),done_count=int(torch.stack(done_counts).sum()),
                       actor_grad_norm_mean=float(norms[::2].mean()),actor_grad_norm_max=float(norms[::2].max()),
                       critic_grad_norm_mean=float(norms[1::2].mean()),critic_grad_norm_max=float(norms[1::2].max()),
                       action_std=alg.actor.output_std.detach().mean(0).cpu().tolist())
            row['physical_reward_time']={k:float(torch.stack([v[k] for v in physical_rows]).mean()) for k in physical_rows[0]}
            if recipe is not None:
                row['synergy_head']=dict(parameter_l2=float(torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in alg.actor.synergy_head.parameters()]))),
                    gradient_l2=float(torch.linalg.vector_norm(torch.cat([p.grad.detach().flatten() for p in alg.actor.synergy_head.parameters()]))),
                    Adam_steps=[int(alg.optimizer.state[p]['step']) for p in alg.actor.synergy_head.parameters()],
                    all_original_leg_arm_parameters_trainable=all(p.requires_grad for p in alg.actor.mlp.parameters()),residual_in_Gaussian_mean=True)
            row['physical_reward_time']['hard_excess_max']=max(float(v['hard_excess_max']) for v in physical_rows)
            assert len(dynamic_masks)==24 and len(physical_rows)==24
            cohort={}
            masks=torch.stack(dynamic_masks)
            # Native update cleared storage; cohort reward/count remains directly observable.
            for label in ('dynamic','fixed'):
                count=sum(int(v[label+'_samples']) for v in physical_rows)
                cohort[label]=dict(samples=count,EE_mse=sum(float(v[label+'_EE_squared_error_sum']) for v in physical_rows)/count if count else None,
                    mean_native_reward=sum(float(v[label+'_reward_sum']) for v in physical_rows)/count if count else None)
            row['Uniform_coverage']=dict(cohorts=cohort,episode_membership_draws_fixed_dynamic=env.unwrapped.command_manager.get_term('ee_pose').membership_draws.cpu().tolist(),mask_shape=list(masks.shape),sampling_mechanism_unchanged=True)
            physical_rows.clear()
            dynamic_masks.clear()
            if gradient_audit is not None:gradient_audit.after_update(row['completed_updates'],loss,row['kl_mean'])
            rows.append(row)
            with (OUT/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('STAGE2_UPDATE '+json.dumps({k:row[k] for k in ['local_update','completed_updates','rollout_seconds','gae_seconds','ppo_update_seconds','kl_mean','learning_rate','adam_step']}),flush=True)
            checkpoint=None
            if len(rows)%100==0:
                checkpoint=save_periodic(row['completed_updates']);now=time.perf_counter();window=rows[-100:]
                phase=dict(local_updates=[len(rows)-99,len(rows)],completed_updates=row['completed_updates'],
                           wall_seconds=now-phase_start,transitions=100*4096*24,full_loop_transitions_per_second=100*4096*24/(now-phase_start),
                           kl_mean=sum(r['kl_mean'] for r in window)/100,learning_rate=alg.learning_rate,
                           reward_mean=sum(r['reward_mean'] for r in window)/100,
                           value_loss_mean=sum(r['losses']['value'] for r in window)/100,
                           actor_grad_norm_max=max(r['actor_grad_norm_max'] for r in window),critic_grad_norm_max=max(r['critic_grad_norm_max'] for r in window),checkpoint=checkpoint)
                write_json(OUT/('phase_'+str(len(rows))+'.json'),phase);print('STAGE2_PHASE '+json.dumps(phase),flush=True);phase_start=now
            write_json(OUT/'latest.json',dict(local_update=len(rows),completed_updates=row['completed_updates'],unix=time.time(),
                       training_started_unix=train_start_unix,training_elapsed_seconds=time.perf_counter()-training_start,
                       checkpoint=checkpoint,finite=True,ppo_updates_budget=a.updates))
            rollout_steps=0;kl_values=[];grad_norms=[];reward_means=[];done_counts=[]
            return loss

        alg.act=observed_act;alg.process_env_step=observed_process;alg.compute_returns=observed_returns
        alg.update=observed_update;alg.actor.get_kl_divergence=observed_kl;alg.optimizer.step=observed_step
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();training_start=start;phase_start=start;train_start_unix=time.time()
        time_stopped=False
        try:runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False)
        except PlannedTimeStop:
            time_stopped=True
            write_json(OUT/'time_stop.json',dict(reason='Independent time reserve',completed_local_updates=len(rows),rollout_boundary=True))
        torch.cuda.synchronize();loop_seconds=time.perf_counter()-start
        actual_updates=len(rows);completed=parent_updates+actual_updates
        assert actual_updates>0 and (actual_updates==a.updates or time_stopped)
        assert optimizer_calls==actual_updates*20 and adam_steps(alg)==completed*20
        assert runner.current_learning_iteration in (completed-1,completed)
        assert not equal_tree(alg.actor.state_dict(),cp['actor_state_dict']) and not equal_tree(alg.critic.state_dict(),cp['critic_state_dict'])
        assert not equal_tree(alg.optimizer.state_dict(),cp['optimizer_state_dict'])
        new_transitions=a.num_envs*24*actual_updates
        if actual_updates%100:save_periodic(completed)
        assert len(checkpoint_records)==(actual_updates+99)//100
        target=OUT/'checkpoints'/('cp_'+str(completed)+'.pt');checkpoint_seconds=sum(r['save_seconds'] for r in checkpoint_records)
        infos=torch.load(target,map_location='cpu',weights_only=False)['infos']
        saved=torch.load(target,map_location='cpu',weights_only=False)
        assert saved['iter']==completed-1 and saved['infos']['completed_updates']==completed
        assert equal_tree(saved['actor_state_dict'],alg.actor.state_dict()) and equal_tree(saved['critic_state_dict'],alg.critic.state_dict())
        assert equal_tree(saved['optimizer_state_dict'],alg.optimizer.state_dict())
        assert saved['infos']['stage2_identity']==identity
        evidence_after_save=rng_proof(saved)
        result=dict(passed=True,kind='stage2_native_PPO_window',identity=identity,
                    updates=actual_updates,requested_slice_updates=a.updates,time_stopped=time_stopped,parent_completed_updates=parent_updates,completed_updates=completed,optimizer_steps=optimizer_calls,
                    initial_adam_step=initial_adam_step,final_adam_step=adam_steps(alg),new_transitions=new_transitions,
                    physical_control_steps=physical_control_steps,reset_control_steps=reset_control_steps,
                    collected_training_transition_slots=(physical_control_steps-reset_control_steps)*a.num_envs,
                    total_environment_steps=infos['environment_steps'],stage2_new_transitions=infos['stage2_new_transitions'],
                    native_runner_learn_called=True,learner_iteration=runner.current_learning_iteration,
                    training_loop_seconds=loop_seconds,full_ppo_loop_transitions_per_second=new_transitions/loop_seconds,
                    rollout_seconds=sum(r['rollout_seconds'] for r in rows),gae_seconds=sum(r['gae_seconds'] for r in rows),
                    ppo_update_seconds=sum(r['ppo_update_seconds'] for r in rows),checkpoint_seconds=checkpoint_seconds,
                    loop_and_checkpoint_transitions_per_second=new_transitions/loop_seconds,
                    checkpoint_schedule=checkpoint_records,training_started_unix=train_start_unix,training_finished_unix=time.time(),checkpoint=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,learner_and_adam_equal=True,rng=evidence_after_save),
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    actor_changed=True,critic_changed=True,adam_changed=True,finite_checks_passed=True,
                    kl_min=min(r['kl_mean'] for r in rows),kl_max=max(r['kl_max'] for r in rows),physx=rows[-1]['physx'],
                    no_physx_state_restore_claim=True,headless=True,enable_cameras=False,disable_viewport_updates=True,
                    timing_scope='Continuous native learn including observers and periodic checkpoints; excludes startup/reset/CP load',
                    limitation='Bounded original PPO slice; task capability is evaluated separately under unchanged A/T0/T1/X.')
        write_json(OUT/'result.json',result);print('STAGE2_PASS '+json.dumps(result),flush=True)
        # Hold the completed process briefly so the host can read kernel high-water marks.
        time.sleep(2)
    finally:env.close()
except SystemExit:
    raise
except BaseException as exc:
    (OUT/'error.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),physical_control_steps=physical_control_steps,physical_transition_slots_including_reset=physical_control_steps*a.num_envs,optimizer_calls_observed=optimizer_calls,reset_control_steps=globals().get('reset_control_steps'),complete_PPO_transitions=sum(r['transitions'] for r in globals().get('rows',[])),partial_rollout_steps=globals().get('rollout_steps'),note='Real reward-compute control steps; reset and incomplete rollout separately identified. Not all collected slots entered PPO updates.'),indent=2))
    print(traceback.format_exc(),flush=True)
    raise
finally:
    if app is not None:app.close()
