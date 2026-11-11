#!/usr/bin/env python3
"""Bounded real 18-joint Isaac/RSL5 experiment. No frozen-JIT high-level policy."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
BASE=REPO/'benchmarks/go2_piper/wbc_sgs'
JIT=REPO/'mujoco/deploy/policy/go2_piper/wbc/policy.pt'
JIT_SHA='c5f57c79bc01d8b8f2c0846163b149e3f53de39291770a1ccb230ef3be6b800e'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=str)+'\n')


def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def canonical(x):
    if isinstance(x,dict):return {k:canonical(v) for k,v in sorted(x.items())}
    if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
    if callable(x):return x.__module__+'.'+x.__qualname__
    if x is None or isinstance(x,(bool,int,float,str)):return x
    return str(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['calibration','interface','smoke','train','evaluate'],required=True)
    p.add_argument('--method',choices=['uniform','sgs'],default='uniform')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pool',type=Path,default=BASE/'task_pool.json')
    p.add_argument('--protocol',type=Path,default=BASE/'success_protocol.json')
    p.add_argument('--num-envs',type=int,default=128)
    p.add_argument('--updates',type=int,default=3)
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--checkpoint',type=Path,nargs='*',default=[])
    p.add_argument('--resume',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--max-wall-seconds',type=float,default=900.)
    a=p.parse_args()
    assert 1<=a.num_envs<=256 and 1<=a.updates<=256 and a.max_wall_seconds<=1200
    a.output.mkdir(parents=True,exist_ok=False)
    assert sha(JIT)==JIT_SHA
    start=time.monotonic()
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(headless=True,device='cuda:0')
    import importlib.metadata as md
    import numpy as np
    import torch
    import gymnasium as gym
    import LeggedManip_Lab.tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.go2_piper.agents.rsl_rl_ppo_cfg import Go2PiperWBCPPORunnerCfg
    from rsl_rl.runners import OnPolicyRunner
    from benchmarks.go2_piper.wbc_sgs.sampler import TaskSampler
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    cfg=parse_env_cfg('GO2-PIPER-WBC-SGS',device='cuda:0',num_envs=a.num_envs)
    cfg.task_pool_path=str(a.pool);cfg.success_protocol_path=str(a.protocol)
    cfg.sampler_mode='evaluation' if a.stage in ['calibration','evaluate'] else a.method
    cfg.sampler_seed=a.seed;cfg.seed=a.seed;cfg.episode_log_path=str(a.output/'raw_episodes.jsonl')
    if a.stage in ['evaluate','calibration']:
        assert a.num_envs==len(json.loads(a.pool.read_text())['tasks'])
    env=gym.make('GO2-PIPER-WBC-SGS',cfg=cfg)
    u=env.unwrapped;vec=RslRlVecEnvWrapper(env)
    assert vec.num_actions==18 and vec.get_observations()['policy'].shape[-1]==210
    protocol=json.loads(a.protocol.read_text())
    assert abs(u.step_dt-.02)<1.e-10 and abs(cfg.episode_length_s-protocol['episode_s'])<1.e-10
    agent=handle_deprecated_rsl_rl_cfg(Go2PiperWBCPPORunnerCfg(),md.version('rsl-rl-lib')).to_dict()
    agent.update(device='cuda:0',seed=a.seed,obs_groups={'actor':['policy'],'critic':['critic']},
                 experiment_name='go2_piper_wbc_sgs',logger='tensorboard',save_interval=100000)
    # Both arms clone the mature mean but train a new stochastic 18D actor.
    # Smaller initial exploration is preregistered for bounded fine-tuning, not tuned per arm.
    agent['actor']['distribution_cfg']['init_std']=.2
    json_write(a.output/'agent.json',canonical(agent))
    env_cfg=canonical(cfg.to_dict())
    for key in ['episode_log_path','sampler_mode','log_dir']:env_cfg.pop(key,None)
    env_fingerprint=hashlib.sha256(json.dumps(env_cfg,sort_keys=True).encode()).hexdigest()
    json_write(a.output/'env_config.json',env_cfg)
    original_cfg=parse_env_cfg('GO2-PIPER-WBC',device='cuda:0',num_envs=a.num_envs)
    reward_definition=canonical(cfg.rewards.to_dict())
    assert reward_definition==canonical(original_cfg.rewards.to_dict()), 'Original WBC reward must be retained'
    source_paths=list(BASE.glob('*.py'))+list((REPO/'source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/wbc_sgs').glob('*.py'))
    source_paths+=[Path(__file__).resolve()]
    source_hashes={str(x.relative_to(REPO)):sha(x) for x in sorted(source_paths)}
    manifest=dict(status='RUNNING',stage=a.stage,method=a.method,seed=a.seed,num_envs=a.num_envs,
        pool_sha256=sha(a.pool),protocol_sha256=sha(a.protocol),mature_wbc_sha256=sha(JIT),
        environment_sha256=env_fingerprint,agent_sha256=sha(a.output/'agent.json'),
        original_reward_equal=True,reward_sha256=hashlib.sha256(json.dumps(reward_definition,sort_keys=True).encode()).hexdigest(),
        source_sha256=source_hashes,
        action_dim=18,physics_hz=200,policy_hz=50,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        versions={x:md.version(x) for x in ['torch','isaaclab','rsl-rl-lib']},
        safety_scope=protocol['safety_scope'], common_initialization='Mature actor mean; random critic seed+1000; trainable Gaussian std=.2')
    json_write(a.output/'manifest.json',manifest)
    def deadline():
        if time.monotonic()-start>a.max_wall_seconds:raise TimeoutError('Bounded experiment deadline exceeded')
    @torch.inference_mode()
    def evaluate_actor(policy,label):
        u.context=label
        u.sampler=TaskSampler(len(u.pool['tasks']),a.seed,'evaluation')
        u.episode_length_buf[:]=0;u.reset_buf[:]=False
        obs,_=vec.reset()
        task_ids=u.wbc_task_ids.clone()
        assert task_ids.cpu().tolist()==list(range(vec.num_envs))
        before=len(u.episode_records);first={}
        with torch.inference_mode(), (a.output/(label+'_trace.jsonl')).open('w') as trace:
            for step in range(u.max_episode_length+1):
                deadline()
                action=policy(obs)
                assert action.shape==(vec.num_envs,18) and bool(torch.isfinite(action).all())
                obs,_,_,_=vec.step(action)
                for r in u.episode_records[before:]:
                    first.setdefault(r['env_id'],r)
                before=len(u.episode_records)
                if step%10==0:
                    trace.write(json.dumps(dict(step=step+1,task_ids=u.wbc_task_ids.cpu().tolist(),
                        errors=u.last_joint_errors.cpu().tolist()))+'\n')
                if len(first)==vec.num_envs:break
        assert len(first)==vec.num_envs
        results=[first[i] for i in range(vec.num_envs)]
        assert [r['task_id'] for r in results]==task_ids.cpu().tolist()
        json_write(a.output/(label+'_episodes.json'),results)
        return dict(label=label,n=len(results),**{k:sum(r[k] for r in results)/len(results)
                    for k in ['locomotion','manipulation','safety','joint_success']},
                    mean_rmse=np.mean([r['rmse'] for r in results],axis=0).tolist())
    result={}
    if a.stage=='calibration':
        jit=torch.jit.load(str(JIT),map_location=u.device).eval()
        result=evaluate_actor(lambda obs:jit(obs['policy']),'mature_calibration')
    elif a.stage=='interface':
        old=u.wbc_task_ids.clone();cmd={n:u.command_manager.get_command(n).clone() for n in ['base_velocity','ee_pose']}
        for name in cmd:u.command_manager.get_term(name).time_left[:]=0
        u.command_manager.compute(12.)
        assert torch.equal(old,u.wbc_task_ids)
        assert all(torch.equal(cmd[n],u.command_manager.get_command(n)) for n in cmd)
        initial_counts=sum(u.sampler.completed_counts)
        u._reset_idx(torch.tensor([1,3],device=u.device))
        mask=torch.ones(a.num_envs,dtype=torch.bool,device=u.device);mask[[1,3]]=False
        assert torch.equal(old[mask],u.wbc_task_ids[mask])
        assert sum(u.sampler.completed_counts)==initial_counts
        obs=vec.get_observations();jit=torch.jit.load(str(JIT),map_location=u.device).eval()
        with torch.inference_mode():
            for step in range(550):
                deadline();obs,_,_,_=vec.step(jit(obs['policy']))
        assert u.joint_metrics.steps.max()>=501
        # Real physics fault: low-base contact must credit failure to the OLD ID before reset.
        old_task=int(u.wbc_task_ids[0]);n=len(u.episode_records)
        with torch.inference_mode():
            root=u.scene['robot'].data.root_state_w[[0]].clone();root[:,2]=u.scene.env_origins[0,2]+.05
            u.scene['robot'].write_root_pose_to_sim(root[:,:7],env_ids=torch.tensor([0],device=u.device))
            for _ in range(5):
                obs,_,_,_=vec.step(jit(obs['policy']))
                if any(r['env_id']==0 for r in u.episode_records[n:]):break
        faults=[r for r in u.episode_records[n:] if r['env_id']==0]
        assert faults and faults[0]['task_id']==old_task and not faults[0]['joint_success'] and not faults[0]['safety']
        assert sum(u.sampler.completed_counts)==len(u.episode_records)
        result=dict(status='PASS',both_real_sources_fixed=True,forced_timer_expiry_ignored=True,
                    longer_than_10_seconds_verified=True,subset_reset_correspondence=True,
                    terminal_old_task_credit=True,physical_fault_record=faults[0],
                    completed_counts_match_episode_log=True)
    else:
        # Construct both networks AFTER reseeding, independent of simulator startup RNG use.
        torch.manual_seed(a.seed+1000)
        runner=OnPolicyRunner(vec,agent,log_dir=None,device='cuda:0')
        actor=runner.alg.actor
        jit=torch.jit.load(str(JIT),map_location=u.device).eval()
        mature=jit.state_dict();state=actor.state_dict()
        assert all(k in state and state[k].shape==v.shape for k,v in mature.items())
        state.update(mature);actor.load_state_dict(state)
        with torch.inference_mode():
            obs=vec.get_observations();mean=actor(obs)
            assert torch.allclose(mean,jit(obs['policy']),atol=1.e-6,rtol=1.e-5)
        del jit
        assert all(p.requires_grad for p in actor.parameters())
        initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        manifest.update(initial_actor_sha256=tensor_digest(initial),initial_critic_sha256=tensor_digest(runner.alg.critic.state_dict()),
                        trainable_actor_parameters=sum(p.numel() for p in actor.parameters()))
        completed=0
        if a.resume:
            loaded=torch.load(a.resume,weights_only=False,map_location=u.device)
            runner.alg.load(loaded,None,True)
            completed=int(loaded['iter'])
            with torch.inference_mode():
                u.restore_sampler_checkpoint(loaded['infos']['environment'])
            torch.set_rng_state(loaded['infos']['torch_rng'].cpu())
            torch.cuda.set_rng_state_all([x.cpu() for x in loaded['infos']['cuda_rng']])
            random.setstate(loaded['infos']['python_rng'])
            np.random.set_state(loaded['infos']['numpy_rng'])
            manifest['resume_from']=str(a.resume)
            manifest['restored_actor_sha256']=tensor_digest(actor.state_dict())
            assert manifest['restored_actor_sha256']==loaded['infos']['actor_sha256']
            assert u.sampler.state_dict()==loaded['infos']['environment']['sampler']
            assert len(runner.alg.optimizer.state)>0
        def save_checkpoint():
            state=runner.alg.save()
            state.update(iter=completed,infos=dict(environment=u.sampler_checkpoint(),
                actor_sha256=tensor_digest(actor.state_dict()),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),manifest=manifest))
            path=a.output/f'checkpoint_{completed:04d}.pt';torch.save(state,path)
            return path
        if a.stage=='evaluate':
            assert a.checkpoint, 'Explicit evaluation checkpoints required'
            scores=[]
            for path in a.checkpoint:
                loaded=torch.load(path,weights_only=False,map_location=u.device)
                assert loaded['infos']['manifest']['pool_sha256']==manifest['pool_sha256']
                assert loaded['infos']['manifest']['protocol_sha256']==manifest['protocol_sha256']
                runner.alg.load(loaded,dict(actor=True,critic=True,optimizer=False),True)
                runner.alg.eval_mode()
                score=evaluate_actor(actor,path.stem)
                score.update(checkpoint=str(path),checkpoint_sha256=sha(path),updates=loaded['iter'],
                             train_method=loaded['infos']['manifest']['method'])
                scores.append(score)
            result=dict(scores=scores,evaluation_distribution='exactly one deterministic episode per each of 48 fixed tasks; no sampler adaptation')
        else:
            if a.stage=='train':
                assert a.plan is not None
                plan=json.loads(a.plan.read_text())
                assert plan['updates']==a.updates and plan['num_envs']==a.num_envs and plan['seed']==a.seed
                assert plan['pool_sha256']==manifest['pool_sha256'] and plan['protocol_sha256']==manifest['protocol_sha256']
                assert plan['source_sha256']==source_hashes
                assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(), 'Commit reviewed experiment before either pilot arm'
                manifest['plan_sha256']=sha(a.plan)
            save_checkpoint()
            obs=vec.get_observations();runner.alg.train_mode()
            rows=[];training_start=time.monotonic()
            with (a.output/'learning.jsonl').open('w') as log, (a.output/'sampling.jsonl').open('w') as sampling:
                for _ in range(a.updates):
                    deadline();t=time.monotonic();record_start=len(u.episode_records)
                    with torch.inference_mode():
                        for _ in range(agent['num_steps_per_env']):
                            actions=runner.alg.act(obs)
                            obs,reward,done,extra=vec.step(actions)
                            assert bool(torch.isfinite(reward).all())
                            runner.alg.process_env_step(obs,reward,done,extra)
                        runner.alg.compute_returns(obs)
                    loss=runner.alg.update();completed+=1
                    assert all(bool(torch.isfinite(p).all()) for p in actor.parameters())
                    end=actor.mlp[-1].weight
                    grad=end.grad.detach().norm(dim=1).cpu().tolist()
                    records=u.episode_records[record_start:]
                    row=dict(update=completed,transitions=completed*vec.num_envs*agent['num_steps_per_env'],
                        elapsed_s=time.monotonic()-training_start,iteration_s=time.monotonic()-t,
                        loss={k:float(v) for k,v in loss.items()},learning_rate=runner.alg.learning_rate,
                        output_gradient_norm_by_joint=grad,completed_episodes=len(u.episode_records),
                        batch_terminal_joint_success=sum(r['joint_success'] for r in records)/len(records) if records else None)
                    log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
                    sampling.write(json.dumps(dict(update=completed,transitions=row['transitions'],**u.sampler.summary()))+'\n');sampling.flush()
                    if completed in [16,32,64,96,128] or completed==a.updates:save_checkpoint()
                final_path=save_checkpoint() if not (a.output/f'checkpoint_{completed:04d}.pt').exists() else a.output/f'checkpoint_{completed:04d}.pt'
            last_key='mlp.6.weight'
            changes=(actor.state_dict()[last_key].detach().cpu()-initial[last_key]).norm(dim=1).tolist()
            assert min(changes)>0 and max(grad[:12])>0 and max(grad[12:])>0
            assert sum(u.sampler.completed_counts)==len(u.episode_records)
            result=dict(completed_updates=completed,additional_updates=a.updates,
                transitions=a.updates*vec.num_envs*agent['num_steps_per_env'],
                training_wall_s=time.monotonic()-training_start,mean_iteration_s=sum(r['iteration_s'] for r in rows)/len(rows),
                output_weight_change_by_joint=changes,actor_changed=True,leg_and_arm_gradients_nonzero=True,
                final_checkpoint=str(final_path),final_sampler=u.sampler.summary(),
                completed_episodes=len(u.episode_records),
                terminal_success_rate=sum(r['joint_success'] for r in u.episode_records)/len(u.episode_records) if u.episode_records else None,
                resume_verified=bool(a.resume),resume_semantics='Optimizer/sampler/RNG resume; partial physical episodes restart same task, not bitwise simulator replay')
    assert sha(JIT)==JIT_SHA
    assert all(sha(REPO/name)==h for name,h in source_hashes.items()), 'Experiment source changed during run'
    manifest.update(status='COMPLETE',wall_s=time.monotonic()-start,result=result)
    json_write(a.output/'manifest.json',manifest)
    print('WBC_SGS_RESULT '+json.dumps(manifest),flush=True)
    vec.close();launcher.app.close()


if __name__=='__main__':
    try:
        main()
    except BaseException:
        # Kit's shutdown hook can otherwise turn an uncaught Python error into exit 0.
        import os
        import traceback
        traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush()
        os._exit(1)
