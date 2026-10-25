"""Explicit from-scratch T1 easy/smooth entry; no checkpoint/load/resume argument."""
from publication_paths import research_path
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import os
from release_support import release_source_hashes
import shlex
import subprocess
import sys
import traceback

os.environ.pop('NATIVE_WBC_DIAGNOSTICS',None)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode',choices=['audit','train'],required=True)
parser.add_argument('--out',required=True)
parser.add_argument('--seed',type=int,default=21)
parser.add_argument('--num_envs',type=int,default=4096)
parser.add_argument('--updates',type=int,default=5000)
parser.add_argument('--interface_audit',help='Required for training: completed audit with matching source hashes and RNG-neutral observer')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.mode=='train' and (args.num_envs!=4096 or args.updates!=5000):
    parser.error('This authorized recipe is exactly 4096 environments and 5000 updates')
if args.mode=='train' and not args.interface_audit:
    parser.error('Training requires a completed, source-bound interface/RNG audit')
if args.mode=='audit' and not 4<=args.num_envs<=64:
    parser.error('Interface audit uses 4..64 environments and zero optimizer updates')
out=Path(args.out)
if out.exists(): parser.error('Refusing existing output directory')
app=AppLauncher(args).app

import torch
import yaml
import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks
from task_suite.config import build_cfg
from telemetry import plain,write_json,sha,snapshot,model_sha
from scratch_observer import ScratchTelemetry,InterfaceObserver

UPSTREAM='4e12109d1eccb1d9da2e4a35f189fabea3f13677'
NATIVE=Path(research_path('LeggedManip_Lab_upstream_wbc_20261023'))
NATIVE_OUTPUT=Path(research_path('go2_piper_upstream_wbc_20261023'))


def yaml_tree(path):
    def node(n):
        if isinstance(n,yaml.MappingNode): return {node(k):node(v) for k,v in n.value}
        if isinstance(n,yaml.SequenceNode): return [node(x) for x in n.value]
        if n.tag.startswith('tag:yaml.org,2002:python/name:'): return n.tag
        return n.value
    return node(yaml.compose(Path(path).read_text()))


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for k in sorted(set(a)|set(b)):
            result+=differences(a.get(k),b.get(k),path+'.'+k)
        return result
    if a!=b: return [{'path':path,'native':a,'scratch':b}]
    return []


def main():
    out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]
    worktree=NATIVE
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=NATIVE,text=True).strip()==UPSTREAM
    assert subprocess.check_output(['git','diff',UPSTREAM,'--','source','scripts'],cwd=worktree)==b''
    cfg=build_cfg('T1',difficulty='easy',num_envs=args.num_envs,seed=args.seed,ee_mode='smooth')
    cfg.sim.device=args.device or 'cuda:0'; cfg.log_dir=str(out)
    agent=load_cfg_from_registry('GO2-PIPER-WBC','rsl_rl_cfg_entry_point')
    agent=handle_deprecated_rsl_rl_cfg(agent,importlib.metadata.version('rsl-rl-lib'))
    agent.seed=args.seed;agent.device=cfg.sim.device;agent.max_iterations=args.updates
    agent.save_interval=5000;agent.resume=False
    reference=Path(os.environ['LEGGED_NATIVE_PARAMS'])
    native_agent=yaml.safe_load((reference/'agent.yaml').read_text())
    resolved=plain(agent)
    for name in ('actor','critic','algorithm','num_steps_per_env','clip_actions','check_for_nan'):
        assert resolved[name]==native_agent[name],f'Native recipe mismatch: {name}'
    assert cfg.episode_length_s==20 and cfg.decimation==4 and cfg.sim.dt==.005
    dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
    provenance=release_source_hashes()
    if args.mode=='train':
        admission=Path(args.interface_audit)
        assert json.loads((admission/'summary.json').read_text())['passed']
        rng=json.loads((admission/'observer_rng_audit.json').read_text())
        assert rng['cpu_cuda_rng_unchanged'] and rng['extra_get_observations_calls']==0
        audited=json.loads((admission/'launch.json').read_text())['source_hashes']
        assert audited == provenance, 'Publication source changed after interface audit'
    write_json(out/'launch.json',dict(argv=sys.argv,python=sys.executable,command=shlex.join([sys.executable,*sys.argv]),
        upstream=UPSTREAM,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=worktree,text=True).strip(),
        source_root=str(source),source_hashes=provenance,mode=args.mode,
        requested_updates=args.updates if args.mode=='train' else 0,requested_envs=args.num_envs,
        expected_transitions=5000*4096*24 if args.mode=='train' else 0,
        no_checkpoint_argument=True,no_resume=True,checkpoint_schedule=[0,500,1000,2000,3000,5000],
        interface_audit=args.interface_audit,
        stop_policy='NaN/Inf, OOM, simulation or confirmed pipeline fault; no KL-only or early-success stop',
        native_agent_sha256=sha(reference/'agent.yaml'),native_env_sha256=sha(reference/'env.yaml')))
    write_json(out/'resolved_config_difference.json',dict(
        agent=differences(yaml_tree(reference/'agent.yaml'),yaml_tree(out/'agent.yaml')),
        environment=differences(yaml_tree(reference/'env.yaml'),yaml_tree(out/'env.yaml'))))
    env=RslRlVecEnvWrapper(gym.make('GO2-PIPER-WBC',cfg=cfg),clip_actions=agent.clip_actions)
    raw=env.unwrapped
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out/'rsl_rl') if args.mode=='train' else None,device=agent.device)
    # Fail closed if any later code attempts weight/optimizer transfer.
    load_calls={'count':0}
    def forbid_load(*a,**kw):
        load_calls['count']+=1
        raise RuntimeError('Checkpoint loading is prohibited in this scratch entry')
    runner.load=forbid_load;runner.alg.load=forbid_load
    initial=snapshot(raw,runner)
    native_runtime=json.loads((NATIVE_OUTPUT/'formal_seed21/initial_runtime.json').read_text())
    assert initial['rewards']==native_runtime['rewards'],'Native reward configuration changed'
    assert initial['action_dim']==18 and initial['observation_dims']=={'policy':[210],'critic':[102]}
    assert runner.current_learning_iteration==0 and len(runner.alg.optimizer.state)==0
    assert runner.alg.learning_rate==5e-4 and runner.alg.schedule=='adaptive' and runner.alg.desired_kl==.01
    distribution={n:p.detach().clone() for n,p in runner.alg.actor.distribution.named_parameters()}
    assert distribution and all(bool((p==0).all()) for p in distribution.values()),'Initial log std is not zero'
    assert all(g['lr']==5e-4 for g in runner.alg.optimizer.param_groups)
    initial.update(load_call_count=load_calls['count'],fresh_distribution_parameters=plain(distribution),
        initialization='fresh constructor, no weight/optimizer/counter loading',reward_exactly_native=True,
        initial_direction_groups=plain(torch.bincount(torch.arange(raw.num_envs,device=raw.device)%4,minlength=4)),
        native_actual_command_ranges={k:v['cfg']['ranges'] for k,v in native_runtime['commands'].items()},
        latest_t1_parameters=plain(cfg.task_suite['params']))
    write_json(out/'initialization.json',initial)
    print('SCRATCH_INITIALIZATION',json.dumps({k:initial[k] for k in ('optimizer_state_entries','observation_dims','action_dim','actor_sha256','critic_sha256','fresh_distribution_parameters','initial_direction_groups','native_actual_command_ranges','latest_t1_parameters')}),flush=True)
    from audit_geometry_runtime import audit
    audit(raw,out)
    interface=InterfaceObserver(env,runner)
    try:
        if args.mode=='audit':
            obs=env.get_observations()
            # Demonstrate read-only capture/check leaves CPU and CUDA RNG intact.
            cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
            interface.capture_input(obs);interface.check_input()
            rng_unchanged=bool(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()))
            assert rng_unchanged
            getter=env.get_observations
            def forbidden_getter():
                raise RuntimeError('Observer tried to recompute observations during a step')
            env.get_observations=forbidden_getter
            with torch.inference_mode():
                for _ in range(160):
                    obs,_,_,_=env.step(runner.alg.act(obs))
                # Explicit synthetic reset probe, not a robot capability rollout.
                ids=torch.tensor([0,2],device=raw.device)
                others=torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device);others[ids]=False
                raw.episode_length_buf[ids]=137
                pose=raw.command_manager.get_term('ee_pose')
                pose._update_command()
                before=pose.command.clone()
                raw._reset_idx(ids)
                expected=torch.tensor(cfg.task_suite['params']['ee_center'],device=raw.device).repeat(2,1)
                expected[:,1]+=cfg.task_suite['params']['ee_amplitude'][1]
                defect=float((pose.command[ids,:3]-expected).abs().max())
                unchanged=bool(torch.equal(pose.command[others],before[others]))
                write_json(out/'reset_probe.json',dict(reset_phase_error_m=defect,nonreset_commands_unchanged=unchanged,
                    note='Synthetic clock/reset probe; no optimizer updates, not a physical capability rollout'))
                # Restore coherent manager output/history before the timeout test.
                raw.command_manager.compute(raw.step_dt)
                raw.obs_buf=raw.observation_manager.compute(update_history=True)
                env.get_observations=getter
                obs=env.get_observations()
                env.get_observations=forbidden_getter
                raw.episode_length_buf[1]=raw.max_episode_length-1
                env.step(runner.alg.act(obs))
            env.get_observations=getter
            write_json(out/'observer_rng_audit.json',dict(cpu_cuda_rng_unchanged=rng_unchanged,
                extra_get_observations_calls=0,getter_trap_steps=161,
                method='native PPO.act input capture; observation getter disabled during every audited step; zero optimizer updates'))
            write_json(out/'interface.json',interface.report())
            assert defect<1e-6 and unchanged,'Reset command phase defect; fix before training'
            assert len(runner.alg.optimizer.state)==0 and runner.current_learning_iteration==0
            write_json(out/'summary.json',dict(passed=True,completed_updates=0,interface_steps=interface.steps))
        else:
            telemetry=ScratchTelemetry(runner,out)
            telemetry.save('cp_0.pt')
            try:
                runner.learn(num_learning_iterations=5000,init_at_random_ep_len=False)
                assert telemetry.update_count==5000
                write_json(out/'summary.json',dict(completed_updates=5000,environment_steps=5000*4096*24,
                    load_calls=load_calls['count'],completed=True,recipe_unchanged=True))
            finally:
                write_json(out/'interface.json',interface.report())
                telemetry.close()
    finally:
        interface.close();env.close()
    print('SCRATCH_COMPLETE',str(out),flush=True)


try:
    main()
except BaseException as exc:
    if out.exists():
        write_json(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    raise
finally:
    app.close()
