"""Recipe V1 A: source-bound interface audit or one capped native-PPO scratch pilot."""
from publication_paths import research_path
import argparse
from pathlib import Path
import os
from release_support import release_source_hashes
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode',choices=['audit','train'],required=True)
parser.add_argument('--out',required=True)
parser.add_argument('--admission',required=True)
parser.add_argument('--interface-audit')
parser.add_argument('--num_envs',type=int,default=4096)
parser.add_argument('--updates',type=int,default=1200)
parser.add_argument('--seed',type=int,default=21)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();out=Path(args.out)
if out.exists():parser.error('Refusing existing output')
if args.mode=='train' and (args.num_envs!=4096 or not 1<=args.updates<=1200 or not args.interface_audit):
    parser.error('Train requires 4096 envs, <=1200 updates, and a passed source-bound audit')
if args.mode=='audit' and not 4<=args.num_envs<=64:parser.error('Audit uses4..64 envs and zero updates')
app=AppLauncher(args).app

import importlib.metadata
import json
import subprocess
import sys
import traceback
import torch
import yaml
import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks
from recipe_v1.config import build_recipe
from recipe_v1.control import ActuatorMonitor
from recipe_v1.telemetry import RecipeInterface,RecipeTelemetry
from telemetry import plain,write_json,sha,snapshot
from recipe_v1.provenance import differences,yaml_tree

NATIVE=Path(research_path('LeggedManip_Lab_upstream_wbc_20261023'))
NATIVE_RUNTIME=Path(research_path('go2_piper_upstream_wbc_20261023/formal_seed21/initial_runtime.json'))


def source_hashes(source):
    return release_source_hashes()


def main():
    out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]
    admission=json.loads(Path(args.admission).read_text())
    assert admission['passed']
    assert admission['control_source_sha256']==sha(source/'recipe_v1/control.py')
    cfg=build_recipe(admission=args.admission,num_envs=args.num_envs,seed=args.seed)
    cfg.sim.device=args.device or 'cuda:0';cfg.log_dir=str(out)
    agent=handle_deprecated_rsl_rl_cfg(load_cfg_from_registry('GO2-PIPER-WBC','rsl_rl_cfg_entry_point'),importlib.metadata.version('rsl-rl-lib'))
    agent.seed=args.seed;agent.device=cfg.sim.device;agent.max_iterations=args.updates;agent.save_interval=1200;agent.resume=False
    reference=Path(os.environ['LEGGED_NATIVE_PARAMS'])
    native_agent=yaml.safe_load((reference/'agent.yaml').read_text())
    resolved=plain(agent)
    for name in ('actor','critic','algorithm','num_steps_per_env','clip_actions','check_for_nan'):
        assert resolved[name]==native_agent[name],name
    assert cfg.decimation==4 and cfg.sim.dt==.005 and cfg.episode_length_s==20.
    provenance=source_hashes(source)
    if args.mode=='train':
        audit=Path(args.interface_audit)
        assert json.loads((audit/'summary.json').read_text())['passed']
        assert json.loads((audit/'observer_rng_audit.json').read_text())['cpu_cuda_rng_unchanged']
        audited=json.loads((audit/'launch.json').read_text())['source_hashes']
        assert provenance==audited,'Recipe code/config changed after interface audit'
    dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
    write_json(out/'launch.json',dict(argv=sys.argv,source=str(source),source_hashes=provenance,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        admission=str(args.admission),admission_sha256=sha(args.admission),mode=args.mode,
        requested_updates=args.updates if args.mode=='train' else 0,num_envs=args.num_envs,
        expected_completed_update_transitions=args.updates*4096*24 if args.mode=='train' else 0,
        checkpoint_schedule=[0,200,500,800,1200],stage='A',automatic_stage_upgrade=False))
    write_json(out/'resolved_diff.json',dict(agent=differences(yaml_tree(reference/'agent.yaml'),yaml_tree(out/'agent.yaml')),
        environment=differences(yaml_tree(reference/'env.yaml'),yaml_tree(out/'env.yaml'))))
    env=RslRlVecEnvWrapper(gym.make('GO2-PIPER-WBC',cfg=cfg),clip_actions=agent.clip_actions)
    raw=env.unwrapped;runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out/'rsl_rl') if args.mode=='train' else None,device=agent.device)
    load_calls={'count':0}
    def forbidden_load(*a,**kw):
        load_calls['count']+=1
        raise RuntimeError('Training/audit entry prohibits all checkpoint loading')
    runner.load=forbidden_load;runner.alg.load=forbidden_load
    initial=snapshot(raw,runner);reference_runtime=json.loads(NATIVE_RUNTIME.read_text())
    assert initial['action_dim']==18 and initial['observation_dims']=={'policy':[210],'critic':[102]}
    assert runner.current_learning_iteration==0 and len(runner.alg.optimizer.state)==0
    assert runner.alg.learning_rate==5e-4 and runner.alg.schedule=='adaptive' and runner.alg.desired_kl==.01
    distribution={n:p.detach().clone() for n,p in runner.alg.actor.distribution.named_parameters()}
    assert distribution and all(bool((p==0).all()) for p in distribution.values())
    changed={'track_lin_vel_xy_exp','track_ang_vel_z_exp','track_base_height_exp'}
    for name,value in reference_runtime['rewards'].items():
        if name not in changed:assert initial['rewards'][name]==value,'Unapproved reward change: '+name
    initial.update(load_calls=0,initialization='new actor/critic/distribution/Adam/counters; no weights loaded',fresh_distribution=plain(distribution),
        approved_reward_changes=sorted(changed|{'low_height_l2','raw_action_tail_l2'}),
        admitted_ee_points=admission['admitted_ee_points'])
    write_json(out/'initialization.json',initial)
    print('RECIPE_A_INITIALIZATION',json.dumps({k:initial[k] for k in ['load_calls','optimizer_state_entries','actor_sha256','critic_sha256','observation_dims','action_dim']}),flush=True)
    interface=RecipeInterface(env,runner);monitor=ActuatorMonitor(env)
    try:
        if args.mode=='audit':
            term=raw.action_manager.get_term('joint_pos');original=term.raw_actions.clone()
            bounds=raw.scene['robot'].data.soft_joint_pos_limits[:,term._joint_ids]
            extrema=[]
            for value in (-1000.,0.,1000.):
                term.process_actions(torch.full_like(original,value))
                excess=torch.maximum((bounds[...,0]+.02-term.processed_actions).clamp_min(0),(term.processed_actions-bounds[...,1]+.02).clamp_min(0)).max()
                assert float(excess)<1e-6;extrema.append({'raw':value,'max_target_excess_rad':float(excess)})
            term.process_actions(original)
            obs=env.get_observations()
            cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
            interface.capture_input(obs);interface.check_input()
            unchanged=bool(torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state()))
            assert unchanged
            getter=env.get_observations
            def forbid_getter():raise RuntimeError('Observer performed an extra observation read')
            env.get_observations=forbid_getter
            pool=torch.tensor(admission['admitted_ee_points'],device=raw.device)
            with torch.inference_mode():
                for index in range(650):
                    obs,_,_,_=env.step(runner.alg.act(obs))
                    if index%64==0:
                        target=raw.command_manager.get_command('ee_pose')[:,:3]
                        nearest=(target[:,None]-pool[None]).norm(dim=-1).amin(-1)
                        assert float(nearest.max())<1e-6
                raw.episode_length_buf[0]=raw.max_episode_length-1
                env.step(runner.alg.act(obs))
            env.get_observations=getter
            assert len(runner.alg.optimizer.state)==0 and runner.current_learning_iteration==0
            write_json(out/'observer_rng_audit.json',dict(cpu_cuda_rng_unchanged=unchanged,extra_get_observations_calls=0,getter_trap_steps=651))
            write_json(out/'synthetic_target_extrema.json',extrema)
            write_json(out/'summary.json',dict(passed=True,completed_updates=0,steps=interface.steps,all_sampled_goals_from_admitted_pool=True))
        else:
            telemetry=RecipeTelemetry(runner,out,monitor);telemetry.save('cp_0.pt')
            try:
                runner.learn(num_learning_iterations=args.updates,init_at_random_ep_len=False)
                assert telemetry.update_count==args.updates
                write_json(out/'summary.json',dict(completed=True,completed_updates=telemetry.update_count,
                    environment_steps=telemetry.update_count*4096*24,load_calls=load_calls['count'],stage='A',automatic_B_C=False))
            finally:telemetry.close()
        write_json(out/'interface.json',interface.report())
        print('RECIPE_A_COMPLETE',str(out),flush=True)
    finally:
        monitor.close();interface.close();env.close()


try:
    main()
except BaseException as exc:
    print(traceback.format_exc(),flush=True)
    if out.exists():write_json(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    raise
finally:app.close()
