"""One source-bound A+ continuation from own CP1200, with native PPO untouched."""
import argparse
from pathlib import Path
from publication_paths import research_path
from release_support import release_source_hashes
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['audit','train'],required=True);p.add_argument('--out',required=True)
p.add_argument('--admission',required=True);p.add_argument('--interface-audit');p.add_argument('--num_envs',type=int,default=4096)
p.add_argument('--additional-updates',type=int,default=600);p.add_argument('--seed',type=int,default=21)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();out=Path(args.out)
if out.exists():p.error('New output required')
if args.mode=='train' and (args.num_envs!=4096 or args.additional_updates!=600 or not args.interface_audit):p.error('Train requires4096 envs,600 additional updates and passed audit')
if args.mode=='audit' and args.num_envs!=32:p.error('Audit requires32 envs')
app=AppLauncher(args).app
import importlib.metadata,json,subprocess,sys,traceback
import torch
import gymnasium as gym
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import LeggedManip_Lab.tasks
from recipe_v1.config import build_recipe
from recipe_v1.control import ActuatorMonitor
from recipe_v1.telemetry import RecipeInterface
from recipe_v1.provenance import differences,yaml_tree
from recipe_aplus.config import build_aplus
from recipe_aplus.state import restore_native,restore_rng,equal_tree,tensor_hash
from recipe_aplus.telemetry import APlusTelemetry
from telemetry import plain,write_json,sha,snapshot


def hashes(source):
    return release_source_hashes()


def main():
    out.mkdir(parents=True,exist_ok=False);source=Path(__file__).resolve().parents[1]
    spec=json.loads((source/'recipe_aplus/spec.json').read_text());checkpoint=Path(research_path(spec['parent_checkpoint']))
    assert sha(checkpoint)==spec['parent_sha256']
    cfg=build_aplus(admission=args.admission,num_envs=args.num_envs,seed=args.seed);cfg.sim.device=args.device or 'cuda:0';cfg.log_dir=str(out)
    base=build_recipe(admission=args.admission,num_envs=args.num_envs,seed=args.seed)
    for field in ['rewards','actions','observations','events','terminations','scene']:
        assert plain(getattr(cfg,field))==plain(getattr(base,field)),field
    agent=handle_deprecated_rsl_rl_cfg(load_cfg_from_registry('GO2-PIPER-WBC','rsl_rl_cfg_entry_point'),importlib.metadata.version('rsl-rl-lib'))
    agent.seed=args.seed;agent.device=cfg.sim.device;agent.max_iterations=1800;agent.save_interval=1200;agent.resume=False
    parent=Path(research_path('go2_piper_recipe_v1_20261025/pilot_A_seed21'))
    import yaml
    parent_agent=yaml.safe_load((parent/'agent.yaml').read_text());resolved=plain(agent)
    for name in ['actor','critic','algorithm','num_steps_per_env','clip_actions','check_for_nan']:assert resolved[name]==parent_agent[name],name
    source_hashes=hashes(source)
    if args.mode=='train':
        audit=Path(args.interface_audit)
        assert json.loads((audit/'summary.json').read_text())['passed']
        assert json.loads((audit/'launch.json').read_text())['source_hashes']==source_hashes,'Training code changed after audit'
    dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
    write_json(out/'resolved_diff_from_V1.json',dict(environment=differences(yaml_tree(parent/'env.yaml'),yaml_tree(out/'env.yaml')),
        agent=differences(yaml_tree(parent/'agent.yaml'),yaml_tree(out/'agent.yaml'))))
    write_json(out/'launch.json',dict(argv=sys.argv,source=str(source),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),source_hashes=source_hashes,
        parent_checkpoint=str(checkpoint),parent_sha256=sha(checkpoint),mode=args.mode,additional_update_budget=600,first_new_completed_update=1201,last_completed_update=1800,
        expected_new_transitions=58982400,expected_cumulative_transitions=176947200,checkpoint_schedule=[1400,1600,1800],startup_ramp_applied=False))
    env=RslRlVecEnvWrapper(gym.make('GO2-PIPER-WBC',cfg=cfg),clip_actions=agent.clip_actions)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out/'rsl_rl') if args.mode=='train' else None,device=agent.device)
    cp,evidence=restore_native(runner,checkpoint)
    raw=env.unwrapped
    assert raw.action_manager.total_action_dim==18 and plain(raw.observation_manager.group_obs_dim)=={'policy':[210],'critic':[102]}
    initial=snapshot(raw,runner);initial.update(evidence)
    write_json(out/'continuation_initialization.json',initial);write_json(out/'resume_evidence.json',evidence)
    observer=RecipeInterface(env,runner);monitor=ActuatorMonitor(env)
    try:
        if args.mode=='audit':
            term=raw.command_manager.get_term('base_velocity');ids=torch.arange(raw.num_envs,device=raw.device)
            # Audit-only direct sampling. Never runs in the training process.
            term.sample_counts.zero_()
            for _ in range(2048):term.pick(ids)
            counts=term.sample_counts.reshape(3,4).cpu();total=int(counts.sum());nonzero=int(counts[1:].sum())
            assert abs(float(counts[0].sum())/total-.1)<.01
            assert abs(float(counts[1].sum())/nonzero-.2)<.01
            for row in counts[1:]:assert bool((row.float()/row.sum()-.25).abs().max()<.015)
            write_json(out/'sampling_audit.json',dict(counts_kind_direction=counts.tolist(),draws=total,zero_fraction=float(counts[0].sum())/total,low_given_nonzero=float(counts[1].sum())/nonzero))
            restore_rng(cp['infos']);obs,_=env.reset()
            before=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state().clone()
            observer.capture_input(obs);observer.check_input()
            assert torch.equal(before,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
            getter=env.get_observations
            def forbid():raise RuntimeError('Extra observer observation getter')
            env.get_observations=forbid
            with torch.inference_mode():
                for _ in range(650):obs,_,_,_=env.step(runner.alg.act(obs))
                raw.episode_length_buf[0]=raw.max_episode_length-1;env.step(runner.alg.act(obs))
            env.get_observations=getter
            assert equal_tree(runner.alg.actor.state_dict(),cp['actor_state_dict']) and equal_tree(runner.alg.critic.state_dict(),cp['critic_state_dict'])
            assert equal_tree(runner.alg.optimizer.state_dict(),cp['optimizer_state_dict'])
            write_json(out/'summary.json',dict(passed=True,optimizer_updates=0,steps=observer.steps,source_bound=True,observer_rng_unchanged=True,extra_observation_reads=0))
        else:
            telemetry=APlusTelemetry(runner,out,monitor,checkpoint)
            restore_rng(cp['infos']);telemetry.save('resume_1200.pt')
            assert tensor_hash(torch.get_rng_state())==tensor_hash(cp['infos']['rng_cpu'])
            # New PhysX episode; this is learner-state continuation, not a physical snapshot.
            env.reset()
            try:
                runner.learn(num_learning_iterations=600,init_at_random_ep_len=False)
                assert telemetry.update_count==1800
                write_json(out/'summary.json',dict(completed=True,parent_completed_updates=1200,additional_updates=600,completed_updates=1800,
                    new_transitions=58982400,cumulative_transitions=176947200,stage='A+',startup_ramp_applied=False))
            finally:telemetry.close()
        write_json(out/'interface.json',observer.report());print('APLUS_COMPLETE',out,flush=True)
    finally:monitor.close();observer.close();env.close()


try:main()
except BaseException as exc:
    print(traceback.format_exc(),flush=True)
    if out.exists():write_json(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    raise
finally:app.close()
