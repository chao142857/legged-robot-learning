"""Prepared shared B2/B3 assembly. Import after AppLauncher in a native checkout.

The only controller and composition wrapper is B3's. The only feedback boundary
and sampler are B1's. This module builds neither IK nor SGS implementations.
"""
import copy
import json
from pathlib import Path
import sys
from .dependencies import ROOT, DEFAULTS, verify, sha, enable_owner_imports
from .task_commands import make_native_phase_command, phases_from_assignments
from .metrics import EpisodeMetrics


def common_inputs():
    verify(); enable_owner_imports()
    executed=DEFAULTS['sgs']/'reference/executed'
    for p in (executed,executed/'tools'):
        if str(p) not in sys.path: sys.path.insert(0,str(p))
    protocol=json.loads((ROOT/'protocol/common_protocol.json').read_text())
    support=json.loads((ROOT/'protocol/task_support.json').read_text())
    from shared_sgs import TaskPool, TaskSpec
    pool=TaskPool([TaskSpec.from_content(c) for c in support['content']],support['support_version'])
    return protocol,pool


def configure(pool, phases, num_envs, seed, device, out):
    from task_suite.config import build_cfg
    cfg=build_cfg('T1',difficulty='easy',num_envs=num_envs,seed=seed,ee_mode='smooth')
    cfg.factorial_initial_phases=list(phases)
    cfg.commands.base_velocity.class_type=make_native_phase_command()
    cfg.sim.device=device; cfg.log_dir=str(out)
    if cfg.episode_length_s!=20 or cfg.sim.dt!=.005 or cfg.decimation!=4:
        raise RuntimeError('Native timestep/horizon changed')
    return cfg


def assemble_training(*,method,run_id,sampler_seed,device,out,pair_path,pair_sha256):
    if method not in ('B2','B3'): raise ValueError('Shared leg-PPO/IK methods are B2 and B3')
    protocol,pool=common_inputs()
    from shared_sgs import SamplerSession,T1_RULE_VERSION
    session=SamplerSession(pool,run_id=run_id,seed=sampler_seed,mode='sgs' if method=='B2' else 'uniform',feedback_rule_version=T1_RULE_VERSION)
    n=protocol['budget']['num_envs']
    initial=session.assign(range(n),initial_task_ids={i:pool.ids[i%len(pool.ids)] for i in range(n)})
    cfg=configure(pool,phases_from_assignments(initial,pool),n,21,device,out)
    agent=copy.deepcopy(protocol['recipe']);agent['device']=device;agent['resume']=False
    if sha(pair_path)!=pair_sha256: raise RuntimeError('Shared untrained pair artifact changed')
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from shared_ik.runtime import LegPPOArmIKWrapper
    from shared_ik.initialization import load_untrained_pair
    from integration.pre_reset import PreResetObserver
    from telemetry import physical_suite
    import LeggedManip_Lab.tasks
    from isaaclab.utils.io import dump_yaml
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
    env=LegPPOArmIKWrapper(gym.make('GO2-PIPER-WBC',cfg=cfg),clip_actions=agent['clip_actions'])
    runner=OnPolicyRunner(env,copy.deepcopy(agent),log_dir=str(out/'rsl_rl'),device=device)
    pair=load_untrained_pair(runner,pair_path)
    if env.num_actions!=12 or env.unwrapped.action_manager.total_action_dim!=18:
        raise RuntimeError('PPO12/native18 dimension binding mismatch')
    if len(runner.alg.optimizer.state) or runner.current_learning_iteration:
        raise RuntimeError('Not a fresh optimizer/counter')
    def forbid_load(*a,**kw):raise RuntimeError('Training checkpoint loading prohibited in scratch entry')
    runner.load=forbid_load;runner.alg.load=forbid_load
    from .logging import ContextLog
    log=ContextLog(out/'episodes_and_assignments.jsonl',dict(run_id=run_id,support_version=pool.support_version,
        support_sha256=pool.support_sha256,protocol_version=protocol['protocol_version'],
        protocol_sha256=sha(ROOT/'protocol/common_protocol.json')))
    boundary=PreResetObserver(env,session,EpisodeMetrics(n,1000),physical_suite,log=log)
    return dict(protocol=protocol,pool=pool,session=session,cfg=cfg,agent=agent,env=env,runner=runner,boundary=boundary,log=log,pair=pair)


def make_telemetry(assembly,out):
    import random
    import numpy as np
    import torch
    import os
    from telemetry import TrainingTelemetry,write_json,model_sha
    from scratch_observer import ScratchTelemetry
    env,runner,boundary=assembly['env'],assembly['runner'],assembly['boundary']
    class PairedTelemetry(ScratchTelemetry):
        # Keep native generic diagnostics. Legacy training direction breakdown
        # derives phase from env_id and is invalid after sampled reassignment.
        reward=TrainingTelemetry.reward
        def save(self,filename):
            target=self.out/filename
            if target.exists():raise FileExistsError(target)
            cp=self.runner.alg.save()
            raw=env.unwrapped
            cp.update(iter=self.update_count-1,infos=dict(completed_updates=self.update_count,
                environment_steps=self.update_count*raw.num_envs*24,
                initialization='same native untrained12 pair; empty new Adam',
                protocol_sha256=sha(ROOT/'protocol/common_protocol.json'),support_sha256=assembly['pool'].support_sha256,
                dependency_lock_sha256=sha(ROOT/'protocol/dependencies.lock.json'),
                run_id=assembly['session'].run_id,sampling=boundary.state_dict(),controller=env.controller_state_dict(),
                native_action18=raw.action_manager.action.clone(),native_prev_action18=raw.action_manager.prev_action.clone(),
                phase_offsets=raw.command_manager.get_term('base_velocity').phase_offsets.clone(),
                rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
                restore_scope='module state; full simulator/commands not serialized, resume requires explicit censor/reset',
                learning_rate=runner.alg.learning_rate))
            temporary=target.with_suffix('.partial');torch.save(cp,temporary);os.replace(temporary,target)
            write_json(target.with_suffix('.json'),dict(path=str(target),sha256=sha(target),completed_updates=self.update_count,
                actor_sha256=model_sha(runner.alg.actor),critic_sha256=model_sha(runner.alg.critic)))
    return PairedTelemetry(runner,out)
