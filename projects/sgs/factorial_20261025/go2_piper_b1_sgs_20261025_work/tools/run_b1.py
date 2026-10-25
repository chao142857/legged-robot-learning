"""Prepared B0r/B1 paired scratch entry; never invoked in the CPU-only round.

Fail before importing Isaac unless the source-bound common review/physical
interface record has been supplied. No checkpoint/resume argument exists.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admit(path, protocol_path):
    record = json.loads(Path(path).read_text())
    for key in ('baseline_review_bound', 'phase4_and_label_bound', 'native_boundary_verified', 'formal_run_authorized'):
        if record.get(key) is not True:
            raise RuntimeError('Pending common binding: '+key)
    if record['common_protocol_sha256'] != sha(protocol_path):
        raise RuntimeError('Common protocol changed since binding')
    for item in record['source_files']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Source changed since binding: '+item['path'])
    return record


def main():
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--admission', required=True)
    early.add_argument('--protocol', required=True)
    known, _ = early.parse_known_args()
    admit(known.admission, known.protocol)
    # Deliberately after all CPU-readable admission checks.
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(parents=[early], description=__doc__)
    parser.add_argument('--method', choices=('B0r', 'B1'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--sampler-seed', type=int, default=9021)
    parser.add_argument('--pair-initialization-json')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists(): parser.error('Refusing existing output directory')
    if not os.environ.get('B1_NATIVE_WORKTREE'): parser.error('Independent native checkout required')
    worktree = Path(os.environ['B1_NATIVE_WORKTREE']).resolve()
    if subprocess.check_output(['git','rev-parse','HEAD'], cwd=worktree, text=True).strip() != '4e12109d1eccb1d9da2e4a35f189fabea3f13677':
        parser.error('Native source commit mismatch')
    if subprocess.check_output(['git','diff','HEAD','--','source','scripts'], cwd=worktree):
        parser.error('Native source/scripts have modifications')
    protocol = json.loads(Path(args.protocol).read_text())
    budget = protocol['budget']
    assert (budget['num_envs'], budget['rollout_steps'], budget['updates'], budget['transitions']) == (4096,24,5000,491520000)
    app = AppLauncher(args).app
    try:
        import random
        import numpy as np
        import torch
        import yaml
        import gymnasium as gym
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        import LeggedManip_Lab.tasks
        from task_suite.config import build_cfg
        from telemetry import snapshot, model_sha, write_json, physical_suite, TrainingTelemetry
        from scratch_observer import ScratchTelemetry, InterfaceObserver
        from b2.task_commands import make_native_phase_command, phases_from_assignments
        from b2.metrics import EpisodeMetrics
        from shared_sgs import TaskSpec, TaskPool, SamplerSession, T1_RULE_VERSION
        from integration.pre_reset import PreResetObserver

        support = json.loads((Path(args.protocol).parent/protocol['task_support_file']).read_text())
        pool = TaskPool([TaskSpec.from_content(c) for c in support['content']], support['support_version'])
        session = SamplerSession(pool, run_id=args.run_id, seed=args.sampler_seed,
                                  mode='sgs' if args.method == 'B1' else 'uniform', feedback_rule_version=T1_RULE_VERSION)
        initial = session.assign(range(4096), initial_task_ids={i:pool.ids[i%4] for i in range(4096)})
        cfg = build_cfg('T1', difficulty='easy', num_envs=4096, seed=21, ee_mode='smooth')
        cfg.factorial_initial_phases = phases_from_assignments(initial, pool)
        cfg.commands.base_velocity.class_type = make_native_phase_command()
        cfg.sim.device = args.device or 'cuda:0'
        out.mkdir(parents=True, exist_ok=False)
        cfg.log_dir = str(out)
        agent = yaml.safe_load((ROOT/'reference/resolved/agent.yaml').read_text())
        agent['device'] = cfg.sim.device
        assert agent['seed'] == 21 and agent['num_steps_per_env'] == 24 and not agent['resume']
        dump_yaml(str(out/'env.yaml'), cfg); dump_yaml(str(out/'agent.yaml'), agent)
        env = RslRlVecEnvWrapper(gym.make('GO2-PIPER-WBC', cfg=cfg), clip_actions=agent['clip_actions'])
        runner = OnPolicyRunner(env, copy.deepcopy(agent), log_dir=str(out/'rsl_rl'), device=agent['device'])
        def forbidden(*a, **kw): raise RuntimeError('Trained checkpoint loading is prohibited')
        runner.load = forbidden; runner.alg.load = forbidden
        raw = env.unwrapped
        init = snapshot(raw, runner)
        assert init['action_dim'] == 18 and init['observation_dims'] == {'policy':[210], 'critic':[102]}
        assert len(runner.alg.optimizer.state) == 0 and runner.current_learning_iteration == 0
        assert runner.alg.learning_rate == .0005 and runner.alg.schedule == 'adaptive' and runner.alg.desired_kl == .01
        assert all(bool((p == 0).all()) for p in runner.alg.actor.distribution.parameters())
        init.update(method=args.method, run_id=args.run_id, sampler_seed=args.sampler_seed,
                    initialization='fresh constructor; no trained weights; paired hashes required',
                    protocol_sha256=sha(args.protocol), support_sha256=pool.support_sha256)
        if args.pair_initialization_json:
            paired = json.loads(Path(args.pair_initialization_json).read_text())
            for key in ('actor_sha256','critic_sha256'):
                assert init[key] == paired[key], 'Paired fresh initialization mismatch: '+key
        elif args.method == 'B1':
            raise RuntimeError('B1 requires the B0r fresh initialization JSON for bitwise comparison')
        write_json(out/'initialization.json', init)
        write_json(out/'launch.json', dict(method=args.method, run_id=args.run_id, seed=21, sampler_seed=args.sampler_seed,
            budget=budget, protocol_sha256=sha(args.protocol), support_sha256=pool.support_sha256,
            initialization='fresh constructor, no checkpoint loading', reviewed_baseline_modified=False))
        metrics = EpisodeMetrics(4096, 1000)
        assignment_log = (out/'episodes_and_assignments.jsonl').open('x', buffering=1)
        boundary = PreResetObserver(env, session, metrics, physical_suite, log=assignment_log)
        for token in initial:
            from dataclasses import asdict
            boundary.emit('assignment', **asdict(token), probability=.25, initialization='paired_balanced_CP0')
        interface = InterfaceObserver(env, runner)

        class B1Telemetry(ScratchTelemetry):
            # Generic native diagnostics remain valid with arbitrary task phases.
            # Legacy env-index-only training direction diagnostics are not reused.
            reward = TrainingTelemetry.reward
            def save(self, filename):
                target = self.out/filename
                if target.exists(): raise FileExistsError(target)
                saved = self.runner.alg.save()
                saved.update(iter=self.update_count-1, infos=dict(completed_updates=self.update_count,
                    environment_steps=self.update_count*4096*24, initialization='from scratch, paired constructor hashes',
                    protocol_sha256=sha(args.protocol), support_sha256=pool.support_sha256,
                    rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all(), learning_rate=self.runner.alg.learning_rate,
                    sampling=boundary.state_dict(), command_state='native command/physics state not serialized; resume must censor/reset',
                    run_id=args.run_id))
                temporary = target.with_suffix('.partial')
                torch.save(saved, temporary); os.replace(temporary, target)
                write_json(target.with_suffix('.json'), dict(path=str(target), sha256=sha(target),
                    completed_updates=self.update_count, actor_sha256=model_sha(self.runner.alg.actor),
                    critic_sha256=model_sha(self.runner.alg.critic), optimizer_state_entries=len(self.runner.alg.optimizer.state)))

        telemetry = B1Telemetry(runner, out)
        try:
            telemetry.save('cp_0.pt')
            runner.learn(num_learning_iterations=5000, init_at_random_ep_len=False)
            assert telemetry.update_count == 5000 and sum(session.transitions) == 491520000
            for row in session.censor_active('training_cutoff'): boundary.emit('censored', **row)
            write_json(out/'sampling_final.json', session.summary())
            write_json(out/'summary.json', dict(completed_updates=5000, environment_steps=sum(session.transitions),
                completed=True, method=args.method, no_sgs_benefit_claim=True))
        except BaseException:
            for row in session.censor_active('interruption'): boundary.emit('censored', **row)
            write_json(out/'sampling_interrupted.json', session.summary())
            raise
        finally:
            telemetry.close(); interface.close(); boundary.close(); assignment_log.close(); env.close()
    finally:
        app.close()


if __name__ == '__main__':
    main()
