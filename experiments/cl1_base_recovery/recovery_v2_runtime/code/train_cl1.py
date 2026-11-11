"""Native PPO counterfactual forks and frozen-Teacher transfer, isolated V0.2."""
import argparse
import json
from pathlib import Path
import statistics
import time
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--mode', choices=('branch', 'transfer'), default='branch')
parser.add_argument('--source', type=Path)
parser.add_argument('--resume', type=Path)
parser.add_argument('--probability', type=Path)
parser.add_argument('--decision-source', type=Path)
parser.add_argument('--branch', default='uniform')
parser.add_argument('--target', type=int, required=True)
parser.add_argument('--student-seed', type=int, required=True)
parser.add_argument('--teacher-in', type=Path)
parser.add_argument('--transfer-kind', choices=('uniform', 'fresh', 'meta'))
parser.add_argument('--smoke', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def train():
    import copy
    import gymnasium as gym
    import numpy as np
    import torch
    import LeggedManip_Lab.tasks
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    from commands import SequenceVelocity, SequenceWBC
    from validation import command_equivalence
    from common import ROOT, PROTOCOL, Pool, digest, emit, file_sha, model_cfg
    from common import plain, restore_rng, rng_state, tensor_hash, write
    from fork_state import capture, restore, tree_hash
    from checkpoint_v02 import capture_environment
    from fork_v02 import partial_features, learning_progress, restored_owner
    from neural_adapter import NeuralAdapter
    from neural_v01 import NeuralSampler, DYNAMIC_FIELDS
    from neural_core import distribution_stats, state_hash
    from protocol import CONFIG, TRAIN_POOL_SHA

    branch = args.mode == 'branch'
    assert args.target > 0 and args.target <= 10000
    assert not (args.source and args.resume)
    assert branch or args.source is None
    assert not branch or (args.source or args.resume) and args.probability
    assert branch or args.transfer_kind is not None and args.teacher_in is not None
    if not args.smoke:
        assert args.target % 100 == 0
    out, dataset = args.out, args.dataset
    dataset.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/'prepared/manifest.json').read_text())
    binding = manifest['binding']
    pool = Pool('wbc', 'train')
    assert pool.sha == TRAIN_POOL_SHA and pool.n == 32768
    teacher = None
    if not branch:
        from cl1_proxy import Teacher
        static = np.load(ROOT/'prepared/command_features.npy', allow_pickle=False)
        teacher = Teacher(static, binding, out, args.student_seed)
        snapshot = torch.load(args.teacher_in, map_location='cpu', weights_only=False)
        teacher.load(snapshot['teacher'])
        teacher_identity = tree_hash(teacher.state())
    if branch:
        probability = np.load(args.probability, allow_pickle=False)
        assert probability.shape == (pool.n,)
        assert probability.min() >= .2/pool.n-1e-15
    else:
        # All three new students receive an identical first physical reset.
        # Their sampling intervention begins after model initialization/dev0.
        probability = np.full(pool.n, 1/pool.n, dtype=np.float64)
    decision = dict(version=0, probability=probability.copy(), distribution=distribution_stats(probability))
    cfg, agent = model_cfg('GO2-PIPER-WBC')
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 4096, args.student_seed, 'cuda:0'
    cfg.log_dir = str(out/'policy')
    native = plain(cfg.to_dict())
    baseline = json.loads((ROOT/'parent/reference_configuration.json').read_text())
    expected = copy.deepcopy(baseline['native_environment'])
    expected['log_dir'], expected['seed'] = native['log_dir'], args.student_seed
    expected_agent = copy.deepcopy(baseline['agent'])
    expected_agent['seed'] = agent.seed
    write(out/'native_comparison.json', dict(passed=expected == native and expected_agent == plain(agent.to_dict()),
        expected_environment=expected, actual_environment=native,
        allowed_differences=['log directory', 'explicit student seed']))
    assert expected == native, 'Native environment differs'
    assert expected_agent == plain(agent.to_dict()), 'Native PPO differs'
    agent.seed = args.student_seed
    write(out/'configuration.json', dict(native_environment=native, agent=plain(agent.to_dict()),
        mode=args.mode, branch=args.branch, student_seed=args.student_seed, target=args.target,
        teacher_frozen=True, transfer_kind=args.transfer_kind, teacher_checkpoint=str(args.teacher_in),
        probability_file=str(args.probability), source_checkpoint=str(args.source or args.resume),
        protocol=PROTOCOL, binding=binding, num_envs=4096, rollout=24, smoke=args.smoke))
    cfg.commands.ee_pose.class_type = SequenceWBC
    cfg.commands.base_velocity.class_type = SequenceVelocity
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    completed = optimizer_calls = base_update = 0
    evaluation_seconds = evaluation_steps = 0
    dev_before = None
    phase = 'initialized'
    latest_row = None
    previous_transitions = np.zeros(pool.n, dtype=np.int64)
    source_hash = None
    start_wall = time.monotonic()
    env = gym.make('GO2-PIPER-WBC', cfg=cfg)
    try:
        raw = env.unwrapped
        adapter = NeuralAdapter(raw, pool, out, decision)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        assert wrapped.num_actions == 18 and raw.max_episode_length == 1000
        for name in ('ee_pose', 'base_velocity'):
            term = raw.command_manager.get_term(name)
            assert not term.cfg.curriculum_enabled and plain(term.cfg.ranges) == plain(term.cfg.limit_ranges)

        def save_pt(path, value):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            assert not path.exists(), f'Never overwrite evidence: {path}'
            temporary = path.with_suffix('.tmp')
            torch.save(value, temporary)
            temporary.replace(path)

        class Runner(OnPolicyRunner):
            def save(self, path, infos=None):
                path = Path(path)
                if path.parent == out/'policy':
                    return
                adapter.owner.check()
                state = self.alg.save()
                state.update(iter=completed-1, infos=dict(completed_updates=completed,
                    optimizer_calls=optimizer_calls, environment_transitions=completed*4096*24,
                    adaptive_learning_rate=self.alg.learning_rate, rng=rng_state(),
                    adapter=adapter.state_dict(), environment=capture_environment(completed, raw, self.logger, capture),
                    environment_snapshot_available=completed > 0,
                    checkpoint_scope='full_resume' if completed > 0 else 'initialization_only_recreate_from_recorded_seed',
                    protocol_sha256=digest(PROTOCOL), pool_sha256=pool.sha, binding=binding,
                    student_seed=args.student_seed, base_update=base_update, new_updates=completed-base_update,
                    mode=args.mode, branch=args.branch, transfer_kind=args.transfer_kind,
                    source_checkpoint_sha256=source_hash, latest_row=latest_row, teacher=teacher.state() if teacher else None,
                    decision=decision, teacher_frozen=True, phase=phase,
                    dev_before=dev_before, evaluation_seconds=evaluation_seconds,
                    evaluation_steps=evaluation_steps, previous_transitions=previous_transitions,
                    experience=str(dataset), configuration_sha256=file_sha(out/'configuration.json')))
                save_pt(path, state)

        runner = Runner(wrapped, agent.to_dict(), log_dir=str(out/'policy'), device=agent.device)
        actor, critic = runner.alg.actor, runner.alg.critic
        initialization = dict(actor_sha256=tensor_hash(actor.state_dict()), critic_sha256=tensor_hash(critic.state_dict()),
            optimizer_empty=not runner.alg.optimizer.state, seed=args.student_seed,
            pool_sha256=pool.sha, initial_sampler_uniform=not branch)
        write(out/'initialization.json', initialization)
        equivalence = command_equivalence(raw.command_manager.get_term('ee_pose'))
        write(out/'command_equivalence.json', equivalence)
        assert equivalence['passed']

        def frozen_actor():
            path = out/'actors'/f'student_{completed:05d}.pt'
            if not path.exists():
                save_pt(path, dict(actor_state_dict={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                    infos=dict(completed_updates=completed, protocol_sha256=digest(PROTOCOL),
                        pool_sha256=pool.sha, binding=binding, student_seed=args.student_seed)))
            return path

        def evaluate(path, kind):
            nonlocal evaluation_seconds, evaluation_steps
            identity = (tree_hash(adapter.state_dict()), tree_hash(rng_state()), tree_hash(teacher.state()))
            request = out/'evaluation_requests'/f'{completed:05d}-{kind}.json'
            assert not request.exists()
            write(request, dict(update=completed, kind=kind, checkpoint=str(path), checkpoint_sha256=file_sha(path)))
            tick = time.monotonic()
            acknowledgement = request.with_suffix('.done.json')
            while not acknowledgement.exists():
                if time.monotonic()-tick > 1800:
                    raise TimeoutError(str(request))
                time.sleep(.5)
            reply = json.loads(acknowledgement.read_text())
            assert reply['passed'] and reply['checkpoint_sha256'] == file_sha(path) and reply['kind'] == kind
            assert identity == (tree_hash(adapter.state_dict()), tree_hash(rng_state()), tree_hash(teacher.state()))
            evaluation_seconds += time.monotonic()-tick
            evaluation_steps += reply['executed_environment_transitions']
            if kind == 'test':
                return None  # Formal results cannot condition Teacher decisions.
            assert len(reply['scores']) == 512
            assert reply['pool_sha256'] == binding['development_pool_sha256']
            return reply

        def global_state(row=None, dev=None):
            dynamic = adapter.owner.engine.features(adapter.owner.control_steps, adapter.owner.transitions)
            recent = dynamic[:, DYNAMIC_FIELDS.index('recent_feedback_observed')] > .5
            g = np.zeros(10, dtype=np.float32)
            g[0], g[1] = completed/10000., runner.alg.learning_rate/1e-3
            g[2] = row['action_std'] if row is not None else agent.to_dict()['actor']['distribution_cfg']['init_std']
            if row is not None:
                g[3] = row['losses'].get('value', row['losses'].get('value_loss', 0.))
                g[4] = (row['mean_training_reward'] or 0)/200.
            g[5] = recent.mean()
            g[6] = dynamic[recent, DYNAMIC_FIELDS.index('recent_good_fraction')].mean() if recent.any() else 0.
            g[7] = dynamic[recent, DYNAMIC_FIELDS.index('recent_failure')].mean() if recent.any() else 0.
            if dev is not None:
                g[8], g[9] = dev['continuous_joint_score'], 1.
            return dynamic, g

        def new_decision(row=None):
            dynamic, g = global_state(row, dev_before)
            engine = adapter.owner.engine
            partial, partial_receipt = partial_features(adapter.state_dict())
            progress, progress_observed = learning_progress(engine, adapter.owner.control_steps)
            teacher.dev = dev_before
            result = teacher.decide(dynamic, g, engine.version+1,
                progress=progress, progress_observed=progress_observed, partial=partial)
            if args.transfer_kind == 'uniform':
                result['probability'] = np.full(pool.n, 1/pool.n, dtype=np.float64)
                result['distribution'] = distribution_stats(result['probability'])
            result.update(partial_receipt=partial_receipt, student_update=completed,
                teacher_frozen=True, transfer_kind=args.transfer_kind,
                feedback_snapshot=engine.state_dict(), transitions_snapshot=adapter.owner.transitions.copy())
            assert tree_hash(teacher.state()) == teacher_identity
            return result

        def preserve_decision():
            path = dataset/f'cycle_{decision["version"]:04d}'
            path.mkdir(exist_ok=False)
            save_pt(path/'decision.pt', dict(decision=decision, active_episodes=adapter.owner.token(np.flatnonzero(adapter.owner.task >= 0)),
                student_seed=args.student_seed, completed_updates=completed, source_checkpoint_sha256=source_hash,
                binding=binding, assignment_file=str(adapter.assign_file), feedback_file=str(adapter.feedback_file),
                assignment_start_bytes=adapter.assign_file.stat().st_size, feedback_start_bytes=adapter.feedback_file.stat().st_size))
            write(path/'decision.json', dict(version=decision['version'], completed_updates=completed,
                distribution=decision['distribution'], decision_sha256=file_sha(path/'decision.pt'),
                teacher_frozen=True, source_checkpoint_sha256=source_hash))

        receipt = None
        source = args.resume or args.source
        if source:
            restore_started = time.monotonic()
            source_hash = file_sha(source)
            saved = torch.load(source, map_location='cpu', weights_only=False)
            info = saved['infos']
            assert info['student_seed'] == args.student_seed
            assert info['pool_sha256'] == pool.sha and info['protocol_sha256'] == digest(PROTOCOL)
            completed = int(info['completed_updates'])
            base_update = completed  # CL1 segment-local cost; cumulative optimizer/control counts stay unchanged
            optimizer_calls = int(info['optimizer_calls'])
            assert 0 < completed < args.target
            assert args.smoke or not branch or args.target-completed == 100
            runner.alg.load(saved, None, True)
            runner.alg.learning_rate = info['adaptive_learning_rate']
            assert all(g['lr'] == runner.alg.learning_rate for g in runner.alg.optimizer.param_groups)
            for path in (adapter.assign_file, adapter.feedback_file):
                path.rename(path.with_name('bootstrap_'+path.name))
                path.touch()
            # The discarded bootstrap sampler may contain lazily created
            # attributes absent from a converted historical checkpoint.
            adapter.owner = restored_owner(pool, info['adapter']['owner'])
            adapter.restore(info['adapter'])
            restore(raw, runner.logger, info['environment'])
            actual = capture(raw, runner.logger)
            assert tree_hash(actual) == tree_hash(info['environment'])
            restored_adapter = adapter.state_dict()
            for section in ('owner', 'metrics', 'task_tensor', 'cursor', 'terms', 'command_stats', 'counts'):
                assert tree_hash(restored_adapter[section]) == tree_hash(info['adapter'][section]), f'Adapter restore differs: {section}'
            assert tensor_hash(actor.state_dict()) == tensor_hash(saved['actor_state_dict'])
            assert tensor_hash(critic.state_dict()) == tensor_hash(saved['critic_state_dict'])
            assert tree_hash(runner.alg.optimizer.state_dict()) == tree_hash(saved['optimizer_state_dict'])
            restore_rng(info['rng'])
            assert tree_hash(rng_state()) == tree_hash(info['rng'])
            receipt = dict(passed=True, source_checkpoint=str(source), source_checkpoint_sha256=source_hash,
                actor_sha256=tensor_hash(actor.state_dict()), critic_sha256=tensor_hash(critic.state_dict()),
                optimizer_sha256=tree_hash(runner.alg.optimizer.state_dict()), environment_sha256=tree_hash(actual),
                global_rng_sha256=tree_hash(rng_state()), adapter_restored_sha256=tree_hash(adapter.state_dict()),
                original_preparation=info.get('v02_preparation'), completed_updates=completed,
                active_episodes=int((adapter.owner.task >= 0).sum()),
                scope='Exact exposed tensors, manager/observation history, PPO/optimizer/global RNG restore. PhysX hidden solver caches are not serializable.')
            if args.resume:
                assert info['mode'] == args.mode and info['branch'] == args.branch
                assert tree_hash(adapter.state_dict()) == tree_hash(info['adapter'])
                decision = info['decision']
                latest_row = info.get('latest_row')
                assert np.array_equal(decision['probability'], probability) if branch else True
                previous_transitions = info['previous_transitions']
                dev_before = info['dev_before']
                evaluation_seconds, evaluation_steps = 0., 0  # Count only this resumed segment's evaluation cost
                if teacher:
                    teacher.load(info['teacher'])
                    teacher_identity = tree_hash(teacher.state())
            else:
                assert info['v02_preparation']['source_checkpoint_sha256']
                assert adapter.owner.engine.version == -1
                decision = dict(version=0, probability=probability.copy(), distribution=distribution_stats(probability),
                    probability_sha256=file_sha(args.probability), decision_source=str(args.decision_source),
                    decision_source_sha256=file_sha(args.decision_source) if args.decision_source else None,
                    teacher_frozen=True, source_checkpoint_sha256=source_hash)
                previous_transitions = adapter.owner.transitions.copy()
                adapter.owner.engine.install(probability, 0)
                adapter.owner.cycle_version_exposure = {}
            clone = NeuralSampler(pool.n, 0)
            clone.load_state_dict(adapter.owner.engine.state_dict())
            probe = copy.deepcopy(adapter.owner.engine)
            assert np.array_equal(clone.sample(4096), probe.sample(4096))
            receipt['next_4096_sampler_draws_equal'] = True
            receipt['installed_sampler_rng_sha256'] = tree_hash(adapter.owner.engine.rng.bit_generator.state)
            receipt['exposed_state_restore_seconds'] = time.monotonic()-restore_started
            write(out/'restore_receipt.json', receipt)
            runner.current_learning_iteration = completed
            get_observations = wrapped.get_observations
            def starting_observations():
                wrapped.get_observations = get_observations
                return TensorDict(raw.obs_buf, batch_size=[4096], device=raw.device)
            wrapped.get_observations = starting_observations
            del saved, info, actual, clone, probe
        else:
            assert not branch and not runner.alg.optimizer.state
            shared = out.parent/f'student_seed{args.student_seed}_initialization.json'
            if shared.exists():
                assert json.loads(shared.read_text()) == initialization, 'Transfer student initialization differs'
            else:
                write(shared, initialization)
            dev_before = evaluate(frozen_actor(), 'development')
            decision = new_decision()
            adapter.owner.engine.install(decision['probability'], decision['version'])
            adapter.owner.cycle_version_exposure = {}
            previous_transitions = adapter.owner.transitions.copy()
        if args.resume:
            decision = new_decision(latest_row)
            adapter.owner.engine.install(decision['probability'], decision['version'])
            adapter.owner.cycle_version_exposure = {}
        preserve_decision()
        phase = 'ready'
        runner.save(out/'checkpoints'/f'update_{completed:05d}.pt')
        frozen_actor()
        write(out/'startup_timing.json', dict(ready_unix=time.time(), train_entry_to_ready_seconds=time.monotonic()-start_wall, evaluation_seconds_before_learning=evaluation_seconds, restored=args.resume is not None))
        write(dataset/'run_manifest.json', dict(student_seed=args.student_seed, mode=args.mode,
            base_update=base_update, target=args.target, source_checkpoint_sha256=source_hash,
            teacher_frozen=True, branch=args.branch, transfer_kind=args.transfer_kind,
            binding=binding, created_unix=time.time(), full_probability_file=str(args.probability)))

        if receipt is not None:
            native_act, native_env_step = runner.alg.act, wrapped.step
            receipt['first_rollout_signatures'] = []
            def audited_act(obs):
                actions = native_act(obs)
                if 'first_action_sha256' not in receipt:
                    receipt['first_action_sha256'] = tensor_hash({'actions': actions})
                return actions
            def audited_env_step(actions):
                result = native_env_step(actions)
                obs, reward, done, extras = result
                receipt['first_rollout_signatures'].append(tree_hash((dict(obs.items()), reward, done)))
                if len(receipt['first_rollout_signatures']) == 24:
                    runner.alg.act, wrapped.step = native_act, native_env_step
                    write(out/'restore_receipt.json', receipt)
                return result
            runner.alg.act, wrapped.step = audited_act, audited_env_step

        native_update, native_step, native_log = runner.alg.update, runner.alg.optimizer.step, runner.logger.log
        def optimizer_step(*a, **kw):
            nonlocal optimizer_calls
            result = native_step(*a, **kw)
            optimizer_calls += 1
            return result
        def counted_update():
            nonlocal completed
            losses = native_update()
            completed += 1
            assert optimizer_calls == completed*20 and adapter.owner.control_steps == completed*24
            assert all(np.isfinite(float(v)) for v in losses.values()), 'Nonfinite native PPO loss'
            values = [*actor.parameters(), *critic.parameters()]
            values.extend(v for item in runner.alg.optimizer.state.values() for v in item.values()
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.is_cuda)
            if not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):
                raise FloatingPointError('Nonfinite native PPO model or optimizer')
            return losses
        runner.alg.optimizer.step, runner.alg.update = optimizer_step, counted_update

        def empirical_stats(counts):
            if not counts.sum():
                return None
            p = counts/counts.sum()
            positive = p[p > 0]
            return dict(entropy=float(-(positive*np.log(positive)).sum()),
                effective_sequences=float(1/(p@p)), tv_uniform=float(np.abs(p-1/len(p)).sum()/2),
                sampled_sequences=int((counts > 0).sum()),
                top10_mass=float(np.sort(p)[-len(p)//10:].sum()))

        def record_boundary(row):
            nonlocal decision, previous_transitions, phase, dev_before
            adapter.owner.check()
            path = dataset/f'cycle_{decision["version"]:04d}'
            draws, durations, counts = adapter.owner.cycle_decisions()
            np.savez_compressed(path/'real_draws.npz', decision_id=draws[:, 0].astype(np.int64),
                env_id=draws[:, 1].astype(np.int32), sequence_id=draws[:, 2].astype(np.int32),
                generation=draws[:, 3].astype(np.int64), assigned_control_step=draws[:, 4].astype(np.int64),
                probability=draws[:, 5], teacher_version=np.full(len(draws), decision['version'], dtype=np.int64),
                exposure_steps_until_boundary=durations, counts=counts,
                executed_transitions_per_sequence=adapter.owner.transitions-previous_transitions,
                active_decision_ids=adapter.owner.decision_id, active_versions=adapter.owner.teacher_version,
                active_sequence_ids=adapter.owner.task, active_steps=adapter.owner.steps)
            actor_path = frozen_actor()
            outcome = dict(completed_updates=completed, distribution=decision['distribution'],
                sampled_decisions=len(draws), zero_physics_exposure_draws=int((durations == 0).sum()),
                cycle_version_exposure=copy.deepcopy(adapter.owner.cycle_version_exposure),
                teacher_frozen=True, teacher_gradient_norm=0., teacher_parameter_delta=0.,
                actual_exposure=empirical_stats(counts),
                assignment_end_bytes=adapter.assign_file.stat().st_size,
                feedback_end_bytes=adapter.feedback_file.stat().st_size, actor_checkpoint=str(actor_path))
            if not branch:
                phase = 'awaiting_development'
                dev_after = evaluate(actor_path, 'development')
                assert dev_after['initial_sha256'] == dev_before['initial_sha256']
                outcome.update(dev_before=dev_before['continuous_joint_score'], dev_after=dev_after['continuous_joint_score'],
                    dev_strict_success=dev_after['joint_success_rate'])
                dynamic_after, global_after = global_state(row, dev_after)
                teacher.complete(decision, dynamic_after, global_after, dev_after, completed)
                dev_before = dev_after
                assert tree_hash(teacher.state()) == teacher_identity
            write(path/'outcome.json', outcome)
            emit(out/'sampling_cycles.jsonl', outcome)
            previous_transitions = adapter.owner.transitions.copy()
            if completed < args.target and not branch:
                decision = new_decision(row)
                adapter.owner.engine.install(decision['probability'], decision['version'])
                adapter.owner.cycle_version_exposure = {}
                preserve_decision()
            phase = 'complete' if completed == args.target else 'ready'
            checkpoint = out/'checkpoints'/f'update_{completed:05d}.pt'
            runner.save(checkpoint)
            write(path/'COMMITTED.json', dict(passed=True, completed_updates=completed,
                checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint),
                files={p.name: file_sha(p) for p in sorted(path.iterdir()) if p.is_file()}))
            emit(dataset/'index.jsonl', dict(version=int(path.name.split('_')[-1]), path=str(path),
                completed_updates=completed, commit_sha256=file_sha(path/'COMMITTED.json')))
            if not branch and completed % 1000 == 0:
                evaluate(actor_path, 'test')

        def recorded_log(*a, **kw):
            nonlocal latest_row
            native_log(*a, **kw)
            row = dict(completed_updates=completed, optimizer_calls=optimizer_calls,
                environment_transitions=completed*4096*24, new_updates=completed-base_update,
                new_environment_transitions=(completed-base_update)*4096*24, unix=time.time(),
                branch=args.branch, mode=args.mode, teacher_version=adapter.owner.engine.version,
                mean_training_reward=statistics.mean(runner.logger.rewbuffer) if runner.logger.rewbuffer else None,
                losses=plain(kw.get('loss_dict', {})), learning_rate=runner.alg.learning_rate,
                action_std=float(kw['action_std'].mean()), native_feedback_episodes=adapter.owner.accepted)
            latest_row = row
            emit(out/'updates.jsonl', row)
            write(out/'latest.json', row)
            if completed == base_update+1 or completed % 20 == 0:
                print('NSGS_V02_UPDATE '+json.dumps(row), flush=True)
            if completed == args.target or not branch and completed % 100 == 0:
                record_boundary(row)
        runner.logger.log = recorded_log
        try:
            runner.learn(num_learning_iterations=args.target-completed, init_at_random_ep_len=False)
        except BaseException:
            try:
                runner.save(out/'checkpoints'/f'fault_update_{completed:05d}.pt')
            except BaseException:
                (out/'fault_checkpoint_error.txt').write_text(traceback.format_exc())
            raise
        assert completed == args.target and optimizer_calls == completed*20
        final = out/'checkpoints'/f'update_{completed:05d}.pt'
        write(out/'result.json', dict(passed=True, completed_updates=completed, new_updates=completed-base_update,
            environment_transitions=completed*4096*24, new_environment_transitions=(completed-base_update)*4096*24,
            checkpoint=str(final), checkpoint_sha256=file_sha(final), actor_checkpoint=str(frozen_actor()),
            optimizer_calls=optimizer_calls, branch=args.branch, mode=args.mode, teacher_frozen=True,
            teacher_proxy_sha256=tree_hash(teacher.state()) if teacher else None, actual_teacher_network_sha256=decision.get('teacher_sha256'), actual_teacher_outer_version=decision.get('teacher_version'),
            wall_seconds=time.monotonic()-start_wall, evaluation_seconds=evaluation_seconds,
            evaluation_environment_transitions=evaluation_steps, summary=adapter.summary(), experience=str(dataset)))
        print('NSGS_V02_TRAIN_PASS '+str(completed), flush=True)
    finally:
        env.close()


args.out.mkdir(parents=True, exist_ok=False)
try:
    train()
except BaseException:
    (args.out/'fault.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close()
