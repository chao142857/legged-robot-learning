"""Native WBC PPO with E3 scheduling and common-prefix checkpoint branching."""
import argparse
import json
from pathlib import Path
import statistics
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=("A", "B", "C", "D", "prefix"), required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--resume", type=Path)
parser.add_argument("--smoke", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def train():
    import gymnasium as gym
    import numpy as np
    import torch
    import LeggedManip_Lab.tasks
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    from commands import SequenceVelocity, SequenceWBC
    from common import ROOT, PARAMETERS, PROTOCOL, Pool, digest, emit, file_sha, model_cfg
    from common import plain, restore_rng, rng_state, tensor_hash, write
    from design import ENVS, FORK, INTERVAL, MAP_DIR, ROLLOUT, SEED, TOTAL, map_change, map_key, mapping_is_balanced
    from e3_adapter import E3Adapter
    from fork_state import capture, restore, tree_hash
    from grouped_sampling import mapping_hash

    out = args.out
    arm = args.arm
    fork = 4 if args.smoke else FORK
    target = fork if arm == "prefix" else (8 if args.smoke else TOTAL)
    assert (args.resume is not None) == (arm in ("B", "D"))
    completed = fork if args.resume else 0
    optimizer_calls = completed * 20
    specification_path = MAP_DIR / "manifest.json"
    specification_hash = file_sha(specification_path)
    specification = json.loads(specification_path.read_text())
    pool = Pool("wbc", "train")
    assert pool.sha == specification["train_pool_sha256"]
    assert digest(PROTOCOL) == specification["protocol_sha256"] and PARAMETERS == specification["sgs_parameters"]

    def key_for(update):
        if args.smoke and arm == "D":
            return "m01" if update >= 5 else "m00"
        return map_key(arm, update)

    def load_mapping(key):
        path = MAP_DIR / f"{key}.npy"
        binding = specification["maps"][key]
        assert file_sha(path) == binding["file_sha256"]
        value = np.load(path, allow_pickle=False)
        assert mapping_is_balanced(value) and mapping_hash(value) == binding["mapping_sha256"]
        return value

    key = key_for(completed)
    cfg, agent = model_cfg("GO2-PIPER-WBC")
    cfg.scene.num_envs = ENVS
    cfg.seed = SEED
    cfg.sim.device = "cuda:0"
    cfg.log_dir = str(out / "policy")
    native, agent_config = plain(cfg.to_dict()), plain(agent.to_dict())
    baseline = json.loads((Path("/reference/runs/formal-wbc-uniform") / "configuration.json").read_text())
    previous_config = dict(baseline["native_environment"])
    previous_config["log_dir"] = native["log_dir"]
    assert previous_config == native and baseline["agent"] == agent_config, "Native environment/PPO changed"
    cfg.commands.ee_pose.class_type = SequenceWBC
    cfg.commands.base_velocity.class_type = SequenceVelocity
    write(out / "configuration.json", dict(
        native_environment=native, agent=agent_config, arm=arm, seed=SEED, target=target,
        num_envs=ENVS, rollout=ROLLOUT, smoke=args.smoke, protocol=PROTOCOL, sgs_parameters=PARAMETERS,
        mapping_manifest_sha256=specification_hash, pool_sha256=pool.sha,
    ))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    env = gym.make("GO2-PIPER-WBC", cfg=cfg)
    try:
        raw = env.unwrapped
        adapter = E3Adapter(raw, pool, out, load_mapping(key))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        assert wrapped.num_actions == 18 and raw.max_episode_length == 1000
        for name in ("ee_pose", "base_velocity"):
            term = raw.command_manager.get_term(name)
            assert not term.cfg.curriculum_enabled and plain(term.cfg.ranges) == plain(term.cfg.limit_ranges)
        receipt = None
        source_hash = None
        previous_counts = np.zeros(pool.n, dtype=np.int64)
        previous_probability = np.full(pool.n, 1 / pool.n)
        previous_boundary = 0
        paused = 0.

        class Runner(OnPolicyRunner):
            def save(self, path, infos=None):
                adapter.owner.check()
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                state = self.alg.save()
                state["iter"] = self.current_learning_iteration
                state["infos"] = dict(
                    completed_updates=completed, optimizer_calls=optimizer_calls,
                    environment_transitions=completed * ENVS * ROLLOUT,
                    adaptive_learning_rate=self.alg.learning_rate, rng=rng_state(),
                    adapter=adapter.state_dict(), environment=capture(raw, self.logger),
                    protocol_sha256=digest(PROTOCOL), pool_sha256=pool.sha, map_key=key,
                    mapping_manifest_sha256=specification_hash,
                    previous_counts=previous_counts, previous_probability=previous_probability,
                    previous_boundary=previous_boundary, source_checkpoint_sha256=source_hash,
                    arm=arm, smoke=args.smoke,
                )
                temp = path.with_suffix(".pt.tmp")
                torch.save(state, temp)
                temp.replace(path)

        runner = Runner(wrapped, agent.to_dict(), log_dir=str(out / "policy"), device=agent.device)
        actor, critic = runner.alg.actor, runner.alg.critic
        if args.resume:
            source_hash = file_sha(args.resume)
            state = torch.load(args.resume, weights_only=False, map_location="cpu")
            info = state["infos"]
            assert state["iter"] == fork - 1 and info["completed_updates"] == fork
            assert info["optimizer_calls"] == optimizer_calls and info["arm"] == "prefix"
            assert info["environment_transitions"] == fork * ENVS * ROLLOUT
            assert info["smoke"] == args.smoke and info["map_key"] == key
            assert info["mapping_manifest_sha256"] == specification_hash
            assert info["protocol_sha256"] == digest(PROTOCOL) and info["pool_sha256"] == pool.sha
            runner.alg.load(state, None, True)
            runner.alg.learning_rate = info["adaptive_learning_rate"]
            assert all(g["lr"] == runner.alg.learning_rate for g in runner.alg.optimizer.param_groups)
            assert tensor_hash(actor.state_dict()) == tensor_hash(state["actor_state_dict"])
            assert tensor_hash(critic.state_dict()) == tensor_hash(state["critic_state_dict"])
            assert tree_hash(runner.alg.optimizer.state_dict()) == tree_hash(state["optimizer_state_dict"])
            adapter.assign_file.rename(out / "bootstrap_assignments.jsonl")
            adapter.assign_file.touch()
            adapter.restore(info["adapter"])
            assert adapter.owner.control_steps == fork * ROLLOUT
            restore(raw, runner.logger, info["environment"])
            restored_environment = capture(raw, runner.logger)
            for section, value in info["environment"].items():
                assert tree_hash(restored_environment[section]) == tree_hash(value), f"Restore differs: {section}"
            assert tree_hash(adapter.state_dict()) == tree_hash(info["adapter"]), "Adapter restore differs"
            runner.current_learning_iteration = fork
            previous_counts = info["previous_counts"]
            previous_probability = info["previous_probability"]
            previous_boundary = info["previous_boundary"]
            restore_rng(info["rng"])
            assert tree_hash(rng_state()) == tree_hash(info["rng"])
            receipt = dict(
                source_checkpoint_sha256=source_hash, completed_updates=fork,
                actor_sha256=tensor_hash(actor.state_dict()), critic_sha256=tensor_hash(critic.state_dict()),
                optimizer_sha256=tree_hash(runner.alg.optimizer.state_dict()),
                adaptive_learning_rate=runner.alg.learning_rate,
                environment_sha256=tree_hash(restored_environment), rng_sha256=tree_hash(rng_state()),
                adapter_sha256=tree_hash(info["adapter"]), active_episodes=int((adapter.owner.task >= 0).sum()),
                bootstrap_resets_discarded=True, rollout_signatures=[],
                scope="Isaac Lab scene/tensors/managers, observation history, PPO/Adam, sampler/journal and RNG. "
                      "PhysX internal contact solver caches are not checkpointed; B/D use identical restore.",
            )
            del state, info, restored_environment

            get_observations = wrapped.get_observations

            def starting_observations():
                # The checkpoint already contains the native next-rollout observation.
                wrapped.get_observations = get_observations
                return TensorDict(raw.obs_buf, batch_size=[ENVS], device=raw.device)

            wrapped.get_observations = starting_observations
        else:
            assert not runner.alg.optimizer.state and adapter.owner.engine.journal_size == 0
            initialization = dict(actor_sha256=tensor_hash(actor.state_dict()), critic_sha256=tensor_hash(critic.state_dict()),
                                  seed=SEED, pool_sha256=pool.sha, optimizer_empty=True, history_empty=True)
            shared = out.parent / "initialization.json"
            if shared.exists():
                assert json.loads(shared.read_text()) == initialization, "Unmatched seed-42 initialization"
            else:
                write(shared, initialization)
            write(out / "initialization.json", initialization)

        native_act, native_step = runner.alg.act, wrapped.step
        if receipt is not None:
            def audited_act(obs):
                actions = native_act(obs)
                if "first_action_sha256" not in receipt:
                    receipt["first_action_sha256"] = tensor_hash({"actions": actions})
                return actions

            def audited_env_step(actions):
                next_obs, reward, done, extras = native_step(actions)
                receipt["rollout_signatures"].append(tree_hash((dict(next_obs.items()), reward, done)))
                if len(receipt["rollout_signatures"]) == ROLLOUT:
                    wrapped.step = native_step
                    runner.alg.act = native_act
                    write(out / "branch_restore.json", receipt)
                return next_obs, reward, done, extras

            runner.alg.act = audited_act
            wrapped.step = audited_env_step

        native_update = runner.alg.update
        native_optimizer_step = runner.alg.optimizer.step
        native_log = runner.logger.log

        def counted_step(*pos, **kw):
            nonlocal optimizer_calls
            result = native_optimizer_step(*pos, **kw)
            optimizer_calls += 1
            return result

        def counted_update():
            nonlocal completed
            before = optimizer_calls
            losses = native_update()
            completed += 1
            assert optimizer_calls - before == 20 and optimizer_calls == completed * 20
            assert adapter.owner.control_steps == completed * ROLLOUT
            if not all(np.isfinite(float(value)) for value in losses.values()):
                raise FloatingPointError("Nonfinite native PPO loss")
            values = [*actor.parameters(), *critic.parameters()]
            values.extend(value for state in runner.alg.optimizer.state.values() for value in state.values()
                          if isinstance(value, torch.Tensor) and value.is_floating_point() and value.is_cuda)
            if not bool(torch.stack([torch.isfinite(value).all() for value in values]).all()):
                raise FloatingPointError("Nonfinite actor, critic or Adam state")
            return losses

        runner.alg.optimizer.step = counted_step
        runner.alg.update = counted_update

        def distribution():
            nonlocal previous_counts, previous_probability, previous_boundary
            engine = adapter.owner.engine
            counts = np.asarray(engine.sample_counts, dtype=np.int64)
            counts_delta = counts - previous_counts
            probability = np.asarray(engine.probabilities())
            rates = engine.group_rates()
            ages = np.asarray([adapter.owner.control_steps - value for window in engine.window_steps for value in window])
            exposure = counts_delta / counts_delta.sum() if counts_delta.sum() else np.zeros(pool.n)
            top = np.sort(exposure)[::-1]
            elapsed_boundary = previous_boundary
            boundary_tv = float(np.abs(previous_probability - probability).sum() / 2)
            previous_counts, previous_probability, previous_boundary = counts, probability, completed
            destination = out / "sampling" / f"update_{completed:05d}.npz"
            destination.parent.mkdir(exist_ok=True)
            with destination.with_suffix(".tmp").open("wb") as file:
                np.savez_compressed(file, probability=probability, group_success_rate=rates,
                                    group_feedback_counts=[len(h) for h in engine.outcomes],
                                    sampled_total=counts, sampled_since_previous=counts_delta,
                                    completed_total=engine.completed_counts, success_total=engine.success_counts,
                                    transitions_total=adapter.owner.transitions)
            destination.with_suffix(".tmp").replace(destination)
            write(destination.with_suffix(".json"), dict(
                completed_updates=completed, from_boundary=elapsed_boundary, active_mapping=key,
                weights=specification["maps"][key], native_terminal_feedback=engine.journal_size,
                probability_quantiles=np.quantile(probability, [0, .01, .1, .5, .9, .99, 1]),
                group_success_rate_quantiles=np.quantile(rates, [0, .1, .25, .5, .75, .9, 1]),
                group_success_rate_hist=np.histogram(rates, bins=np.linspace(0, 1, 21))[0],
                group_history_diagnostics=engine.diagnostics(adapter.owner.control_steps),
                feedback_age_updates_quantiles=np.quantile(ages / ROLLOUT, [0, .1, .5, .9, .99, 1]) if len(ages) else None,
                probability_effective_sequences=float(1 / (probability @ probability)),
                probability_tv_from_previous=boundary_tv,
                sampled_episodes=int(counts_delta.sum()), unique_sampled=int((counts_delta > 0).sum()),
                sample_top1_share=float(top[:int(.01 * pool.n)].sum()), sample_top10_share=float(top[:int(.1 * pool.n)].sum()),
                sample_effective_sequences=float(1 / (exposure @ exposure)) if exposure.any() else None,
            ))

        def evaluate(path, kind="heldout"):
            nonlocal paused
            before = (tree_hash(adapter.owner.state_dict()), tree_hash(rng_state()))
            request = out / "evaluation_requests" / f"{completed:05d}-{kind}.json"
            write(request, dict(update=completed, kind=kind, checkpoint=str(path), checkpoint_sha256=file_sha(path)))
            ack = request.with_suffix(".done.json")
            start = time.monotonic()
            while not ack.exists():
                if time.monotonic() - start > 1200:
                    raise TimeoutError(f"Timed out awaiting {request}")
                time.sleep(.5)
            reply = json.loads(ack.read_text())
            paused += time.monotonic() - start
            assert reply["passed"] and reply["checkpoint_sha256"] == file_sha(path)
            assert before == (tree_hash(adapter.owner.state_dict()), tree_hash(rng_state()))

        def recorded_log(*pos, **kw):
            nonlocal key
            native_log(*pos, **kw)
            new_key = key_for(completed)
            if new_key != key:
                engine = adapter.owner.engine
                old_mapping = engine.mapping.copy()
                old_probability = np.asarray(engine.probabilities())
                old_rng = engine.rng.getstate()
                old_task = adapter.owner.task.copy()
                assert adapter.pending is None and engine.switch(load_mapping(new_key))
                assert old_rng == engine.rng.getstate() and np.array_equal(old_task, adapter.owner.task)
                new_probability = np.asarray(engine.probabilities())
                emit(out / "mapping_switches.jsonl", dict(
                    completed_updates=completed, before=key, after=new_key,
                    before_weights=specification["maps"][key], after_weights=specification["maps"][new_key],
                    **map_change(old_mapping, engine.mapping), feedback_events=engine.journal_size,
                    probability_total_variation=float(np.abs(old_probability - new_probability).sum() / 2),
                    future_rollouts=target > completed,
                ))
                key = new_key
            row = dict(
                completed_updates=completed, optimizer_calls=optimizer_calls, active_mapping=key,
                environment_transitions=completed * ENVS * ROLLOUT, unix=time.time(),
                mean_training_reward=statistics.mean(runner.logger.rewbuffer) if runner.logger.rewbuffer else None,
                mean_episode_length=statistics.mean(runner.logger.lenbuffer) if runner.logger.lenbuffer else None,
                losses=plain(kw.get("loss_dict", {})), native_feedback_episodes=adapter.owner.accepted,
                learning_rate=runner.alg.learning_rate,
            )
            emit(out / "updates.jsonl", row)
            write(out / "latest.json", row)
            if completed % 20 == 0 or completed == 1:
                print("E3_UPDATE " + json.dumps(row), flush=True)
            boundary = completed % INTERVAL == 0 or completed == target or (args.smoke and completed == 5)
            if boundary:
                distribution()
                path = out / "checkpoints" / f"update_{completed:05d}.pt"
                runner.save(path)
                if completed % 1000 == 0 or (args.smoke and arm == "prefix" and completed == fork):
                    evaluate(path)
                if not args.smoke and arm != "prefix" and completed == TOTAL:
                    evaluate(path, "exposure")

        runner.logger.log = recorded_log
        if args.resume is None:
            distribution()
        start = time.monotonic()
        runner.learn(num_learning_iterations=target - completed, init_at_random_ep_len=args.resume is None)
        assert completed == target and optimizer_calls == completed * 20
        adapter.owner.check()
        final = out / "checkpoints" / f"update_{completed:05d}.pt"
        write(out / "result.json", dict(
            passed=True, arm=arm, smoke=args.smoke, completed_updates=completed, optimizer_calls=optimizer_calls,
            environment_transitions=completed * ENVS * ROLLOUT,
            native_feedback_episodes=adapter.owner.accepted, active_mapping=key, checkpoint=str(final),
            checkpoint_sha256=file_sha(final), source_checkpoint_sha256=source_hash,
            wall_seconds=time.monotonic() - start, paused_evaluation_seconds=paused,
        ))
        print(f"E3_TRAIN_PASS arm={arm} updates={completed}", flush=True)
    finally:
        env.close()


args.out.mkdir(parents=True, exist_ok=False)
try:
    train()
except BaseException:
    (args.out / "fault.txt").write_text(traceback.format_exc())
    raise
finally:
    app.close()
