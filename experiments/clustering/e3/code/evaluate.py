"""Evaluate frozen actors on the fixed held-out pool or the fixed train diagnostic panel."""
import argparse
import json
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--kind", choices=("heldout", "exposure"), required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def evaluate():
    import gymnasium as gym
    import numpy as np
    import torch
    import LeggedManip_Lab.tasks
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from adapter import SequenceAdapter
    from commands import SequenceVelocity, SequenceWBC
    from common import PROTOCOL, Pool, digest, file_sha, model_cfg, tensor_hash, write
    from design import MAP_DIR

    started = time.monotonic()
    manifest_path = MAP_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert digest(PROTOCOL) == manifest["protocol_sha256"]
    if args.kind == "heldout":
        pool = Pool("wbc", "evaluation")
        assert pool.n == 1024 and pool.sha == manifest["evaluation_pool_sha256"]
        selected = np.arange(1024)
        seed = 922042
    else:
        pool = Pool("wbc", "train")
        assert pool.sha == manifest["train_pool_sha256"]
        panel = manifest["final_exposure_panel"]
        selected = np.tile(panel["sequence_ids"], panel["repetitions"])
        pool.ee, pool.velocity = pool.ee[selected], pool.velocity[selected]
        pool.n = len(selected)
        pool.identity += ":e3-exposure"
        seed = 51002042

    cfg, agent = model_cfg("GO2-PIPER-WBC")
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = pool.n, seed, "cuda:0"
    cfg.commands.ee_pose.class_type = SequenceWBC
    cfg.commands.base_velocity.class_type = SequenceVelocity
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    checkpoint_hash = file_sha(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    info = checkpoint["infos"]
    assert info["pool_sha256"] == manifest["train_pool_sha256"]
    assert info["mapping_manifest_sha256"] == file_sha(manifest_path)
    assert info["protocol_sha256"] == digest(PROTOCOL)

    env = gym.make("GO2-PIPER-WBC", cfg=cfg)
    try:
        raw = env.unwrapped
        adapter = SequenceAdapter(raw, "wbc", "evaluation", pool, args.out)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        assert wrapped.num_actions == 18 and raw.max_episode_length == 1000
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
        actor = runner.alg.actor
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        update = info["completed_updates"]
        del checkpoint, info
        policy = runner.get_inference_policy(device=raw.device)
        actor_hash = tensor_hash(actor.state_dict())
        obs = wrapped.get_observations()
        initial_hash = tensor_hash({
            "robot": raw.scene["robot"].data.root_state_w,
            "joints": raw.scene["robot"].data.joint_pos,
            **{f"obs/{k}": v for k, v in obs.items()},
        })
        with torch.inference_mode():
            for step in range(1, raw.max_episode_length + 3):
                obs, _, dones, _ = wrapped.step(policy(obs))
                policy.reset(dones)
                if len(adapter.first_episode_rows) == pool.n:
                    break
        adapter.owner.check()
        assert actor_hash == tensor_hash(actor.state_dict()) and file_sha(args.checkpoint) == checkpoint_hash
        rows = sorted(adapter.first_episode_rows, key=lambda row: row["env_id"])
        assert len(rows) == pool.n
        for i, row in enumerate(rows):
            assert row["env_id"] == row["sequence_id"] == i
            row["sequence_id"] = int(selected[i])
            row["replicate"] = i // 1024
        values = np.asarray([r["values"] for r in rows])
        n = values[:, 0]
        fraction = np.divide(values[:, 1], n, out=np.zeros_like(n), where=n > 0)
        mature = (n >= PROTOCOL["min_scored_steps"]) & (values[:, 7] == 0)
        rmse = np.sqrt(np.sum(values[:, 2:6] ** 2 * n[:, None], axis=0) / n.sum())
        write(args.out / "episode_results.json", rows)
        result = dict(
            passed=True, kind=args.kind, update=update, checkpoint_sha256=checkpoint_hash, actor_sha256=actor_hash,
            pool_sha256=pool.sha, episodes=len(rows), initial_sha256=initial_hash, seed=seed,
            protocol_sha256=digest(PROTOCOL), model_weights_unchanged=True,
            training_history_updates=adapter.owner.accepted,
            joint_success_rate=float(values[:, 6].mean()), native_failure_rate=float(values[:, 7].mean()),
            mean_joint_good_time=float(fraction[mature].mean()) if mature.any() else None,
            joint_good_time_pooled=float(values[:, 1].sum() / n.sum()),
            pooled_rmse=dict(zip(("velocity_m_s", "yaw_rad_s", "ee_position_m", "ee_orientation_rad"), rmse.tolist())),
            mature_nonfailure_episodes=int(mature.sum()), wall_seconds=time.monotonic() - started,
            episode_results_sha256=file_sha(args.out / "episode_results.json"),
        )
        write(args.out / "result.json", result)
        print("E3_EVAL_PASS " + json.dumps(result), flush=True)
    finally:
        env.close()


args.out.mkdir(parents=True, exist_ok=False)
try:
    evaluate()
except BaseException:
    (args.out / "fault.txt").write_text(traceback.format_exc())
    raise
finally:
    app.close()
