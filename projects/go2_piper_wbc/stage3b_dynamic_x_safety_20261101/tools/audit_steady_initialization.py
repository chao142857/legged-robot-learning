"""CPU-only actual-run pairing audit; never loads a model or launches GPU."""
import argparse, hashlib, json, pathlib


def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def audit(root):
    plan=read(root/'STEADY_KL001_PLAN_FROZEN.json');out=[]
    for seed in plan['training_seeds']:
        runs=[pathlib.Path(read(root/(prefix+'-seed'+str(seed)+'-window-to2_RUN.json'))['run'])
              for prefix in ('legKL002','steadyKL001')]
        configs=[read(r/'config.json') for r in runs]
        assert all(c['checkpoint_sha256']==plan['common_parent_sha256'] for c in configs)
        assert configs[0]['leg_retention']['weight']==configs[1]['leg_retention']['weight']==.02
        rows=[json.loads((r/'ppo/updates.jsonl').open().readline()) for r in runs]
        assert len(rows[0]['physical_reward_time'])==len(rows[1]['physical_reward_time'])==94
        assert rows[0]['physical_reward_time']==rows[1]['physical_reward_time']
        assert rows[0]['Uniform_coverage']==rows[1]['Uniform_coverage']
        assert rows[0]['transitions']==rows[1]['transitions']
        restore=read(runs[1]/'ppo/restore-evidence.json');zero=read(runs[1]/'ppo/zero-mean-attachment.json')
        assert all(restore[k] for k in ('actor_bitwise_equal','critic_bitwise_equal','adam_bitwise_equal','saved_rng_replay_passed','rng_restored_after_probe'))
        assert all(v==0 for case in zero.values() for k,v in case.items() if k.endswith('max_error'))
        assert all(case['full_mean_JIT_equal'] for case in zero.values())
        result=read(runs[1]/'ppo/result.json');assert result['passed'] and result['updates']==2
        out.append(dict(training_seed=seed,matched_control_run=str(runs[0]),new_run=str(runs[1]),
            common_parent_sha256=plan['common_parent_sha256'],restore=restore,zero_attachment=zero,
            actual_PPO_updates=result['updates'],native_optimizer_steps=result['optimizer_steps'],
            first_rollout_original_physics_rewards_and_Uniform_exact=True,original_fields=94,
            update_log_sha256=[sha(r/'ppo/updates.jsonl') for r in runs],
            frozen_launch_sha256=[sha(r/'launch.json') for r in runs]))
    return dict(CPU_only=True,seeds=out,limitation='Only the first pre-update rollout is required identical. Later on-policy batches diverge; neither identical PhysX hidden-state restoration nor identical later PPO gradients is claimed.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.root),indent=2))
