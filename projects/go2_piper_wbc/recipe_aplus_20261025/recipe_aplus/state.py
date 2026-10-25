"""Restore all saved native learner state and the RNG actually present in V1."""
import hashlib
import torch


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def equal_tree(a,b):
    if isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def restore_rng(infos):
    torch.set_rng_state(infos['rng_cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in infos['rng_cuda']])


def restore_native(runner,path,expected_updates=1200):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    infos=runner.load(str(path),strict=True,map_location=runner.device)
    assert infos['completed_updates']==expected_updates and cp['iter']==expected_updates-1
    assert equal_tree(runner.alg.actor.state_dict(),cp['actor_state_dict'])
    assert equal_tree(runner.alg.critic.state_dict(),cp['critic_state_dict'])
    assert equal_tree(runner.alg.optimizer.state_dict(),cp['optimizer_state_dict'])
    runner.current_learning_iteration=expected_updates
    runner.alg.learning_rate=float(cp['infos']['learning_rate'])
    assert all(g['lr']==runner.alg.learning_rate for g in runner.alg.optimizer.param_groups)
    steps=[float(v['step']) for v in runner.alg.optimizer.state.values() if 'step' in v]
    assert steps and min(steps)==max(steps)==expected_updates*20
    restore_rng(cp['infos'])
    cpu_hash=tensor_hash(torch.get_rng_state());gpu_hash=[tensor_hash(x) for x in torch.cuda.get_rng_state_all()]
    assert cpu_hash==tensor_hash(cp['infos']['rng_cpu'])
    assert gpu_hash==[tensor_hash(x) for x in cp['infos']['rng_cuda']]
    # Prove the saved stream can be replayed, then leave it at its original boundary.
    a=torch.rand(16);b=torch.rand(16,device=runner.device)
    restore_rng(cp['infos']);assert torch.equal(a,torch.rand(16)) and torch.equal(b,torch.rand(16,device=runner.device))
    restore_rng(cp['infos'])
    evidence=dict(actor_bitwise_equal=True,critic_bitwise_equal=True,distribution_in_actor_state=True,
        adam_bitwise_equal=True,adam_step_min=min(steps),adam_step_max=max(steps),
        learning_rate=runner.alg.learning_rate,next_zero_based_runner_iteration=expected_updates,
        first_new_completed_update=expected_updates+1,cpu_rng_sha256=cpu_hash,cuda_rng_sha256=gpu_hash,
        saved_rng_replay_passed=True,rng_restored_after_probe=True,
        limitation='V1 contains Torch CPU/CUDA RNG, not PhysX/env/history or Python/NumPy RNG; physical env is reset, not bitwise resumed')
    return cp,evidence
