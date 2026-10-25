"""The B2/B3 untrained 12D pair uses installed native RSL-RL constructors."""
import hashlib
from pathlib import Path
import torch


def model_sha(model):
    h=hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_untrained_pair(runner, path):
    """Copy only the common fresh model state; never import an optimizer/counter.

    The caller must bind the artifact SHA256 in its run manifest before calling.
    This is an untrained initialization, with a distinct format from trained CPs.
    """
    pair=torch.load(Path(path), map_location=runner.device, weights_only=False)
    if pair.get('schema') != 'b2_b3_native_untrained_pair_v1' or pair.get('completed_updates') != 0 or pair.get('optimizer_state_entries') != 0 or pair.get('action_dim') != 12:
        raise ValueError('Only the shared untrained 12D pair artifact is accepted')
    if runner.current_learning_iteration != 0 or len(runner.alg.optimizer.state):
        raise ValueError('Pair initialization requires a freshly constructed runner')
    runner.alg.actor.load_state_dict(pair['actor_state_dict'],strict=True)
    runner.alg.critic.load_state_dict(pair['critic_state_dict'],strict=True)
    if model_sha(runner.alg.actor) != pair['actor_sha256'] or model_sha(runner.alg.critic) != pair['critic_sha256']:
        raise ValueError('Shared pair model hash mismatch')
    return dict(actor_sha256=pair['actor_sha256'],critic_sha256=pair['critic_sha256'],
                completed_updates=0,optimizer_state_entries=len(runner.alg.optimizer.state),
                provenance='native constructors on CPU; exact shared untrained model state')
