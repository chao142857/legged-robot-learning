"""Validate our own PPO recovery files before selecting a checkpoint."""
from pathlib import Path


def validate_checkpoint(path):
    import torch
    path = Path(path)
    data = torch.load(path,map_location='cpu',weights_only=False)
    required = ['actor_state_dict','critic_state_dict','optimizer_state_dict','iter']
    if not all(key in data for key in required):
        raise ValueError('Incomplete actor/critic/optimizer checkpoint')
    iteration = int(data['iter'])
    if iteration < 0 or iteration != int(path.stem.split('_')[-1]):
        raise ValueError('Checkpoint filename and stored iteration disagree')

    def tensors(value):
        if torch.is_tensor(value):
            yield value
        elif isinstance(value,dict):
            for child in value.values():
                yield from tensors(child)
        elif isinstance(value,(tuple,list)):
            for child in value:
                yield from tensors(child)

    for key in required[:-1]:
        values = list(tensors(data[key]))
        if not values or not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError(f'Empty or nonfinite state: {key}')
    return iteration


def select_usable_checkpoint(paths):
    rejected = []
    ordered = sorted(paths,key=lambda p:(int(p.stem.split('_')[-1]),p.stat().st_mtime),reverse=True)
    for path in ordered:
        try:
            validate_checkpoint(path)
            return path,rejected
        except Exception as exc:
            rejected.append({'path':str(path),'error':repr(exc)})
    return None,rejected
