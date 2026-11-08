import hashlib
import numpy as np
import torch

def tree_hash(value):
    """Deterministic typed hash, avoiding tensor-to-JSON expansion."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            result.update(f"array:{item.dtype}:{item.shape}:".encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            result.update(f"dict:{len(item)}:".encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        else:
            result.update(f"{type(item).__name__}:{item!r};".encode())

    visit(value)
    return result.hexdigest()
