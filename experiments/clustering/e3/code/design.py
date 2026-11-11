"""Immutable E3 schedule and command-only feature normalization."""
from pathlib import Path
import numpy as np

SIZE = 32768
GROUPS = 4096
CAPACITY = 8
ENVS = 4096
ROLLOUT = 24
SEED = 42
TOTAL = 10000
FORK = 2000
INTERVAL = 500
RANDOM_SEED = 20261104
LOCO = {"velocity_time", "standing_time", "initial_reset_standing_pulse", "velocity_exact_dwell"}
MANIP = {"position_time", "reference_SO3_time", "perturbation_SO3_time", "EE_switch_times", "EE_exact_dwell"}


def map_key(arm, completed):
    if arm == "A":
        return "A"
    if arm in ("B", "prefix"):
        return "m00"
    if arm == "C":
        return "C"
    if arm == "D" and 0 <= completed <= TOTAL:
        return f"m{max(0, (completed - FORK) // INTERVAL):02d}"
    raise ValueError((arm, completed))


def normalized_blocks(matrix, dimensions):
    assert set(dimensions) == LOCO | MANIP and matrix.shape[1] == sum(dimensions.values())
    columns = {"loco": [], "manip": []}
    offset = 0
    for name, size in dimensions.items():
        columns["loco" if name in LOCO else "manip"].extend(range(offset, offset + size))
        offset += size
    blocks = {}
    scales = {}
    for name, ids in columns.items():
        block = matrix[:, ids]
        # E||x-x'||² = 2 * sum(Var(x)); all normalization uses only commands.
        rms = float(np.sqrt(2 * np.var(block, axis=0, dtype=np.float64).sum()))
        if not np.isfinite(rms) or rms <= 0:
            raise ValueError(f"Invalid {name} feature scale")
        blocks[name] = block / rms
        scales[name] = rms
    return blocks, scales


def weighted_features(blocks, manip):
    if not 0 <= manip <= 1:
        raise ValueError("Manip weight outside [0,1]")
    # Squared Euclidean distance is the specified weighted sum of normalized squared distances.
    return np.concatenate((np.sqrt(1 - manip) * blocks["loco"], np.sqrt(manip) * blocks["manip"]), axis=1)


def mapping_is_balanced(mapping):
    return (
        mapping.shape == (SIZE,) and np.issubdtype(mapping.dtype, np.integer)
        and mapping.min() == 0 and mapping.max() == GROUPS - 1
        and np.all(np.bincount(mapping) == CAPACITY)
    )


def map_change(old, new):
    """Compare member sets, irrespective of numerical group labels."""
    members = np.argsort(old, kind="stable").reshape(-1, CAPACITY)
    overlap = np.sum(new[members[old]] == new[:, None], axis=1)
    return {
        "sequences_with_changed_group_members": int((overlap < CAPACITY).sum()),
        "mean_retained_peers": float((overlap - 1).mean()),
        "old_groups_intact": int((overlap == CAPACITY).sum() // CAPACITY),
    }


MAP_DIR = Path("/workspace/maps")
