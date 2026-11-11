"""Build the 19 balanced maps once from the original training commands."""
import json
from pathlib import Path
import numpy as np

from command_features import balanced_partition, features
from common import ROOT, Pool, PROTOCOL, PARAMETERS, digest, file_sha, write
from design import MAP_DIR, RANDOM_SEED, LOCO, MANIP, mapping_is_balanced, map_change, normalized_blocks, weighted_features
from grouped_sampling import mapping_hash


def main():
    reference = Path("/e2reference/artifacts")
    e2_manifest = reference / "groups/mapping_manifest.json"
    assert file_sha(e2_manifest) == "80d754fdb906e26ef0c57879565e63c582b4ba1c956ae559a10f4d43d44d5cb1"
    e2 = json.loads(e2_manifest.read_text())
    random = np.load(reference / f"groups/mapping_random_{RANDOM_SEED}.npy", allow_pickle=False)
    assert mapping_hash(random) == e2["mappings"][f"random_{RANDOM_SEED}"]["mapping_sha256"]
    pool = Pool("wbc", "train")
    assert pool.sha == e2["pool_sha256"]
    meta = json.loads((reference / "groups/feature_definition.json").read_text())
    assert meta["pool_sha256"] == pool.sha
    matrix, feature_meta = features(pool.ee, pool.velocity, meta["reference"])
    assert feature_meta["block_dimensions"] == meta["block_dimensions"]
    blocks, scales = normalized_blocks(matrix, feature_meta["block_dimensions"])
    del matrix
    MAP_DIR.mkdir(exist_ok=False)
    maps = {}

    def save(key, mapping, weight=None):
        assert mapping_is_balanced(mapping)
        path = MAP_DIR / f"{key}.npy"
        with path.open("xb") as file:
            np.save(file, mapping, allow_pickle=False)
        maps[key] = dict(loco=None if weight is None else 1 - weight, manip=weight,
                         mapping_sha256=mapping_hash(mapping), file_sha256=file_sha(path))

    save("A", random)
    previous = None
    for index in range(17):
        key = f"m{index:02d}"
        manip = .5 + index / 32
        mapping = balanced_partition(weighted_features(blocks, manip), capacity=8, seed=RANDOM_SEED)
        save(key, mapping, manip)
        if previous is not None:
            maps[key]["change_from_previous"] = map_change(previous, mapping)
        previous = mapping
        print(f"E3_MAP {key} manip={manip} {maps[key]['mapping_sha256']}", flush=True)
    save("C", balanced_partition(weighted_features(blocks, .8), capacity=8, seed=RANDOM_SEED), .8)
    evaluation = Pool("wbc", "evaluation")
    assert evaluation.n == 1024 and evaluation.sha != pool.sha
    # A common command-only train panel for final exposure-stratified analysis.
    panel = sorted(np.random.default_rng(20261108).choice(pool.n, 1024, replace=False).tolist())
    write(MAP_DIR / "manifest.json", dict(
        maps=maps, train_pool_sha256=pool.sha, evaluation_pool_sha256=evaluation.sha,
        features=feature_meta, loco_blocks=sorted(LOCO), manip_blocks=sorted(MANIP),
        scales_rms_pairwise_distance=scales,
        distance="Squared Euclidean: lambda_L * normalized_d_L^2 + lambda_M * normalized_d_M^2",
        partition_seed=RANDOM_SEED, e2_random_mapping_manifest_sha256=file_sha(e2_manifest),
        protocol_sha256=digest(PROTOCOL), sgs_parameters=PARAMETERS,
        final_exposure_panel=dict(seed=20261108, sequence_ids=panel, repetitions=2),
    ))
    print("E3_MAPS_PASS 19", flush=True)


if __name__ == "__main__":
    main()
