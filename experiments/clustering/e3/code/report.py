"""Summarize complete learning curves and exposure-stratified final replays."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ROOT, file_sha, write

RUNS = ROOT / "runs"
ARMS = ("A", "B", "C", "D")
NAMES = {"A": "Random", "B": "LocoManip", "C": "EE-priority", "D": "Scheduled"}


def read(path):
    return json.loads(path.read_text())


def table(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage(arm, update):
    return "prefix" if arm in ("B", "D") and update <= 2000 else arm


def sampling(arm, update):
    directory = RUNS / stage(arm, update) / "sampling"
    path = directory / f"update_{update:05d}.npz"
    with np.load(path, allow_pickle=False) as data:
        return read(path.with_suffix(".json")), {key: data[key] for key in data.files}


def exposure_metrics(arm, ids):
    directory = RUNS / "evaluations" / arm / "10000-exposure"
    result = read(directory / "result.json")
    path = directory / "episode_results.json"
    assert file_sha(path) == result["episode_results_sha256"] and result["episodes"] == len(ids) * 2
    rows = read(path)
    assert [row["sequence_id"] for row in rows] == ids * 2
    data = np.asarray([row["values"] for row in rows]).reshape(2, len(ids), 9)
    valid = (data[:, :, 0] >= 100) & (data[:, :, 7] == 0)
    fraction = data[:, :, 1] / np.maximum(data[:, :, 0], 1)
    return result["initial_sha256"], dict(success=data[:, :, 6], failure=data[:, :, 7],
                                         position=np.where(valid, data[:, :, 4], np.nan),
                                         joint_time=np.where(valid, fraction, np.nan))


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=False)
    manifest = read(ROOT / "maps/manifest.json")
    initial = read(RUNS / "evaluation_initial_heldout.json")
    progress = {}
    for arm in (*ARMS, "prefix"):
        progress[arm] = {}
        for line in (RUNS / arm / "updates.jsonl").read_text().splitlines():
            row = json.loads(line)
            progress[arm][row["completed_updates"]] = row
    learning = []
    exposure = []
    exposures = {}
    for arm in ARMS:
        previous_samples = None
        for update in range(1000, 10001, 1000):
            source = stage(arm, update)
            result = read(RUNS / "evaluations" / source / f"{update:05d}-heldout/result.json")
            assert result["pool_sha256"] == manifest["evaluation_pool_sha256"] and result["episodes"] == 1024
            assert result["initial_sha256"] == initial["initial_sha256"]
            train = progress[source][update]
            learning.append(dict(
                arm=arm, update=update, source_run=source, episodes=result["episodes"],
                joint_success_pct=result["joint_success_rate"] * 100,
                native_failure_pct=result["native_failure_rate"] * 100,
                ee_position_mm=result["pooled_rmse"]["ee_position_m"] * 1000,
                ee_orientation_rad=result["pooled_rmse"]["ee_orientation_rad"],
                velocity_m_s=result["pooled_rmse"]["velocity_m_s"],
                yaw_rad_s=result["pooled_rmse"]["yaw_rad_s"],
                joint_good_time_pct=(result["mean_joint_good_time"] * 100
                                     if result["mean_joint_good_time"] is not None else None),
                training_reward=train["mean_training_reward"], checkpoint_sha256=result["checkpoint_sha256"],
            ))
        for update in range(500, 10001, 500):
            stat, data = sampling(arm, update)
            weights = stat["weights"]
            exposure.append(dict(
                arm=arm, update=update, mapping=stat["active_mapping"],
                loco=weights["loco"], manip=weights["manip"],
                probability_effective_sequences=stat["probability_effective_sequences"],
                probability_tv_from_previous=stat["probability_tv_from_previous"],
                sampled_episodes=stat["sampled_episodes"], unique_sampled=stat["unique_sampled"],
                sample_top1_share=stat["sample_top1_share"], sample_top10_share=stat["sample_top10_share"],
                sample_effective_sequences=stat["sample_effective_sequences"],
            ))
        _, initial_samples = sampling(arm, 2000)
        _, final_samples = sampling(arm, 10000)
        exposures[arm] = final_samples["sampled_total"] - initial_samples["sampled_total"]

    ids = manifest["final_exposure_panel"]["sequence_ids"]
    metrics = {}
    hashes = set()
    for arm in ARMS:
        initial_hash, metrics[arm] = exposure_metrics(arm, ids)
        hashes.add(initial_hash)
    assert len(hashes) == 1
    strata = []
    tv = []
    for arm in ("A", "C", "D"):
        sampled = exposures[arm]
        balanced = exposures["B"]
        tv.append(dict(arm=arm, reference="B", from_update=2000, through_update=10000,
                       sample_total_variation=float(np.abs(sampled / sampled.sum() - balanced / balanced.sum()).sum() / 2)))
        low, high = np.quantile(sampled, [.25, .75])
        for name, mask in (
            ("lowest", sampled[ids] <= low),
            ("middle", (sampled[ids] > low) & (sampled[ids] < high)),
            ("highest", sampled[ids] >= high),
        ):
            assert mask.any()
            for metric, scale in (("success", 100), ("failure", 100), ("position", 1000), ("joint_time", 100)):
                candidate, reference = metrics[arm][metric][:, mask], metrics["B"][metric][:, mask]
                paired = np.isfinite(candidate) & np.isfinite(reference)
                candidate_value, reference_value = candidate[paired], reference[paired]
                assert paired.any()
                strata.append(dict(
                    arm=arm, reference="B", stratum=name, panel_sequences=int(mask.sum()),
                    matched_episodes=int(paired.sum()), metric=metric,
                    mean_postfork_samples_candidate=float(sampled[ids][mask].mean()),
                    mean_postfork_samples_B=float(balanced[ids][mask].mean()),
                    arm_mean=float(candidate_value.mean() * scale),
                    B_mean=float(reference_value.mean() * scale),
                    arm_minus_B=float((candidate_value - reference_value).mean() * scale),
                ))
    table(out / "learning_curves.csv", learning)
    table(out / "sampling_curves.csv", exposure)
    table(out / "exposure_strata.csv", strata)
    table(out / "sample_distribution_shift.csv", tv)
    switches = [json.loads(line) for line in (RUNS / "D/mapping_switches.jsonl").read_text().splitlines()]
    assert [x["completed_updates"] for x in switches] == list(range(2500, 10001, 500))
    assert not switches[-1]["future_rollouts"]
    summary = {
        "completed_updates_per_arm": {arm: read(RUNS / arm / "result.json")["completed_updates"] for arm in ARMS},
        "seed": 42, "exposure_assignment_window": [2000, 10000], "heldout_final": [r for r in learning if r["update"] == 10000],
        "success_curve_mean_1k_to_10k_pct": {
            arm: float(np.trapz([r["joint_success_pct"] for r in learning if r["arm"] == arm]) / 9)
            for arm in ARMS
        },
        "dynamic_switches": switches, "sample_distribution_shift": tv,
        "paired_exposure_strata": strata, "fork": read(RUNS / "B/branch_restore.json"),
        "interpretation": "Exploratory single training seed. Exposure groups are descriptive post-training strata using "
                          "each candidate's post-fork counts, evaluated on a common command-only panel and paired with B. "
                          "Endogeneity of exposure prevents causal claims within strata.",
    }
    write(out / "summary.json", summary)

    panels = (
        ("joint_success_pct", "Strict joint success (%)"), ("joint_good_time_pct", "Joint-good time (%)"),
        ("ee_position_mm", "EE position RMSE (mm)"), ("ee_orientation_rad", "EE orientation RMSE (rad)"),
        ("velocity_m_s", "Base velocity RMSE (m/s)"), ("yaw_rad_s", "Yaw-rate RMSE (rad/s)"),
        ("native_failure_pct", "Native failure (%)"), ("training_reward", "Training episode reward"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
    for axis, (key, label) in zip(axes.ravel(), panels):
        for arm, color, style in zip(ARMS, ("#777777", "#265e87", "#b66429", "#328557"), ("-", "-", "-", "--")):
            selected = [row for row in learning if row["arm"] == arm]
            axis.plot([row["update"] / 1000 for row in selected], [row[key] for row in selected],
                      style, color=color, marker=".", label=f"{arm} {NAMES[arm]}")
        axis.axvline(2, color="0.8", linewidth=.8)
        axis.set_title(label)
        axis.grid(alpha=.2)
    axes[1, 0].set_xlabel("PPO updates (×1,000)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("E3 WBC cluster SGS · seed 42 · fixed strict held-out panel")
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out / "learning_curves.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    main()
