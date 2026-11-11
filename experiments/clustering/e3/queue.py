"""Run engineering acceptance, then the authorized serial four-arm experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from host import E2, KIT, ROOT, run, write

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-id", default="smoke-v1")
parser.add_argument("--smoke-only", action="store_true")
args = parser.parse_args()
assert args.smoke_id.startswith("smoke-v") and args.smoke_id[7:].isdigit()


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def status(phase, **detail):
    write(ROOT / "E3_STATUS.json", dict(phase=phase, unix=time.time(), **detail))


def in_container(path):
    return "/workspace/" + str(path.relative_to(ROOT))


def read(path):
    return json.loads(path.read_text())


def train_arm(arm, base, smoke=False):
    out = base / arm
    name = base.name + "-" + arm.lower()
    arguments = [
        "/workspace/code/train.py", "--arm", arm, "--out", in_container(out),
        "--headless", "--kit_args", KIT,
    ]
    if smoke:
        arguments.append("--smoke")
    if arm in ("B", "D"):
        fork = 4 if smoke else 2000
        cp = base / "prefix/checkpoints" / f"update_{fork:05d}.pt"
        assert cp.is_file()
        arguments += ["--resume", in_container(cp)]

    def tick(process):
        for req in sorted((out / "evaluation_requests").glob("*.json")):
            ack = req.with_suffix(".done.json")
            if req.name.endswith(".done.json") or ack.exists():
                continue
            request = read(req)
            checkpoint = ROOT / Path(request["checkpoint"]).relative_to("/workspace")
            assert checkpoint.parent == out / "checkpoints" and sha(checkpoint) == request["checkpoint_sha256"]
            evaluation = base / "evaluations" / arm / req.stem
            run(name + "-eval-" + req.stem, [
                "/workspace/code/evaluate.py", "--checkpoint", request["checkpoint"],
                "--kind", request["kind"], "--out", in_container(evaluation),
                "--headless", "--kit_args", KIT,
            ])
            result = read(evaluation / "result.json")
            assert result["passed"] and result["update"] == request["update"]
            assert result["kind"] == request["kind"] and result["checkpoint_sha256"] == request["checkpoint_sha256"]
            assert result["training_history_updates"] == 0 and result["model_weights_unchanged"]
            expected = base / f"evaluation_initial_{request['kind']}.json"
            identity = {key: result[key] for key in ("initial_sha256", "pool_sha256", "seed", "episodes", "protocol_sha256")}
            if expected.exists():
                assert read(expected) == identity
            else:
                write(expected, identity)
            write(ack, result)

    status("gpu_smoke" if smoke else "formal_training", arm=arm, run=str(out))
    run(name, arguments, on_tick=tick)
    if (out / "fault.txt").exists():
        raise RuntimeError((out / "fault.txt").read_text())
    result = read(out / "result.json")
    assert result["passed"] and result["smoke"] == smoke and result["arm"] == arm
    return result


def preflight():
    source_path = E2 / "artifacts/E2_source_versions.json"
    assert sha(source_path) == "33f77b63567b5b60e8f500b849c004c1788d39413e9ba15ad714d07e1d6c3af5"
    source = read(source_path)
    for name, expected in source["code_files"].items():
        assert sha(E2 / name) == expected, name
    maps = read(ROOT / "maps/manifest.json")
    assert len(maps["maps"]) == 19
    for name, entry in maps["maps"].items():
        assert sha(ROOT / "maps" / f"{name}.npy") == entry["file_sha256"], name
    files = [ROOT / "host.py", ROOT / "queue.py", *sorted((ROOT / "code").glob("*.*")),
             *sorted((ROOT / "tests").glob("*.py"))]
    write(ROOT / "source_manifest.json", dict(
        integration_sha256="20261104-e3",
        files={str(path.relative_to(ROOT)): sha(path) for path in files},
        e2_source_manifest_sha256=sha(source_path), native_commit=source["native_commit"],
        isaaclab_commit=source["isaaclab_commit"], maps_manifest_sha256=sha(ROOT / "maps/manifest.json"),
    ))
    status("cpu_acceptance")
    run(args.smoke_id + "-cpu", ["-m", "unittest", "discover", "-s", "/workspace/tests", "-v"], gpu=False)


def check_branches(base):
    b = read(base / "B/branch_restore.json")
    d = read(base / "D/branch_restore.json")
    assert b == d, f"B/D restored state or first rollout differs: {[key for key in b if b[key] != d.get(key)]}"
    assert len(b["rollout_signatures"]) == 24 and b["active_episodes"] == 4096
    return b


def smoke_acceptance():
    base = ROOT / args.smoke_id
    base.mkdir(exist_ok=False)
    prefix = train_arm("prefix", base, smoke=True)
    b = train_arm("B", base, smoke=True)
    d = train_arm("D", base, smoke=True)
    branch = check_branches(base)
    assert prefix["completed_updates"] == 4 and b["completed_updates"] == d["completed_updates"] == 8
    assert b["source_checkpoint_sha256"] == d["source_checkpoint_sha256"] == prefix["checkpoint_sha256"]
    assert d["native_feedback_episodes"] > prefix["native_feedback_episodes"] > 0
    switches = [json.loads(line) for line in (base / "D/mapping_switches.jsonl").read_text().splitlines()]
    assert len(switches) == 1 and switches[0]["completed_updates"] == 5
    assert switches[0]["before"] == "m00" and switches[0]["after"] == "m01"
    assert switches[0]["feedback_events"] > 0 and d["active_mapping"] == "m01"
    receipt = dict(passed=True, prefix=prefix, B=b, D=d, branch=branch, switch=switches[0],
                   source_manifest_sha256=sha(ROOT / "source_manifest.json"))
    write(base / "ACCEPTANCE.json", receipt)
    print("E3_ENGINEERING_ACCEPTANCE_PASS", flush=True)


def formal():
    base = ROOT / "runs"
    base.mkdir(exist_ok=False)
    for arm in ("A", "prefix", "B", "C", "D"):
        result = train_arm(arm, base)
        assert result["completed_updates"] == (2000 if arm == "prefix" else 10000)
    branch = check_branches(base)
    run("formal-report", ["/workspace/code/report.py"], gpu=False)
    status("complete", run=str(base), branch=branch["source_checkpoint_sha256"],
           summary=str(ROOT / "results/summary.json"))


try:
    preflight()
    smoke_acceptance()
    if args.smoke_only:
        status("engineering_passed", acceptance=str(ROOT / args.smoke_id / "ACCEPTANCE.json"))
    else:
        formal()
except BaseException as exc:
    status("failed", error=repr(exc), traceback=traceback.format_exc())
    raise
