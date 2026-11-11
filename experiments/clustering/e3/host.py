"""YangLab runner, using the same immutable simulator image as E2."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
E2 = Path("/data/go2_piper_sgs_e2_20261104")
spec = importlib.util.spec_from_file_location("e2_docker", E2 / "code/docker_job.py")
docker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docker)
docker.ROOT = ROOT
KIT = docker.KIT


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def run(label, arguments, gpu=True, on_tick=None):
    command, name = docker.command(label, arguments, gpu=gpu)
    insert = command.index("--workdir")
    command[insert:insert] = [
        "--mount", f"type=bind,src={E2},dst=/e2reference,readonly",
        "--mount", f"type=bind,src={E2 / 'pools'},dst=/workspace/pools,readonly",
    ]
    out = ROOT / "jobs" / label
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    record = dict(label=label, container=name, started_unix=start, command=command)
    write(out / "launch.json", record)
    print(f"E3_JOB_START {label}", flush=True)
    with (out / "console.log").open("xb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if on_tick:
                    on_tick(process)
                time.sleep(.5)
        except BaseException:
            if process.poll() is None:
                subprocess.run(["docker", "stop", "--time", "20", name], check=False)
                process.wait()
            raise
    record.update(exit_code=process.returncode, wall_seconds=time.time() - start)
    write(out / "result.json", record)
    print(f"E3_JOB_END {label} exit={process.returncode}", flush=True)
    if process.returncode:
        raise RuntimeError(f"{label} exited {process.returncode}: {out / 'console.log'}")


if __name__ == "__main__":
    import sys
    label, script = sys.argv[1:3]
    run(label, [script, *sys.argv[3:]], gpu=label != "prepare")
