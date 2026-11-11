"""Isolated native-host orchestration; no Docker or changes to installed environments."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
EXPERIENCE=ROOT/"experience"
KIT=" ".join([f"--/persistent/isaac/asset_root/{key}={ROOT/'assets'}" for key in ("cloud","default","nvidia")]+["--/app/extensions/registryEnabled=false"])

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(path)

def run(label,arguments,gpu=True,on_tick=None):
    command=["bash",str(ROOT/"code/python.sh"),*arguments]
    out=ROOT/"jobs"/label
    out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    record=dict(label=label,host="ubuntu-robot",started_unix=start,command=command,gpu=gpu)
    write(out/"launch.json",record)
    print("NSGS_JOB_START "+label,flush=True)
    error=None
    max_gpu_mib=0
    next_monitor=0
    with (out/"console.log").open("xb") as log:
        job_env=dict(os.environ)
        if not gpu:job_env["CUDA_VISIBLE_DEVICES"]=""
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True,env=job_env)
        record["pid"]=process.pid
        write(out/"launch.json",record)
        try:
            while process.poll() is None:
                if gpu and time.monotonic()>=next_monitor:
                    probe=subprocess.run(["nvidia-smi","--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
                    if probe.returncode==0:
                        values=[float(v.strip()) for v in probe.stdout.splitlines()[0].split(",")]
                        max_gpu_mib=max(max_gpu_mib,values[0])
                        with (out/"gpu.jsonl").open("a") as f:
                            f.write(json.dumps(dict(unix=time.time(),memory_mib=values[0],utilization=values[1],temperature_c=values[2],power_w=values[3]))+"\n")
                    next_monitor=time.monotonic()+5
                if on_tick: on_tick(process)
                time.sleep(.5)
        except BaseException as exc:
            error=repr(exc)
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
                try: process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            record.update(exit_code=process.returncode,wall_seconds=time.time()-start,finished_unix=time.time(),
                          max_total_gpu_mib=max_gpu_mib,error=error)
            write(out/"result.json",record)
    print("NSGS_JOB_END "+label+" exit="+str(process.returncode),flush=True)
    if process.returncode:
        raise RuntimeError(f"{label}: exit {process.returncode}, preserved log {out/'console.log'}")
    return record
