"""Isolated, persistent V0.2 jobs using the existing immutable Isaac image."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
E2 = Path('/data/go2_piper_sgs_e2_20261104')
EXPERIENCE = ROOT/'experience'
spec = importlib.util.spec_from_file_location('e2_docker', E2/'code/docker_job.py')
docker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docker)
docker.ROOT = ROOT
KIT = docker.KIT


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def inside(value):
    value = str(value)
    for source, target in ((str(ROOT), '/workspace'), (str(EXPERIENCE), '/experience')):
        if value == source or value.startswith(source+'/'):
            return target+value[len(source):]
    return value


def outside(value):
    value = str(value)
    for source, target in (('/workspace', str(ROOT)), ('/experience', str(EXPERIENCE))):
        if value == source or value.startswith(source+'/'):
            return Path(target+value[len(source):])
    return Path(value)


def run(label, arguments, gpu=True, on_tick=None):
    command, name = docker.command('v02-'+label, [inside(a) for a in arguments], gpu=gpu)
    index = command.index('--workdir')
    command[index:index] = [
        '--mount', f'type=bind,src={E2},dst=/e2reference,readonly',
        '--mount', f'type=bind,src={E2/"pools"},dst=/workspace/pools,readonly',
        '--mount', 'type=bind,src=/data,dst=/data,readonly',
        '--mount', f'type=bind,src={EXPERIENCE},dst=/experience',
    ]
    out = ROOT/'jobs'/label
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    record = dict(label=label, container=name, started_unix=start, command=command, gpu=gpu)
    write(out/'launch.json', record)
    print('V02_JOB_START '+label, flush=True)
    with (out/'console.log').open('xb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if on_tick:
                    on_tick(process)
                time.sleep(.5)
        except BaseException:
            if process.poll() is None:
                subprocess.run(['docker', 'stop', '--time', '30', name], check=False)
                process.wait()
            raise
    record.update(exit_code=process.returncode, finished_unix=time.time(), wall_seconds=time.time()-start)
    inspected = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name], capture_output=True, text=True)
    if inspected.returncode == 0:
        record['container_state'] = json.loads(inspected.stdout)
    write(out/'result.json', record)
    print('V02_JOB_END '+label+' exit='+str(process.returncode), flush=True)
    if process.returncode:
        raise RuntimeError(f'{label} exit={process.returncode}; evidence retained at {out}')
    return record
