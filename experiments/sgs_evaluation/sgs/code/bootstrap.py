"""Upload only the small integration source; use existing remote runtime/assets."""
import hashlib,json,pathlib,shlex,subprocess,zlib
base=pathlib.Path(__file__).resolve().parent
names=['common.py','vendor_sampler.py','sampling.py','prepare.py','commands.py','metrics.py','adapter.py','job.py','docker_job.py','pipeline.py','report.py','python.sh','start.sh','tmux.sh']
files={n:(base/n).read_text() for n in names}
for n,v in files.items():
    if n.endswith('.py'):compile(v,n,'exec')
hashes={n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()}
manifest=dict(native_commit='4e12109d1eccb1d9da2e4a35f189fabea3f13677',isaaclab_commit='b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8',sgs_repository='https://github.com/chao142857/legged-robot-learning',sgs_repository_commit='e67f5b000513f9f059afeac110179560a4d23af6',sgs_core_path='projects/sgs/factorial_20261025/go2_piper_b1_sgs_20261025_work/shared_sgs/vendor/sampler.py',files=hashes,integration_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())
(base/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
code="""import sys,zlib,json,pathlib,subprocess
r=pathlib.Path('/data/go2_piper_sgs_20261102')
assert not (r/'STARTED.json').exists(),'Already started; do not overwrite running code'
d=json.loads(zlib.decompress(sys.stdin.buffer.read()))
for n in ('code','pools','audit','runs','evaluations','replays'):(r/n).mkdir(parents=True,exist_ok=True)
for n,v in d['files'].items():(r/'code'/n).write_text(v)
(r/'source_manifest.json').write_text(json.dumps(d['manifest'],indent=2)+'\\n')
print('Uploaded source',d['manifest']['integration_sha256'],flush=True)
subprocess.run(['bash',str(r/'code/start.sh')],check=True)
"""
payload=zlib.compress(json.dumps(dict(files=files,manifest=manifest)).encode())
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=3','-o','ControlPath=none','yanglab','python3 -c '+shlex.quote(code)]
print('Compressed source payload bytes:',len(payload),flush=True)
r=subprocess.run(cmd,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)
