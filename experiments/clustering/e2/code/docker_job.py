import json, os, pathlib, shlex, subprocess, sys, time
ROOT=pathlib.Path('/data/go2_piper_sgs_e2_20261104')
BASE=pathlib.Path('/home/tongji/alex/go2_piper_yanglab')
IMAGE='sha256:8ee9b2bb8e6aae08dace71ee1e2141a4d62bcc63e4cf0eac28b347111a486a87'
KIT='--/persistent/isaac/asset_root/cloud=/node-assets --/persistent/isaac/asset_root/default=/node-assets --/persistent/isaac/asset_root/nvidia=/node-assets --/app/extensions/registryEnabled=false'
def command(label,script_args,gpu=True):
    state=ROOT/'runtime'/label
    for n in ('config','data','kit-data','kit-logs','tmp'): (state/n).mkdir(parents=True,exist_ok=True)
    name='e2-'+json.loads((ROOT/'source_manifest.json').read_text())['integration_sha256'][:8]+'-'+label
    c=['docker','run','--pull=never','--name',name,'--network=none','--log-driver=none',
       '--cap-drop=ALL','--security-opt=no-new-privileges','--pids-limit=2048','--cpus=16',
       '--memory=24g','--memory-swap=24g','--shm-size=8g','--user','1000:1000']
    if gpu:c+=['--gpus','all']
    for key,value in dict(NVIDIA_DRIVER_CAPABILITIES='compute,utility,graphics',XDG_CACHE_HOME='/home/alex/.cache',
                          XDG_CONFIG_HOME='/home/alex/.config',XDG_DATA_HOME='/home/alex/.local/share',
                          CUDA_CACHE_PATH='/home/alex/.cache/cuda',TMPDIR='/native-tmp',TZ='Asia/Shanghai').items():
        c+=['--env',key+'='+value]
    mounts=[(ROOT,'/workspace',False),(pathlib.Path('/data/go2_piper_continue15k_20261103'),'/continue_reference',True),(pathlib.Path('/data/go2_piper_sgs_e1_20261102'),'/e1reference',True),(pathlib.Path('/data/go2_piper_sgs_20261102'),'/reference',True),(pathlib.Path('/data/go2_piper_native_10k_20261101'),'/native',True),(BASE/'src/go2_piper_isaaclab','/home/alex/dog/go2_piper_isaaclab',True),
            (BASE/'assets/runtime_closure','/node-assets',True),(BASE/'env/cache/runtime-shared','/home/alex/.cache',False),
            (state/'config','/home/alex/.config',False),(state/'data','/home/alex/.local/share',False),
            (state/'kit-data','/home/alex/dog/go2_piper_env/lib/python3.11/site-packages/isaacsim/kit/data',False),
            (state/'kit-logs','/home/alex/dog/go2_piper_env/lib/python3.11/site-packages/isaacsim/kit/logs',False),
            (BASE/'env/cache/runtime-shared/kit-cache','/home/alex/dog/go2_piper_env/lib/python3.11/site-packages/isaacsim/kit/cache',False),
            (state/'tmp','/native-tmp',False)]
    for src,dst,ro in mounts:
        c+=['--mount','type=bind,src='+str(src)+',dst='+dst+(',readonly' if ro else '')]
    c+=['--workdir','/workspace','--entrypoint','/bin/bash',IMAGE,'/workspace/code/python.sh']+script_args
    return c,name
def run(label,args,gpu=True,training=False):
    c,name=command(label,args,gpu)
    out=ROOT/'jobs'/label;out.mkdir(parents=True,exist_ok=True)
    manifest=dict(label=label,command=c,command_shell=shlex.join(c),container_name=name,started_unix=time.time(),image=IMAGE)
    (out/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('START '+label+' '+shlex.join(args),flush=True)
    with (out/'console.log').open('xb') as f:
        proc=subprocess.Popen(c,stdout=f,stderr=subprocess.STDOUT)
        (out/'docker_cli_pid.txt').write_text(str(proc.pid)+'\n')
        while proc.poll() is None:
            if training:
                requests=ROOT/'runs'/label/'evaluation_requests'
                for request in sorted(requests.glob('*.json')):
                    if request.name.endswith('.done.json') or request.with_suffix('.done.json').exists():continue
                    job=json.loads(request.read_text());evaluation=ROOT/'evaluations'/label/request.stem
                    rc_eval=run(label+'-eval-'+request.stem,['/workspace/code/evaluation.py','--role','evaluate','--kind',job['kind'],'--checkpoint',job['checkpoint'],'--num_envs','1024','--out','/workspace/'+str(evaluation.relative_to(ROOT)),'--headless','--kit_args',KIT])
                    data=dict(exit_code=rc_eval,checkpoint_sha256=job['checkpoint_sha256'],result=str(evaluation/'result.json'))
                    tmp=request.with_suffix('.done.tmp');tmp.write_text(json.dumps(data)+'\n');tmp.replace(request.with_suffix('.done.json'))
            time.sleep(.25)
        rc=proc.returncode
    manifest.update(exit_code=rc,finished_unix=time.time(),wall_seconds=time.time()-manifest['started_unix'])
    result=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
    if result.returncode==0:
        inspect_data=json.loads(result.stdout)[0]
        (out/'container_state.json').write_text(json.dumps(inspect_data['State'],indent=2)+'\n')
        manifest['container_state']=inspect_data['State']
    (out/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('FINISH '+label+' rc='+str(rc),flush=True)
    return rc
if __name__=='__main__':
    raise SystemExit(run(sys.argv[1],sys.argv[2:]))
