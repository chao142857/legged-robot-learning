import fcntl, hashlib, json, os, pathlib, time, traceback
from docker_job import ROOT, IMAGE, KIT, run

def write(p,x):
    p=ROOT/p; p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(x,indent=2)+'\n'); tmp.replace(p)
def read(p): return json.loads((ROOT/p).read_text())
def main():
    assert not (ROOT/'STARTED.json').exists(), 'Do not duplicate or overwrite E1'
    write('STARTED.json',dict(unix=time.time(),pid=os.getpid(),formal_order=['flat-sgs-e1','wbc-sgs-e1'],updates_each=10000,
                             short_updates_each=200,num_envs=4096,rollout=24,seed=42,image=IMAGE,authorization='User: 按照md启动实验; existing authorization: automatic full training after engineering checks'))
    manifest=read('source_manifest.json')
    for n,h in manifest['files'].items(): assert hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest()==h,n
    write('status.json',dict(state='preflight',unix=time.time()))
    assert run('prepare',['/workspace/code/prepare_e1.py'],gpu=False)==0
    shorts=[]
    for kind in ('flat','wbc'):
        label='short-'+kind+'-sgs-e1'
        write('status.json',dict(state='engineering_short',run=label,completed=[x['task'] for x in shorts],unix=time.time()))
        assert run(label,['/workspace/code/train_e1.py','--role','train','--kind',kind,'--mode','sgs','--stage','short',
                         '--updates','200','--num_envs','4096','--out','/workspace/runs/'+label,'--headless','--kit_args',KIT],training=True)==0
        r=read('runs/'+label+'/result.json'); tests=read('runs/'+label+'/injected_engineering_checks.json')
        assert r['passed'] and r['completed_updates']==200 and r['environment_transitions']==19660800
        assert r['restoration']['passed'] and r['rng_roundtrip'] and tests['passed']
        assert r['summary']['stage_transitions']==2 and r['summary']['stage_crossing_episodes']>0
        assert r['summary']['strict_shadow_compared_episodes']==r['summary']['feedback_accepted']
        assert read('evaluations/'+label+'/00200/result.json')['train_history_updates']==0
        shorts.append(r)
    write('ENGINEERING_SHORT_PASS.json',dict(passed=True,results=shorts,success_rate_not_a_gate=True,short_state_discarded=True,unix=time.time()))
    completed=[]
    for kind in ('flat','wbc'):
        label='formal-'+kind+'-sgs-e1'
        write('status.json',dict(state='formal_training',run=label,completed=[r['task'] for r in completed],unix=time.time()))
        assert run(label,['/workspace/code/train_e1.py','--role','train','--kind',kind,'--mode','sgs','--stage','formal',
                         '--updates','10000','--num_envs','4096','--out','/workspace/runs/'+label,'--headless','--kit_args',KIT],training=True)==0
        r=read('runs/'+label+'/result.json')
        assert r['passed'] and r['completed_updates']==10000 and r['environment_transitions']==983040000
        for n in range(1000,10001,1000):
            e=read('evaluations/'+label+'/'+str(n).zfill(5)+'/result.json')
            assert e['train_history_updates']==0 and e['episodes']==1024
        completed.append(r); write('formal_results.json',completed)
    for kind in ('flat','wbc'):
        label='formal-'+kind+'-sgs-e1'
        write('status.json',dict(state='representative_replay',run=label,unix=time.time()))
        assert run('video-'+kind+'-sgs-e1',['/workspace/code/evaluation.py','--role','video','--kind',kind,'--mode','sgs',
                   '--checkpoint','/workspace/runs/'+label+'/policy/model_9999.pt','--num_envs','1',
                   '--out','/workspace/replays/'+label,'--headless','--kit_args',KIT])==0
    write('status.json',dict(state='reporting',unix=time.time()))
    assert run('final-report',['/workspace/code/report_e1.py'],gpu=False)==0
    write('COMPLETED.json',dict(state='complete',formal_runs=2,updates=20000,formal_environment_transitions=1966080000,finished_unix=time.time()))
    write('status.json',read('COMPLETED.json'))

if __name__=='__main__':
    with (ROOT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: main()
        except BaseException as exc:
            fault=dict(state='fault',error=repr(exc),traceback=traceback.format_exc(),unix=time.time())
            write('PIPELINE_FAULT.json',fault); write('status.json',fault); print(traceback.format_exc(),flush=True); raise
