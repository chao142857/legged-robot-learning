import hashlib,json,pathlib,subprocess,time,traceback,os
from docker_job import ROOT,IMAGE,KIT,run

def write(p,x):
    p=ROOT/p;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (ROOT/'STARTED.json').exists(),'Never duplicate/reinitialize an existing run'
    write('STARTED.json',dict(unix=time.time(),pid=os.getpid(),formal_order=['flat-uniform','flat-sgs','wbc-uniform','wbc-sgs'],updates_each=10000,short_updates_each=200,num_envs=4096,seed=42,image=IMAGE))
    native=pathlib.Path('/data/go2_piper_native_10k_20261101')
    proof=read('source_manifest.json')
    for name,value in proof['files'].items():assert sha(ROOT/'code'/name)==value,name
    upstream=json.loads((native/'audit/upstream_tree.json').read_text())
    checked=[]
    for x in upstream['tree']:
        if x['type']!='blob' or not (x['path'].endswith('.py') and x['path'].startswith(('source/','scripts/rsl_rl/')) or '/assets/go2_piper/' in x['path']):continue
        raw=(native/'LeggedManip_Lab'/x['path']).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==x['sha'];checked.append(x['path'])
    write('audit/native_source_verified.json',dict(commit=upstream['sha'],files=checked,all_match=True))
    write('status.json',dict(state='prepare',unix=time.time()))
    assert run('prepare',['/workspace/code/prepare.py'],gpu=False)==0
    baseline=json.loads((native/'delivery/summary.json').read_text())
    for task in baseline['tasks']:
        kind='flat' if task['task'].endswith('Flat') else 'wbc';checkpoint='/native/'+str(pathlib.Path(task['last_checkpoint']['path']).relative_to(native))
        write('status.json',dict(state='native_10k_metric_and_unit_check',task=kind,unix=time.time()))
        args=['/workspace/code/job.py','--role','baseline','--kind',kind,'--checkpoint',checkpoint,'--num_envs','1024','--out','/workspace/baseline/'+kind,'--headless','--kit_args',KIT]
        assert run('baseline-'+kind,args)==0,'Baseline metric/units engineering failure'
        check=read('baseline/'+kind+'/result.json');assert check['passed'] and check['train_history_updates']==0 and check['episodes']==1024
    candidate=read('audit/success_definition_candidate.json')
    canonical=json.dumps(candidate,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    write('protocol_frozen.json',dict(definition=candidate,definition_sha256=hashlib.sha256(canonical).hexdigest(),frozen_unix=time.time(),native_10k_baseline_checks={k:read('baseline/'+k+'/result.json') for k in ('flat','wbc')},definition_chosen_before_baseline=True,no_threshold_search=True,evaluation_pools=read('pools/manifest.json')))
    order=[('flat','uniform'),('flat','sgs'),('wbc','uniform'),('wbc','sgs')]
    short=[]
    for kind,mode in order:
        label='short-'+kind+'-'+mode
        write('status.json',dict(state='engineering_short',run=label,completed=[x['task']+'-'+x['mode'] for x in short],unix=time.time()))
        args=['/workspace/code/job.py','--role','train','--kind',kind,'--mode',mode,'--stage','short','--updates','200','--num_envs','4096','--out','/workspace/runs/'+label,'--headless','--kit_args',KIT]
        assert run(label,args,training=True)==0,'Engineering short program fault: '+label
        result=read('runs/'+label+'/result.json');assert result['passed'] and result['completed_updates']==200 and result['environment_transitions']==19660800 and result['restoration']['passed'] and result['rng_roundtrip']
        short.append(result)
    for kind in ('flat','wbc'):
        pair=[s for s in short if s['task'].lower().endswith(kind)]
        assert pair[0]['initialization']['actor_sha256']==pair[1]['initialization']['actor_sha256']
        assert pair[0]['initialization']['critic_sha256']==pair[1]['initialization']['critic_sha256']
    # Each short run also tests the external evaluator with its learner resident.
    # Evaluation scores never decide the engineering pass.
    write('ENGINEERING_SHORT_PASS.json',dict(passed=True,short_results=short,cpu=read('audit/cpu_engineering_tests.json'),success_rate_not_a_gate=True,formal_authorization='User explicitly authorized automatic formal training after engineering pass.',unix=time.time()))
    formal=[]
    for kind,mode in order:
        label='formal-'+kind+'-'+mode
        write('status.json',dict(state='formal_training',run=label,completed=[s['task']+'-'+s['mode'] for s in formal],unix=time.time()))
        args=['/workspace/code/job.py','--role','train','--kind',kind,'--mode',mode,'--stage','formal','--updates','10000','--num_envs','4096','--out','/workspace/runs/'+label,'--headless','--kit_args',KIT]
        assert run(label,args,training=True)==0,'Formal program/resource/nonfinite failure: '+label
        result=read('runs/'+label+'/result.json');assert result['passed'] and result['completed_updates']==10000 and result['environment_transitions']==983040000
        evaluated=list((ROOT/'evaluations'/label).glob('*/result.json'));assert len(evaluated)==10 and all(json.loads(p.read_text())['train_history_updates']==0 for p in evaluated)
        formal.append(result);write('formal_results.json',formal)
    for kind,mode in order:
        label='formal-'+kind+'-'+mode;checkpoint='/workspace/runs/'+label+'/policy/model_9999.pt'
        write('status.json',dict(state='representative_replay',run=label,unix=time.time()))
        assert run('video-'+kind+'-'+mode,['/workspace/code/job.py','--role','video','--kind',kind,'--mode',mode,'--checkpoint',checkpoint,'--num_envs','1','--out','/workspace/replays/'+label,'--headless','--kit_args',KIT])==0
    write('status.json',dict(state='reporting',unix=time.time()))
    assert run('final-report',['/workspace/code/report.py'],gpu=False)==0
    write('COMPLETED.json',dict(state='complete',formal_runs=4,updates=40000,formal_environment_transitions=3932160000,finished_unix=time.time()))
    write('status.json',read('COMPLETED.json'))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        fault=dict(state='fault',error=repr(exc),traceback=traceback.format_exc(),unix=time.time())
        write('PIPELINE_FAULT.json',fault);write('status.json',fault);print(traceback.format_exc(),flush=True);raise
