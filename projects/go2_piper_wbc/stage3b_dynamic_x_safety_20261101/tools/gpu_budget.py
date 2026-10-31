"""Stage3b independent time-only SUM budget; activated only after Stage3 closes.

Research allocations are elastic. The last3600s remain unavailable to research.
No actor/optimizer/transition quota, old-ledger balance transfer or parallel GPU.
"""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,shutil,subprocess,time
APPROVED_ROOT=pathlib.Path('/data/go2_piper_wbc_stage3b_dynamic_x_20261101').resolve()
APPROVED_GRANT_ID='stage3b-original-uniform-dynamicX-safety-20261101-28800'

def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
    p=pathlib.Path(p);t=p.with_suffix(p.suffix+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def stamp(s):return datetime.datetime.fromisoformat(s[:26]+'+00:00').timestamp()
def read_state(cid):return json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]['State']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def job_states(jobs):
    if not jobs:return {}
    cids=[j['cid'] for j in jobs]
    assert len(cids)==len(set(cids)), 'Each container must be charged exactly once'
    lines=subprocess.check_output(['docker','inspect','--format','{{json .State}}',*cids],text=True).splitlines()
    assert len(lines)==len(cids)
    return dict(zip(cids,[json.loads(line) for line in lines]))

class Budget:
    def __init__(self,root):
        self.p=pathlib.Path(root);self.a=read(self.p/'AUTHORIZATION.json')
        assert self.p.resolve()==APPROVED_ROOT and self.a['grant_id']==APPROVED_GRANT_ID,'This single authorization cannot be cloned to a fresh directory or grant'
        self.cap=self.a['approved_GPU_seconds_SUM'];self.reserve=self.a['final_reserve_seconds']
        assert self.a['approved'] and self.cap==28800 and self.reserve==3600
        assert sha(self.p/'RESOURCE_PLAN.json')==self.a['plan_sha256']
        old=pathlib.Path(self.a['prior_closed_stage3_root']);assert old!=self.p
        done=read(old/'COMPLETED.json');assert done['clock']['cap_seconds']==21600 and not done['clock']['active']
        required={'LEDGER.json','GPU_CLOCK.json','COMPLETED.json','FROZEN_SELECTION.json','FINAL_MATRIX.json','CONSECUTIVE_NODE_FINAL.json','RESOURCES_FINAL.json','FINAL_RESTORE_RUN.json'}
        assert required <= self.a['prior_closed_file_sha256'].keys(), 'Complete original-stage closure evidence required'
        assert not read(old/'GPU_CLOCK.json')['active']
        assert not (old/'FINALIZER_ERROR.json').exists(), 'Resolve original-stage finalizer error first'
        restore=pathlib.Path(read(old/'FINAL_RESTORE_RUN.json')['run'])/'ppo/result.json'
        assert read(restore)['passed'] and read(restore)['updates']==0
        assert done['clock']['GPU_container_SUM_seconds']<=21600
        for name,expected in self.a['prior_closed_file_sha256'].items():assert sha(old/name)==expected,(name,'Closed Stage3 history changed')
        assert self.a.get('actor_update_cap') is None and self.a.get('transition_cap') is None
    def meter(self):
        total=0.;active=[];jobs=read(self.p/'LEDGER.json')['jobs'];states=job_states(jobs);now=time.time()
        for j in jobs:
            assert pathlib.Path(j['run']).parent==self.p/'runs' and j['grant_id']==self.a['grant_id']
            s=states[j['cid']]
            if not s['StartedAt'].startswith('0001'):total+=(now if s['Running'] else stamp(s['FinishedAt']))-stamp(s['StartedAt'])
            if s['Running']:active.append(j)
        d=dict(GPU_container_SUM_seconds=total,cap_seconds=self.cap,final_reserve_seconds=self.reserve,active=active,grant_id=self.a['grant_id'])
        if not (self.p/'COMPLETED.json').exists():write(self.p/'GPU_CLOCK.json',d)
        return d
    def phase_seconds(self,phase):
        total=0.;jobs=[j for j in read(self.p/'LEDGER.json')['jobs'] if j['phase']==phase];states=job_states(jobs);now=time.time()
        for j in jobs:
            s=states[j['cid']]
            if not s['StartedAt'].startswith('0001'):total+=(now if s['Running'] else stamp(s['FinishedAt']))-stamp(s['StartedAt'])
        return total
    def launch(self,cfg,label,purpose='research',timeout=3600):
        cfg=dict(cfg,grant_id=self.a['grant_id'])
        with (self.p/'launch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);assert not (self.p/'COMPLETED.json').exists()
            assert shutil.disk_usage(self.p).free>=15*1024**3,'Preserve history; stop for real disk shortage rather than deleting evidence'
            clock=self.meter();assert not clock['active'];phase=cfg.get('phase','P4' if purpose=='final' else 'P2')
            assert phase in ('P1','P2','P3','P4') and (phase=='P4')==(purpose=='final')
            if purpose!='final':assert not (self.p/'RESEARCH_CLOSED.json').exists()
            entry=cfg.get('entry','train_entry.py' if cfg['mode']=='train' else 'evaluate_entry.py')
            assert entry in ('train_entry.py','evaluate_entry.py','component_probe_entry.py')
            if purpose=='final' and cfg['mode']=='train':assert cfg['updates']==0,'Final reserve cannot hide research training'
            if entry=='component_probe_entry.py':assert cfg['mode']=='probe' and not cfg.get('resume'),'Diagnostic composition never trains or becomes a learned-policy success'
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Foreign GPU compute active; never preempt'
            os.kill(read(self.p/'WATCHER.json')['pid'],0)
            left=self.cap-clock['GPU_container_SUM_seconds']-(self.reserve if purpose!='final' else 0);assert left>60
            ck=pathlib.Path(cfg['checkpoint']);assert sha(ck)==cfg['checkpoint_sha256']
            extras=[]
            for key,destination in [('diagnostic_parent','/diagnostic_parent/checkpoint.pt'),('leg_anchor','/leg_anchor/checkpoint.pt')]:
                if cfg.get(key+'_checkpoint'):
                    q=pathlib.Path(cfg[key+'_checkpoint']);assert sha(q)==cfg[key+'_sha256'];extras+=['--mount',f'type=bind,src={q},dst={destination},readonly']
            run=self.p/'runs'/(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'-'+label);run.mkdir(parents=True);shutil.copytree(self.p/'active',run/'source');write(run/'config.json',cfg)
            b=read(self.p/'BASE_CONTAINER.json');cmd=b['command'][:b['command'].index('--entrypoint')];cmd[1]='create';cmd[cmd.index('--name')+1]='wbc-stage3b-'+run.name.lower()
            i=0
            while i<len(cmd):
                if cmd[i]=='--mount':
                    v=cmd[i+1];parts=dict(x.split('=',1) for x in v.split(',') if '=' in x)
                    if parts['dst'].startswith('/eval') or parts['dst']=='/node-stage3e':del cmd[i:i+2];continue
                    if 'readonly' not in v:
                        local=run if parts['dst']=='/runs' else run/'runtime'/parts['dst'].strip('/').replace('/','_');local.mkdir(parents=True,exist_ok=True);cmd[i+1]=f"type=bind,src={local},dst={parts['dst']}"
                i+=1
            cmd+=extras+['--log-driver=none','--mount',f'type=bind,src={run}/source,dst=/stage2-active,readonly','--mount',f'type=bind,src={run}/config.json,dst=/stage2-config.json,readonly','--mount',f'type=bind,src={ck},dst=/eval/checkpoint.pt,readonly']
            if cfg.get('resume'):cmd+=['--mount',f'type=bind,src={ck},dst=/resume/checkpoint.pt,readonly']
            cmd+=['--entrypoint','/bin/bash',b['image_id'],'-c','exec /usr/bin/timeout --kill-after=5s "$@" >>/runs/console.log 2>&1','stage3b',str(int(min(timeout,left-15))),'/bin/bash',b['python_wrapper'],'/stage2-active/'+entry]
            if cfg['mode']=='train':
                cmd+=['--num-envs',str(cfg['num_envs']),'--updates',str(cfg['updates'])]
                if cfg.get('resume'):cmd+=['--resume-sha',cfg['checkpoint_sha256']]
            versions={f.name:sha(f) for f in (run/'source').glob('*') if f.is_file()};write(run/'launch.json',dict(command=cmd,source_sha256=versions,grant_id=self.a['grant_id']))
            cid=subprocess.check_output(cmd,text=True).strip();ledger=read(self.p/'LEDGER.json');ledger['jobs'].append(dict(run=str(run),cid=cid,purpose=purpose,mode=cfg['mode'],phase=phase,grant_id=self.a['grant_id']));write(self.p/'LEDGER.json',ledger)
            subprocess.run(['docker','start',cid],check=True);return run
    def watch(self):
        assert not (self.p/'COMPLETED.json').exists(),'Never reopen a closed grant'
        write(self.p/'WATCHER.json',dict(pid=os.getpid(),independent_of_training=True,hard_SUM_seconds=self.cap))
        while not (self.p/'COMPLETED.json').exists():
            c=self.meter()
            for j in c['active']:
                r=pathlib.Path(j['run'])
                if c['GPU_container_SUM_seconds']>=self.cap-self.reserve and j['purpose']!='final':
                    f=r/'TIME_STOP'
                    if not f.exists():f.touch()
                    if time.time()-f.stat().st_mtime>30:subprocess.run(['docker','stop','--time','3',j['cid']],check=False)
                if c['GPU_container_SUM_seconds']>=self.cap-10:subprocess.run(['docker','stop','--time','2',j['cid']],check=False)
            if c['GPU_container_SUM_seconds']>=self.cap-1 and not c['active']:
                write(self.p/'HARD_BUDGET_REACHED.json',dict(clock=c,new_GPU_launch_forbidden=True));return
            time.sleep(2)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();Budget(a.root).watch()
