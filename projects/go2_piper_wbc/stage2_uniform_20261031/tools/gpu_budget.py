"""Independent Stage2 GPU-container SUM meter. Native PPO is external."""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,shutil,subprocess,time

def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
 p=pathlib.Path(p);tmp=p.with_suffix(p.suffix+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def stamp(s):return datetime.datetime.fromisoformat(s[:26]+'+00:00').timestamp()
class Budget:
 def __init__(self,root):
  self.p=pathlib.Path(root);self.a=read(self.p/'AUTHORIZATION.json');self.cap=self.a['approved_GPU_seconds_SUM'];self.reserve=self.a['final_reserve_seconds']
  assert self.a['approved'] and self.a['host_change_approved'] and self.cap==14400
  assert hashlib.sha256((self.p/'BUDGET_REQUEST.json').read_bytes()).hexdigest()==self.a['public_budget_view_sha256']
 def meter(self):
  total=0.;active=[]
  for j in read(self.p/'LEDGER.json')['jobs']:
   s=read_state(j['cid'])
   if not s['StartedAt'].startswith('0001'):total+=(time.time() if s['Running'] else stamp(s['FinishedAt']))-stamp(s['StartedAt'])
   if s['Running']:active.append(j)
  v=dict(GPU_container_SUM_seconds=total,cap_seconds=self.cap,reserve_seconds=self.reserve,active=active);write(self.p/'GPU_CLOCK.json',v);return v
 def launch(self,cfg,label,purpose='research',timeout=1200):
  with (self.p/'launch.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);assert not (self.p/'COMPLETED.json').exists();m=self.meter();assert not m['active']
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Foreign GPU compute active'
   os.kill(read(self.p/'WATCHER.json')['pid'],0)
   left=self.cap-m['GPU_container_SUM_seconds']-(self.reserve if purpose!='final' else 0);assert left>60
   ck=pathlib.Path(cfg['checkpoint']);assert hashlib.sha256(ck.read_bytes()).hexdigest()==cfg['checkpoint_sha256']
   run=self.p/'runs'/(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'-'+label);run.mkdir(parents=True);shutil.copytree(self.p/'active',run/'source');write(run/'config.json',cfg)
   b=read(self.p/'BASE_CONTAINER.json');cmd=b['command'][:b['command'].index('--entrypoint')];cmd[1]='create';cmd[cmd.index('--name')+1]='wbc-stage2-'+run.name.lower()
   i=0
   while i<len(cmd):
    if cmd[i]=='--mount':
     v=cmd[i+1];parts=dict(x.split('=',1) for x in v.split(',') if '=' in x)
     if parts['dst'].startswith('/eval') or parts['dst']=='/node-stage3e':del cmd[i:i+2];continue
     if 'readonly' not in v:
      local=run if parts['dst']=='/runs' else run/'runtime'/parts['dst'].strip('/').replace('/','_');local.mkdir(parents=True,exist_ok=True);cmd[i+1]=f"type=bind,src={local},dst={parts['dst']}"
    i+=1
   cmd+=['--log-driver=none','--mount',f'type=bind,src={run}/source,dst=/stage2-active,readonly','--mount',f'type=bind,src={run}/config.json,dst=/stage2-config.json,readonly','--mount',f'type=bind,src={ck},dst=/eval/checkpoint.pt,readonly','--entrypoint','/bin/bash',b['image_id'],'-c','exec /usr/bin/timeout --kill-after=5s "$@" >>/runs/console.log 2>&1','stage2',str(int(min(timeout,left-15))),'/bin/bash',b['python_wrapper'],'/stage2-active/train_entry.py' if cfg['mode']=='train' else '/stage2-active/evaluate_entry.py']
   if cfg['mode']=='train':
    cmd+=['--num-envs',str(cfg['num_envs']),'--updates',str(cfg['updates'])]
    if cfg.get('resume'):cmd+=['--resume-sha',cfg['checkpoint_sha256']]
   if cfg.get('resume'):cmd[2:2]=['--mount',f'type=bind,src={ck},dst=/resume/checkpoint.pt,readonly']
   write(run/'launch.json',dict(command=cmd,source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (run/'source').glob('*.py')}));cid=subprocess.check_output(cmd,text=True).strip();d=read(self.p/'LEDGER.json');d['jobs'].append(dict(run=str(run),cid=cid,purpose=purpose,mode=cfg['mode']));write(self.p/'LEDGER.json',d);subprocess.run(['docker','start',cid],check=True);return run
 def watch(self):
  write(self.p/'WATCHER.json',dict(pid=os.getpid()))
  while not (self.p/'COMPLETED.json').exists():
   m=self.meter()
   for j in m['active']:
    r=pathlib.Path(j['run'])
    if m['GPU_container_SUM_seconds']>=self.cap-self.reserve and j['purpose']!='final':
     f=r/'TIME_STOP'
     if not f.exists():f.touch()
     if time.time()-f.stat().st_mtime>30:subprocess.run(['docker','stop','--time','5',j['cid']],check=False)
    if m['GPU_container_SUM_seconds']>=self.cap-10:subprocess.run(['docker','stop','--time','2',j['cid']],check=False)
   time.sleep(2)
def read_state(cid):return json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]['State']
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);v=a.parse_args();Budget(v.root).watch()
