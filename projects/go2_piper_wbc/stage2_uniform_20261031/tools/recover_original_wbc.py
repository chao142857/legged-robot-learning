"""Bounded, read-only recovery inventory for the ORIGINAL PPO WBC host.
Does not import a simulator, load a policy, launch training, or alter remote files.
"""
import argparse,datetime,hashlib,json,pathlib,shlex,subprocess
REMOTE = r'''
import pathlib,subprocess,json,hashlib,datetime

def cmd(argv,timeout=15,limit=12000):
 try:
  r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
  return dict(returncode=r.returncode,stdout=r.stdout[:limit],stderr=r.stderr[:1000])
 except Exception as e:return dict(error=repr(e))
root=pathlib.Path('/home/alex/dog')
out=dict(host=cmd(['hostname']),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=cmd(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),GPU_compute=cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),docker=cmd(['docker','ps','--format','{{.Names}} {{.Status}}']))
worktrees=['LeggedManip_Lab_B0_20261026','LeggedManip_Lab_recipe_aplus_20261025','LeggedManip_Lab_B0_jointScratch_20261027','LeggedManip_Lab_scheduledMix_safetyDiag_20261028']
out['instructions']={};out['worktrees']=[]
for q in [pathlib.Path('/home/alex/AGENTS.md'),root/'AGENTS.md']:
 if q.exists():out['instructions'][str(q)]=q.read_text()
for name in worktrees:
 w=root/name;s=w/'task_suite_experiment'
 if not w.exists():out['worktrees'].append(dict(path=str(w),exists=False));continue
 for q in [w/'AGENTS.md',s/'AGENTS.md']:
  if q.exists():out['instructions'][str(q)]=q.read_text()
 files={}
 for f in ['recipe_b0/spec.json','recipe_b0/config.py','recipe_b0/commands.py','recipe_v1/spec.json','recipe_v1/control.py','recipe_aplus/spec.json','recipe_aplus/startup.py','tools/run_B0.py']:
  q=s/f
  if q.exists():files[f]=dict(sha256=hashlib.sha256(q.read_bytes()).hexdigest(),bytes=q.stat().st_size)
 out['worktrees'].append(dict(path=str(s),commit=cmd(['git','-C',str(s),'rev-parse','HEAD']),branch=cmd(['git','-C',str(s),'branch','--show-current']),status=cmd(['git','-C',str(s),'status','--short']),files=files))
if root.exists():
 ds=sorted([p for p in root.iterdir() if p.is_dir() and (p.name.startswith('LeggedManip_Lab') or p.name.startswith('go2_piper_'))],key=lambda x:x.stat().st_mtime,reverse=True)
 out['latest_roots']=[dict(path=str(p),mtime=p.stat().st_mtime) for p in ds[:30]]
 paths=[root/'go2_piper_recipe_aplus_20261025/initialization_retry_v1/train_seed21/cp_1800.pt',root/'go2_piper_B0_20261026/train_seed21/cp_2000.pt',root/'go2_piper_B0_20261026/train_seed21/cp_2200.pt']
 out['candidate_files']=[]
 for q in paths:
  d=dict(path=str(q),exists=q.exists())
  if q.exists():
   h=hashlib.sha256()
   with q.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   d.update(bytes=q.stat().st_size,sha256=h.hexdigest())
  out['candidate_files'].append(d)
 ps=cmd(['ps','-eo','pid,etime,args'],limit=None);ps['stdout']='\n'.join(x for x in ps.get('stdout','').splitlines() if any(t in x for t in ['run_B0','run_recipe','OnPolicy','gpu_supervisor','scheduledMix','Isaac','train.py']))[:8000]
 out['relevant_processes']=ps
print(json.dumps(out))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='ubuntu-robot');p.add_argument('--out',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'evidence');a=p.parse_args()
 started=datetime.datetime.now(datetime.timezone.utc);out=dict(time_utc=started.isoformat(),host=a.host,read_only=True)
 try:
  r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',a.host,'python3 -u -c '+shlex.quote(REMOTE)],capture_output=True,text=True,timeout=150)
  out['returncode']=r.returncode
  if r.returncode==0:out['status']='CONNECTED';out['inventory']=json.loads(r.stdout)
  else:out.update(status='SSH_FAILED',stderr=r.stderr[-2000:])
 except subprocess.TimeoutExpired:out.update(status='SSH_FAILED',stderr='Whole-call150s deadline reached; remote status unknown')
 a.out.mkdir(parents=True,exist_ok=True);target=a.out/('HOST_'+started.strftime('%Y%m%dT%H%M%SZ')+'.json');target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(status=out['status'],evidence=str(target),returncode=out.get('returncode')),ensure_ascii=False))
 if out['status']=='CONNECTED':print(json.dumps(out['inventory'],ensure_ascii=False))
 return 0 if out['status']=='CONNECTED' else 2
if __name__=='__main__':raise SystemExit(main())
