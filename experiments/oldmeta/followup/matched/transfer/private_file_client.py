"""Resumable peer download of one approved file; configuration stays in RAM."""
import sys,json,urllib.request,hashlib,time,traceback,os
from pathlib import Path
config=json.load(sys.stdin);url=config['url'];SIZE=428464362;SHA=config['sha256']
root=Path('/home/alex/dog/go2_piper_nsgs_v02_matched_seed48_20261108');p=Path('/media/alex/Sharedata/go2_piper_nsgs_v02_matched_seed48_20261108/matched_sources.tar.gz.partial')
def request(endpoint,headers=None):
 return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request(url+'/'+endpoint,headers=headers or {}),timeout=40)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
try:
 for attempt in range(6):
  try:
   n=p.stat().st_size;assert n<=SIZE
   with request('prefix/'+str(n)) as r:assert json.load(r)['sha256']==digest(p)
   print('PREFIX_SHA_VERIFIED',n,flush=True)
   if n<SIZE:
    with request('file',{'Range':'bytes='+str(n)+'-'}) as r,p.open('ab',buffering=0) as f:
     assert r.status==206 and r.headers['X-File-SHA256']==SHA
     assert r.headers['Content-Range']==f'bytes {n}-{SIZE-1}/{SIZE}'
     last=n
     while n<SIZE:
      b=r.read(min(1<<20,SIZE-n));assert b;f.write(b);n+=len(b)
      if n-last>=8*1024*1024:print('RECEIVED',n,'/',SIZE,flush=True);last=n
   assert p.stat().st_size==SIZE and digest(p)==SHA
   final=p.with_suffix('');assert not final.exists();p.rename(final)
   receipt=dict(passed=True,bytes=SIZE,sha256=SHA,pid=os.getpid(),source='YangLab existing Tailscale peer',transport='One-file HTTP Range over encrypted Tailscale; peer IP allowlist; in-memory nonce; source auto-expiry',no_mac_runtime_dependency=True)
   (root/'PEER_TRANSFER_COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n')
   with request('done') as r:assert r.read()==b'OK'
   print('PEER_TRANSFER_COMPLETE',SIZE,SHA,flush=True);break
  except Exception as e:
   print('RETRY',attempt,type(e).__name__,flush=True)
   if attempt==5:raise
   time.sleep(5)
except BaseException:
 (root/'PEER_TRANSFER_FAULT.txt').write_text(traceback.format_exc());raise
