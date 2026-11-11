"""Fixed small-file transfer over existing encrypted Tailnet, exact peer only."""
import http.server,json,time,threading,subprocess,hmac,urllib.request,urllib.error,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'peer_private.json').read_text())
if sys.argv[1]=='serve':
 class Handler(http.server.BaseHTTPRequestHandler):
  def log_message(self,*a):pass
  def do_GET(self):
   if self.client_address[0]!=cfg['peer_ip'] or not hmac.compare_digest(self.headers.get('X-Peer-Key',''),cfg['key']):self.send_error(403);return
   ready=ROOT/'delivery/receipt.json'
   if self.path=='/ready':
    payload=ready.read_bytes() if ready.exists() else b'{"ready":false}'
   elif self.path=='/teacher' and ready.exists():payload=(ROOT/'delivery/CleanMeta8.delta.pt').read_bytes()
   else:self.send_error(404);return
   assert len(payload)<=262144
   self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
 server=http.server.ThreadingHTTPServer((cfg['bind_ip'],cfg['port']),Handler)
 def prepare():
  deadline=time.monotonic()+48*3600
  while time.monotonic()<deadline:
   if (ROOT/'CLEANMETA8_ACCEPTANCE.json').exists() and not (ROOT/'delivery/receipt.json').exists():
    subprocess.run(['bash',str(ROOT/'code/python.sh'),str(ROOT/'code/compact_teacher.py'),'export'],check=True)
   time.sleep(20)
  server.shutdown()
 threading.Thread(target=prepare,daemon=True).start();server.serve_forever()
else:
 # Ignore ambient HTTP proxy for this one verified private Tailnet endpoint.
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 def get(path):
  q=urllib.request.Request(f"http://{cfg['source_ip']}:{cfg['port']}{path}",headers={'X-Peer-Key':cfg['key']})
  with opener.open(q,timeout=15) as f:
   data=f.read(262145);assert len(data)<=262144;return data
 deadline=time.monotonic()+48*3600
 while time.monotonic()<deadline:
  try:
   r=json.loads(get('/ready'))
   if r.get('passed'):
    blob=get('/teacher');assert len(blob)==r['bytes'] and hashlib.sha256(blob).hexdigest()==r['sha256']
    out=ROOT/'delivery';out.mkdir(exist_ok=True);assert not (out/'CleanMeta8.delta.pt').exists();(out/'CleanMeta8.delta.pt').write_bytes(blob);(out/'receipt.json').write_text(json.dumps(r,indent=2));print('TEACHER_RECEIVED',len(blob),flush=True);break
  except (OSError,urllib.error.URLError,TimeoutError):
   if sys.argv[1]=='probe':raise
  if sys.argv[1]=='probe':print('PRIVATE_PEER_PROBE_OK',flush=True);break
  time.sleep(30)
 else:raise TimeoutError('CleanMeta8 not received within 48h; no training started')
