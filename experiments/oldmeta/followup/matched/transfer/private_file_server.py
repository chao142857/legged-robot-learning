"""One consented file, private Tailscale peer only, nonce in RAM, automatic expiry."""
import http.server,hashlib,secrets,time,threading,json,os,sys
from pathlib import Path
P=Path('/data/go2_piper_nsgs_v02_matched_export_20261108/matched_sources.tar.gz')
SIZE=428464362;SHA='9fd501b688c052cbf54c71608dc5aa9ee65e09099387a282c768c673c0e8d267'
assert P.stat().st_size==SIZE
NONCE=secrets.token_urlsafe(36);started=time.time()
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  if self.client_address[0]!='100.99.32.57' or not self.path.startswith('/'+NONCE+'/'):
   self.send_error(403);return
  route=self.path[len(NONCE)+2:]
  if route.startswith('prefix/'):
   n=int(route.split('/')[-1]);assert 0<=n<=SIZE;h=hashlib.sha256()
   with P.open('rb') as f:
    while n:
     b=f.read(min(n,1<<20));assert b;h.update(b);n-=len(b)
   data=json.dumps(dict(sha256=h.hexdigest())).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
  if route=='done':
   self.send_response(200);self.send_header('Content-Length','2');self.end_headers();self.wfile.write(b'OK');threading.Thread(target=self.server.shutdown,daemon=True).start();return
  if route!='file':self.send_error(404);return
  value=self.headers.get('Range','bytes=0-');assert value.startswith('bytes=')
  lo,hi=value[6:].split('-');lo=int(lo);hi=int(hi) if hi else SIZE-1;assert 0<=lo<=hi<SIZE
  self.send_response(206);self.send_header('Content-Length',str(hi-lo+1));self.send_header('Content-Range',f'bytes {lo}-{hi}/{SIZE}');self.send_header('X-File-SHA256',SHA);self.end_headers()
  with P.open('rb') as f:
   f.seek(lo);left=hi-lo+1
   try:
    while left:
     b=f.read(min(1<<20,left));self.wfile.write(b);left-=len(b)
   except (BrokenPipeError,ConnectionResetError):pass
class Server(http.server.ThreadingHTTPServer):
 def handle_error(self,*a):pass
server=Server(('100.69.48.93',0),Handler)
timer=threading.Timer(10800,server.shutdown);timer.daemon=True;timer.start()
print(json.dumps(dict(port=server.server_address[1],nonce=NONCE,pid=os.getpid(),expires_unix=started+10800)),flush=True)
server.serve_forever();timer.cancel();server.server_close()
