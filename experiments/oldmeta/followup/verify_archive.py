"""Check lightweight package, reconstruction overlays, and all70 fixed-test points."""
import ast,hashlib,json
from pathlib import Path

def main():
 root=Path(__file__).resolve().parent;base=root.parent/'nsgs_v02_20261106'
 package=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
 for name,e in package['files'].items():
  p=root/name;b=p.read_bytes();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],name
  if p.suffix=='.py':ast.parse(b,filename=name)
  if p.suffix=='.json':json.loads(b)
 for label in ('matched','executed_yanglab_seed48'):
  overlay=root/label;manifest=json.loads((overlay/'source_manifest.json').read_text())
  for name,sha in manifest['files'].items():
   p=overlay/name
   if not p.exists():p=base/name
   assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,(label,name)
 a=json.loads((root/'evidence/completed_curves.json').read_text());assert len(a['series'])==7
 for s in a['series']:
  assert [p['update'] for p in s['points']]==list(range(1000,10001,1000))
  assert all(0<=p['joint_success_rate']<=1 and p['checkpoint_sha256'] for p in s['points'])
 for p in (root/'evidence/completed_runs').glob('*_result.json'):
  r=json.loads(p.read_text());assert r['passed'] and r['completed_updates']==10000 and r['environment_transitions']==983040000
 assert len(list((root/'evidence/completed_runs').glob('*_result.json')))==7
 sources=json.loads((root/'matched/sources.json').read_text())['sources']
 assert [(s['seed'],s['updates']) for s in sources]==[(seed,u) for u in (1000,3000,6000,9000) for seed in (43,44)]
 print('PASS:',len(package['files']),'files; reconstructed source hashes;70 test points;7 completed10k runs;8 matched sources')
if __name__=='__main__':main()
