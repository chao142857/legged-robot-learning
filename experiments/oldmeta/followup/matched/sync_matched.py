from pathlib import Path
import tarfile,io,base64,subprocess,json,hashlib
w=Path(__file__).resolve().parent
root='/home/alex/dog/go2_piper_nsgs_v02_matched_seed48_20261108'
original=json.loads((w.parent/'legged-nsgs-v02-publish/projects/sgs/nsgs_v02_20261106/sources.json').read_text())['sources']
receipts=json.loads((w/'source_export_manifest.json').read_text())['sources'];sources=[]
assert len(original)==len(receipts)==8
for s,r in zip(original,receipts):
 assert s['seed']==r['student_seed'] and s['updates']==r['updates'] and s['sha256']==r['original_source_checkpoint_sha256']
 sources.append(dict(seed=s['seed'],updates=s['updates'],path=root+f"/starts/seed{s['seed']}-cp{s['updates']:05d}/prepared.pt",sha256=r['exported_prepared_sha256'],bytes=r['exported_bytes'],family='matched-oldmeta-prepared',original_source=s,original_prepared_sha256=r['original_prepared_sha256'],progress_provenance_decision_sha256=r['progress_row_source_decision_sha256']))
(w/'sources.json').write_text(json.dumps(dict(sources=sources),indent=2)+'\n')
files=[p for d in ('code','tests') for p in (w/d).iterdir() if p.suffix in ('.py','.sh')]+[w/n for n in ('controller.py','host.py','tmux.sh','PROTOCOL.json','import_stream.py')]
(w/'source_manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n');files += [w/n for n in ('source_manifest.json','sources.json','source_export_manifest.json')]
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(w)))
script='''from pathlib import Path
import tarfile,io,base64
r=Path(ROOT)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as t:
 assert all(p.isfile() and '..' not in Path(p.name).parts and not p.name.startswith('/') for p in t.getmembers())
 t.extractall(r)
print('MATCHED_CODE_SYNCED')
'''.replace('ROOT',repr(root)).replace('PAYLOAD',repr(base64.b64encode(b.getvalue()).decode()))
r=subprocess.run(['ssh','ubuntu-robot','python3 -'],input=script.encode(),capture_output=True);print(r.returncode,r.stdout.decode(),r.stderr.decode()[-2000:]);assert r.returncode==0
