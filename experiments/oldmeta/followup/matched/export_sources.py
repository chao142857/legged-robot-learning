"""Read-only export of required full-resume states; original checkpoints retained."""
from pathlib import Path
import copy,json,hashlib,tarfile
import torch
from cpu_hash import tree_hash
source=Path('/source');out=Path('/export');inputs=out/'bundle';inputs.mkdir(exist_ok=False)
rows=json.loads((source/'sources.json').read_text())['sources'];receipts=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
fields=('actor_state_dict','critic_state_dict','optimizer_state_dict')
for index,s in enumerate(rows):
 label=f"seed{s['seed']}-cp{s['updates']:05d}";p=source/'starts'/label/'prepared.pt';original_sha=sha(p)
 a=torch.load(p,map_location='cpu',weights_only=False);i=a['infos']
 keep_hash={k:tree_hash(a[k]) for k in fields}
 keep_hash.update({k:tree_hash(i[k]) for k in ('adapter','environment','rng','v02_context')})
 removed={k:tree_hash(i.pop(k)) for k in ('source_adapter_v02','source_sampler_v02') if k in i}
 assert all(tree_hash(a[k])==v for k,v in keep_hash.items() if k in fields)
 assert all(tree_hash(i[k])==v for k,v in keep_hash.items() if k not in fields)
 dest=inputs/'starts'/label;dest.mkdir(parents=True)
 torch.save(a,dest/'prepared.pt')
 point=Path('/old-experience/meta-eight-starts')/f'point_{index:02d}'/'decision.pt'
 d=torch.load(point,map_location='cpu',weights_only=False)
 assert d['source_original']['sha256']==s['sha256']
 row=d['source_progress_row'];assert row['completed_updates']==s['updates'] and row['native_feedback_episodes']==i['adapter']['owner']['accepted']
 (dest/'progress_row.json').write_text(json.dumps(row,indent=2)+'\n')
 (dest/'source.json').write_text(json.dumps(s,indent=2)+'\n')
 receipt=dict(index=index,student_seed=s['seed'],updates=s['updates'],original_prepared_path=str(p).replace('/source',str(Path('/data/go2_piper_nsgs_v02_20261106')),1),original_prepared_sha256=original_sha,original_source_checkpoint=s['path'],original_source_checkpoint_sha256=s['sha256'],exported_prepared_sha256=sha(dest/'prepared.pt'),exported_bytes=(dest/'prepared.pt').stat().st_size,unchanged_required_state_hashes=keep_hash,removed_unused_duplicate_metadata=removed,progress_row_source_decision_sha256=sha(point),progress_row_sha256=sha(dest/'progress_row.json'))
 (dest/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt)
 assert sha(p)==original_sha
 print('EXPORTED',label,receipt['exported_bytes'],flush=True)
 del a,i,d
(inputs/'source_export_manifest.json').write_text(json.dumps(dict(passed=True,sources=receipts,scope='All actor/critic/optimizer/adapter/environment/RNG/context states retained bitwise. Only two unused duplicate provenance fields omitted; original full files remain authoritative.'),indent=2)+'\n')
archive=out/'matched_sources.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as t:
 for p in sorted(inputs.rglob('*')):
  if p.is_file():t.add(p,arcname=str(p.relative_to(inputs)))
summary=dict(passed=True,bundle_bytes=archive.stat().st_size,bundle_sha256=sha(archive),source_count=8,uncompressed_export_bytes=sum(x['exported_bytes'] for x in receipts),sources=[{k:s[k] for k in ('student_seed','updates','original_prepared_sha256','exported_prepared_sha256','exported_bytes')} for s in receipts])
(out/'bundle_receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
print('EXPORT_COMPLETE',json.dumps(summary),flush=True)
