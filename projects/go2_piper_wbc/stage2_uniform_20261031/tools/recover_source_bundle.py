"""Reassemble exact original task-suite bytes from an existing authorized cache.
No download, simulator import, policy load, source modification or GPU execution.
"""
import argparse,hashlib,json,pathlib,shutil

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def recover(cache,launch,out,extra=None):
 if out.exists():raise FileExistsError('Fresh output required; historical bundles are immutable')
 manifest=json.loads(launch.read_text());checked=[]
 for name,expected in manifest['source_hashes'].items():
  rel=pathlib.PurePosixPath(name)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid relative source name')
  f=cache/name
  if not f.is_file() and extra is not None:f=extra/name
  if not f.is_file():raise FileNotFoundError('Missing original source: '+name)
  if digest(f)!=expected:raise ValueError('Original launch SHA mismatch: '+name)
  checked.append((name,f,expected))
 out.mkdir(parents=True)
 for name,f,_ in checked:
  target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
 proof=dict(source_commit=manifest['source_commit'],files=[dict(path=name,sha256=sha) for name,_,sha in checked],all_original_source_bytes_match=True,weights_or_assets_included=False,simulator_verified=False)
 (out/'RECOVERY_MANIFEST.json').write_text(json.dumps(proof,indent=2)+'\n')
 return proof
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--launch',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--extra',type=pathlib.Path);a=p.parse_args()
 r=recover(a.cache,a.launch,a.out,a.extra);print(json.dumps(dict(source_commit=r['source_commit'],files=len(r['files']),all_original_source_bytes_match=True,weights_or_assets_included=False)))
