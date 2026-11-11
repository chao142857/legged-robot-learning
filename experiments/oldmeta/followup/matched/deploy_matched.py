"""Create an isolated queue from cached assets. Full state import is separate and consented."""
from pathlib import Path
import tarfile,io,json,hashlib,shutil,sys,base64
r=Path('/home/alex/dog/go2_piper_nsgs_v02_matched_seed48_20261108')
data=Path('/media/alex/Sharedata/go2_piper_nsgs_v02_matched_seed48_20261108')
old=Path('/home/alex/dog/go2_piper_nsgs_v02_seed48_20261107')
r.mkdir(exist_ok=False);data.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)),mode='r:gz') as t:
 assert all(p.isfile() and '..' not in Path(p.name).parts and not p.name.startswith('/') for p in t.getmembers())
 t.extractall(r)
for name in ('assets','pools','deps'):(r/name).symlink_to(old/name,target_is_directory=True)
for name in ('runs','experience','starts'):
 (data/name).mkdir();(r/name).symlink_to(data/name,target_is_directory=True)
for name in ('teachers','prepared','parent'):(r/name).mkdir()
for name in ('command_features.npy','feature_normalization.npz','wbc_development.npz'):(r/'prepared'/name).symlink_to(old/'prepared'/name)
shutil.copy2(old/'prepared/manifest.json',r/'prepared/manifest.json')
shutil.copy2(old/'parent/reference_configuration.json',r/'parent/reference_configuration.json')
shutil.copy2(old/'teachers/old-meta.pt',r/'teachers/old-meta.pt')
shutil.copy2(old/'runs/student_seed48_initialization.json',r/'runs/student_seed48_initialization.json')
for kind in ('development','test'):shutil.copy2(old/f'evaluation_identity_{kind}.json',r/f'evaluation_identity_{kind}.json')
(r/'provenance.json').write_text(json.dumps(dict(previous_base=str(old),original_sources_yanglab='/data/go2_piper_nsgs_v02_20261106',source_seeds=[43,44],source_updates=[1000,3000,6000,9000],source_order='phase then seed',student48_previously_observed=True,hardware='RTX5060 Laptop; old48 teacher arms RTX3060',old_results_preserved=True,new_teacher_experience=str(r/'experience'),data_partition=str(data),teacher_update_versions=list(range(8,16))),indent=2)+'\n')
print('SETUP_ONLY',r,'DATA',data,flush=True)
