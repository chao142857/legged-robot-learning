"""Assemble explicit runtime sources from pinned Stage3 code plus small patches.

CPU only. Does not connect, initialize a study, start Docker or consume a grant.
"""
import argparse,hashlib,json,pathlib,subprocess,tempfile

PROJECT=pathlib.Path(__file__).resolve().parents[1]
def digest(raw):return hashlib.sha256(raw).hexdigest()
def assemble(base):
    base=pathlib.Path(base);reuse=json.loads((PROJECT/'SOURCE_REUSE.json').read_text());tools={}
    for f in (PROJECT/'tools').glob('*.py'):
        if f.name=='build_payload.py' or f.name.startswith('test_'):continue
        source=f.read_text();compile(source,f.name,'exec')
        tools[f.name]=dict(source=source,sha256=digest(source.encode()))
    for item in reuse['transforms']:
        source=(base/item['base_file']).read_bytes();assert digest(source)==item['base_sha256'],item['base_file']+' template source mismatch'
        patch=PROJECT/item['patch'];assert digest(patch.read_bytes())==item['patch_sha256']
        with tempfile.TemporaryDirectory(prefix='stage3b-explicit-source-') as td:
            work=pathlib.Path(td);f=work/item['base_file'];f.write_bytes(source)
            subprocess.run(['patch','-s','-p0','-i',str(patch.resolve())],cwd=work,stdin=subprocess.DEVNULL,
                capture_output=True,text=True,check=True,timeout=10)
            updated=f.read_bytes();assert digest(updated)==item['target_sha256'],item['target_file']+' runtime source mismatch'
        text=updated.decode();compile(text,item['target_file'],'exec');tools[item['target_file']]=dict(source=text,sha256=digest(updated))
    return dict(user_approved=True,approved_GPU_seconds_SUM=28800,final_reserve_seconds=3600,actor_update_cap=None,transition_cap=None,
        grant_id='stage3b-original-uniform-dynamicX-safety-20261101-28800',public_base_commit=reuse['base_commit'],
        approval_text='User Stage3b authorization: independent28800 GPU-container SUM seconds; final3600 reserved; counts accounting only; close Stage3 first; no competing GPU; simulation only.',
        formal_seeds=json.loads((PROJECT/'FORMAL_SEEDS_PREREGISTRATION.json').read_text()),tools=tools)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-tools',type=pathlib.Path,default=PROJECT.parent/'stage3_dynamic_x_20261031/tools');p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();d=assemble(a.base_tools);a.out.write_text(json.dumps(d,indent=2)+'\n');print('Explicit source payload:',len(d['tools']),'tools; no GPU launch')
