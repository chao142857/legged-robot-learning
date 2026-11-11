"""Export/import <256 KiB Teacher delta, reusing the exact cached encoder."""
import argparse,copy,json,hashlib
from pathlib import Path
import torch,numpy as np
from common import ROOT,write,file_sha
from cpu_hash import tree_hash
from teacher_v02 import Teacher

def export():
 acceptance=json.loads((ROOT/'CLEANMETA8_ACCEPTANCE.json').read_text());assert acceptance['passed']
 source=ROOT/'teachers/CleanMeta8.pt';assert file_sha(source)==acceptance['checkpoint_sha256']
 t=torch.load(source,map_location='cpu',weights_only=False)['teacher'];state=copy.deepcopy(t);encoder=state.pop('encoder')
 out=ROOT/'delivery';out.mkdir(exist_ok=True);path=out/'CleanMeta8.delta.pt'
 assert not path.exists()
 torch.save(dict(state_without_encoder=state,encoder_sha256=tree_hash(encoder),teacher_full_state_sha256=tree_hash(t),source_checkpoint_sha256=file_sha(source),acceptance=acceptance),path)
 assert path.stat().st_size<262144
 write(out/'receipt.json',dict(passed=True,sha256=file_sha(path),bytes=path.stat().st_size,full_state_sha256=tree_hash(t),source_checkpoint_sha256=file_sha(source)))
def install():
 p=ROOT/'delivery/CleanMeta8.delta.pt';r=json.loads((ROOT/'delivery/receipt.json').read_text());assert file_sha(p)==r['sha256']
 d=torch.load(p,map_location='cpu',weights_only=False);fresh=torch.load(ROOT/'teachers/Fresh.pt',map_location='cpu',weights_only=False)['teacher']
 assert tree_hash(fresh['encoder'])==d['encoder_sha256'];s=d['state_without_encoder'];s['encoder']=copy.deepcopy(fresh['encoder'])
 assert tree_hash(s)==d['teacher_full_state_sha256']==r['full_state_sha256']
 t=Teacher(np.load(ROOT/'prepared/command_features.npy'),fresh['binding']);t.load(s);assert tree_hash(t.state())==r['full_state_sha256']
 assert t.updates==8 and d['acceptance']['passed']
 out=ROOT/'teachers/CleanMeta8.pt';assert not out.exists();torch.save(dict(teacher=t.state(),cleanmeta8_source=d['source_checkpoint_sha256']),out)
 write(ROOT/'IMPORT_ACCEPTANCE.json',dict(passed=True,teacher_full_state_sha256=tree_hash(t.state()),checkpoint_sha256=file_sha(out),original_source_checkpoint_sha256=d['source_checkpoint_sha256'],encoder_reused_exactly=True,transferred_bytes=p.stat().st_size))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['export','install']);globals()[a.parse_args().action]()
