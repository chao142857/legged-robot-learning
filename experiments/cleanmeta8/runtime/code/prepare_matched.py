"""Verify all eight imports and exact old Teacher inheritance; no feedback replay."""
import argparse,json
import numpy as np
import torch
from common import ROOT,file_sha,write,digest,PROTOCOL
from cpu_hash import tree_hash
from teacher_v02 import Teacher,state_hash
from matched_sources import EXPECTED,OLD_META_SHA,source_progress,verify_required,wait_for_import

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-index',type=int);args=p.parse_args()
    sources=json.loads((ROOT/'sources.json').read_text())['sources']
    assert [(s['seed'],s['updates']) for s in sources]==EXPECTED
    if args.source_index is not None:
        s=sources[args.source_index];path=__import__('pathlib').Path(s['path'])
        wait_for_import(path,ROOT)
        assert file_sha(path)==s['sha256']
        receipt=json.loads((path.parent/'export_receipt.json').read_text())
        assert receipt['exported_prepared_sha256']==s['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False);i=saved['infos']
        hashes=verify_required(saved,receipt);source_progress(path,i)
        assert (i['student_seed'],i['completed_updates'])==(s['seed'],s['updates'])
        assert i['pool_sha256']=='86f39ce6b71d17d84c47681fe5e44d0fd7efac605ef366c3d8e91916b3eb56db'
        assert i['protocol_sha256']==digest(PROTOCOL)
        assert i['v02_preparation']['source_checkpoint_sha256']==receipt['original_source_checkpoint_sha256']
        assert i['adapter']['owner']['engine']['version']==-1
        write(path.parent/'import_acceptance.json',dict(passed=True,source_index=args.source_index,
            original_source_preserved=True,all_required_states_preserved=True,required_state_hashes=hashes,
            checkpoint_sha256=file_sha(path),progress_snapshot_only_past_feedback=True))
        print('MATCHED_SOURCE_IMPORT_PASS',s['seed'],s['updates'],flush=True);return
    old=ROOT/'teachers/old-meta.pt';assert file_sha(old)==OLD_META_SHA
    manifest=json.loads((ROOT/'prepared/manifest.json').read_text())
    t=Teacher(np.load(ROOT/'prepared/command_features.npy',allow_pickle=False),manifest['binding'])
    a=torch.load(old,map_location='cpu',weights_only=False)['teacher'];t.load(a)
    assert t.updates==8 and t.used_versions==set(str(v) for v in range(8))
    assert tree_hash(t.state())==tree_hash(a) and t.optimizer.state_dict()['state']
    write(ROOT/'teachers/inheritance.json',dict(passed=True,old_teacher_file_sha256=file_sha(old),
        old_teacher_network_sha256=state_hash(t.net.state_dict()),old_teacher_state_sha256=tree_hash(a),
        old_updates=8,new_versions=list(range(8,16)),optimizer_state_retained=True,private_rng_retained=True,
        source_grid=EXPECTED,old_outcomes_not_reused_for_gradient=True))
    print('MATCHED_INHERITANCE_PASS: old8 ->16, same original seed43/44 stage grid',flush=True)
if __name__=='__main__':main()
