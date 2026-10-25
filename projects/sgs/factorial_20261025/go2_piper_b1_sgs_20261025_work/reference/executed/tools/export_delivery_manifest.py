"""Index final evidence on its execution host; never copy raw assets or models."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    report=json.loads((a.root/'analysis_smoke_v1/report.json').read_text())
    selected=[r for r in report['runs'] if
        r['run'].startswith('frozen_') and r['run'].endswith('_v3') or
        r['run'].startswith(('post_','train_','candidate_smoke_v1/','diagnostic_first_update_'))]
    entries=[]
    for row in selected:
        run=a.root/row['run']
        producer=json.loads((run/'manifest.json').read_text())
        files={}
        for name in ['manifest.json','protocol.json','summary.json','transfer.json','candidate_identity.json',
                     'geometry.json','geometry_samples.npz','trajectories.npz','episodes.csv',
                     'updates.jsonl','cp_0000.pt','cp_final.pt','first_update_probe.jsonl','probe_manifest.json']:
            path=run/name
            if not path.is_file(): continue
            digest=sha(path)
            if name in producer and digest!=producer[name]['sha256']:
                raise ValueError('Evidence differs from producer manifest: '+str(path))
            files[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':digest,
                         'producer_manifest_match':name in producer}
        log=a.root/'logs'/(run.name+'.log')
        if row['run'].startswith('candidate_smoke_v1/'):
            log=a.root/'candidate_smoke_v1/logs'/(run.name+'.log')
        if log.is_file():
            files['process_log']={'path':str(log),'bytes':log.stat().st_size,'sha256':sha(log)}
        entries.append(dict(run=row['run'],status=row['status'],checkpoint_role=
            'failed_recipe_diagnostic_only' if row['mode']=='train' else 'frozen_evaluation',files=files))
    result=dict(execution_host='ubuntu-robot',root=str(a.root),raw_files_copied=False,
                suite_source=str(Path(__file__).resolve().parents[1]),exporter_sha256=sha(Path(__file__)),
                analysis_report_sha256=sha(a.root/'analysis_smoke_v1/report.json'),runs=entries,
                original_cp3000_sha256='600a2dab9caf18289bb6ec558f86c4e6e86c352b621f9a6d83ff73f209cc08b6',
                recipe_pass=False,vanilla_learning_demonstrated=False,short_80_started=False,
                SGS_started=False,long_3000_started=False,
                note='v1/v2 preliminary runs remain on host; final physical evidence uses v3. Checkpoints from failed smoke are retained, not accepted deployment models.')
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(a.out),runs=len(entries),bytes=a.out.stat().st_size,sha256=sha(a.out))))

if __name__=='__main__':main()
