import json, pathlib, shutil, hashlib
from common import ROOT, PROTOCOL, PARAMETERS, digest, file_sha, write
from e1_sampling import SCHEDULE

reference=pathlib.Path('/reference')
manifest=json.loads((reference/'source_manifest.json').read_text())
for n,h in manifest['files'].items(): assert file_sha(reference/'code'/n)==h,n
unchanged=['common.py','vendor_sampler.py','sampling.py','commands.py','metrics.py','adapter.py','python.sh','tmux.sh']
for n in unchanged: assert file_sha(ROOT/'code'/n)==file_sha(reference/'code'/n),n
assert file_sha(ROOT/'code/evaluation.py')==file_sha(reference/'code/job.py')
for n in ['flat_train.npz','flat_evaluation.npz','wbc_train.npz','wbc_evaluation.npz','manifest.json']:
    src=reference/'pools'/n; dst=ROOT/'pools'/n
    assert not dst.exists(); shutil.copy2(src,dst); assert file_sha(src)==file_sha(dst)
src=reference/'protocol_frozen.json'; shutil.copy2(src,ROOT/'protocol_frozen.json')
protocol=json.loads(src.read_text()); assert protocol['definition']==PROTOCOL and protocol['definition_sha256']==digest(PROTOCOL)
baseline={}
for k in ('flat','wbc'):
    baseline[k]={}
    for mode in ('sgs','uniform'):
        src=reference/'evaluations'/('formal-'+k+'-'+mode)/'10000'
        result=json.loads((src/'result.json').read_text())
        rows=json.loads((src/'episode_results.json').read_text())
        n=sum(int(x['values'][6]) for x in rows)
        assert result['episodes']==1024 and result['joint_success_rate']==n/1024
        assert n==dict(flat=dict(sgs=226,uniform=326),wbc=dict(sgs=363,uniform=374))[k][mode]
        baseline[k][mode]=dict(successes=n,episodes=1024,result_sha256=file_sha(src/'result.json'),episodes_sha256=file_sha(src/'episode_results.json'))
    baseline[k]['initialization']=json.loads((reference/'runs'/('formal-'+k+'-sgs')/'initialization.json').read_text())
native=pathlib.Path('/native'); upstream=json.loads((native/'audit/upstream_tree.json').read_text()); checked=0
for x in upstream['tree']:
    if x['type']!='blob' or not (x['path'].endswith('.py') and x['path'].startswith(('source/','scripts/rsl_rl/')) or '/assets/go2_piper/' in x['path']): continue
    raw=(native/'LeggedManip_Lab'/x['path']).read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==x['sha']; checked+=1
proof=dict(passed=True,reference_source_unchanged=True,reference_source_digest=manifest['integration_sha256'],unchanged_files=unchanged+['evaluation.py (original job.py)'],
           native_commit=upstream['sha'],native_files_checked=checked,official_protocol_sha256=digest(PROTOCOL),sgs_parameters=PARAMETERS,
           schedule=SCHEDULE,baselines=baseline,pools=json.loads((ROOT/'pools/manifest.json').read_text()),
           no_baseline_overwrite=True,no_prior_policy_or_history_load=True)
write(ROOT/'audit/preflight.json',proof); print('E1_PREFLIGHT_PASS '+json.dumps(proof),flush=True)
