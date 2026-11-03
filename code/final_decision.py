import pathlib,json,time
import numpy as np
from common import ROOT,write,file_sha,digest,PROTOCOL

A=ROOT/'artifacts';f=json.loads((A/'frozen/frozen_summary.json').read_text());b=json.loads((A/'feedback/feedback_summary.json').read_text());g=json.loads((A/'groups/geometry_summary.json').read_text())
t=json.loads((A/'tests_result.json').read_text());p=json.loads((A/'production_checks.json').read_text());manifest=json.loads((A/'groups/mapping_manifest.json').read_text())
assert f['passed'] and b['passed'] and t['passed'] and p['passed']
assert manifest['pool_sha256']==file_sha(ROOT/'pools/wbc_train.npz')
for path in [pathlib.Path('/e1reference/exports/cohort10397_20261103/selection.json'),pathlib.Path('/reference/analysis/metric-diagnostics-20261102/cohort_ids.json')]:
    s=json.loads(path.read_text());assert s['pool_sha256']==manifest['pool_sha256'],str(path)
for root,cps in [(pathlib.Path('/e1reference/exports/cohort10397_20261103'),(4000,8000,10000)),(pathlib.Path('/reference/analysis/metric-diagnostics-20261102'),(8000,10000))]:
    for cp in cps:
        for mode in ('mean','stochastic'):
            meta=json.loads((root/(str(cp)+'-'+mode)/'result.json').read_text())
            cproot=pathlib.Path('/e1reference/runs/formal-wbc-sgs-e1' if 'e1reference' in str(root) else '/reference/runs/formal-wbc-sgs')
            assert meta['checkpoint_sha256']==file_sha(cproot/'checkpoints'/('update_'+str(cp).zfill(5)+'.pt'))
cmd=next(x for x in g['rows'] if x['mapping']=='command')['mean_feature_distance'];geometry=all(cmd<x['mean_feature_distance'] for x in g['rows'] if x['mapping']!='command')
conditions=dict(engineering=t['passed'] and p['passed'],command_geometry=geometry,frozen_control_consistency=f['frozen_GO_condition_passed'],chronological_estimator=b['estimator_GO_condition_passed'])
go=all(conditions.values())
decision=dict(decision='GO' if go else 'NO-GO',formal_PPO_authorized_conditionally=True,formal_PPO_started=False,conditions=conditions,
    recommendation='Run matched fresh RandomGroup and CommandCluster WBC10k automatically.' if go else 'Defer formal PPO. Complete the fixed-policy cluster-complete supplement before reconsidering GO.',
    facts=['All32768 training sequences retained; all4 mappings4096x8; independent1024 evaluation pool excluded.',
           'Exact native timing matched two16-env episodes, including one effective velocity-slot2 tail step after autoreset.',
           'Grouped last100 histories are denser/fresher. Command grouping improves logged Uniform prequential Brier versus all3 Random maps in both late blocks.',
           'Existing fixed-policy panels do not demonstrate robust lower within-group position/time dispersion under the predeclared criteria. No full8-member group is observed in those panels.'],
    inference='Command grouping is a plausible estimator improvement, but fixed-policy group homogeneity and new PPO learning benefit remain unverified.',
    missing=['Cluster-complete frozen measurements for current mappings; sufficiently precise per-sequence strict rates; current15k frozen consistency.'],
    minimum_supplement_file=str(A/'minimal_frozen_supplement_proposal.json'),protocol_sha256=file_sha(ROOT/'protocol/E2_protocol_frozen.json'),
    strict_protocol_sha256=digest(PROTOCOL),mapping_manifest_sha256=file_sha(A/'groups/mapping_manifest.json'),source_binding_checks_passed=True,
    no_threshold_or_reward_changes=True,unix=time.time())
write(A/'E2_GO_NO_GO.json',decision)
write(ROOT/'E2_STATUS.json',dict(phase='GO_pending_formal_launch' if go else 'NO_GO_report_ready',formal_PPO_started=False,decision=decision['decision'],unix=time.time()))
print('E2_DECISION '+json.dumps(decision),flush=True)
