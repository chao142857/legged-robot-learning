"""Prepare six bounded physics-only candidate jobs; never launches them.

Flat profiles retain the exact IK-audited EE/base fields and omit irrelevant
terrain fields. The advanced T2 4m route remains explicitly rejected, unchanged.
"""
from publication_paths import research_path
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')


def prepare(a):
    root=Path(__file__).resolve().parents[1]
    evidence=Path(a.audit).resolve()
    out=Path(a.out).resolve()
    if out.exists():
        raise ValueError('Refusing to reuse existing preparation/output directory')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.tag):
        raise ValueError('Tag must contain only letters, numbers, underscore or hyphen')
    checkpoint=Path(a.checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError('Native checkpoint does not exist')
    source={}
    for level in ['medium','hard']:
        s_path=evidence/level/'summary.json'
        p_path=evidence/level/'profile_overrides.json'
        summary=json.loads(s_path.read_text())
        params=json.loads(p_path.read_text())
        if summary['total']!=270 or summary['exact_count']!=270 or not summary['all_sampled_poses_pass']:
            raise ValueError(level+': all 270 IK samples must pass before this physics-only plan')
        if sha(p_path)!=summary['profile_overrides']['sha256'] or params!=summary['parameters']:
            raise ValueError(level+': exact audited profile content/hash mismatch')
        if summary.get('training_admission') is not False or summary.get('range_admission') is not False:
            raise ValueError('Expected candidate-only evidence with no training/range admission')
        source[level]=dict(summary=summary,params=params,summary_sha256=sha(s_path),profile_sha256=sha(p_path))
    if not source['medium']['summary']['geometry_precheck']['route_supported']:
        raise ValueError('Medium T2 route was rejected by precheck')
    if source['hard']['params']['route_distance']<=3.5 or source['hard']['summary']['geometry_precheck']['route_supported']:
        raise ValueError('Expected advanced T2 route>3.5m rejection to remain intact')
    out.mkdir(parents=True,exist_ok=False)
    for sub in ['profiles','logs','runs']:
        (out/sub).mkdir()
    profiles={}
    for level in ['medium','hard']:
        params=source[level]['params']
        flat={k:v for k,v in params.items() if k.startswith('ee_') or k.startswith('base_')}
        # Exact matching to the full audited candidate is mandatory; this does
        # not invent a reduced-distance hard T2 profile.
        if not all(flat[k]==params[k] for k in flat):
            raise AssertionError('Flat override changed an audited field')
        path=out/'profiles'/f'{level}_flat_ee_base.json'
        write_json(path,flat)
        profiles[level+'_flat']=dict(path=str(path),sha256=sha(path),fields=list(flat),
            source_profile_sha256=source[level]['profile_sha256'],
            excluded_inactive_terrain_and_other_fields={k:v for k,v in params.items() if k not in flat})
    medium_t2=out/'profiles'/'medium_terrain.json'
    write_json(medium_t2,source['medium']['params'])
    profiles['medium_terrain']=dict(path=str(medium_t2),sha256=sha(medium_t2),fields=list(source['medium']['params']),
        source_profile_sha256=source['medium']['profile_sha256'])
    jobs=[]
    for family,terrain,level in [('T0','plane','medium'),('T0','plane','hard'),('T1','plane','medium'),('T1','plane','hard'),
                                  ('T2','stairs_up','medium'),('T2','stairs_down','medium')]:
        profile=profiles[level+('_terrain' if family=='T2' else '_flat')]
        name=f'smoke_{family.lower()}_{terrain}_{level}_smooth_s{a.seed}_{a.tag}'
        jobs.append(dict(name=name,family=family,terrain=terrain,difficulty=level,ee_mode='smooth',
            mode='physics',num_envs=8,steps=500,simulated_seconds_cap=10.,seed=a.seed,
            profile=profile['path'],profile_sha256=profile['sha256'],
            output=str(out/'runs'/name),log=str(out/'logs'/(name+'.log'))))
    plan=dict(schema_version=1,prepared_only=True,suite_root=str(root),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        source_audit_directory=str(evidence),source_summaries={k:v['summary_sha256'] for k,v in source.items()},
        generator_sha256=sha(__file__),runner_sha256=sha(root/'tools/run_candidate_smoke.py'),
        profiles=profiles,jobs=jobs,per_job_timeout_seconds=900,
        rejected_jobs=[dict(family='T2',difficulty='hard',requested_route_distance_m=source['hard']['params']['route_distance'],
            current_limit_m=3.5,reason='Unchanged advanced route exceeds current tile support; no job generated')],
        admission=False,training_jobs=0,SGS_jobs=0,
        scope='Six censored 10-second physics smokes only; no full-episode success, interval admission or PPO learning claim')
    write_json(out/'plan.json',plan)
    print(json.dumps(dict(plan=str(out/'plan.json'),jobs=6,launched=False,training_jobs=0,hard_T2='rejected_route_4m')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True,help='Completed audit_candidate_levels output directory')
    p.add_argument('--out',required=True,help='New directory for prepared profiles, plan and future runs')
    p.add_argument('--checkpoint',default=research_path('go2_piper_upstream_wbc_20261023/formal_seed21/cp_3000.pt'))
    p.add_argument('--seed',type=int,default=7101)
    p.add_argument('--tag',default='candidate_v1')
    prepare(p.parse_args())
