"""Dependency-free parameter sanity audit; never grants physical admission.

Reports timing lower bounds and exact active-profile containment. It deliberately
cannot establish collision, IK, physics, or PPO learning from static numbers.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    profile_path = root / 'task_suite' / 'profiles.json'
    profiles = json.loads(profile_path.read_text())
    easy = profiles['easy']
    checks = []
    def record(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))
    record('active_profile_remains_candidate', easy.get('status') == 'unvalidated_candidate',
           meaning='Candidate label is not a physics acceptance decision')
    for family, levels in profiles['candidate_ranges'].items():
        for level, parameters in levels.items():
            for name, limits in parameters.items():
                record('ordered_finite_range:'+family+':'+level+':'+name,
                       len(limits)==2 and all(math.isfinite(x) for x in limits) and limits[0]<=limits[1], limits=limits)
    mappings = {
        'T0': {'ee_amplitude_m': max(easy['ee_amplitude']), 'ee_period_s':easy['ee_period_s'],
               'ee_center_x_m':easy['ee_center'][0], 'ee_center_z_m':easy['ee_center'][2],
               'orientation_offset_rad':max(abs(v-n) for v,n in zip(easy['ee_orientation_offset_rpy'], [0., .6, 0.]))},
        'T1': {'linear_speed_mps':easy['base_linear_speed'],'yaw_rate_radps':easy['base_yaw_rate'],
               'segment_s':easy['base_segment_s']},
        'T2': {'step_height_m':easy['stairs_height'],'tread_m':easy['stairs_tread'],
               'stone_width_m':easy['stones_width'],'gap_m':easy['stones_gap'],
               'height_variation_m':easy['stones_height'],'route_distance_m':easy['route_distance'],
               'linear_speed_mps':easy['terrain_speed']}}
    for family, values in mappings.items():
        for name, value in values.items():
            bounds=profiles['candidate_ranges'][family]['beginner'][name]
            record('easy_in_beginner:'+family+':'+name,bounds[0]<=value<=bounds[1],value=value,bounds=bounds)
    period=easy['ee_period_s']
    speed_upper=2*math.pi/period*math.sqrt(sum(a*a for a in easy['ee_amplitude']))
    lower_bound=easy['route_distance']/easy['terrain_speed']
    record('positive_terrain_geometry', all(easy[k]>0 for k in
           ['stairs_height','stairs_tread','stones_width','stones_gap','route_distance']))
    record('episode_exceeds_constant_speed_route_lower_bound', easy['episode_s_terrain']>lower_bound,
           lower_bound_s=lower_bound, episode_s=easy['episode_s_terrain'],
           caveat='Does not include acceleration, falls, waypoint turns, settling or braking')
    record('flat_episode_has_two_smooth_cycles',easy['episode_s_flat']>=2*period,
           cycles=easy['episode_s_flat']/period)
    source_manifest={}
    syntax={}
    for path in sorted(list((root/'tools').glob('*.py'))+list((root/'task_suite').glob('*.py'))):
        relative=str(path.relative_to(root))
        source=path.read_text()
        source_manifest[relative]=hashlib.sha256(source.encode()).hexdigest()
        try:
            ast.parse(source,filename=relative)
            syntax[relative]='parse_pass'
        except SyntaxError as err:
            syntax[relative]=str(err)
            record('python_syntax:'+relative,False,error=str(err))
    return dict(scope='Static parameter bounds, timing lower bound, Python syntax and source identity only',
                physical_admission=False, kinematics_admission=False, learning_demonstrated=False,
                profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                all_checks_pass=all(c['passed'] for c in checks),checks=checks,
                smooth_command_speed_upper_bound_mps=speed_upper,
                source_sha256=source_manifest,syntax=syntax)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.root)
    with args.out.open('x') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','source_sha256','syntax')}))
    if not result['all_checks_pass']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
