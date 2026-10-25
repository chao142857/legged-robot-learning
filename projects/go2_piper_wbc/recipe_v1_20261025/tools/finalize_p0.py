"""CPU admission gate from the paired frozen rollouts and corrected workspace evidence."""
import argparse
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',required=True)
parser.add_argument('--out')
args=parser.parse_args()
root=Path(args.root);source=Path(__file__).resolve().parents[1]
spec=json.loads((source/'recipe_v1/spec.json').read_text())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
control_sha=sha(source/'recipe_v1/control.py')
workspace=read(root/'p0_workspace_screening_named_v2/workspace_admission.json')
comparisons=[];checks={}
for seed,folder in [(8101,'p0_control'),(8102,'p0_workspace_and_repeat')]:
    original=root/folder/f'native_original_seed{seed}'
    constrained=root/folder/f'native_clamped_seed{seed}'
    a=read(original/'summary.json');b=read(constrained/'summary.json')
    t0=read(original/'tracking_metrics.json')['windows']['all']
    t1=read(constrained/'tracking_metrics.json')['windows']['all']
    c=read(constrained/'control_summary.json')
    for p in (original,constrained):
        protocol=read(p/'protocol.json')
        assert protocol['source_hashes']['recipe_v1/control.py']==control_sha
        assert read(p/'interface.json')['passed']
        assert read(p/'summary.json')['actor_unchanged'] and read(p/'summary.json')['critic_unchanged']
    delta={d:abs(t1['directions'][d]['command_plateau']['gain']-t0['directions'][d]['command_plateau']['gain']) for d in ('forward','backward','left','right')}
    row=dict(seed=seed,episodes=a['num_episodes'],original_complete=a['complete_count'],clamped_complete=b['complete_count'],
             original_ee_rmse_m=t0['ee']['position_rmse_m'],clamped_ee_rmse_m=t1['ee']['position_rmse_m'],
             direction_gain_absolute_change=delta,shadow_low_terminations=c['shadow_low_termination_episodes'],
             target_limit_excess_max_rad=c['target_limit_excess_max_rad'],
             actuator_substep_counts=[c['actuator_substep_counts_min'],c['actuator_substep_counts_max']],
             original_path=str(original),constrained_path=str(constrained))
    gate=spec['p0_control_gate']
    row['passed']=(a['complete_count']-b['complete_count'])/a['num_episodes']<=gate['native_completion_loss_max_fraction'] and max(delta.values())<=gate['native_direction_gain_absolute_change_max'] and row['clamped_ee_rmse_m']-row['original_ee_rmse_m']<=gate['native_ee_rmse_increase_max_m'] and row['shadow_low_terminations']/a['num_episodes']<=gate['native_shadow_low_termination_max_fraction'] and row['target_limit_excess_max_rad']<=gate['processed_target_excess_max_rad'] and row['actuator_substep_counts']==[4.,4.]
    comparisons.append(row)
inventory=read(root/'p0_inventory_v3/inventory.json')
checks.update(native_retention_both_seeds=all(x['passed'] for x in comparisons),
    workspace_cloud=workspace['passed'],interface_dimensions=inventory['observation_dims']=={'policy':[210],'critic':[102]} and inventory['action_dim']==18,
    FK_matches_physics=workspace['fk_validation']['passed'])
failed=read(root/'p0_control/failed_original_seed8101/control_summary.json')
result=dict(schema='recipe_v1_P0_admission',passed=all(checks.values()),checks=checks,
    recipe_spec_sha256=sha(source/'recipe_v1/spec.json'),
    comparisons=comparisons,admitted_ee_points=workspace['admitted_ee_points'],
    control_source_sha256=control_sha,workspace_evidence=str(root/'p0_workspace_screening_named_v2/workspace_admission.json'),
    workspace_evidence_sha256=sha(root/'p0_workspace_screening_named_v2/workspace_admission.json'),
    raw_tail_regularizer_justification=dict(failed_raw_abs_max=max(x['raw_abs_max'] for x in failed['per_joint']),
        failed_shadow_target_clip_fraction=sum(x['shadow_would_clip_fraction'] for x in failed['per_joint'])/18,
        selected_weight=-.0001,formula='mean(relu(abs(raw_action)-4)^2)',unit='dimensionless raw action units'),
    physical_sampling_scope='27 tested fixed anchors only, each with four initial medium movement directions; no continuous EE volume is claimed',
    known_probe_corrections='Inventory v1/v2 omitted Gym reset; v3 passed. Initial CPU IK used PhysX last-six bounds; corrected by named mapping without modifying physical trajectories. Per-joint saturation columns remapped in corrected workspace evidence.',
    A_to_B_gate=spec['A_to_B_gate'],automatic_B_or_C=False)
target=Path(args.out) if args.out else root/'p0_admission.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'checks':checks,'admitted_points':len(result['admitted_ee_points'])}))
if not result['passed']:raise SystemExit(1)
