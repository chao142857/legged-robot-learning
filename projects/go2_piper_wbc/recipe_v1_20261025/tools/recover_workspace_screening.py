"""Correct named-joint CPU screening; preserve all original physical trajectories."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
from recipe_v1.kinematics import screen_cloud,validate_dynamic_fk,rotation_wxyz

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--physical',required=True)
parser.add_argument('--inventory',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
physical=Path(args.physical);out=Path(args.out)
out.mkdir(parents=True,exist_ok=False)
inventory=json.loads(Path(args.inventory).read_text())
model=json.loads((physical/'kinematic_model.json').read_text())
protocol=json.loads((physical/'protocol.json').read_text())
reset=json.loads((physical/'reset_state.json').read_text())
data=np.load(physical/'trajectories.npz')
cloud=protocol['task']['workspace_cloud']
bounds=[x['soft_limits_rad'] for x in inventory['joint_table'] if x['name'].startswith('joint')]
assert len(bounds)==6
base_id=inventory['body_names'].index('base');link_id=inventory['body_names'].index('link0')
qb=inventory['body_quaternion_world'][0][base_id]
ql=inventory['body_quaternion_world'][0][link_id]
mount=(rotation_wxyz(qb).inv()*rotation_wxyz(ql)).as_quat()
mount=mount[[3,0,1,2]].tolist()
offset=reset['body_pos'][0][link_id][2]-reset['root_state'][0][2]
validation=validate_dynamic_fk(model['chain'],data)
assert validation['passed'],validation
result=screen_cloud(model['chain'],cloud,bounds,offset,mount)
(out/'kinematic_screening.json').write_text(json.dumps(result,indent=2)+'\n')
mapping={name:i for a in inventory['actuators'].values() for name,i in zip(a['joint_names'],a['joint_indices'])}
ids=[mapping[x['name']] for x in inventory['joint_table']]
valid=data['valid'];after=valid&(data['age_steps']*.02>1.)
admitted=[];stats=[]
for i,entry in enumerate(result):
 sl=slice(i*4,(i+1)*4);mask=after[:,sl]
 durations=data['valid'][:,sl].sum(0)*.02
 failed=np.zeros(4,dtype=bool)
 for key in data.files:
  if key.startswith('termination_') and key!='termination_time_out':failed|=(data[key][:,sl]&data['valid'][:,sl]).any(0)
 def v(key):return data['control_'+key][:,sl][mask]
 ee=data['ee_position_error'][:,sl][mask]
 margin=v('soft_margin')[:,12:]
 # This source trajectory predates the corrected saturation column mapping.
 sat=data['control_actuator_saturation_substep_fraction'][:,sl][:,:,ids][mask][:,12:]
 error=v('pd_target_error')[:,12:]
 stats_row=dict(target_m=entry['target_m'],complete_count=int(((durations>=20.)&~failed).sum()),
   ee_after_1s_rmse_m=float(np.sqrt(np.mean(ee**2))),ee_error_p95_m=float(np.quantile(ee,.95)),
   arm_soft_margin_p01_rad=float(np.quantile(margin,.01)),arm_saturation_fraction=float(sat.mean()),
   arm_pd_abs_error_mean_rad=float(abs(error).mean()),arm_pd_abs_error_p95_rad=float(np.quantile(abs(error),.95)),
   actual_hard_limit_violation_fraction=float((v('hard_limit_excess')>.001).mean()),
   kinematically_admitted=entry['kinematically_admitted'])
 stats_row['physically_admitted']=stats_row['complete_count']==4 and stats_row['ee_after_1s_rmse_m']<.075 and stats_row['ee_error_p95_m']<.10 and stats_row['arm_soft_margin_p01_rad']>.005 and stats_row['arm_saturation_fraction']<.05 and stats_row['arm_pd_abs_error_mean_rad']<.2 and stats_row['arm_pd_abs_error_p95_rad']<.45 and stats_row['actual_hard_limit_violation_fraction']<.001
 stats_row['admitted']=stats_row['kinematically_admitted'] and stats_row['physically_admitted']
 if stats_row['admitted']:admitted.append(entry['target_m'])
 stats.append(stats_row)
receipt=dict(source=str(physical),source_trajectory_sha256=hashlib.sha256((physical/'trajectories.npz').read_bytes()).hexdigest(),
 correction='IK bounds formerly sliced the PhysX last six joints, inadvertently including a calf. Correct bounds use named action-joint order. Saturation statistics were remapped. Actual physical control constraints already used the correct action joint IDs.',
 fk_validation=validation,arm_bounds_rad=bounds,action_to_asset_joint_indices=ids,
 cloud_statistics=stats,admitted_ee_points=admitted,
 scope='Exact finite anchor cloud only. No continuous volume or dynamic trajectory is admitted by corner/IK tests.',
 passed=len(admitted)>=8)
(out/'workspace_admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'kinematic_points':sum(x['kinematically_admitted'] for x in result),'physical_points':sum(x['physically_admitted'] for x in stats),'admitted':len(admitted)}),flush=True)
