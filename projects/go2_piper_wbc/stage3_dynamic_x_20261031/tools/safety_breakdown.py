"""CPU attribution of native full20s strict-safety failures; never changes scores."""
import argparse,csv,json,pathlib
import numpy as np
from gpu_budget import read,write

def analyze(run):
    directory=pathlib.Path(run)/'evaluation';summary=read(directory/'summary.json')
    with (directory/'episodes.csv').open() as f:episodes=list(csv.DictReader(f))
    z=np.load(directory/'trajectories.npz',allow_pickle=False);valid=z['valid'].astype(bool);age=z['age_steps']*.02
    masks=dict(overspeed=z['joint_velocity_limit_ratio']>1.0001,hard_position=z['joint_position_limit_excess']>.001,nonfoot=z['nonfoot_max_force']>1,clearance=z['base_clearance']<.15)
    failures=[];matches=0
    names=[v['name'] for v in read(directory/'control_summary.json')['per_joint']]
    for e,episode in enumerate(episodes):
        measured={k:v[:,e]&valid[:,e] for k,v in masks.items()}
        safe=episode['complete']=='True' and not any(m.any() for m in measured.values())
        matches+=int(safe==(episode['safe']=='True'))
        if safe:continue
        issues={}
        for name,m in measured.items():
            t=age[:,e][m]
            if not len(t):continue
            issues[name]=dict(first_s=float(t.min()),last_s=float(t.max()),frames=int(m.sum()),duration_s=float(m.sum()*.02),after1s_frames=int((t>1.).sum()))
        hard=z['control_hard_limit_excess'][:,e][valid[:,e]].max(0)
        failures.append(dict(env=e,complete=episode['complete']=='True',issues=issues,
            hard_position_joints=[names[j] for j in np.flatnonzero(hard>.001)],
            peak_velocity_limit_ratio=float(z['joint_velocity_limit_ratio'][:,e][valid[:,e]].max()),
            max_hard_excess_rad=float(z['joint_position_limit_excess'][:,e][valid[:,e]].max()),
            min_clearance_m=float(z['base_clearance'][:,e][valid[:,e]].min())))
    assert matches==len(episodes)==summary['num_episodes'],'Derived interpretation differs from native score; investigate, do not rescore'
    return dict(run=str(run),strict_safe=summary['safe_count'],complete=summary['complete_count'],n=summary['num_episodes'],native_score_reproduced=matches,failures=failures,
        note='Full20s and original startup retained. after1s is attribution only. Velocity joint identity is not inferred from a scalar maximum ratio.')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--labels',nargs='+',required=True);a.add_argument('--out',required=True);c=a.parse_args();p=pathlib.Path(c.root);index=read(p/'EVAL_INDEX.json')
    rows=[dict(label=label,**analyze(index[label])) for label in c.labels];write(c.out,dict(rows=rows));print(json.dumps(rows))
