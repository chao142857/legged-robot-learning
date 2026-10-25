"""Render retained P0 physical evidence on CPU, without simulation or RTX."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    screen=json.loads((root/'p0_workspace_screening_named_v2/workspace_admission.json').read_text())
    # Paired first-episode plots exclude all post-reset samples, including failed CP5000.
    cases=['native_original_seed8101','native_clamped_seed8101','failed_original_seed8101','failed_clamped_seed8101']
    fig,axes=plt.subplots(3,4,figsize=(15,9),constrained_layout=True)
    for col,name in enumerate(cases):
        directory=root/'p0_control'/name;z=np.load(directory/'trajectories.npz')
        valid=z['valid'].astype(bool);m=valid&(z['age_steps']*.02>=1.)
        raw=z['control_raw_action'][m];q=z['control_processed_target'][m]
        axes[0,col].hist(raw.ravel(),bins=80,log=True);axes[0,col].set_title(name.replace('_seed8101',''));axes[0,col].set_xlabel('Raw action (dimensionless)')
        for j in range(12,18):axes[1,col].hist(q[:,j],bins=50,histtype='step',label=f'joint{j-11}')
        axes[1,col].set_xlabel('Arm processed target (rad)')
        for e in range(4):
            em=valid[:,e];axes[2,col].plot(np.arange(len(valid))[em]*.02,z['base_clearance'][em,e],label=f'env{e}')
        axes[2,col].axhline(.16,ls='--',c='r');axes[2,col].set_ylim(0,.4);axes[2,col].set_xlim(0,20);axes[2,col].set_xlabel('First-episode time (s)');axes[2,col].set_ylabel('Base clearance (m)')
    axes[1,0].legend(fontsize=7);fig.suptitle('Measured P0 physics: native CP3000 and failed CP5000, original vs bounded targets')
    fig.savefig(out/'P0_control_comparison.png',dpi=140);plt.close(fig)
    # Named workspace is a finite admitted cloud, not a declaration about its continuous hull.
    print('workspace_keys',list(screen))
    cert=json.loads((root/'p0_admission_final.json').read_text());points=np.array(cert['admitted_ee_points'])
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection='3d');ax.scatter(*points.T,c=points[:,2],s=65)
    ax.set_xlabel('EE link0 X (m)');ax.set_ylabel('EE link0 Y (m)');ax.set_zlabel('EE world Z (m)')
    ax.set_title('27 admitted anchors: IK + four-direction physical rollouts\nOnly these points are admitted; continuous hull is untested')
    fig.savefig(out/'P0_workspace_cloud.png',dpi=140);plt.close(fig)
    # Full 20s native first episode, A protocol, fixed medium command and EE target switches.
    z=np.load(root/'evaluator_acceptance_A/trajectories.npz');t=np.arange(len(z['valid']))*.02
    fig,axes=plt.subplots(5,4,figsize=(15,11),constrained_layout=True)
    for e in range(4):
        m=z['valid'][:,e]
        for row,j in enumerate((0,2)):
            axes[row,e].plot(t[m],z['base_cmd'][m,e,j],label='target');axes[row,e].plot(t[m],z['base_actual'][m,e,j],label='actual');axes[row,e].set_ylabel(('vx (m/s)','yaw (rad/s)')[row])
        for j,axis in enumerate('xyz'):
            axes[j+2,e].plot(t[m],z['ee_cmd'][m,e,j],label='target');axes[j+2,e].plot(t[m],z['ee_actual_mixed'][m,e,j],label='actual');axes[j+2,e].set_ylabel(f'EE {axis} (m)')
        axes[0,e].set_title(f'env{e}');axes[-1,e].set_xlabel('Time (s)')
        for ax in axes[:,e]:ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle('Native CP3000 under Recipe V1 A evaluation: measured headless physics, 20s')
    fig.savefig(out/'P0_native_A_trajectory.png',dpi=130);plt.close(fig)
    (out/'source_index.json').write_text(json.dumps(dict(p0_control=str(root/'p0_control'),workspace=str(root/'p0_workspace_and_repeat'),named_workspace=str(root/'p0_workspace_screening_named_v2'),native_A=str(root/'evaluator_acceptance_A'),kind='CPU plots from measured first-episode physics; no RTX video'),indent=2)+'\n')


if __name__=='__main__':main()
