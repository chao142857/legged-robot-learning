"""Rebuild the model-free scientific figure from published numerical results."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    native = json.loads((root/'projects/go2_piper_wbc/native_20261023/results/evaluation_pooled.json').read_text())
    pilot = json.loads((root/'projects/sgs/historical_48task_20261019/results/fixed_learning_curve.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11.6,3.9),layout='constrained')
    for mode,color,marker in [('deterministic','#2463ad','o'),('stochastic','#ba681a','s')]:
        rows=sorted((x for x in native if x['mode']==mode and x['window']=='after_1s' and x['group']=='all'),key=lambda x:x['checkpoint'])
        axes[0].plot([x['checkpoint'] for x in rows],[100*x['ee_position_rmse'] for x in rows],label=mode.capitalize(),color=color,marker=marker)
    axes[0].set(title='Native Go2 + Piper WBC',xlabel='Completed PPO updates',ylabel='EE position RMSE (cm)')
    axes[0].text(.97,.95,'First episode, age > 1 s\nPooled over three evaluation seeds',transform=axes[0].transAxes,va='top',ha='right',fontsize=8.5,color='#555555')
    for method,color,marker,style in [('uniform','#2463ad','o','-'),('sgs','#ba681a','x','--')]:
        rows=sorted((x for x in pilot if x['method']==method),key=lambda x:x['update'])
        axes[1].plot([x['update'] for x in rows],[100*x['joint_success']/x['n'] for x in rows],label=method.upper() if method=='sgs' else 'Uniform',color=color,marker=marker,linestyle=style,markersize=7)
    axes[1].set(title='Fixed-48-task warm-start pilot',xlabel='Completed PPO updates',ylabel='Joint success (%)',ylim=(-5,105))
    axes[1].text(.97,.95,'Both methods: 0/48 after updates\nNo observed SGS benefit',transform=axes[1].transAxes,va='top',ha='right',fontsize=8.5,color='#555555')
    for ax in axes:
        ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,loc='center right')
    fig.suptitle('Separate simulation protocols · one training seed each',fontsize=12,fontweight='bold')
    out=args.out or root/'docs/figures/simulation_results.svg'
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,metadata={'Creator':'legged-robot-learning; tools/plot_results.py','Date':None})
    plt.close(fig)
    print(out)


if __name__=='__main__':
    main()
