"""CPU-only static scientific plots of verified formal evaluation summaries."""
from pathlib import Path
import json,math,os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
a=json.loads((root.parent/'evidence/completed_curves.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#94a3b8','svg.fonttype':'none'})
colors={'uniform':'#5b6472','fresh':'#d97706','old-meta':'#2563eb','new-meta':'#059669'}
labels={'uniform':'Uniform','fresh':'Fresh Teacher (untrained, frozen)','old-meta':'Meta (8 starts, frozen)','new-meta':'New Meta (+8 seed46 starts, frozen)'}
max_y=max(p['joint_success_rate']*100 for s in a['series'] for p in s['points'] if p)
ymax=max(45,math.ceil((max_y+5)/5)*5)
for seed in (47,48):
    fig,ax=plt.subplots(figsize=(11.4,6.7),facecolor='white')
    fig.subplots_adjust(left=.085,right=.94,top=.83,bottom=.235)
    series=[s for s in a['series'] if s['student_seed']==seed]
    for s in series:
        arm=s['arm'];xs=np.arange(1,11);ys=np.array([p['joint_success_rate']*100 if p else np.nan for p in s['points']])
        label=labels[arm]
        if seed==48 and arm=='old-meta':label='Old '+label
        if np.isnan(ys).all():label+=' — first 1k test pending'
        ax.plot(xs,ys,color=colors[arm],lw=2.3,marker='o',markersize=5.7,
                linestyle='--' if arm=='new-meta' else '-',label=label,zorder=3)
        if np.isfinite(ys[-1]):
            ax.annotate(f'{ys[-1]:.2f}%',xy=(10,ys[-1]),xytext=(10.14,ys[-1]),
                        ha='left',va='center',color=colors[arm],fontsize=11,fontweight='bold')
    if seed==47:
        ax.axvline(7.7,color='#94a3b8',ls=':',lw=1.1,zorder=1)
        ax.text(7.77,ymax-1.8,'Meta GPU\nmigration',fontsize=9,color='#64748b',va='top')
    ax.set(xlim=(.75,10.9),ylim=(0,ymax),xticks=np.arange(1,11),yticks=np.arange(0,ymax+1,5),
           xlabel='PPO updates (thousands)',ylabel='Strict joint success rate (%)')
    ax.grid(axis='both',alpha=.4,color='#d1d9e3',lw=.7);ax.set_axisbelow(True)
    ax.legend(loc='upper left',frameon=False,fontsize=10,handlelength=2.5,labelspacing=.75)
    fig.text(.085,.95,f'Student seed {seed}  |  Go2-Piper WBC',fontsize=18,fontweight='bold',color='#15283d')
    fig.text(.085,.897,'Strict 5 cm / 80% protocol · fixed 1,024-sequence test set · mean actions · unsmoothed checkpoints',fontsize=10.5,color='#526578')
    if seed==47:
        note='Meta moved from RTX 3060 to RTX 5060 Laptop after update 7,700; Uniform and Fresh ran on RTX 3060.\nCross-GPU initial tensor bytes differ. All reported points use the same test pool, seeds and scoring protocol.'
    else:
        note='All four arms completed 10,000 updates. New Meta added 8 starts from one Student seed46.\nUniform: RTX 5060 Laptop; Fresh / Old Meta / New Meta: RTX 3060. Cross-GPU initial tensor bytes differ.'
    fig.text(.085,.12,note,fontsize=9,color='#526578',linespacing=1.6,va='top')
    fig.text(.085,.025,'Snapshot: 2026-11-08 16:52 (Asia/Shanghai)  |  One Student seed per plot; no seed-level uncertainty estimate.',fontsize=8.5,color='#718096')
    stem=f'seed{seed}_strict_joint_success_20261108_final'
    fig.savefig(root/(stem+'.png'),dpi=240,facecolor='white')
    fig.savefig(root/(stem+'.svg'),facecolor='white')
    plt.close(fig)
print(json.dumps(dict(passed=True,plots=['seed47','seed48'],source_points=sum(p is not None for s in a['series'] for p in s['points']),missing_points_preserved=True,gpu_used=False)))
