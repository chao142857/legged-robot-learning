import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'RESULTS_BEFORE_REPAIR.json').read_text())
fig,ax=plt.subplots(figsize=(8.8,4.8),layout='constrained')
for seed,color in [('43','#2672ba'),('44','#ce6a23')]:
 rows=a['source_curves'][seed];ax.plot([x['update']/1000 for x in rows],[100*x['joint_success_rate'] for x in rows],marker='o',lw=2,color=color,label='CL1 source seed '+seed)
 ax.annotate(f"{100*rows[-1]['joint_success_rate']:.2f}%",(10,100*rows[-1]['joint_success_rate']),xytext=(5,-12 if seed=='43' else 8),textcoords='offset points',color=color)
ax.set(xlabel='Student PPO updates (thousands)',ylabel='Strict joint success (%)',title='CL1 source students: fixed independent test pool',xticks=range(1,11),xlim=(.7,10.8),ylim=(0,40))
ax.grid(alpha=.22);ax.legend(loc='upper left');fig.text(.5,-.02,'Source-training curves; these are not the new seed70158 generalization comparison.',ha='center',fontsize=9)
fig.savefig(ROOT/'source_curves.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'source_curves.svg',bbox_inches='tight')
