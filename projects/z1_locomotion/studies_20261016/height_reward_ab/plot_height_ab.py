from publication_paths import research_path
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(research_path('z1_cp5700_isaaclab_5060/experiments/height_reward_ab_20261016'));s=json.loads((B/'compact-report-input.json').read_text());out=B/'plots';out.mkdir(exist_ok=True)
colors={'original':'#777777','A':'#3676b2','B1':'#d87830'}
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for arm in ('A','B1'):
 h=s['training'][arm]['height_per_update'];axs[0,0].plot([v['update'] for v in h],[v['actual_mean_m'] for v in h],label=arm,color=colors[arm])
 axs[0,0].axhline(.373 if arm=='A' else .384,color=colors[arm],ls='--',alpha=.7)
axs[0,0].set(title='Training: measured height (solid), reward goal (dashed)',xlabel='Additional PPO updates',ylabel='RootZ - same ray mean [m]');axs[0,0].legend()
labels=['original','A','B1']
for ax,key,scale,title in [(axs[0,1],'height_mean_m',1000,'Downstairs: actual mean height'),(axs[1,0],'mean_calf_fraction',100,'Downstairs: calf contact time'),(axs[1,1],'mean_sum_calf_norm_integral_Ns',1,'Downstairs: cumulative calf force-norm proxy')]:
 vals=np.array([[s['cases'][f'{a}_down_s{k}'][key]*scale for k in (456,789,2026)] for a in labels]);means=vals.mean(1)
 ax.bar(labels,means,color=[colors[a] for a in labels],alpha=.75)
 for i,v in enumerate(vals):ax.scatter(np.full(3,i)+[-.08,0,.08],v,s=23,color='black')
 ax.set_title(title);ax.set_ylabel('mm' if key=='height_mean_m' else '%' if key=='mean_calf_fraction' else 'Ns / trial (not vertical impulse)')
 if key=='height_mean_m':ax.set_ylim(350,390)
 for i,v in enumerate(means):ax.text(i,v,f'{v:.2f}',ha='center',va='bottom')
for ax in axs.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('CP5700 height-reward A/B | 100 updates each | 3 paired seeds x16 downstairs trials\nAll three policies complete48/48; strict-safe0/48. Nominal / fixed body command /50Hz.')
fig.savefig(out/'01_height_ab_results.png',dpi=160);plt.close(fig)
d=B/'evaluations/B1_down_s456'
with np.load(d/'telemetry.npz') as n:
 root=n['root_pos'];orig=n['origins'];ray=n['base_ray_hits_w'];step=int(np.argmin(abs((root[:,0,0]-orig[:,0,0])-.9)))
 rel=root[step,0]-orig[step,0];rr=ray[step,0]-orig[step,0];mean=rr[:,2].mean();idx=int(np.clip(np.floor((rel[0]-.9)/.3)+1,0,8));ground=-.1*idx
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
x=np.linspace(.3,1.65,1200);z=-.1*np.clip(np.floor((x-.9)/.3).astype(int)+1,0,8)
ax.plot(x,z,color='#775b40',lw=3,label='Actual10cm staircase surface')
ax.scatter(rr[:,0],rr[:,2],s=80,marker='x',color='#5b9b55',label='Saved ray hits (16rays, projected X-Z)')
ax.axhline(mean,color='gray',ls=':',label=f'Mean ray ground={mean:.3f}m')
ax.axhline(.384+mean,color='#d87830',ls='--',label=f'Actual reward targetZ={.384+mean:.3f}m')
ax.axhline(.384+ground,color='#3676b2',ls='--',label=f'Diagnostic root-local tread+.384={.384+ground:.3f}m')
ax.scatter([rel[0]],[rel[2]],color='black',s=70,label=f'Actual root at{(step+1)*.02:.2f}s')
ax.set(xlabel='X relative to terrain origin [m]',ylabel='Z relative to origin [m]',title='Measured edge mixing: scanner mean vs tread directly under root\nDiagnostic reference only; reward function and physics unchanged')
ax.legend(loc='lower left',fontsize=9);ax.grid(alpha=.15);fig.savefig(out/'02_measured_scanner_edge.png',dpi=160)
