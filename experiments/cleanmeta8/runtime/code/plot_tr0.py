"""Static scientific plot of actual gate evidence; no generated imagery."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT

c=json.loads((ROOT/'CANDIDATE_RESULT.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[2,1]})
rows=c['original_changes'];labels=['H%d'%i for i in range(8)]+['V44','V46','V47','V48','Fresh H']
axes[0].bar(np.arange(len(rows)),[100*x['tv'] for x in rows],color='#3478b8',label='Original-step = Sampling-TR')
axes[0].axhline(2,color='#c74235',linestyle='--',linewidth=1.6,label='Added TV limit: 2%')
axes[0].set_xticks(np.arange(len(rows)),labels,rotation=40,ha='right');axes[0].set_ylabel('TV on full mixed sequence distribution (%)')
axes[0].set_ylim(0,2.3);axes[0].set_title('Same frozen input for old/new parameters');axes[0].legend(frameon=False,loc='upper left')
raw=c['original_candidate_audit'];x=np.arange(len(raw));tv=[100*max(y['tv'] for y in a['changes']) for a in raw]
axes[1].bar(x,tv,color=['#c66b63','#348b65'][:len(x)])
axes[1].axhline(2,color='#c74235',linestyle='--',linewidth=1.6)
axes[1].set_xticks(x,['LR %.3g'%(.01*a['learning_rate_multiplier']) for a in raw]);axes[1].set_ylabel('Max reference TV (%)')
for i,a in enumerate(raw):axes[1].text(i,tv[i]+.08,'KL %.5f\n%s'%(a['preference_kl'],'old guard rejects' if a['preference_kl']>.05 else 'accepted by both'),ha='center',fontsize=9)
axes[1].set_ylim(0,max(tv)*1.35);axes[1].set_title('Existing preference-KL bound: 0.05')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.suptitle('nSGS-TR0: added TV bound did not change the accepted update',fontsize=14)
fig.text(.5,.01,'13 fixed inputs; actual32768-way probabilities after20% Uniform mixing. No12k validation launched.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.05,1,.93]);out=ROOT/'figures';out.mkdir(exist_ok=True)
for suffix in ('png','svg'):fig.savefig(out/('tr0_real_tv_gate.'+suffix),dpi=180,bbox_inches='tight')
plt.close(fig)
print('TR0_REAL_TV_PLOT_SAVED',flush=True)
