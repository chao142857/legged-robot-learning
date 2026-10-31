"""Standard scientific plots of actual Stage3 physics, without diagnostic/policy mixing."""
import argparse,json,pathlib,hashlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read(p):return json.loads(p.read_text())
def main(p,out):
    out.mkdir(parents=True,exist_ok=True);e=read(p/'P1_CAUSAL_EVIDENCE.json');cases={c['label']:c for c in e['cases']};protocols=['X','T0','T1']
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');x=np.arange(3);width=.36
    for offset,suffix,label,color in [(-width/2,'s0.0','Frozen CP1800','#50687d'),(width/2,'s0.5','Physical diagnostic correction','#2a8f87')]:
        rows=[cases['P1-parent-'+r+'-'+suffix] for r in protocols]
        axes[0].bar(x+offset,[r['native_X_after1s']['actual_fitted_amplitude_m']*1000 for r in rows],width,label=label,color=color)
    axes[0].axhline(8,ls='--',c='#c66338',lw=1,label='8mm necessary gate')
    axes[0].set(xticks=x,xticklabels=protocols,ylabel='Real EE X fitted amplitude (mm)',title='Actual20s PhysX;32/32 safe per case');axes[0].legend(fontsize=7)
    a=[];b=[]
    for r in protocols:
        c=next(q for q in cases['P1-parent-'+r+'-s0.0']['counterfactuals'] if q['mode']=='coherent_history');a.append(c['sum_of_individual_abs_m']*1000);b.append(c['effective_sum_abs_m']*1000)
    axes[1].bar(x-width/2,a,width,label='Sum individual |Jx dq|',color='#b69459');axes[1].bar(x+width/2,b,width,label='|Sum Jx dq|',color='#50687d')
    axes[1].set(xticks=x,xticklabels=protocols,ylabel='Local target-counterfactual X (mm)',title='Same actual state; coherent goal history');axes[1].legend(fontsize=7)
    pairs={r['treatment'].split('-')[2]:r for r in e['pairs']};total=[pairs[r]['native_reward_rate_delta_sum'] for r in protocols]
    axes[2].bar(x,total,color='#2a8f87');axes[2].axhline(0,c='#333',lw=.7);axes[2].set(xticks=x,xticklabels=protocols,ylabel='Original27-term reward rate change',title='Corrected minus frozen; after5s common-live')
    fig.suptitle('P1 causal diagnosis: physical intervention is NOT learned PPO success',fontsize=11)
    fig.savefig(out/'P1_PHYSICS-v1.svg',metadata=dict(Date=None));plt.close(fig)
    curve=p/'PHYSICAL_CURVE.json'
    if curve.exists():
        rows=read(curve)['rows'];fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
        for branch,color in [('synergy','#2a8f87'),('matchedNative','#50687d')]:
            for seed,marker in [(6181031,'o'),(6281031,'s')]:
                data=[]
                for r in rows:
                    m=re.match(r'^'+branch+r'([0-9]+)-seed21-X-'+str(seed)+r'$',r['label'])
                    if m:data.append((int(m.group(1)),r))
                    elif r['label']=='frozen1800-X-'+str(seed):data.append((0,r))
                data.sort(key=lambda q:q[0])
                for ax,key,factor in zip(axes,['synchronous_gain','actual_fitted_amplitude_m','correlation'],[1,1000,1]):
                    ax.plot([q[0] for q in data],[q[1]['X_after1s'][key]*factor for q in data],marker=marker,color=color,ls='-' if seed==6181031 else '--',label=branch+' dev'+str(seed))
        for ax,title,gate in zip(axes,['X synchronous gain','Real X amplitude (mm)','X correlation'],[.35,8,.5]):
            ax.axhline(gate,c='#c66338',ls=':',lw=1);ax.set(xlabel='Additional warm-start PPO updates',ylabel=title);ax.grid(alpha=.2)
        axes[0].legend(fontsize=6);fig.suptitle('Actual original20s X physics; unchanged original safety gates reported separately',fontsize=11)
        fig.savefig(out/('LEARNING-'+str(len(rows))+'rows.svg'),metadata=dict(Date=None));plt.close(fig)
    for f in out.glob('*.svg'):
        meta=dict(file=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        f.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--out',required=True);v=a.parse_args();main(pathlib.Path(v.root),pathlib.Path(v.out))
