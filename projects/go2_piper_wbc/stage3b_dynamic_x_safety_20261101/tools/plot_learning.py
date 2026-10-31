"""CPU scientific curves of actual native20s physics; no diagnostic mixing."""
import argparse,hashlib,json,pathlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gpu_budget import read

def main(root,out):
    p=pathlib.Path(root);old=pathlib.Path(read(p/'AUTHORIZATION.json')['prior_closed_stage3_root']);out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    prior=read(old/'PHYSICAL_CURVE.json')['rows'];now=read(p/'PHYSICAL_CURVE.json')['rows']
    groups={}
    for r in prior:
        match=re.match(r'^(synergy|matchedNative)([0-9]+)-seed([0-9]+)-',r['label'])
        if match:
            family='Old unconstrained synergy' if match[1]=='synergy' else 'Matched native PPO'
            groups.setdefault((family,int(match[3]),int(match[2])),[]).append(r)
    for node in read(p/'CANDIDATE_NODES.json'):
        rows=[r for r in now if r['label'].startswith(node['evaluation_prefix']+'-')]
        if len(rows)==8:groups[(node['recipe_id'],node['training_seed'],node['additional_actual_updates'])]=rows
    families=['Matched native PPO','Old unconstrained synergy']+sorted({k[0] for k in groups if k[0] not in ('Matched native PPO','Old unconstrained synergy')})
    seeds=sorted({k[1] for k in groups});colors=['#5a7188','#8b7b68','#248c77','#8b6099','#bd8f28']
    fig,axes=plt.subplots(len(seeds),4,figsize=(16,3.5*len(seeds)),squeeze=False,layout='constrained');points=[]
    parent={seed:next(r for r in prior if r['label']=='frozen1800-X-'+str(seed)) for seed in (6181031,6281031)}
    for rowno,training in enumerate(seeds):
        for fi,family in enumerate(families):
            steps=sorted(k[2] for k in groups if k[:2]==(family,training))
            for dev,marker in [(6181031,'o'),(6281031,'s')]:
                series=[(0,parent[dev],None)]
                for step in steps:
                    whole=groups[(family,training,step)];x=next(r for r in whole if r['protocol']=='X' and r['seed']==dev);series.append((step,x,whole))
                if len(series)==1:continue
                for col,key,factor in [(0,'synchronous_gain',1),(1,'actual_fitted_amplitude_m',1000),(2,'correlation',1)]:
                    axes[rowno,col].plot([n for n,r,w in series],[r['X_after1s'][key]*factor for n,r,w in series],color=colors[fi%len(colors)],marker=marker,ls='-' if dev==6181031 else '--',label=family+' /'+str(dev))
                safety=[]
                for step,x,whole in series:
                    if whole is None:
                        whole=[r for r in prior if r['label'].startswith('frozen1800-') and r['seed'] in (6181031,6281031)]
                    safety.append(min(r['strict_safe']/r['n'] for r in whole))
                    if step and dev==6181031:
                        hard=sum(r['joint_limit']+r['nonfoot'] for r in whole);retained=all(all(r.get('retention_checks',{}).values()) for r in whole)
                        points.append(dict(family=family,training_seed=training,additional_updates=step,checkpoint_sha256=x['checkpoint_sha256'],hard_nonfoot_events=hard,original_retention=retained,worst_strict_safe_fraction=safety[-1]))
                        for col,value in enumerate([x['X_after1s']['synchronous_gain'],x['X_after1s']['actual_fitted_amplitude_m']*1000,x['X_after1s']['correlation'],safety[-1]]):
                            if hard:axes[rowno,col].scatter([step],[value],s=90,facecolors='none',edgecolors='#c33e3e',zorder=5)
                            elif not retained:axes[rowno,col].scatter([step],[value],s=55,marker='^',color='#d49d31',zorder=5)
                if dev==6181031:axes[rowno,3].plot([n for n,r,w in series],safety,color=colors[fi%len(colors)],marker=marker,label=family)
        for ax,label,gate in zip(axes[rowno],['X synchronous gain','Real X amplitude (mm)','X correlation','Worst full-matrix strict-safe fraction'],[.35,8,.5,.95]):
            ax.axhline(gate,c='#ae5a37',ls=':',lw=1);ax.set(xlabel='Additional common-parent PPO updates',ylabel=label);ax.grid(alpha=.2)
        axes[rowno,0].set_title('Independent continuation trainingseed'+str(training),fontsize=10);axes[rowno,0].legend(fontsize=5.5)
        axes[rowno,3].set_ylim(.8,1.01)
    fig.suptitle('Original20s physics: red rings=actual hard/contact; amber=retention failure. Tracking uses original after1s mask; safety includes startup.',fontsize=10)
    target=out/'STAGE3B_PHYSICAL_LEARNING.svg';fig.savefig(target,metadata=dict(Date=None));plt.close(fig)
    record=dict(file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,points=points,
        scope='Separate training seeds and physics seeds; no frozen-composition diagnostic scores; no extrapolation beyond evaluated native control checkpoints.')
    target.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.root,a.out)
