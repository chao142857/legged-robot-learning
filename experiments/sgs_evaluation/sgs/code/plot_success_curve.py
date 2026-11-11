import argparse,json,pathlib,os,math
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(out/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,FuncFormatter

data=json.loads(pathlib.Path(a.input).read_text())
steps=np.arange(1000,10001,1000)
groups=[('formal-flat-uniform','Flat - Uniform','#0072B2','o','-'),
        ('formal-flat-sgs','Flat - SGS','#009475','s','--'),
        ('formal-wbc-uniform','WBC - Uniform','#D55E00','o','-'),
        ('formal-wbc-sgs','WBC - SGS','#A34F91','D','--')]
values={}
for key,_,_,_,_ in groups:
    values[key]=np.array([np.nan if data['runs'][key][str(n)] is None else data['runs'][key][str(n)]['joint_success_rate']*100 for n in steps])
    for n,v in zip(steps,values[key]):
        if not np.isnan(v):
            assert 0<=v<=100 and abs(v/100*1024-round(v/100*1024))<1e-8
    if key!='formal-wbc-sgs':assert np.isfinite(values[key]).sum()==10

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':'#26364A','text.color':'#203047','xtick.color':'#43526A','ytick.color':'#43526A','svg.fonttype':'path'})
fig=plt.figure(figsize=(12.4,10.5),facecolor='white')
fig.text(.085,.972,'Joint success rate at every 1,000 PPO updates',fontsize=20,fontweight='bold',va='top')
stamp=data['as_of_beijing'][:16].replace('T',' ')
fig.text(.085,.935,'Independent evaluation | 1,024 episodes per point | Snapshot: '+stamp+' (UTC+08:00)',fontsize=10.7,color='#65738A')
ax=fig.add_axes([.085,.455,.855,.385])
max_value=max(float(np.nanmax(v)) for v in values.values());upper=max(50,10*math.ceil((max_value+2)/10))
for key,label,color,marker,style in groups:
    y=values[key]
    ax.plot(steps,y,color=color,marker=marker,linestyle=style,linewidth=2.4,markersize=6.1,label=label,markeredgecolor='white',markeredgewidth=.7,zorder=4)
    finite=np.flatnonzero(np.isfinite(y))
    if len(finite) and finite[-1]==9:
        ax.annotate(f'{y[-1]:.2f}%',(10000,y[-1]),xytext=(9,0),textcoords='offset points',color=color,va='center',fontsize=10.5,fontweight='bold')
ax.set_xlim(800,10900);ax.set_ylim(0,upper)
ax.set_xticks(steps);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{int(x):,}'))
ax.set_yticks(np.arange(0,upper+1,10));ax.yaxis.set_major_formatter(PercentFormatter(xmax=100,decimals=0))
ax.set_xlabel('Completed PPO updates',fontsize=12,labelpad=10);ax.set_ylabel('Joint success rate',fontsize=12,labelpad=9)
ax.grid(axis='y',color='#DDE5EE',linewidth=.8,zorder=0);ax.grid(axis='x',color='#EDF1F5',linewidth=.5,zorder=0)
for side in ('top','right'):ax.spines[side].set_visible(False)
for side in ('left','bottom'):ax.spines[side].set_color('#B7C4D4')
ax.tick_params(axis='both',labelsize=10.5,length=0,pad=7)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.51,.907),ncol=4,frameon=False,fontsize=11,handlelength=2.8,columnspacing=1.6)
last=np.flatnonzero(np.isfinite(values['formal-wbc-sgs']))
if len(last)<10:
    checkpoint=int(steps[last[-1]]) if len(last) else 0
    ax.text(.985,.045,f'WBC - SGS: available through {checkpoint:,} updates\nLater checkpoints left blank',transform=ax.transAxes,ha='right',va='bottom',fontsize=10.5,color='#6F5770',bbox=dict(facecolor='white',edgecolor='none',pad=5))

table_ax=fig.add_axes([.085,.085,.855,.292]);table_ax.axis('off')
headers=['PPO updates']+[g[1] for g in groups]
rows=[]
for i,n in enumerate(steps):rows.append([f'{n:,}']+['' if np.isnan(values[k][i]) else f'{values[k][i]:.2f}%' for k,_,_,_,_ in groups])
table=table_ax.table(cellText=rows,colLabels=headers,cellLoc='center',colLoc='center',bbox=[0,0,1,1],colWidths=[.16,.21,.21,.21,.21])
table.auto_set_font_size(False);table.set_fontsize(11.3)
for (row,col),cell in table.get_celld().items():
    cell.set_edgecolor('#DDE5EE');cell.set_linewidth(.55)
    if row==0:
        cell.set_facecolor('#24354D' if col==0 else groups[col-1][2]);cell.set_text_props(color='white',fontweight='bold',fontsize=11)
    else:
        cell.set_facecolor('#F2F6FA' if row%2==1 else 'white');cell.set_text_props(color='#26364A',fontweight='bold' if col==0 else 'normal')
        if col==4 and rows[row-1][col]=='':cell.set_facecolor('#F9FAFC')
fig.text(.085,.397,'Checkpoint values',fontweight='bold',fontsize=12)
fig.text(.085,.045,'Same fixed evaluation set and random seed within each task. Values are percentages.',fontsize=10,color='#65738A')
fig.text(.085,.024,'Blank cells = no completed evaluation at snapshot time. No smoothing or extrapolation.',fontsize=10,color='#65738A')
for extension in ('png','svg'):fig.savefig(out/('joint_success_by_1000.'+extension),dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.16)
plt.close(fig)
verification=dict(as_of_beijing=data['as_of_beijing'],points={k:int(np.isfinite(v).sum()) for k,v in values.items()},blank_cells=int(np.isnan(values['formal-wbc-sgs']).sum()),table=rows,episodes_per_point=1024,missing_values_preserved=True)
(out/'plot_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification),flush=True)
for path in out.glob('joint_success_by_1000.*'):print(path,path.stat().st_size,flush=True)
