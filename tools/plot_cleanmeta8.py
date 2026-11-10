"""Regenerate the published two-seed figure from lightweight verified JSON."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=ROOT/'projects/sgs/cleanmeta8_20261109/evidence/completed_curves.json')
 parser.add_argument('--output',type=Path,default=ROOT/'docs/figures/cleanmeta8_learning_curves.svg')
 parser.add_argument('--preview',type=Path);args=parser.parse_args()
 data=json.loads(args.data.read_text());series={(x['student_seed'],x['method']):x for x in data['series']}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','svg.hashsalt':'cleanmeta8-public-20261110'})
 fig,axes=plt.subplots(1,2,figsize=(11.8,4.7),sharey=True)
 styles=[('uniform','Uniform','#2563eb','o'),('fresh','Fresh','#64748b','s'),('cleanmeta8','CleanMeta8 (corrected)','#df5b36','^')]
 for ax,seed in zip(axes,(48,49)):
  for method,label,color,marker in styles:
   points=series[seed,method]['points'];assert [x['update'] for x in points]==list(range(1000,10001,1000))
   ax.plot([x['update']/1000 for x in points],[100*x['joint_success_rate'] for x in points],label=label,color=color,marker=marker,markersize=4.5,linewidth=2)
  ax.set_title(f'Student seed {seed}',fontweight='bold',pad=10)
  ax.set_xlim(.7,10.3);ax.set_ylim(0,45);ax.set_xticks(range(1,11));ax.set_yticks(range(0,46,5))
  ax.set_xlabel('PPO updates (thousands)');ax.grid(axis='y',color='#e2e8f0',linewidth=.7);ax.set_axisbelow(True)
  for side in ('top','right'):ax.spines[side].set_visible(False)
  for side in ('bottom','left'):ax.spines[side].set_color('#cbd5e1')
 axes[0].set_ylabel('Strict joint success (%)')
 fig.suptitle('Corrected CleanMeta8: completed 10k transfers',fontsize=15,fontweight='bold',y=.975)
 fig.text(.5,.907,'5 cm / 80% criterion · 1,024 fixed test sequences · mean actions',ha='center',fontsize=10,color='#475569')
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.875),ncol=3,frameon=False)
 fig.text(.065,.068,'Seed48: Uniform/CleanMeta8 on RTX5060 Laptop; Fresh on RTX3060. Seed49: all methods on RTX3060.',fontsize=8,color='#475569')
 fig.text(.065,.027,'One corrected Teacher, two Student seeds; reused baselines. Simulation only. Mixed final-checkpoint advantage.',fontsize=8,color='#475569')
 fig.subplots_adjust(left=.075,right=.985,bottom=.205,top=.72,wspace=.13)
 args.output.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(args.output,metadata={'Date':'2026-11-10','Creator':'Legged Robot Learning; tools/plot_cleanmeta8.py'})
 if args.preview:fig.savefig(args.preview,dpi=140,facecolor='white')
 plt.close(fig)
 print(args.output)
if __name__=='__main__':main()
