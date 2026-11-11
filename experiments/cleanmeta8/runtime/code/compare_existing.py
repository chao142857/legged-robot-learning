"""Final CPU-only comparison against preserved seed48 results; no training input."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import numpy as np
from common import ROOT,write,file_sha

def main():
    baseline=ROOT/'comparison_baselines.json';a=json.loads(baseline.read_text())
    identity=json.loads((ROOT/'evaluation_identity_test.json').read_text())
    assert identity['pool_sha256']=='d48b5850285e26b9cec447a0bb4bd6c84493afec5efa0c7725ddcbacbd714afa'
    assert identity['seed']==922042 and identity['episodes']==1024
    points=[]
    for p in sorted((ROOT/'evaluations').glob('transfer-seed48-new-meta-*-test/result.json')):
        r=json.loads(p.read_text())
        assert r['passed'] and r['kind']=='test' and r['training_history_updates']==0 and r['model_weights_unchanged']
        assert {k:r[k] for k in identity}==identity
        points.append(dict(update=r['update'],joint_success_rate=r['joint_success_rate'],continuous_joint_score=r['continuous_joint_score'],pooled_rmse=r['pooled_rmse'],native_failure_rate=r['native_failure_rate'],source_path=str(p),source_sha256=file_sha(p)))
    assert [p['update'] for p in points]==list(range(1000,10001,1000))
    existing=[s for s in a['series'] if s['student_seed']==48]
    def area(ps):return float(np.trapezoid([p['joint_success_rate'] for p in ps],x=[p['update'] for p in ps])/9000) if hasattr(np,'trapezoid') else float(np.trapz([p['joint_success_rate'] for p in ps],x=[p['update'] for p in ps])/9000)
    final=points[-1];comparisons={}
    for s in existing:
        assert [p['update'] for p in s['points']]==list(range(1000,10001,1000))
        old=s['points'][-1]
        comparisons[s['arm']]=dict(strict_final_difference_pp=100*(final['joint_success_rate']-old['joint_success_rate']),continuous_final_difference_pp=100*(final['continuous_joint_score']-old['continuous_joint_score']),ee_position_rmse_difference_cm=100*(final['pooled_rmse']['ee_position_m']-old['pooled_rmse']['ee_position_m']),strict_area_difference_pp=100*(area(points)-area(s['points'])))
    write(ROOT/'results/comparison_existing_seed48.json',dict(passed=True,new_arm='matched-source-continued-meta',student_seed=48,points=points,comparisons=comparisons,baseline_sha256=file_sha(baseline),test_identity=identity,interpretation='Exploratory follow-up on already-inspected seed48. Same-GPU Uniform initialization matched; prior Teacher arms trained on3060, this arm on5060. No significance or isolated data-count causality claim.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans'})
    fig,ax=plt.subplots(figsize=(11,6.4));fig.subplots_adjust(bottom=.22,right=.96)
    labels={'uniform':'Uniform (5060)','fresh':'Fresh frozen (3060)','old-meta':'Old Meta frozen (3060)','new-meta':'Seed46-continued Meta frozen (3060)'}
    colors={'uniform':'#64748b','fresh':'#d97706','old-meta':'#2563eb','new-meta':'#059669'}
    for s in existing:ax.plot([p['update']/1000 for p in s['points']],[p['joint_success_rate']*100 for p in s['points']],marker='o',label=labels[s['arm']],color=colors[s['arm']],alpha=.8)
    ax.plot([p['update']/1000 for p in points],[p['joint_success_rate']*100 for p in points],marker='o',lw=2.8,label='Matched43/44-continued Meta frozen (5060)',color='#9333ea')
    ax.set(xlabel='PPO updates (thousands)',ylabel='Strict joint success (%)',title='Student seed48 | matched-source Meta follow-up',xticks=range(1,11),ylim=(0,None));ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=9)
    fig.text(.10,.055,'Fixed1,024-sequence strict5cm/80% mean-action test; no smoothing.\nPreviously inspected Student seed48; GPU and PhysX reproducibility limits apply.',fontsize=9,color='#64748b')
    for suffix in ('svg','png'):fig.savefig(ROOT/'results'/('comparison_existing_seed48.'+suffix),dpi=200)
    plt.close(fig)
    print('FINAL_MATCHED_SOURCE_COMPARISON_PASS',flush=True)
if __name__=='__main__':main()
