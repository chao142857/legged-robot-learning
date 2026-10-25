"""Produce compact paired tables from completed microscope JSON; no torch needed."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics as st


def run(root):
    meta=json.loads((root/'METADATA.json').read_text())
    if not meta.get('completed'):raise ValueError('Only summarize completed evidence')
    rows=json.loads((root/'RESULTS.json').read_text());cal=json.loads((root/'CALIBRATION.json').read_text())
    clip_path=root/'CLIP_DECOMPOSITION.json'
    clip_fields=['unclipped_loss_change','clipping_penalty_change','unclipped_nonlinear_remainder']
    if clip_path.exists():
        clip={(r['fold'],r['window'],r['partition'],r['scale']):r for r in json.loads(clip_path.read_text())['rows']}
        for row in rows:
            cr=clip[row['fold'],row['window_start'],row['partition'],row['lr_scale']]
            row.update({'heldout_native_'+k:cr[k] for k in clip_fields})
    pairs=[]
    for key,grp in itertools.groupby(sorted(rows,key=lambda r:(r['fold'],r['window_start'],r['partition'])),lambda r:(r['fold'],r['window_start'],r['partition'])):
        full,small=sorted(grp,key=lambda r:-r['lr_scale'])
        if (full['lr_scale'],small['lr_scale'])!=(1.,.1):raise ValueError('Missing paired candidate')
        p=dict(fold=key[0],window=key[1],partition=key[2],**full['roles'],minibatch_n=full['minibatch_n'])
        for label,r in [('full',full),('small',small)]:
            p.update({label+'_'+k:r[k] for k in ['lr_actual','actor_delta_norm','critic_delta_norm','cosine_delta_negative_policy_grad',
                'cosine_delta_negative_actor_loss_grad','native_policy_g_dot_delta','actor_loss_g_dot_delta','native_actor_loss_grad_norm',
                'critic_grad_norm','update_kl_pre','policy_gradient_pre_post_cosine','heldout_policy_gradient_pre_post_cosine',
                'heldout_native_actual_loss_change','heldout_native_first_order_loss_change','heldout_native_nonlinear_remainder',
                'heldout_native_change_per_delta_norm']})
            for k in clip_fields:
                if 'heldout_native_'+k in r:p[label+'_heldout_native_'+k]=r['heldout_native_'+k]
            for k in ('kl_mean','kl_max','clip_fraction','active_clip_fraction'):
                p[label+'_heldout_'+k]=r['heldout_after'][k]
            for e,est in enumerate(meta['estimators']):
                for j,obj in enumerate(('L','M','S')):
                    for mode,short in [('raw','raw'),('centered_native_std','norm')]:
                        for measure,tag in [('actual_loss_change','actual'),('first_order_loss_change','predicted'),('nonlinear_remainder','nonlinear')]:
                            p[f'{label}_{est}_{short}_{obj}_{tag}']=r['heldout_changes'][mode][measure][e][j]
            for i,name in enumerate(r['heldout_uncertainty']['labels']):
                lo,hi=r['heldout_uncertainty']['ci95'][i]
                p[label+'_'+name+'_conditional_ci_low']=lo;p[label+'_'+name+'_conditional_ci_high']=hi
        p.update(delta_pair_cosine=full['paired_delta_cosine'],delta_norm_ratio=full['paired_delta_norm_ratio'],lr_ratio=full['paired_actual_lr_ratio'])
        pairs.append(p)
    with (root/'PAIRED_COMPARISON.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]),lineterminator='\n');writer.writeheader();writer.writerows(pairs)
    keys=['actor_delta_norm','cosine_delta_negative_policy_grad','cosine_delta_negative_actor_loss_grad','heldout_kl_mean','heldout_clip_fraction',
        'heldout_native_actual_loss_change','heldout_native_first_order_loss_change','heldout_native_nonlinear_remainder',
        'probe_gae_finite_raw_L_actual','probe_gae_finite_raw_M_actual','probe_gae_finite_raw_S_actual']
    if clip_path.exists():keys.extend('heldout_native_'+k for k in clip_fields)
    summary=[]
    for w in meta['starts']:
        subset=[p for p in pairs if p['window']==w]
        entry=dict(window=w,paired_minibatches=len(subset),folds=3,metrics={})
        for label in ('full','small'):
            entry['metrics'][label]={}
            for k in keys:
                seedmeans=[st.mean(p[label+'_'+k] for p in subset if p['fold']==fold) for fold in range(3)]
                entry['metrics'][label][k]=dict(mean=st.mean(seedmeans),seed_mean_range=[min(seedmeans),max(seedmeans)],seed_means=seedmeans)
        entry['counts']=dict(
            full_native_loss_improved=sum(p['full_heldout_native_actual_loss_change']<0 for p in subset),
            small_native_loss_improved=sum(p['small_heldout_native_actual_loss_change']<0 for p in subset),
            full_native_linear_prediction_improved=sum(p['full_heldout_native_first_order_loss_change']<0 for p in subset),
            full_overshoot=sum(p['full_heldout_native_first_order_loss_change']<0<p['full_heldout_native_actual_loss_change'] for p in subset),
            small_overshoot=sum(p['small_heldout_native_first_order_loss_change']<0<p['small_heldout_native_actual_loss_change'] for p in subset))
        summary.append(entry)
    cal_summary=[]
    for w in meta['calibration_starts']:
        subset=[x for x in cal if x['start']==w]
        cal_summary.append(dict(window=w,rows=len(subset),ranges={k:[min(x[k] for x in subset),max(x[k] for x in subset)] for k in
            ['eligible_envs','native_vs_probe_current_corr','native_vs_probe_finite_corr','raw_baseline_difference_rms','raw_timeout_difference_rms',
             'finite_baseline_difference_rms','native_timeout_removal_rms','standardized_baseline_rms','standardized_timeout_rms',
             'normalizer_change_rms','decomposition_max_error','native_reference_max_error']}))
    result=dict(paired_rows=len(pairs),candidate_rows=len(rows),summary=summary,calibration=cal_summary,
        convention='Mean of 3 seed-role means (4 minibatches each); ranges are across 3 seed-role means, not confidence intervals',
        delta_ratio_range=[min(p['delta_norm_ratio'] for p in pairs),max(p['delta_norm_ratio'] for p in pairs)],
        delta_pair_cosine_min=min(p['delta_pair_cosine'] for p in pairs),lr_ratios=sorted(set(p['lr_ratio'] for p in pairs)))
    (root/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['| 窗口起点 | LR组 | ‖Δθactor‖ | cos(Δθ,−gactor) | 持出KL | clip比例 | 持出native Δloss | 一阶预测 | 非线性余项 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        for label in ('full','small'):
            m=row['metrics'][label]
            values=[m[k]['mean'] for k in ['actor_delta_norm','cosine_delta_negative_actor_loss_grad','heldout_kl_mean','heldout_clip_fraction',
                'heldout_native_actual_loss_change','heldout_native_first_order_loss_change','heldout_native_nonlinear_remainder']]
            lines.append('| '+str(row['window'])+' | '+label+' | '+' | '.join(f'{v:.6g}' for v in values)+' |')
    (root/'PAIRED_TABLE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('paired_rows','candidate_rows','delta_ratio_range','delta_pair_cosine_min','lr_ratios')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');a=p.parse_args();run(Path(a.root))
