"""Frozen development selection; no unseen-seed peeking or mixed lineages."""
from gpu_budget import read
from stage3_pipeline import report
def selection(b):
    curve=report(b,'before_final');nodes=read(b.p/'CANDIDATE_NODES.json') if (b.p/'CANDIDATE_NODES.json').exists() else []
    options=[]
    for node in nodes:
        rows=[r for r in curve['rows'] if r['label'].startswith(node['evaluation_prefix']+'-')]
        if len(rows)!=8:continue
        assert all(r['checkpoint_sha256']==node['checkpoint']['checkpoint_sha256'] for r in rows)
        assert all({r['protocol'] for r in rows if r['seed']==s}=={'A','T0','T1','X'} for s in (6181031,6281031))
        xs=[r for r in rows if r['protocol']=='X']
        safe=all(r['joint_limit']==r['nonfoot']==0 and r['strict_safe']/r['n']>=.95 and r['complete']/r['n']>=.95 for r in rows)
        retained=all(all(r.get('retention_checks',{}).values()) for r in rows)
        options.append(dict(node=node,safe=safe,registered_retention=retained,min_X_gain=min(r['X_after1s']['synchronous_gain'] for r in xs),
            original_all_development_gates=node['decision']['original_all_development_gates']))
    by_sha={v['node']['checkpoint']['checkpoint_sha256']:v for v in options}
    for option in options:
        node=option['node']
        earlier=[n for n in nodes if n['experiment_id']==node['experiment_id'] and n['training_seed']==node['training_seed']
            and n['recipe_id']==node['recipe_id'] and n['additional_actual_updates']<node['additional_actual_updates']]
        previous=max(earlier,key=lambda n:n['additional_actual_updates']) if earlier else None
        previous_option=None if previous is None else by_sha.get(previous['checkpoint']['checkpoint_sha256'])
        option['previous_development_node']=None if previous is None else previous['label']
        option['consecutive_development_pass']=bool(option['safe'] and option['original_all_development_gates'] and previous_option
            and previous_option['safe'] and previous_option['original_all_development_gates'])
    admitted=read(b.p/'FINAL_RECIPE_ADMISSION.json')['recipe_ids'] if (b.p/'FINAL_RECIPE_ADMISSION.json').exists() else None
    pool=[v for v in options if v['safe'] and (admitted is None or v['node']['recipe_id'] in admitted)]
    if admitted is not None:assert pool, 'Explicit replicated-recipe admission has no safe physical candidate'
    if pool:
        best=max(pool,key=lambda v:(v['consecutive_development_pass'],v['original_all_development_gates'],v['registered_retention'],v['min_X_gain']))
        node=best['node'];candidate=dict(node['checkpoint'],labels=[node['evaluation_prefix']],recipe_id=node['recipe_id'],training_seed=node['training_seed'],
            safe=best['safe'],registered_retention=best['registered_retention'],min_X_gain=best['min_X_gain'],actual_checkpoint_iteration=node['actual_checkpoint_iteration'],
            consecutive_development_pass=best['consecutive_development_pass'])
        earlier=[n for n in nodes if n['experiment_id']==node['experiment_id'] and n['training_seed']==node['training_seed']
            and n['recipe_id']==node['recipe_id'] and n['additional_actual_updates']<node['additional_actual_updates']]
        previous=max(earlier,key=lambda n:n['additional_actual_updates']) if earlier else None
        adjacent=None if previous is None else dict(checkpoint=previous['checkpoint'],labels=[previous['evaluation_prefix']],frozen_before_final_seeds=True)
    else:
        old=read(b.p/'HANDOFF.json')['selected_safe_development_node']
        candidate=dict(old['checkpoint'],labels=[old['label']],reason='No new safe full-matrix candidate; retain verified safe original-retained historical initialization',not_new_learning=True)
        adjacent=None
    return dict(candidate=candidate,adjacent=adjacent,all_development_candidates=options,used_final_seeds=False,selection_version='consecutive_original_gates_with_recipe_admission_v3',admitted_recipe_ids=admitted,
        rule='Before new formal seeds: require full8 matrices, zero actual hard/nonfoot events and>=95%safe/complete each; prioritize a current/adjacent pair already passing all original development gates, then single-node original gates, original retention and worst X gain. Both same-experiment/trainingseed nodes must still independently pass the unchanged formal protocols; no capability mixing.')
