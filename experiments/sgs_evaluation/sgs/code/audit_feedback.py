"""Read-only replay of real terminal ledgers against saved sampler histories."""
import collections,json,pathlib,time
import numpy as np,torch
from common import ROOT,write

results=[]
for run in sorted((ROOT/'runs').glob('short-*')):
    if not (run/'result.json').exists():continue
    result=json.loads((run/'result.json').read_text());cp=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    saved=cp['infos']['adapter']['owner'];n=saved['engine']['n'];counts=np.zeros(n,dtype=np.int64);success=np.zeros(n,dtype=np.int64)
    histories=[collections.deque(maxlen=100) for _ in range(n)];last_generation={};assignments={};terminals=0
    with (run/'assignments.jsonl').open() as f:
        for line in f:
            d=json.loads(line);assert d['pool_identity']==saved['pool_identity']
            for e,t,g in zip(d['env_ids'],d['task_ids'],d['generation']):
                key=(e,g);assert key not in assignments
                assignments[key]=t
    with (run/'terminal_feedback.jsonl').open() as f:
        for line in f:
            d=json.loads(line);assert d['pool_identity']==saved['pool_identity']
            for e,t,g,steps,row in zip(d['env_ids'],d['task_ids'],d['generation'],d['physical_steps'],d['rows']):
                assert assignments[(e,g)]==t and g>last_generation.get(e,0)
                last_generation[e]=g;assert steps>0 and steps>=row[0]
                y=int(row[6]);assert y in (0,1)
                if row[7] or row[0]<100:assert y==0
                counts[t]+=1;success[t]+=y;histories[t].append(y);terminals+=1
    assert counts.tolist()==saved['engine']['completed_counts'] and success.tolist()==saved['engine']['success_counts']
    assert [list(h) for h in histories]==saved['engine']['outcomes']
    assert terminals==saved['accepted'] and int(saved['transitions'].sum())==19660800
    row=dict(run=run.name,passed=True,real_terminal_feedback_records=terminals,all_ownership_tokens_match=True,no_duplicate_generations=True,complete_counts_and_success_counts_match_checkpoint=True,last100_history_per_sequence_matches_raw_ledger=True,environment_transitions=int(saved['transitions'].sum()))
    write(run/'real_feedback_ledger_audit.json',row);results.append(row)
write(ROOT/'audit/real_feedback_audits.json',dict(audited_unix=time.time(),results=results))
print(json.dumps(results),flush=True)
