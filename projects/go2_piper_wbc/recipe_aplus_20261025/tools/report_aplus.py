"""Frozen V1 gates and explicit medium-command retention for A+ checkpoints."""
from publication_paths import research_path
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from report_recipe_a import capability,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1];gate=json.loads((source/'recipe_v1/spec.json').read_text())['A_to_B_gate']
    spec=json.loads((source/'recipe_aplus/spec.json').read_text())
    baseline=json.loads(Path(research_path('go2_piper_recipe_v1_20261025/report_A_recovery_v2/physical_results.json')).read_text())
    stages=json.loads((root/'aplus_evaluations.json').read_text());assert stages['state']=='complete' and len(stages['stages'])==21
    rows=[capability(s,gate) for s in stages['stages']]
    for r in rows:
        r['retention_checks']={}
        if r['protocol']=='A':
            b=next(x for x in baseline if x['label']=='cp_1200' and x['protocol']=='A' and x['seed']==r['seed'])
            g=spec['evaluation']['retention_relative_to_parent'];checks=r['retention_checks']
            checks['completion']=r['completion_fraction']>=b['completion_fraction']-g['A_completion_drop_max']
            checks['ee']=r['ee_rmse_m']<=b['ee_rmse_m']+g['A_ee_rmse_increase_max_m']
            for d in ['forward','backward','left','right']:checks[d]=r['directions'][d]['gain'] is not None and abs(r['directions'][d]['gain']-b['directions'][d]['gain'])<=g['A_direction_gain_absolute_change_max']
        r['retention_pass']=all(r['retention_checks'].values())
    nodes={n:all(r['gate_pass'] and r['retention_pass'] for r in rows if r['label']==f'cp_{n}' and r['protocol'] in ['A','T1']) for n in [1400,1600,1800]}
    pairs=[list(pair) for pair in [(1400,1600),(1600,1800)] if all(nodes[n] for n in pair)]
    decision=dict(node_pass=nodes,passed_pairs=pairs,latest_policy_pass=nodes[1800],low_speed_gate_pass=bool(pairs) and nodes[1800],
        B_started=False,B_requires_dynamic_physical_admission_and_preregistration=True,old_V1_gates_unchanged=True)
    for name,obj in [('physical_results.json',rows),('gate_decision.json',decision)]: (out/name).write_text(json.dumps(obj,indent=2)+'\n')
    train=[json.loads(x) for x in (root/'train_seed21/updates.jsonl').read_text().splitlines()]
    assert [x['completed_updates'] for x in train]==list(range(1201,1801))
    audit=[]
    for n in [1400,1600,1800]:
        cp=root/f'train_seed21/cp_{n}.pt';row=json.loads(cp.with_name(cp.stem+'_cpu_audit.json').read_text())
        assert row['finite'] and row['adam_step_min']==row['adam_step_max']==n*20 and digest(cp)==row['sha256'];audit.append(row)
    (out/'training_verification.json').write_text(json.dumps(dict(parent=1200,completed_updates=1800,new_updates=600,new_transitions=58982400,cumulative_transitions=176947200,checkpoints=audit),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for protocol in ['A','T1']:
        fig,axs=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
        series=[(d+' gain',lambda r,d=d:r['directions'][d]['gain']) for d in ['forward','backward','left','right']]+[('EE RMSE (m)',lambda r:r['ee_rmse_m']),('20s completion',lambda r:r['completion_fraction'])]
        for ax,(title,f) in zip(axs.flat,series):
            for seed in [8101,8102,8103]:
                rr=[next(x for x in baseline if x['label']=='cp_1200' and x['protocol']==protocol and x['seed']==seed)]+[next(x for x in rows if x['label']==f'cp_{n}' and x['protocol']==protocol and x['seed']==seed) for n in [1400,1600,1800]]
                ax.plot([1200,1400,1600,1800],[f(x) for x in rr],'-o',label=f'seed{seed}')
            ax.set_title(title);ax.grid(alpha=.25);ax.set_xlabel('Cumulative completed updates')
        axs[0,0].legend();fig.suptitle('A+ continuation, frozen '+protocol);fig.savefig(out/f'capability_{protocol}.png',dpi=150);plt.close(fig)
    lines=['# A+冻结能力验收','',f'低速能力与A保持门：{decision["low_speed_gate_pass"]}；本文件不代替B物理准入。','',
        '|策略/协议|完成|EE(cm)|前/后/左/右gain|失败项|','|---|---:|---:|---|---|']
    for r in rows:
        gains='/'.join('--' if x['gain'] is None else f'{x["gain"]:.3f}' for x in r['directions'].values())
        lines.append(f'|{r["label"]}/{r["protocol"]}/s{r["seed"]}|{r["complete_count"]}/64|{r["ee_rmse_m"]*100:.2f}|{gains}|'+','.join(r['failed_checks']+[k for k,v in r['retention_checks'].items() if not v])+'|')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision))


if __name__=='__main__':main()
