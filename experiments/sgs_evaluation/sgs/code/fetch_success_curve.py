import json,pathlib,subprocess,shlex,datetime

out=pathlib.Path(__file__).resolve().parent/'success_curve_snapshot'
out.mkdir(exist_ok=True)
code=r'''
import pathlib,json,datetime
r=pathlib.Path('/data/go2_piper_sgs_20261102')
snapshot=dict(as_of_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),runs={})
for label in ('formal-flat-uniform','formal-flat-sgs','formal-wbc-uniform','formal-wbc-sgs'):
    evaluations={}
    for n in range(1000,10001,1000):
        p=r/'evaluations'/label/str(n).zfill(5)/'result.json'
        if p.exists():
            d=json.loads(p.read_text())
            assert d['passed'] and d['episodes']==1024 and d['train_history_updates']==0
            evaluations[str(n)]={k:d[k] for k in ('joint_success_rate','episodes','pool_sha256','definition_sha256','evaluation_seed')}
            evaluations[str(n)]['source']=str(p)
        else:evaluations[str(n)]=None
    snapshot['runs'][label]=evaluations
print(json.dumps(snapshot))
'''
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=3','-o','ControlPath=none','yanglab','python3 -c '+shlex.quote(code)]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=50)
if r.returncode:raise RuntimeError(r.stderr)
snapshot=json.loads(r.stdout)
for task in ('flat','wbc'):
    refs=[v for name,rows in snapshot['runs'].items() if ('-'+task+'-') in name for v in rows.values() if v is not None]
    assert len({v['pool_sha256'] for v in refs})==1
    assert len({v['evaluation_seed'] for v in refs})==1
all_refs=[v for rows in snapshot['runs'].values() for v in rows.values() if v is not None]
assert len({v['definition_sha256'] for v in all_refs})==1
(out/'joint_success_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print('AS_OF',snapshot['as_of_beijing'])
for n in range(1000,10001,1000):
    values=[]
    for rows in snapshot['runs'].values():
        v=rows[str(n)];values.append('' if v is None else f"{v['joint_success_rate']*100:.2f}%")
    print(n,*values,sep=' | ')
