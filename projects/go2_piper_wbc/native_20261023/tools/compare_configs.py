"""Compare YAML nodes and tags without instantiating Python objects in saved configs."""
from publication_paths import research_path
import hashlib
import json
from pathlib import Path
import yaml

ROOT=Path(research_path('go2_piper_upstream_wbc_20261023'))
runs=list((ROOT/'native_logs/rsl_rl/go2_piper_wbc').iterdir())
native=next(p for p in runs if p.name.endswith('_smoke16'))
formal=next(p for p in runs if 'smoke' not in p.name and p.is_dir())


def flatten(node,path='',out=None):
    if out is None:out={}
    out[path]={'tag':node.tag}
    if isinstance(node,yaml.ScalarNode):out[path]['value']=node.value
    elif isinstance(node,yaml.SequenceNode):
        out[path]['length']=len(node.value)
        for i,v in enumerate(node.value):flatten(v,f'{path}[{i}]',out)
    elif isinstance(node,yaml.MappingNode):
        for k,v in node.value:
            assert isinstance(k,yaml.ScalarNode)
            flatten(v,f'{path}.{k.value}',out)
    return out


allowed={'env.yaml':{'.scene.num_envs','.scene.terrain.num_envs','.log_dir'},'agent.yaml':{'.max_iterations','.run_name'}}
result={}
for name in allowed:
    a=native/'params'/name;b=formal/'params'/name
    x=flatten(yaml.compose(a.read_text()));y=flatten(yaml.compose(b.read_text()))
    if name=='env.yaml':
        assert x['.scene.num_envs']==x['.scene.terrain.num_envs']
        assert y['.scene.num_envs']==y['.scene.terrain.num_envs']
        assert x['.scene.num_envs']['value']=='16' and y['.scene.num_envs']['value']=='4096'
    differences=[dict(path=k,plain_native_smoke=x.get(k),formal=y.get(k)) for k in sorted(x.keys()|y.keys()) if x.get(k)!=y.get(k)]
    assert all(d['path'] in allowed[name] for d in differences),differences
    result[name]=dict(native_path=str(a),formal_path=str(b),native_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
                     formal_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),differences=differences)
(ROOT/'audit/resolved_config_difference.json').write_text(json.dumps(dict(passed=True,configs=result),indent=2)+'\n')
print(json.dumps(result,indent=2))
