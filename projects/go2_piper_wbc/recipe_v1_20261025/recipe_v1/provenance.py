"""Read saved YAML trees without importing executable training entry points."""
from pathlib import Path
import yaml


def yaml_tree(path):
    def node(n):
        if isinstance(n,yaml.MappingNode):return {node(k):node(v) for k,v in n.value}
        if isinstance(n,yaml.SequenceNode):return [node(x) for x in n.value]
        if n.tag.startswith('tag:yaml.org,2002:python/name:'):return n.tag
        return n.value
    return node(yaml.compose(Path(path).read_text()))


def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        result=[]
        for k in sorted(set(a)|set(b)):result+=differences(a.get(k),b.get(k),path+'.'+k)
        return result
    return [{'path':path,'native':a,'recipe_v1':b}] if a!=b else []
