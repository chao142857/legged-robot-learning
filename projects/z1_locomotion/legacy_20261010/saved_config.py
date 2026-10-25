"""Read our own YAML snapshots without executing Python-tag constructors."""
from pathlib import Path
import yaml


class SavedConfigLoader(yaml.SafeLoader):
    pass


def python_metadata(loader,suffix,node):
    if isinstance(node,yaml.SequenceNode):
        value = loader.construct_sequence(node,deep=True)
    elif isinstance(node,yaml.MappingNode):
        value = loader.construct_mapping(node,deep=True)
    else:
        value = loader.construct_scalar(node)
    return {'serialized_python_tag':suffix,'value':value}


SavedConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',
    lambda loader,node:tuple(loader.construct_sequence(node,deep=True)))
SavedConfigLoader.add_multi_constructor('tag:yaml.org,2002:python/',python_metadata)


def load_saved_config(path):
    return yaml.load(Path(path).read_text(),Loader=SavedConfigLoader)
