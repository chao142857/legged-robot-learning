"""Resolve optional, independently obtained experiment inputs.

This module never locates credentials, connects to a machine, downloads assets,
or silently falls back to the author's private filesystem.
"""
import os
from pathlib import Path


def research_path(relative):
    relative = str(relative).removeprefix('private-runtime/')
    overrides = {
        'LeggedManip_Lab_upstream_wbc_20261023': 'LEGGED_NATIVE_CHECKOUT',
        'workspace/basic-locomotion-isaaclab': 'LEGGED_IIT_CHECKOUT',
        'workspace/IsaacLab': 'LEGGED_ISAACLAB_CHECKOUT',
    }
    for prefix, variable in overrides.items():
        if relative == prefix or relative.startswith(prefix + '/'):
            value = os.environ.get(variable)
            if value:
                return str(Path(value) / relative[len(prefix):].lstrip('/'))
    root = os.environ.get('LEGGED_RUNTIME_ROOT')
    if not root:
        raise RuntimeError('Set LEGGED_RUNTIME_ROOT to your own external experiment inputs; '
                           'this release does not include private runtime data.')
    return str(Path(root) / relative)
