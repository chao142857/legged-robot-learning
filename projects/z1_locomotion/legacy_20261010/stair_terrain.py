"""Straight, continuous staircase test fixture with eight risers and level landings."""
import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def staircase(difficulty, cfg):
    h, n, w = cfg.step_height, cfg.num_steps, cfg.step_width
    length, width = cfg.size
    start = cfg.landing_length
    end = start+n*w
    meshes = []
    intervals = [(0,start,n*h if cfg.descending else 0)]
    intervals += [(start+i*w,start+(i+1)*w,
                   (n-i-1)*h if cfg.descending else (i+1)*h) for i in range(n)]
    intervals += [(end,length,0 if cfg.descending else n*h)]
    for x0,x1,z in intervals:
        # Closed adjacent prisms supply real vertical risers, not a heightfield approximation.
        thickness = z+0.25
        mesh = trimesh.creation.box(extents=(x1-x0,width,thickness))
        mesh.apply_translation(((x0+x1)/2,width/2,z-thickness/2))
        meshes.append(mesh)
    origin = np.array((0.9,width/2,n*h if cfg.descending else 0))
    return meshes,origin


@configclass
class StaircaseCfg(SubTerrainBaseCfg):
    function = staircase
    step_height: float = 0.08
    step_width: float = 0.30
    num_steps: int = 8
    landing_length: float = 1.8
    descending: bool = False
