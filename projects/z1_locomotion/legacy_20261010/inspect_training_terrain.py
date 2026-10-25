"""Audit actual generated box surfaces and the native curriculum from a saved training config."""
from publication_paths import research_path
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import numpy as np
from isaaclab.terrains.trimesh import mesh_terrains, mesh_terrains_cfg
from saved_config import load_saved_config
from stair_terrain import StaircaseCfg, staircase

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--env-config',type=Path,required=True)
parser.add_argument('--height',type=float,default=.18)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--repository',type=Path,default=Path(research_path('workspace/basic-locomotion-isaaclab')))
parser.add_argument('--isaaclab',type=Path,default=Path(research_path('workspace/IsaacLab')))
args = parser.parse_args()
cfg = load_saved_config(args.env_config)
generator = cfg['terrain']['terrain_generator']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(meshes, origin, size):
    # These constructors return axis-aligned closed boxes; their top faces are exact surfaces.
    bounds = np.array([mesh.bounds for mesh in meshes])
    assert np.isfinite(bounds).all()
    y = origin[1]
    bounds = bounds[(bounds[:,0,1] <= y) & (bounds[:,1,1] >= y)]
    edges = sorted({float(origin[0]),float(size[0]),
                    *(float(x) for x in bounds[:,:,0].ravel() if origin[0] < x < size[0])})
    segments = []
    for left,right in zip(edges,edges[1:]):
        if right-left < 1e-7:
            continue
        x = (left+right)/2
        selected = bounds[(bounds[:,0,0] <= x) & (bounds[:,1,0] >= x)]
        assert len(selected), (x,y)
        height = float(selected[:,1,2].max())
        segments.append({'from_spawn_m':[left-float(origin[0]),right-float(origin[0])],
                         'surface_z_m':height,'relative_surface_z_m':height-float(origin[2])})
    assert abs(segments[0]['relative_surface_z_m']) < 1e-6
    changes = np.diff([segment['surface_z_m'] for segment in segments])
    return {'origin_m':np.asarray(origin).tolist(),'positive_x_centerline':segments,
            'outward_up_steps':int((changes > 1e-6).sum()),
            'outward_down_steps':int((changes < -1e-6).sum()),'mesh_count':len(meshes)}


surfaces = {}
for name, cls, function in (
    ('pyramid_stairs',mesh_terrains_cfg.MeshPyramidStairsTerrainCfg,mesh_terrains.pyramid_stairs_terrain),
    ('pyramid_stairs_inv',mesh_terrains_cfg.MeshInvertedPyramidStairsTerrainCfg,mesh_terrains.inverted_pyramid_stairs_terrain)):
    source = generator['sub_terrains'][name]
    assert source['step_height_range'][0] <= args.height <= source['step_height_range'][1]
    terrain = cls(size=tuple(source['size']),step_height_range=(args.height,args.height),
        step_width=source['step_width'],platform_width=source['platform_width'],
        border_width=source['border_width'],holes=source['holes'])
    assert not terrain.holes, 'Box-bound profile is intended for the actual closed stair tiles'
    meshes, origin = function(.5,terrain)
    surfaces[name] = profile(meshes,origin,terrain.size)
fixture = StaircaseCfg(size=(8,3),step_height=args.height,descending=True)
meshes,origin = staircase(.5,fixture)
surfaces['evaluation_eight_riser_descent'] = profile(meshes,origin,fixture.size)
assert surfaces['pyramid_stairs']['outward_down_steps'] > 0
assert surfaces['pyramid_stairs_inv']['outward_up_steps'] > 0
assert surfaces['evaluation_eight_riser_descent']['outward_down_steps'] == 8

names = list(generator['sub_terrains'])
weights = np.array([generator['sub_terrains'][name]['proportion'] for name in names],dtype=float)
cumulative = np.cumsum(weights/weights.sum())
columns = ([names[int(np.where(index/generator['num_cols']+.001 < cumulative)[0][0])]
            for index in range(generator['num_cols'])] if generator['curriculum'] else None)
env_source = args.repository/'source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks/locomotion/locomotion_env.py'
geometry_source = Path(inspect.getsourcefile(mesh_terrains.pyramid_stairs_terrain))
result = {'saved_config':str(args.env_config),'saved_config_sha256':sha(args.env_config),
    'iit_commit':subprocess.check_output(['git','-C',str(args.repository),'rev-parse','HEAD'],text=True).strip(),
    'isaaclab_commit':subprocess.check_output(['git','-C',str(args.isaaclab),'rev-parse','HEAD'],text=True).strip(),
    'env_source_sha256':sha(env_source),'geometry_source_sha256':sha(geometry_source),
    'fixture_source_sha256':sha(Path(__file__).parent/'stair_terrain.py'),
    'audit_source_sha256':sha(Path(__file__)),'sample_height_m':args.height,
    'curriculum_enabled':generator['curriculum'],'terrain_columns':columns,
    'curriculum_error_thresholds_percent':{
        'up':cfg.get('terrain_curriculum_move_up_error_percent',20.),
        'down':cfg.get('terrain_curriculum_move_down_error_percent',50.)},
    'terrain_column_counts':{name:columns.count(name) for name in names} if columns is not None else None,
    'surfaces':surfaces,
    'source_review':'Pinned native _reset_idx advances/demotes using command-relative linear velocity error '
        'and the recorded thresholds, without an explicit native termination/complete-episode gate. '
        'Native orientation reward estimates terrain pitch from front/back ray heights; it is not simply an upright penalty.',
    'limits':'Static surface audit along outward +X from the spawn platform, not observed training trajectories. '
        'Random yaw, changing commands, radial motion and curriculum rows affect actual exposure. '
        'A representative 18cm mesh and source review do not prove locomotion success or a failure cause.'}
assert result['iit_commit'] == '8bda7c7df8b223acf649e0b212635a92c8528900'
assert result['isaaclab_commit'] == 'ae37b028ea415c91ea2bc32609efcd759ed2b974'
assert geometry_source.resolve().is_relative_to(args.isaaclab.resolve())
for repository, source in ((args.repository,env_source),(args.isaaclab,geometry_source)):
    relative = source.resolve().relative_to(repository.resolve())
    committed = subprocess.check_output(['git','-C',str(repository),'show','HEAD:'+str(relative)])
    assert hashlib.sha256(committed).hexdigest() == sha(source), 'Reviewed source differs from pinned Git blob'
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps({'columns':result['terrain_column_counts'],'surfaces':{
    name:{key:data[key] for key in ('origin_m','outward_up_steps','outward_down_steps')}
    for name,data in surfaces.items()}},indent=2))
