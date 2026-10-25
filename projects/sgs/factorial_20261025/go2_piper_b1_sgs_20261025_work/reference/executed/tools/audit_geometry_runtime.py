"""Read-only initial-state audit against actual terrain mesh and USD collisions.

Call audit(env, out) after env/runner initialization. Does not step, reset, change
configuration or draw random samples. Full ray grids stay on the execution host.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_stats(values):
    a = np.asarray(values)
    finite = np.isfinite(a)
    return dict(count=int(a.size), finite_count=int(finite.sum()),
                minimum=float(a[finite].min()) if finite.any() else None,
                maximum=float(a[finite].max()) if finite.any() else None,
                mean=float(a[finite].mean()) if finite.any() else None)


def _foot_geometry(env):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    stage = Usd.Stage.Open(env.cfg.scene.robot.spawn.usd_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    result = {}
    for foot in stage.Traverse():
        if not foot.GetName().endswith('_foot'):
            continue
        collisions = []
        for prim in Usd.PrimRange(foot, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not prim.GetAttribute('physics:collisionEnabled').Get() is False:
                entry = dict(path=str(prim.GetPath()), type=prim.GetTypeName())
                if prim.IsA(UsdGeom.Sphere):
                    radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
                    relative = np.asarray(transforms.GetLocalToWorldTransform(prim) * transforms.GetLocalToWorldTransform(foot).GetInverse())
                    center = relative[3, :3]
                    axes = radius * relative[:3, :3]
                    extent = np.linalg.norm(axes, axis=0)
                    entry.update(sphere_radius_m=radius, local_sphere_center=center.tolist(),
                        local_sphere_axes=axes.tolist(), local_aabb_min=(center-extent).tolist(),
                        local_aabb_max=(center+extent).tolist(), bounds_method='analytic transformed Sphere')
                else:
                    try:
                        bound = cache.ComputeRelativeBound(prim, foot).ComputeAlignedRange()
                        lo, hi = np.asarray(bound.GetMin()), np.asarray(bound.GetMax())
                        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo>hi):
                            raise ValueError('USD returned empty or nonfinite collision bounds')
                        entry.update(local_aabb_min=lo.tolist(), local_aabb_max=hi.tolist(), bounds_method='USD collision AABB')
                    except Exception as exc:
                        entry['bounds_error'] = repr(exc)
                collisions.append(entry)
        if collisions:
            result[foot.GetName()] = collisions
    return result


def audit(env, out):
    """Write geometry.json + geometry_samples.npz and return JSON-safe summary."""
    import torch
    from isaaclab.utils.warp import raycast_mesh
    from isaaclab.utils.math import quat_apply
    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.usd
    out = Path(out)
    env = env.unwrapped
    meta = env.cfg.task_suite
    robot = env.scene['robot']
    is_t2 = meta['family'] == 'T2'
    scanner = env.scene.sensors.get('height_scanner')
    mesh = scanner.meshes['/World/ground'] if is_t2 else None
    stage = omni.usd.get_context().get_stage()
    ground = stage.GetPrimAtPath('/World/ground')
    ground_prims = [dict(path=str(p.GetPath()), type=p.GetTypeName(),
                        collision=bool(p.HasAPI(UsdPhysics.CollisionAPI)))
                    for p in Usd.PrimRange(ground) if p.GetTypeName() in ('Plane', 'Mesh')]
    planes = [p for p in Usd.PrimRange(ground) if p.GetTypeName() == 'Plane' and p.HasAPI(UsdPhysics.CollisionAPI)]
    # Native plane is an actual USD/PhysX Plane, so analytical intersection is
    # exact here; never substitute a plane for a generated terrain.
    if not is_t2:
        if len(planes) != 1:
            raise RuntimeError('Expected exactly one collision Plane in native flat terrain')
        plane = planes[0]
        matrix = np.asarray(UsdGeom.Xformable(plane).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        axis = plane.GetAttribute('axis').Get() or 'Z'
        normal = np.eye(3)['XYZ'.index(str(axis))] @ matrix[:3, :3]
        point = matrix[3, :3]
        if abs(normal[2]) < 1e-8:
            raise RuntimeError('Native collision plane cannot support vertical rays')
    def rays(xy):
        xy = np.asarray(xy, dtype=np.float32)
        if not is_t2:
            return point[2] - ((xy-point[:2])*normal[:2]).sum(-1)/normal[2]
        starts = torch.zeros((*xy.shape[:-1], 3), dtype=torch.float32, device=env.device)
        starts[..., :2] = torch.as_tensor(xy, device=env.device)
        starts[..., 2] = 10.
        directions = torch.zeros_like(starts); directions[..., 2] = -1.
        hits, _, _, _ = raycast_mesh(starts, directions, mesh, max_dist=20.)
        return hits[..., 2].detach().cpu().numpy()
    origins = env.scene.env_origins.detach().cpu().numpy()
    unique = np.unique(origins, axis=0)[:16]
    route = meta['route']
    start = np.asarray(route['start_offset'] if route else [0., 0.])
    goal = np.asarray(route['goal_offset'] if route else [0., 0.])
    spacing = .02 if is_t2 else .1
    xs = np.arange(min(start[0], goal[0])-.4, max(start[0], goal[0])+.4+spacing/2, spacing)
    ys = np.arange(-.8, .8+spacing/2, spacing)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    relative = np.stack((xx, yy), axis=-1)
    grid_xy = unique[:, None, None, :2]+relative
    heights = rays(grid_xy)
    route_x = np.linspace(start[0], goal[0], max(2, int(abs(goal[0]-start[0])/.01)+1))
    route_xy = np.stack((route_x, np.zeros_like(route_x)), -1)
    route_z = rays(unique[:, None, :2]+route_xy)
    foot_ids = [i for i, name in enumerate(robot.body_names) if name.endswith('_foot')]
    if len(foot_ids) != 4:
        raise RuntimeError(f'Expected four feet, found {len(foot_ids)}')
    foot_pos = robot.data.body_pos_w[:, foot_ids].detach().cpu().numpy()
    foot_quat = robot.data.body_quat_w[:, foot_ids]
    foot_ground = rays(foot_pos[..., :2])
    foot_bounds = _foot_geometry(env)
    bottom_offsets = []
    bounds_complete = True
    for j, body_id in enumerate(foot_ids):
        rows = foot_bounds.get(robot.body_names[body_id], [])
        valid = [r for r in rows if 'local_aabb_min' in r and np.isfinite(r['local_aabb_min']).all() and np.isfinite(r['local_aabb_max']).all() and np.all(np.asarray(r['local_aabb_min'])<=np.asarray(r['local_aabb_max']))]
        if not valid:
            bounds_complete = False
            bottom_offsets.append(np.full(env.num_envs, np.nan))
            continue
        shape_bottoms = []
        for row in valid:
            if 'local_sphere_axes' in row:
                local = np.vstack((row['local_sphere_center'], row['local_sphere_axes']))
                v = torch.as_tensor(local, dtype=torch.float32, device=env.device)[None].expand(env.num_envs,-1,-1)
                q = foot_quat[:,j,None,:].expand(-1,4,-1)
                world = quat_apply(q.reshape(-1,4),v.reshape(-1,3)).reshape(env.num_envs,4,3)
                # Exact vertical support of transformed sphere (ellipsoid), so
                # unlike a rotated box this cannot invent sphere penetration.
                bottom = world[:,0,2] - world[:,1:,2].norm(dim=-1)
            else:
                corners = np.array([[x,y,z] for x in [row['local_aabb_min'][0],row['local_aabb_max'][0]] for y in [row['local_aabb_min'][1],row['local_aabb_max'][1]] for z in [row['local_aabb_min'][2],row['local_aabb_max'][2]]])
                v = torch.as_tensor(corners, dtype=torch.float32, device=env.device)[None].expand(env.num_envs,-1,-1)
                q = foot_quat[:,j,None,:].expand(-1,8,-1)
                bottom = quat_apply(q.reshape(-1,4),v.reshape(-1,3)).reshape(env.num_envs,8,3)[...,2].amin(-1)
            shape_bottoms.append(bottom)
        bottom_offsets.append(torch.stack(shape_bottoms).amin(0).detach().cpu().numpy())
    bottom_offsets = np.stack(bottom_offsets, axis=-1)
    clearances = foot_pos[...,2]+bottom_offsets-foot_ground
    scan = scanner.data.ray_hits_w.detach().cpu().numpy() if scanner is not None else None
    endpoints = []
    offsets = np.array([[0.,0.],[.2,.15],[.2,-.15],[-.2,.15],[-.2,-.15]])
    for name,xy in [('start',start),('goal',goal)]:
        values = rays(unique[:,None,:2]+xy+offsets)
        endpoints.append(dict(name=name, offsets_m=offsets.tolist(), surface_z_m=values.tolist(),
            finite=bool(np.isfinite(values).all()), support_span_m=(np.max(values,axis=-1)-np.min(values,axis=-1)).tolist(),
            all_probes_on_stone_top=bool(np.all(values>=np.max(heights,axis=(1,2))[:,None]-.12)) if meta['terrain']=='stepping_stones' else None))
    layout_rows=[]
    for i,z in enumerate(route_z):
        delta=np.diff(z); good=np.isfinite(delta)
        changes=delta[good & (np.abs(delta)>.005)]
        row=dict(origin=unique[i].tolist(),finite=bool(np.isfinite(z).all()),height_min_m=float(np.min(z)),height_max_m=float(np.max(z)),
                 measured_elevation_change_m=float(z[-1]-z[0]), nonzero_height_transition_count=int(changes.size),
                 height_transitions_m=np.round(changes,6).tolist())
        if meta['terrain']=='stepping_stones':
            from scipy.ndimage import label
            support=np.isfinite(heights[i]) & (heights[i]>=np.max(heights[i])-.12)
            labels,n=label(support)
            counts=np.bincount(labels.ravel())[1:]
            low=np.isfinite(z)&(z<np.max(z)-.12)
            low_labels,ng=label(low)
            gap_counts=np.bincount(low_labels)[1:]
            row.update(gap_sample_count=int(low.sum()), gap_runs_m=(gap_counts*abs(route_x[1]-route_x[0])).tolist(),
                support_component_count=int(n), sampled_support_component_area_m2=(counts*spacing**2).tolist(),
                support_area_note='Grid-cell estimate in sampled corridor; components touching boundary are censored, not full stone areas')
        layout_rows.append(row)
    checks=dict(all_sampled_mesh_rays_finite=bool(np.isfinite(heights).all() and np.isfinite(route_z).all()),
        height_scanner_finite=bool(np.isfinite(scan).all()) if scan is not None else None,
        reset_foot_body_centers_above_ground=bool(np.all(foot_pos[...,2]-foot_ground>=-.002)),
        foot_collision_bounds_available=bounds_complete,
        reset_collision_clearance_finite=bool(np.isfinite(clearances).all()),
        reset_collision_aabb_clearance_ge_minus_2mm=bool(np.all(clearances>=-.002)) if bounds_complete else None,
        start_goal_rays_finite=all(e['finite'] for e in endpoints))
    if meta['terrain']=='stairs_up':
        checks['route_has_real_ascending_steps']=all(r['nonzero_height_transition_count']>0 and r['measured_elevation_change_m']>.005 for r in layout_rows)
    if meta['terrain']=='stairs_down':
        checks['route_has_real_descending_steps']=all(r['nonzero_height_transition_count']>0 and r['measured_elevation_change_m']<-.005 for r in layout_rows)
    if meta['terrain']=='stepping_stones':
        checks['start_goal_all_probes_on_stone_top']=all(e['all_probes_on_stone_top'] for e in endpoints)
        checks['route_has_real_nonzero_gaps']=all(r['gap_sample_count']>0 for r in layout_rows)
    sample_path=out/'geometry_samples.npz'
    np.savez_compressed(sample_path,origins=origins,sampled_origins=unique,grid_xy_relative=relative,grid_height_world=heights,
        route_xy_relative=route_xy,route_height_world=route_z,foot_pos_world=foot_pos,foot_ground_world=foot_ground,
        foot_collision_aabb_clearance=clearances,scanner_hits_world=scan if scan is not None else np.empty((0,3)))
    result=dict(schema_version=1,task=meta,source_sha256=_sha(__file__),
        scope='Read-only initial-state terrain and collision-bound audit; does not establish dynamic traversability or PPO learning',
        plane_method='Exact intersection with authored collision Plane' if not is_t2 else None,
        ground_prims=ground_prims,grid_spacing_m=spacing,unique_origins_tested=len(unique),all_unique_origins_count=len(np.unique(origins,axis=0)),
        layouts=layout_rows,endpoints=endpoints,checks=checks,foot_geometry=foot_bounds,
        reset_foot_body_center_clearance_m=_finite_stats(foot_pos[...,2]-foot_ground),
        reset_foot_collision_aabb_clearance_m=_finite_stats(clearances),
        collision_clearance_note='Exact transformed Sphere vertical support when available; otherwise conservative rotated collision AABB. Clearance is against terrain at foot body XY (step-edge contact needs full collision check). Body center below surface is separately reported.',
        scan_stats=_finite_stats(scan) if scan is not None else None,
        samples=dict(path=str(sample_path),sha256=_sha(sample_path),bytes=sample_path.stat().st_size),
        geometry_only_checks_pass=all(v is not False for k,v in checks.items() if not k.startswith('reset_')),physical_admission=False)
    with (out/'geometry.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    return result
