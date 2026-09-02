from __future__ import annotations

import os

import mujoco
import newton
from newton import GeoType, JointTargetMode, ShapeFlags
from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
from newton.viewer import ViewerGL, ViewerRerun, ViewerRTX
import torch
import warp as wp

from sim.metalab.backends.newton import friction as _friction
from sim.metalab.backends.newton.viewer import log_world_labels, scale_pick_force
from sim.metalab.contract.asset_path import resolve_asset
from sim.metalab.contract.mjcf_prep import prepare_mjcf, prepare_object_mjcf
from sim.metalab.contract.spec import EnvSpec, RobotSpec
from sim.metalab.runtime.rerun_recording import attach_file_sink, no_viewer_client, tune_batcher


def _wxyz_to_xyzw_transform(pos, quat_wxyz) -> wp.transform:
    w, x, y, z = quat_wxyz
    return wp.transform(wp.vec3(*pos), wp.quat(x, y, z, w))


def _joint_local_index(builder: newton.ModelBuilder, name: str) -> int:
    for i, lbl in enumerate(builder.joint_label):
        if lbl == name or lbl.endswith(f"/{name}"):
            return i
    raise ValueError(f"joint '{name}' not found in builder.joint_label")


def _apply_gain_overrides(builder: newton.ModelBuilder, r: RobotSpec) -> None:
    for jname, ov in r.joint_mode_param.items():
        jidx = _joint_local_index(builder, jname)
        dof = builder.joint_qd_start[jidx]
        if ov.kp != "default":
            builder.joint_target_ke[dof] = float(ov.kp)
        if ov.kv != "default":
            builder.joint_target_kd[dof] = float(ov.kv)
        if ov.armature != "default":
            builder.joint_armature[dof] = float(ov.armature)


def _apply_contact_params(solver, spec: EnvSpec, ov: dict) -> None:
    cp = {k: v for k, v in (spec.contact_params or {}).items()
          if k not in {o.name for o in spec.objects if not o.collision}}
    eq_solref, eq_solimp = ov.get("eq_solref"), ov.get("eq_solimp")
    if not cp and eq_solref is None and eq_solimp is None:
        return
    mjm = solver.mj_model
    if cp:
        obj_prefixes = tuple(o.name for o in spec.objects)
        groups: dict[str, list[int]] = {"robot": [], "table": []}
        static_order = ([f.name for f in spec.fixtures if f.kind == "box"]
                        + [o.name for o in spec.objects if o.fixed for _ in o.parts])
        static_names = list(static_order)
        for g in range(mjm.ngeom):
            bid = int(mjm.geom_bodyid[g])
            bname = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bid == 0:
                if mjm.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    if not static_names:
                        static_names = list(static_order)
                    groups.setdefault(static_names.pop(0), []).append(g)
            elif bname.startswith(obj_prefixes):
                groups.setdefault(next(o for o in obj_prefixes if bname.startswith(o)), []).append(g)
            else:
                groups["robot"].append(g)
        g2s = wp.to_torch(solver.mjc_geom_to_newton_shape).long()
        attrs = solver.model.mujoco
        solref_t = wp.to_torch(attrs.solref)
        solref_mode_t = wp.to_torch(attrs.solref_mode)
        solimp_t = wp.to_torch(attrs.geom_solimp)
        solmix_t = wp.to_torch(attrs.geom_solmix)
        for key, p in cp.items():
            gidx = groups.get(key)
            assert gidx is not None, \
                f"contact params key '{key}' matched no geom (have: {'|'.join(k for k, v in groups.items() if v)})"
            assert gidx, f"contact_params['{key}'] matched no geoms"
            shapes = g2s[:, gidx].flatten().unique()
            if "solref" in p:
                solref_t[shapes] = torch.tensor(p["solref"], device=solref_t.device, dtype=solref_t.dtype)
                solref_mode_t[shapes] = SOLREF_MODE_RAW
            if "solimp" in p:
                solimp_t[shapes] = torch.tensor(p["solimp"], device=solimp_t.device, dtype=solimp_t.dtype)
            if "solmix" in p:
                solmix_t[shapes] = float(p["solmix"])
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    if eq_solref is not None or eq_solimp is not None:
        assert mjm.neq > 0, "eq_solref/eq_solimp declared but the model has no equality constraints"
        if eq_solref is not None:
            t = wp.to_torch(solver.mjw_model.eq_solref)
            t[:] = torch.tensor(eq_solref, device=t.device, dtype=t.dtype)
        if eq_solimp is not None:
            t = wp.to_torch(solver.mjw_model.eq_solimp)
            t[:] = torch.tensor(eq_solimp, device=t.device, dtype=t.dtype)


def tile_worlds(viewer, model, env_spacing: float) -> None:
    spacing = [float(env_spacing)] * 3
    spacing[int(model.up_axis)] = 0.0
    viewer.set_world_offsets(tuple(spacing))


def build_scene(spec: EnvSpec, num_envs: int | None = None, viz: bool = False, viewer_kind: str = "gl",
                device: str = "cuda:0", rrd_path: str | None = None) -> dict:
    n_envs = int(num_envs if num_envs is not None else spec.num_envs)
    ph = spec.physics
    ov = spec.overrides.get("newton", {})
    use_mjc_contacts = bool(ov.get("use_mujoco_contacts", True))
    rigid_gap = float(ov.get("rigid_gap", 0.0))
    robot_object_margin = float(ov.get("robot_object_margin", 0.0))
    object_gap = float(ov.get("object_gap", 0.0))
    newton.use_coord_layout_targets = True

    r = spec.robot
    assert "mjcf" in r.asset, "robot.asset.mjcf required (MJCF is source of truth)"
    base = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=ph.gravity[2])
    newton.solvers.SolverMuJoCo.register_custom_attributes(base)
    base.rigid_gap = rigid_gap
    base.default_shape_cfg.margin = robot_object_margin
    fixed_pose = {n: v for n, v in spec.robot.init_pose.items() if r.joints.get(n) == 0}
    mjcf = prepare_mjcf(str(resolve_asset(r.asset["mjcf"])), r.joints, r.collision, fixed_pose=fixed_pose)
    base.add_mjcf(
        mjcf,
        xform=_wxyz_to_xyzw_transform(r.base_pos, r.base_quat),
        floating=not r.fixed_base,
        up_axis=newton.Axis.Z,
        enable_self_collisions=ph.self_collision,
        convert_mjc_equality_constraints=True,
    )
    sc_exclude = ov.get("self_collision_exclude")
    if sc_exclude:
        assert ph.self_collision, \
            "self_collision_exclude needs physics.self_collision=True — with it False every self-pair is " \
            "already filtered and this knob would be a no-op that reads as if it did something"
        pats = [sc_exclude] if isinstance(sc_exclude, str) else list(sc_exclude)
        hit = {p: 0 for p in pats}
        shapes = []
        for s, bidx in enumerate(base.shape_body):
            if not (base.shape_flags[s] & newton.ShapeFlags.COLLIDE_SHAPES) or int(bidx) < 0:
                continue
            name = base.body_label[int(bidx)]
            for p in pats:
                if p in name:
                    shapes.append(s)
                    hit[p] += 1
                    break
        for p, n in hit.items():
            assert n, (f"self_collision_exclude pattern {p!r} matched no colliding robot shape — "
                       f"bodies are named like {base.body_label[:3]}")
        for i, sa in enumerate(shapes):
            for sb in shapes[i + 1:]:
                base.add_shape_collision_filter_pair(sa, sb)
        print(f"[robot] self_collision_exclude {pats}: {len(shapes)} shapes -> "
              f"{len(shapes) * (len(shapes) - 1) // 2} self-pairs filtered ({hit})", flush=True)

    base.default_shape_cfg.margin = 0.0
    n_robot_joints = base.joint_count
    for fx in spec.fixtures:
        if fx.kind == "box":
            assert fx.size is not None, f"box fixture '{fx.name}' requires size"
            hx, hy, hz = (v / 2.0 for v in fx.size)
            base.add_shape_box(-1, xform=_wxyz_to_xyzw_transform(fx.pos, fx.quat), hx=hx, hy=hy, hz=hz)
    for obj in spec.objects:
        if not obj.fixed:
            continue
        assert obj.parts, f"fixed object '{obj.name}': procedural parts required (an asset would need a joint)"
        for part in obj.parts:
            assert part.shape == "box", f"object '{obj.name}': fixed parts support shape 'box' for now"
            pos = tuple(a + b for a, b in zip(obj.init_pos or (0.0, 0.0, 0.0), part.pos))
            hx, hy, hz = (v / 2.0 for v in part.size)
            base.add_shape_box(-1, xform=_wxyz_to_xyzw_transform(pos, part.quat), hx=hx, hy=hy, hz=hz)
    for jname in (n for _, g in r.action_groups.items() for n in g):
        base.joint_target_mode[base.joint_qd_start[_joint_local_index(base, jname)]] = int(JointTargetMode.POSITION)
    _apply_gain_overrides(base, r)
    coupled = r.coupled_groups()
    mc_on = bool(coupled) and os.environ.get("METALAB_MOTOR_COUPLING", "1") != "0"
    if mc_on:
        for grp in coupled:
            for jname in grp.joints:
                dof = base.joint_qd_start[_joint_local_index(base, jname)]
                base.joint_target_ke[dof] = 0.0
                base.joint_target_kd[dof] = 0.0
    for jname, val in spec.robot.init_pose.items():
        if jname in fixed_pose:
            continue
        base.joint_q[base.joint_q_start[_joint_local_index(base, jname)]] = float(val)

    obj_variants = []
    for obj in spec.objects:
        if obj.fixed:
            continue
        mjcfs = obj.asset.get("mjcf") if obj.asset else None
        assert mjcfs, f"object '{obj.name}': MJCF asset required (asset.mjcf)"
        paths = mjcfs if isinstance(mjcfs, list) else [mjcfs]
        pos = obj.init_pos if obj.init_pos is not None else (0.0, 0.0, 0.0)
        subs = []
        for p in paths:
            ob = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=ph.gravity[2])
            newton.solvers.SolverMuJoCo.register_custom_attributes(ob)
            ob.rigid_gap = rigid_gap
            ob.default_shape_cfg.gap = rigid_gap + object_gap
            ob.default_shape_cfg.margin = robot_object_margin
            ob.add_mjcf(prepare_object_mjcf(str(resolve_asset(p)), obj.mass, obj.collision),
                        xform=_wxyz_to_xyzw_transform(pos, obj.init_quat))
            subs.append(ob)
        obj_variants.append(subs)

    _reduce = []
    if ov.get("hull_maxvert") is not None and not use_mjc_contacts:
        _reduce.append((base, int(ov["hull_maxvert"])))
    if ov.get("object_hull_maxvert") is not None:
        _reduce += [(_ob, int(ov["object_hull_maxvert"])) for _subs in obj_variants for _ob in _subs]
    if _reduce:
        above = int(ov.get("hull_reduce_above", 0))
        for _b, max_hv in _reduce:
            _idx = []
            for _s, _src in enumerate(_b.shape_source):
                if not (_b.shape_flags[_s] & int(ShapeFlags.COLLIDE_SHAPES)):
                    continue
                if _src is None or not hasattr(_src, "vertices") or len(_src.vertices) <= above:
                    continue
                _src.maxhullvert = max_hv
                _idx.append(_s)
            if _idx:
                _b.approximate_meshes(method="convex_hull", shape_indices=_idx)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=ph.gravity[2])
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.rigid_gap = rigid_gap
    obj_variant_w: list[int] = []
    for w in range(n_envs):
        builder.begin_world()
        builder.add_builder(base)
        for i, subs in enumerate(obj_variants):
            v = w % len(subs)
            if i == 0:
                obj_variant_w.append(v)
            builder.add_builder(subs[v])
        builder.end_world()
    if spec.scene.ground:
        builder.add_ground_plane()

    if not use_mjc_contacts:
        for i, t in enumerate(builder.shape_type):
            if t == int(GeoType.MESH) and (builder.shape_flags[i] & int(ShapeFlags.COLLIDE_SHAPES)):
                builder.shape_type[i] = int(GeoType.CONVEX_MESH)

    model = builder.finalize(device=device)
    contact_scale = float(ov.get("contact_scale", 1.0))
    nconmax = int(ov["nconmax"]) if ov.get("nconmax") is not None else round(1024 * contact_scale)
    njmax = int(ov["njmax"]) if ov.get("njmax") is not None else round(256 * contact_scale)
    solver_kw = dict(
        use_mujoco_cpu=False,
        solver=ov.get("solver", "newton"),
        integrator=ov.get("integrator", "implicitfast"),
        cone=ov.get("cone", "elliptic"),
        impratio=ov.get("impratio", 100),
        iterations=ov.get("iterations", ph.solver_iterations),
        ls_iterations=ov.get("ls_iterations", 50),
        ccd_iterations=int(ov.get("ccd_iterations", 16)),
        njmax=njmax, nconmax=nconmax,
        enable_multiccd=bool(ov.get("enable_multiccd", False)),
        use_mujoco_contacts=use_mjc_contacts,
    )
    for _k in ("tolerance", "ls_tolerance", "jacobian"):
        if _k in ov:
            solver_kw[_k] = ov[_k]
    _friction.install()
    solver = newton.solvers.SolverMuJoCo(model, **solver_kw)
    _apply_contact_params(solver, spec, ov)
    contacts_native = None
    if not use_mjc_contacts:
        model.shape_material_ke.fill_(0.0)
        model.shape_material_kd.fill_(0.0)
        contacts_native = model.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    _ctrl_fps = 1.0 / (float(ph.dt) * int(ph.decimation))
    viewer = None
    rerun_viewer = None
    if viz:
        if viewer_kind == "rtx":
            viewer = ViewerRTX()
        elif viewer_kind == "gl":
            viewer = ViewerGL()
        elif viewer_kind == "rerun":
            if rrd_path:
                tune_batcher()
            viewer = ViewerRerun(app_id="metalab", keep_historical_data=True)
            if rrd_path:
                print(f"[newton] rerun recording → {attach_file_sink(viewer, rrd_path, fps=_ctrl_fps)}", flush=True)
        else:
            raise ValueError(f"unsupported --viewer '{viewer_kind}' for newton --viz (use: gl | rtx | rerun)")
        viewer.set_model(model)
        tile_worlds(viewer, model, spec.env_spacing)
        if viewer_kind != "rerun":
            scale_pick_force(viewer)
    if rrd_path and not isinstance(viewer, ViewerRerun):
        tune_batcher()
        with no_viewer_client():
            rerun_viewer = ViewerRerun(app_id="metalab", keep_historical_data=True)
        print(f"[newton] rerun recording → {attach_file_sink(rerun_viewer, rrd_path, fps=_ctrl_fps)}", flush=True)
        rerun_viewer.set_model(model)
        tile_worlds(rerun_viewer, model, spec.env_spacing)
        print(f"[newton] rerun world labels: {log_world_labels(rerun_viewer)}", flush=True)

    return {
        "model": model,
        "solver": solver,
        "state_0": state_0,
        "state_1": state_1,
        "control": control,
        "contacts_native": contacts_native,
        "viewer": viewer,
        "rerun_viewer": rerun_viewer,
        "num_envs": n_envs,
        "device": device,
        "n_robot_joints": n_robot_joints,
        "substeps": ph.substeps,
        "sim_dt": ph.dt / ph.substeps,
        "motor_coupling_on": mc_on,
        "object_variant": obj_variant_w,
        "object_variant_count": len(obj_variants[0]) if obj_variants else 0,
    }
