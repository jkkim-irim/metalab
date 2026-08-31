"""Newton parser (sim.metalab.backends.newton.parser) — EnvSpec (engine-agnostic contract) → newton.Model + SolverMuJoCo (Newton spoke, standalone).

Newton counterpart of the Genesis parser. **Both robot and object load via MJCF `add_mjcf`** (equality/gains/
armature/mass/contact material native; MuJoCo equality converted via `convert_mjc_equality_constraints=True`).
Masked-0 joints load the **same copy** that ``mjcf_prep`` turned into 0° FIXED (byte-identical to genesis → parity).

Scene = **per-world loop**: each world assembles base (robot + static fixtures) + object variant (env i →
variant i%N round-robin) via `begin_world`/`add_builder`/`end_world`. Each world is physically independent
via SolverMuJoCo separate_worlds (spacing=0 overlap at origin is numerically stable). Per-world structure is
uniform (robot + 1 free body), so backend indexing is unchanged.

Only builds the scene (no obs/reward/action logic). Overrides (kp/kv) applied to base builder arrays before finalize.
"""
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
from sim.metalab.backends.newton.picking import scale_pick_force
from sim.metalab.backends.newton.viewer_rerun_sink import (
    install_rrd_sink,
    log_world_labels,
    no_viewer_client,
    tune_batcher,
)
from sim.metalab.contract.asset_path import resolve_asset
from sim.metalab.contract.mjcf_prep import prepare_mjcf, prepare_object_mjcf
from sim.metalab.contract.spec import EnvSpec, RobotSpec


def _wxyz_to_xyzw_transform(pos, quat_wxyz) -> wp.transform:
    """canonical (pos, wxyz) → wp.transform (warp stores quat as xyzw)."""
    w, x, y, z = quat_wxyz
    return wp.transform(wp.vec3(*pos), wp.quat(x, y, z, w))


def _joint_local_index(builder: newton.ModelBuilder, name: str) -> int:
    """Active joint name → joint index within the sub-builder. Labels are 'prefix/Name' or 'Name'."""
    for i, lbl in enumerate(builder.joint_label):
        if lbl == name or lbl.endswith(f"/{name}"):
            return i
    raise ValueError(f"joint '{name}' not found in builder.joint_label")


def _apply_gain_overrides(builder: newton.ModelBuilder, r: RobotSpec) -> None:
    """Apply active-joint kp/kv/armature overrides to the builder's per-dof arrays before finalize.
    ``"default"`` is skipped (keeps the value add_mjcf filled from the MJCF actuator). Assumes 1-DOF joints.

    frictionloss is deliberately absent: Coulomb friction comes from the MJCF only (add_mjcf reads it into
    ``builder.joint_friction`` → SolverMuJoCo's ``dof_frictionloss``) and is not a per-task knob."""
    for jname, ov in r.joint_mode_param.items():
        jidx = _joint_local_index(builder, jname)
        dof = builder.joint_qd_start[jidx]           # 1-DOF: dof start = that joint's sole dof
        if ov.kp != "default":
            builder.joint_target_ke[dof] = float(ov.kp)
        if ov.kv != "default":
            builder.joint_target_kd[dof] = float(ov.kv)
        if ov.armature != "default":
            builder.joint_armature[dof] = float(ov.armature)


def _apply_contact_params(solver, spec: EnvSpec, ov: dict) -> None:
    """Per-asset contact + equality stiffness — legacy isaaclab-newton parity (retired solver_setup/contact_params).

    Robot links and builder-made fixtures (table box) have no MJCF contact block, so they land on MuJoCo's
    default solref 0.02 (~4× softer than the legacy-validated 0.01 → finger/table penetration). OBJECT geoms
    are handled here too — an object name is just another group key, authored in the contract instead of in
    the object's MJCF, so ONE file states an object's contact softness and both engines read the same
    numbers (genesis applies them as geom ``sol_params``). Contact params are written to the
    **Newton source** custom attributes (model.mujoco.solref/geom_solimp/geom_solmix + solref_mode=RAW — exactly
    how add_mjcf marks authored values): the SHAPE_PROPERTIES re-sync kernel reads mjw_model FROM these every
    notify, so direct mjw writes get clobbered on the first friction-DR reset (measured), while source writes are
    permanent. A one-shot notify_model_changed pushes them into mjw_model now (before the first step → baked
    ahead of the CUDA-graph capture). Equality stiffness is the exception: eq_solref/eq_solimp have no Newton-side
    re-sync (writes survive — same as legacy), so they are written to mjw_model directly. Values come from the
    contract: the top-level ``contact_params`` knob {robot|fixture|object: {solref, solimp, solmix}} — shared
    with genesis, which implements the same math — plus overrides.newton.eq_solref/eq_solimp (finger-mimic
    stiffness, legacy [0.0055, 1.0] vs default 0.02), which stay per-engine. Nothing declared → no-op."""
    # A GHOST object (ObjectSpec.collision=False) reaches MuJoCo with no geom at all, so its contact-params
    # entry has nothing to write — drop the key rather than let the fail-loud match below read as a typo.
    cp = {k: v for k, v in (spec.contact_params or {}).items()
          if k not in {o.name for o in spec.objects if not o.collision}}
    eq_solref, eq_solimp = ov.get("eq_solref"), ov.get("eq_solimp")
    if not cp and eq_solref is None and eq_solimp is None:
        return
    mjm = solver.mj_model
    if cp:
        obj_prefixes = tuple(o.name for o in spec.objects)
        # classify every mj geom by its body: world-body BOX = table fixture, object-prefixed body = THAT
        # object, anything else = robot link. Plane/other world geoms stay untouched.
        groups: dict[str, list[int]] = {"robot": [], "table": []}
        # world boxes are emitted in this order by build_scene: box fixtures, then each fixed object's parts.
        # Per WORLD, so the list repeats; rebuilt lazily below as geoms are walked.
        static_order = ([f.name for f in spec.fixtures if f.kind == "box"]
                        + [o.name for o in spec.objects if o.fixed for _ in o.parts])
        static_names = list(static_order)
        for g in range(mjm.ngeom):
            bid = int(mjm.geom_bodyid[g])
            bname = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bid == 0:
                # world-body boxes = static geometry: box fixtures and FIXED objects, in creation order
                if mjm.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    if not static_names:
                        static_names = list(static_order)      # next world's copy of the same static set
                    groups.setdefault(static_names.pop(0), []).append(g)
            elif bname.startswith(obj_prefixes):
                groups.setdefault(next(o for o in obj_prefixes if bname.startswith(o)), []).append(g)
            else:
                groups["robot"].append(g)
        g2s = wp.to_torch(solver.mjc_geom_to_newton_shape).long()   # (nworld, ngeom) → newton shape index
        attrs = solver.model.mujoco
        solref_t = wp.to_torch(attrs.solref)              # (nshape, 2), sentinel (0,0) = not authored
        solref_mode_t = wp.to_torch(attrs.solref_mode)    # (nshape,)  RAW = forward verbatim
        solimp_t = wp.to_torch(attrs.geom_solimp)         # (nshape, 5)
        solmix_t = wp.to_torch(attrs.geom_solmix)         # (nshape,)
        for key, p in cp.items():
            gidx = groups.get(key)
            assert gidx is not None, \
                f"contact params key '{key}' matched no geom (have: {'|'.join(k for k, v in groups.items() if v)})"
            assert gidx, f"contact_params['{key}'] matched no geoms"
            shapes = g2s[:, gidx].flatten().unique()      # all-world newton shapes backing these geoms
            if "solref" in p:
                solref_t[shapes] = torch.tensor(p["solref"], device=solref_t.device, dtype=solref_t.dtype)
                solref_mode_t[shapes] = SOLREF_MODE_RAW
            if "solimp" in p:
                solimp_t[shapes] = torch.tensor(p["solimp"], device=solimp_t.device, dtype=solimp_t.dtype)
            if "solmix" in p:
                solmix_t[shapes] = float(p["solmix"])
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)   # push source → mjw_model now
    if eq_solref is not None or eq_solimp is not None:
        assert mjm.neq > 0, "eq_solref/eq_solimp declared but the model has no equality constraints"
        if eq_solref is not None:
            t = wp.to_torch(solver.mjw_model.eq_solref)
            t[:] = torch.tensor(eq_solref, device=t.device, dtype=t.dtype)
        if eq_solimp is not None:
            t = wp.to_torch(solver.mjw_model.eq_solimp)
            t[:] = torch.tensor(eq_solimp, device=t.device, dtype=t.dtype)


def tile_worlds(viewer, model, env_spacing: float) -> None:
    """Lay the worlds out on the CONTRACT grid in ``viewer``, replacing newton's auto spacing.

    The worlds overlap in physics (separate_worlds, spacing 0 — they cannot interact, so there is nothing to
    separate) and it is the VIEWER that spreads them: ``set_model`` auto-derives a grid from the scene extents
    (``ceil(max_extent × 1.5)``), which makes the same scene tile at a different scale than genesis. Pinning
    the pitch to ``spec.env_spacing`` is what makes the two spokes' views the same grid — the newton
    counterpart of genesis ``scene.envs_offset`` (``scene.build(env_spacing=...)``, likewise render-only).
    Both grids are ``ceil(sqrt(N))`` per row and centered on the origin; only the row/column axes are swapped.
    """
    spacing = [float(env_spacing)] * 3
    spacing[int(model.up_axis)] = 0.0            # never lift a world off the ground plane
    viewer.set_world_offsets(tuple(spacing))     # also latches _user_spacing: set_visible_worlds keeps this grid


def build_scene(spec: EnvSpec, num_envs: int | None = None, viz: bool = False, viewer_kind: str = "gl",
                device: str = "cuda:0", rrd_path: str | None = None) -> dict:
    """EnvSpec → built newton.Model + SolverMuJoCo. Returns handles (consumed by backend):
    {model, solver, state_0, state_1, control, contacts, num_envs, device}.
    """
    n_envs = int(num_envs if num_envs is not None else spec.num_envs)
    ph = spec.physics
    ov = spec.overrides.get("newton", {})
    # Contact pipeline: True (default) = MuJoCo-warp broadphase/narrowphase (legacy hammer path);
    # False = Newton-native collide() feeding the solver via _convert_contacts_to_mjwarp (legacy
    # dexblind_track path — true per-variant collision meshes, no MuJoCo EPA/CCD workspace).
    use_mjc_contacts = bool(ov.get("use_mujoco_contacts", True))
    # Contact-detection gap [m], baked into every shape's geom_gap at add-time. newton's ModelBuilder default
    # is 0.1 (10 cm!) — the broadphase then flags any geom pair within 10 cm as a contact candidate. For the
    # compact dexterous hand this detects ~727 near-touching "contacts"/world (99.9% NOT penetrating), 45× the
    # ~16 real contacts, and physics cost is linear in contact count → ~3× slower for zero fidelity gain (the
    # penetrating-contact set is identical at gap=0, so self-collision still fully works). genesis effectively
    # uses ~0 gap. Default 0.0 = MuJoCo default = detect on actual overlap only; override to reintroduce a
    # small anticipatory margin if a task needs softer contact engagement. Applies to BOTH contact paths.
    rigid_gap = float(ov.get("rigid_gap", 0.0))
    # MARGIN is the other half of newton's band and it is NOT interchangeable with gap: gap decides WHEN a
    # contact is generated, margin decides WHERE the contact surface sits — a pair settles with its surfaces
    # held apart by margin_a + margin_b (newton docs, "Margin and gap semantics"). So it is applied to the
    # ROBOT and the MOVABLE OBJECT only, never to the scenery they rest on: newton's own dense-hand example
    # does exactly that (examples/robot/example_robot_allegro_hand.py sets margin/gap on the builder holding
    # the hand AND the manipulated cube, then adds the ground plane on a builder that sets neither), which
    # keeps the resting pair margin at m rather than 2m. Applied to scenery it would lift every object off
    # the table by another m and walk the lift threshold. Default 0.0 = surfaces touch.
    # AND it is currently 0 for every recipe, because applying it to the ROBOT is worse than the
    # resting-height problem: it inflates every robot shape by m, so a robot-robot pair closes by 2m and
    # the finger self-pairs that newton does NOT filter start touching. collision_filter_parent excludes
    # only DIRECT parent-child, and proto_v4 put the pad on its own *_Fingertip body, so
    # *_Middle_Link <-> *_Fingertip is grandparent-grandchild and survives the filter. Measured at
    # m=0.001 on 4hammers: 436 robot self-contacts and the thumb MCP/IP equality pair oscillating at
    # |qd| ~50 rad/s (index PIP/DIP, the other equality pair, stayed at 0.006 — its meshes are 20x
    # smaller). m=0 with the object gap kept: 192 self-contacts, thumb |qd| 0.001.
    robot_object_margin = float(ov.get("robot_object_margin", 0.0))
    # GAP on the movable object ONLY, added on top of rigid_gap. Targeted differently from the margin above
    # and for a different reason: gap costs contact SLOTS, and putting it on the robot makes the hand pay for
    # its own geoms. Measured on 4hammers (256 envs, zero action, self-collision ON): gap 0.015 builder-wide
    # gave 109 contacts/world and nacon 28,044, against 6.8 and 2,140 at gap 0 — a 16x rise that is almost
    # entirely finger-vs-finger, since the same gap with self-collision OFF gave only 10.8. Pair gap is
    # additive, so object-only still yields the full band on every hand-vs-object and object-vs-table pair,
    # which is where early detection actually helps a grasp. (Constraint rows barely move either way:
    # nefc peak 79 -> 103, because a gap-generated contact is recorded but inactive until the surfaces meet.)
    object_gap = float(ov.get("object_gap", 0.0))
    # coord-layout targets: control.joint_target_q in joint_coord layout (recommended). Must be set before build.
    newton.use_coord_layout_targets = True

    # --- base sub-builder: robot (MJCF, fixed, self-collision ON) + static fixtures (table). Object is per-world. ---
    r = spec.robot
    assert "mjcf" in r.asset, "robot.asset.mjcf required (MJCF is source of truth)"
    base = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=ph.gravity[2])
    newton.solvers.SolverMuJoCo.register_custom_attributes(base)
    base.rigid_gap = rigid_gap      # 0 by default → detect on overlap only (see rigid_gap note above)
    base.default_shape_cfg.margin = robot_object_margin   # robot only — zeroed again below
    # init_pose split: ACTIVE entries → runtime joint_q default below; MASK-0 entries → weld angle baked
    # into the prepared MJCF (posed-then-welded fixed joint, e.g. waist pitch held at 45° for free).
    fixed_pose = {n: v for n, v in spec.robot.init_pose.items() if r.joints.get(n) == 0}
    mjcf = prepare_mjcf(str(resolve_asset(r.asset["mjcf"])), r.joints, r.collision, fixed_pose=fixed_pose)
    base.add_mjcf(
        mjcf,
        xform=_wxyz_to_xyzw_transform(r.base_pos, r.base_quat),
        floating=not r.fixed_base,                   # allex: fixed_base=True → FIXED root
        up_axis=newton.Axis.Z,
        # From the CONTRACT (physics.self_collision), not hardcoded — genesis has always read it
        # (backends/genesis/parser.py RigidOptions) and newton pinning it to True was a silent sim2sim
        # divergence. newton excludes adjacent parent-child pairs on its own either way.
        enable_self_collisions=ph.self_collision,
        convert_mjc_equality_constraints=True,       # convert finger IP/DIP follower equality
    )
    # self_collision_exclude: body-name substrings whose shapes stop colliding with EACH OTHER, while
    # physics.self_collision stays on for everything else. newton's own switch is all-or-nothing — with
    # enable_self_collisions=False import_mjcf filters EVERY robot self-pair (import_mjcf.py:3346-3352) —
    # so a subset is expressed by adding just that subset's pairs to the same filter list. Pairs BETWEEN
    # groups are left alone: a finger touching the forearm is real, and _NO_TOUCH_LINKS terminates on it.
    # Measured at 4096 envs (hull 32/64, substeps 3): all self-collision on 18.98 ms, all off 15.08 ms.
    sc_exclude = ov.get("self_collision_exclude")
    if sc_exclude:
        assert ph.self_collision, \
            "self_collision_exclude needs physics.self_collision=True — with it False every self-pair is " \
            "already filtered and this knob would be a no-op that reads as if it did something"
        pats = [sc_exclude] if isinstance(sc_exclude, str) else list(sc_exclude)
        # The patterns name ONE group: every matched shape stops colliding with every other matched shape.
        # So ["R_Thumb", "R_Index", ...] drops thumb<->index too, not just each finger's own links.
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

    base.default_shape_cfg.margin = 0.0              # the static fixtures/table added below are scenery
    n_robot_joints = base.joint_count                # per-world robot joint count (object free joint comes after)
    # Static geometry → world(body=-1) shapes on the base sub-builder (cloned per world): box fixtures and
    # FIXED procedural objects (a table) land in the same place, which is what keeps them out of both the
    # robot and the free-object shape masks the backend builds for friction/contact targeting.
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
    for jname in (n for _, g in r.action_groups.items() for n in g):   # active joints = position PD
        base.joint_target_mode[base.joint_qd_start[_joint_local_index(base, jname)]] = int(JointTargetMode.POSITION)
    _apply_gain_overrides(base, r)                   # kp/kv override
    # Motor-to-joint coupled PD: zero the coupled joints' native kp/kv at the BUILDER level so the
    # coupled law (backend substep kernel → control.joint_f) fully owns them. The mjc position
    # actuator stays structurally present (indices/effort plumbing unchanged) with gain/bias 0, and
    # every JOINT_DOF_PROPERTIES re-sync re-derives from joint_target_ke/kd — so it cannot resurrect.
    # METALAB_MOTOR_COUPLING=0 keeps native diagonal PD (build-time toggle, gravcomp-style).
    coupled = r.coupled_groups()                     # control_mode=motor → hand + arm(wrist) groups (else [])
    mc_on = bool(coupled) and os.environ.get("METALAB_MOTOR_COUPLING", "1") != "0"
    if mc_on:
        for grp in coupled:                          # zero native PD for every coupled joint (2- or 3-DOF)
            for jname in grp.joints:
                dof = base.joint_qd_start[_joint_local_index(base, jname)]
                base.joint_target_ke[dof] = 0.0
                base.joint_target_kd[dof] = 0.0
    for jname, val in spec.robot.init_pose.items():        # joint_q = per-world default; reset restores to this value
        if jname in fixed_pose:
            continue                                 # mask-0 entry — already welded at this angle in the MJCF
        base.joint_q[base.joint_q_start[_joint_local_index(base, jname)]] = float(val)

    # --- object variant sub-builders (MJCF; freejoint honored → FREE 6-DOF root). The variants must keep a
    # UNIFORM shape count (same geom count per variant) or SolverMuJoCo's homogeneous-world check rejects them.
    obj_variants = []                                # per object: [variant sub-builder ...] (movable only)
    for obj in spec.objects:
        if obj.fixed:
            continue                                 # already welded into `base` above as static shapes
        mjcfs = obj.asset.get("mjcf") if obj.asset else None
        assert mjcfs, f"object '{obj.name}': MJCF asset required (asset.mjcf)"
        paths = mjcfs if isinstance(mjcfs, list) else [mjcfs]
        pos = obj.init_pos if obj.init_pos is not None else (0.0, 0.0, 0.0)
        subs = []
        for p in paths:
            ob = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=ph.gravity[2])
            newton.solvers.SolverMuJoCo.register_custom_attributes(ob)
            ob.rigid_gap = rigid_gap
            ob.default_shape_cfg.gap = rigid_gap + object_gap   # explicit cfg wins over rigid_gap
            ob.default_shape_cfg.margin = robot_object_margin
            # contract mass (+consistent inertia) is written into a prepared copy — see prepare_object_mjcf
            ob.add_mjcf(prepare_object_mjcf(str(resolve_asset(p)), obj.mass, obj.collision),
                        xform=_wxyz_to_xyzw_transform(pos, obj.init_quat))
            subs.append(ob)
        obj_variants.append(subs)

    # Collision-mesh hull reduction — MECHANISM only; the numbers are task knobs:
    #   `hull_maxvert`        cap for the ROBOT + static meshes, NATIVE path only. mjwarp hill-climbs a hull
    #                         adjacency graph and so does not care how many vertices a hull has, while newton's
    #                         GJK scans them linearly — the cap buys speed on one path and only pulls the
    #                         contact surface in on the other. None (or absent) = no reduction at all.
    #   `object_hull_maxvert` cap for the OBJECT's collision meshes, on BOTH paths. Under mjwarp contacts this
    #                         does NOT change the colliding hull — import_mjcf stamps newton's
    #                         Mesh.MAX_HULL_VERTICES (64) on every mesh that omits MJCF `maxhullvert` and the
    #                         exporter forwards it, so MuJoCo's hull is already 64 verts per mesh (measured:
    #                         nmeshgraph/nmesh = 938.0 exactly, one graph slot size for a 239- and a 1436-vert
    #                         mesh). What it buys there is the stored VERTEX ARRAY that per-env geometry DR
    #                         duplicates once per world: 973 -> 256 verts = 79.7 -> 21 MB at 4096 envs.
    #   `hull_reduce_above`   only reduce meshes with MORE than this many vertices (0 = every collision mesh) —
    #                         keeps the reduction off meshes that are already proper (dense) hulls, where it
    #                         would only pull the contact surface in for no speed gain.
    # WHY a cap is worth declaring (and what it costs): see the CONVEX_MESH block after the assembly, plus the
    # per-task measurements next to the knob in the contract.
    # Must run BEFORE the worlds are assembled (add_builder COPIES the sub-builder, so reducing afterwards would
    # leave every world's copy untouched) and per sub-builder rather than on the assembled model (one hull
    # computation per distinct mesh, not per world).
    _reduce = []
    if ov.get("hull_maxvert") is not None and not use_mjc_contacts:
        _reduce.append((base, int(ov["hull_maxvert"])))
    if ov.get("object_hull_maxvert") is not None:
        _reduce += [(_ob, int(ov["object_hull_maxvert"])) for _subs in obj_variants for _ob in _subs]
    if _reduce:
        above = int(ov.get("hull_reduce_above", 0))          # 0 = reduce every collision mesh
        for _b, max_hv in _reduce:
            _idx = []
            for _s, _src in enumerate(_b.shape_source):
                # COLLIDING meshes only — same selection newton's own default makes. VISUAL meshes must keep
                # their full vertex set: nothing collides against them, and `approximate_meshes` both hulls
                # AND retypes what it touches, so including them silently decimated the render (measured:
                # every visual mesh cut down to the collision cap) for zero physics benefit.
                if not (_b.shape_flags[_s] & int(ShapeFlags.COLLIDE_SHAPES)):
                    continue
                if _src is None or not hasattr(_src, "vertices") or len(_src.vertices) <= above:
                    continue
                _src.maxhullvert = max_hv
                _idx.append(_s)
            if _idx:
                _b.approximate_meshes(method="convex_hull", shape_indices=_idx)

    # --- multi-world assembly: world w = base + object variant w%N (env i → variant i%N fixed = legacy clone plan).
    # use_mujoco_contacts=True (same as legacy isaaclab; the mjwarp limitation where variant collision may
    # broadcast to world-0 exists, but legacy trained this way too). Uniform per-world structure → backend indexing unchanged.
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=ph.gravity[2])
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.rigid_gap = rigid_gap   # ground plane / any shape added on this top-level builder
    obj_variant_w: list[int] = []                    # per world: variant index of the FIRST movable object
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
        builder.add_ground_plane()                   # global (world -1) ground, shared across worlds

    if not use_mjc_contacts:
        # import_mjcf registers every mesh as GeoType.MESH (trimesh) — newton's native narrowphase then runs
        # triangle-pair contacts on mesh-mesh (measured: >1M pairs → nefc overflow → NaN). Collision meshes
        # must take the convex-convex (GJK) path instead, which is MuJoCo's mesh=convex-hull semantics.
        # Visual-only shapes keep MESH (rendering unchanged); the MuJoCo-contacts path is unaffected (mjw
        # treats mesh as convex regardless, and CONVEX_MESH also maps to mjGEOM_MESH).
        #
        # Retyping ALONE is only half the job, and the missing half cost 40x: newton's CONVEX_MESH support
        # function is a LINEAR SCAN over every vertex (geometry/support_function.py) — unlike MuJoCo, which
        # hill-climbs a hull adjacency graph and so does not care how many vertices a hull has. newton's own
        # answer is to keep hulls small (``Mesh.MAX_HULL_VERTICES = 64``), and import_mjcf duly records
        # maxhullvert=64 on every mesh — but that value is inert until something calls the reduction. Nothing
        # did, so GJK was scanning the raw asset: our ALLEX collision meshes are NOT all hand-authored hulls
        # (measured: Forearm_Collision3.stl = 91,253 verts, is_convex=False, volume 8.8% of its own hull), and
        # 91k-vertex support queries x ~1757 pairs/world = 251 ms in narrow_phase_kernel_gjk_mpr alone.
        # approximate_meshes() does BOTH halves: replaces each collision mesh with its convex hull (capped at
        # the mesh's maxhullvert) AND retypes it to CONVEX_MESH. Applied to the per-world sub-builders BEFORE
        # world assembly, so the hull is computed once per distinct mesh instead of once per world.
        # (the hull reduction itself runs BEFORE world assembly — see above.)
        for i, t in enumerate(builder.shape_type):
            if t == int(GeoType.MESH) and (builder.shape_flags[i] & int(ShapeFlags.COLLIDE_SHAPES)):
                builder.shape_type[i] = int(GeoType.CONVEX_MESH)

    model = builder.finalize(device=device)
    # SolverMuJoCo knobs = declared via contract overrides["newton"] (defaults = below = current behavior unchanged).
    # Exposing via contract instead of hardcoding lets E3 physics-fidelity matching (impratio/iterations/timing etc.)
    # be tuned in the task YAML.
    # Legacy hammer reference (60a5e58 simulation.py): impratio 1.0·iterations 30·ls_iterations 30·tolerance 1e-6
    #   + timing hz200/substep2/decim4 (differs from current 120/2/2 — match via physics·overrides if reproduction needed).
    # --- Large-num_envs memory (why newton OOM'd at 8192 while genesis fit) ---
    # The dominant cost is NOT the persistent contact/constraint buffers (those are ~1.3 GB @8192). It is the
    # TRANSIENT convex-narrowphase EPA workspace allocated every step in mujoco_warp collision_convex.py:
    #   epa_pr = wp.empty((naccdmax, 6 + MJ_MAX_EPAFACES·ccd_iterations), vec3),  naccdmax = nconmax·num_worlds.
    # With MJ_MAX_EPAFACES=5 and the default ccd_iterations=35 that is (6+175)·12 B = 2172 B/contact-slot; six
    # such arrays total ~4996 B/slot → ~42 GB @ (1024·8192) and the first (epa_pr, 18.22 GB) blows the 32 GB card.
    # `ccd_iterations` is the clean lever: it scales the EPA workspace linearly WITHOUT changing which contacts
    # are detected (ncon is unchanged across the sweep) — only EPA penetration-depth precision on deep convex
    # interpenetrations. 16 is mujoco_warp's own box-box default (well-precedented, low risk); 12 fits 8192 with
    # headroom. genesis ignores all of this (it doesn't pre-size a per-candidate EPA polytope workspace).
    #
    # nconmax/njmax are the PER-WORLD contact/constraint caps (pre-allocated × num_worlds). newton's auto-estimate
    # (None) is derived from the INITIAL hand-open state — far too small for a grasp (mujoco_warp "broadphase
    # overflow" drops contacts; measured grasp peak nconmax≈850, nefc≈80/world vs auto ~256/~0). `contact_scale`
    # (default 1.0) scales both caps to the task's contact richness. Note nconmax also scales the EPA workspace
    # (via naccdmax), so it cannot be cut below the grasp peak — hence ccd_iterations is the primary memory lever.
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
        ccd_iterations=int(ov.get("ccd_iterations", 16)),   # EPA workspace lever (default 35 → 16, ~2× less transient mem)
        njmax=njmax, nconmax=nconmax,
        # Contact points per geom pair: newton defaults this OFF, which leaves every mesh/box pair with a
        # SINGLE point (only box-box keeps a manifold) — one point cannot resist tilt, so a finger pressing a
        # face slides into it and the render shows deep interpenetration while the buffers sit idle. True
        # restores up to 4 points. Costs contact/constraint rows in proportion, so njmax must follow (a
        # pyramidal contact is 4 efc rows) — the backend's budget assert names the number when it does not.
        # Requires geom margin 0, which the use_mujoco_contacts path already forces.
        enable_multiccd=bool(ov.get("enable_multiccd", False)),
        use_mujoco_contacts=use_mjc_contacts,        # True → MuJoCo collision (hand self-contacts ~800/world)
    )                                                # either path: update_contacts feeds SensorContact from mjw
    for _k in ("tolerance", "ls_tolerance", "jacobian"):   # pass only if specified (unspecified = MuJoCo default;
        if _k in ov:                                       # jacobian auto = sparse iff nv>32 — allex nv=33)
            solver_kw[_k] = ov[_k]
    _friction.install()      # geometric-mean contact friction (runtime/physics/friction.py) — before any step
    solver = newton.solvers.SolverMuJoCo(model, **solver_kw)
    _apply_contact_params(solver, spec, ov)          # legacy isaaclab-newton contact/equality stiffness parity
    contacts_native = None
    if not use_mjc_contacts:
        # Newton-native collision (legacy dexblind_track path): the backend calls model.collide() each control
        # step and passes these contacts into solver.step (_convert_contacts_to_mjwarp). The convert kernel
        # derives per-contact solref from shape ke/kd whenever BOTH are > 0 — ShapeConfig defaults them to
        # 2500/100, which would silently override the injected geom_solref — so zero them (legacy
        # *_CONTACT_KE/KD = 0.0 rule: ke/kd=0 → geom_solref/solimp/solmix mixing, same as MuJoCo contacts).
        model.shape_material_ke.fill_(0.0)
        model.shape_material_kd.fill_(0.0)
        contacts_native = model.contacts()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)  # initialize body_q

    # --viz: open a real GL viewer (symmetric with genesis `show_viewer`). If display/GL is unavailable,
    # `ViewerGL()` raises → **loud fail** — never silently fall back to headless (fail-loud convention). Backend renders each step.
    # Policy rate: one .rrd frame per control step, so this is the sequence fps the replay must run at.
    _ctrl_fps = 1.0 / (float(ph.dt) * int(ph.decimation))
    viewer = None
    rerun_viewer = None          # set whenever rrd_path is given (unless --viz rerun already records)
    if viz:
        # --viewer selects the interactive Newton viewer: gl = OpenGL/pyglet (default),
        # rtx = OVRTX real-time path tracer (needs the `ovrtx` package; presents in a pyglet window),
        # rerun = rerun.io viewer (ships geometry, not pixels — remote-friendly, needs no X server).
        if viewer_kind == "rtx":
            viewer = ViewerRTX()
        elif viewer_kind == "gl":
            viewer = ViewerGL()
        elif viewer_kind == "rerun":
            # keep_historical_data=True is REQUIRED for replay: ViewerRerun feeds `not keep_historical_data`
            # straight into `static=` on every transform/line/scalar log, and a static entity carries no
            # timeline — recording with the default (False) yields ONE frozen frame plus scalars.
            # record_to_rrd is deliberately NOT passed: the constructor's own client launch replaces that
            # file sink (see viewer_rerun_sink), so the recording is installed after the fact instead.
            if rrd_path:
                tune_batcher()                             # before the constructor's rr.init
            viewer = ViewerRerun(app_id="metalab", keep_historical_data=True)
            if rrd_path:
                print(f"[newton] rerun recording → {install_rrd_sink(viewer, rrd_path, fps=_ctrl_fps)}", flush=True)
        else:
            raise ValueError(f"unsupported --viewer '{viewer_kind}' for newton --viz (use: gl | rtx | rerun)")
        viewer.set_model(model)
        tile_worlds(viewer, model, spec.env_spacing)    # contract grid, not newton's extent-derived one
        if viewer_kind != "rerun":
            # Mouse grab: newton's stock spring is ~500 N/m at the picked link with a clamp derived from the
            # whole 43.7 kg articulation, so a right-drag hurls ALLEX around. Weaken it (see picking.py).
            # rerun is view-only — it has no `.picking`, so this would assert.
            scale_pick_force(viewer)
    if rrd_path and not isinstance(viewer, ViewerRerun):
        # The recording viewer stands on its OWN, independent of any interactive or offscreen one: an .rrd is
        # asked for by path, not by viz mode. (When --viz rerun is up, that viewer already carries the sink
        # above — a second one would log the same scene twice.) Every env is tiled and visible at once in the
        # replay, on the same contract grid the interactive viewer draws.
        tune_batcher()                                     # before the constructor's rr.init
        with no_viewer_client():                           # headless: no gRPC/web server, no browser tab
            rerun_viewer = ViewerRerun(app_id="metalab", keep_historical_data=True)
        # Sink FIRST, then the scene: set_sinks only affects LATER logs, and with no client launched there is
        # no interim sink to catch anything logged in between.
        print(f"[newton] rerun recording → {install_rrd_sink(rerun_viewer, rrd_path, fps=_ctrl_fps)}", flush=True)
        rerun_viewer.set_model(model)
        tile_worlds(rerun_viewer, model, spec.env_spacing)
        print(f"[newton] rerun world labels: {log_world_labels(rerun_viewer)}", flush=True)

    # Contacts is lazily allocated by the backend **after** SensorContact creation (the force attr is requested
    # on the model at sensor creation, and Contacts must be built after the sensor to reflect it — sensor_contact example order).
    return {
        "model": model,
        "solver": solver,
        "state_0": state_0,
        "state_1": state_1,
        "control": control,
        "contacts_native": contacts_native,          # None on the MuJoCo-contacts path
        "viewer": viewer,
        "rerun_viewer": rerun_viewer,                # None unless an rrd_path was given
        "num_envs": n_envs,
        "device": device,
        "n_robot_joints": n_robot_joints,
        "substeps": ph.substeps,
        "sim_dt": ph.dt / ph.substeps,
        "motor_coupling_on": mc_on,                  # backend constructs MotorCoupledPDHand when True
        "object_variant": obj_variant_w,             # [] when the scene has no movable object
        "object_variant_count": len(obj_variants[0]) if obj_variants else 0,
    }
