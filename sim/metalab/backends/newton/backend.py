"""NewtonBackend — wraps newton.Model/State/Control/Contacts as :class:`sim.metalab.runtime.backend.SimBackend`.

env_driver talks to Newton only through this adapter (read+control+step+reset). Newton state is a **flat +
per-world contiguous** tensor (joint_q/joint_qd/body_q, shape=(coord/dof/body_count,)), so joint/body names
are resolved to a per-world local index once, then expanded to all-world index tensors via ``w*per_world +
local`` for gather. Returns follow the canonical convention (wxyz, world, SI, ``(N, ...)`` torch); Newton
transforms are xyzw, converted at body_quat.

reset uses SolverMuJoCo.reset(per-world mask) to restore model default (=init_pose baked) + clear warm-start
(randomization is future work). Contact forces are read via SensorContact (net force per body).
"""
from __future__ import annotations

import os

import newton
from newton import JointType
import numpy as np
import torch
import warp as wp

from sim.metalab.api.frames import quat_conj, quat_mul, quat_rotate, wxyz_to_xyzw, xyzw_to_wxyz
from sim.metalab.backends.newton import gravcomp as _gravcomp
from sim.metalab.backends.newton.mjw_object_scale import install as _install_mjw_object_scale
from sim.metalab.backends.newton.viewer_origin import add_origin_axes
from sim.metalab.backends.newton.viewer_rerun_sink import log_contact_arrows, mark_step
from sim.metalab.backends.newton.viewer_scale import make_scale_syncs
from sim.metalab.drive.motor2joint.motor_coupling import (
    MotorCoupledPDArm,
    MotorCoupledPDHand,
    load_arm_group,
    load_hand_group,
)


@wp.kernel
def _write_body_wrench(body_idx: wp.array(dtype=wp.int32), wrench: wp.array(dtype=wp.spatial_vector),
                       body_f: wp.array(dtype=wp.spatial_vector)):
    """Scatter the per-env external wrench into body_f (after clear_forces). Launched every substep so it
    lives INSIDE the CUDA graph — fixed buffers, events only rewrite values (unfired envs = 0 → no-op).
    A host-side torch write here would permanently disable graph capture (measured: the lifted-force
    interval event latched _has_wrench and silently ran the whole training eager).

    ACCUMULATES, never assigns: body_f is the shared external-force buffer and newton's own producers add
    into it (the viewer's mouse grab is a wp.atomic_add in viewer/kernels.py). Assigning made this kernel the
    last writer of the object's slot, so an unfired env (wrench = 0) silently ZEROED a mouse-grab force on
    the object — the hammer could not be dragged in standalone. After clear_forces the buffer is 0, so with
    no viewer the sum is identical to the old assignment."""
    i = wp.tid()
    wp.atomic_add(body_f, body_idx[i], wrench[i])


@wp.kernel(enable_backward=False)
def _scatter_penetration(
    nacon: wp.array(dtype=wp.int32),            # (1,) active contact count, ACROSS ALL WORLDS
    contact_dist: wp.array(dtype=float),        # (naconmax,) nearest-point distance; negative = penetrating
    contact_geom: wp.array(dtype=wp.vec2i),     # (naconmax,) the two geom ids per contact
    contact_worldid: wp.array(dtype=wp.int32),  # (naconmax,) which world the contact is in
    geom_tip: wp.array(dtype=wp.int32),         # (ngeom,) output column of that geom's link, -1 = not one
    geom_counterpart: wp.array(dtype=wp.int32),  # (ngeom,) 1 = geom belongs to the target
    out: wp.array2d(dtype=float),               # OUT: (nworld, K) deepest overlap [m], pre-zeroed
):
    """Reduce mjwarp's flat contact list to the deepest overlap per (world, link) against one counterpart.

    One thread per contact SLOT, not per link: the list is world-major-mixed and its length varies per step,
    so the only pass that sees every contact once is over the slots. Hence atomic_max — several contacts of
    one link land on the same output cell (a link carries several geoms, and a geom pair yields up to 4
    points), and the winner is the worst spot.

    Separating pairs (dist > 0) are skipped rather than clamped so a pre-zeroed cell keeps reading 0: with
    ``geom_margin`` forced to 0 they should not appear at all, but ``rigid_gap`` widens the detection
    envelope if a task ever sets it, and then "far apart" must not overwrite a real overlap elsewhere."""
    tid = wp.tid()
    if tid >= nacon[0]:                         # inactive slot — stale values, not this step's contacts
        return
    pen = -contact_dist[tid]                    # flip to the api/contact.py sign: positive = overlap
    if pen <= 0.0:
        return
    g = contact_geom[tid]
    w = contact_worldid[tid]
    k0 = geom_tip[g[0]]
    if k0 >= 0 and geom_counterpart[g[1]] == 1:
        wp.atomic_max(out, w, k0, pen)
    k1 = geom_tip[g[1]]                         # either side may be the sensing link (pair order is mjwarp's)
    if k1 >= 0 and geom_counterpart[g[0]] == 1:
        wp.atomic_max(out, w, k1, pen)


class NewtonBackend:
    def __init__(self, spec, handles: dict, num_envs: int):
        self.spec = spec
        self.model = handles["model"]
        self.solver = handles["solver"]
        self.state_0 = handles["state_0"]
        self.state_1 = handles["state_1"]
        self.control = handles["control"]
        self._contacts = None                            # lazy alloc (on first contact_force, after sensor creation)
        self._contacts_native = handles.get("contacts_native")   # Newton-native collision buffer (None = MuJoCo contacts)
        # per-step contact cache: update_contacts (the expensive mjw→Newton contact re-processing) runs ONCE
        # per state change; all obs/reward/terminate contact reads within a policy step share it (legacy parity
        # — isaaclab updated contact sensors once inside the CUDA graph). _state_epoch bumps on every state_0
        # mutation (step_n, reset_idx, set_object_pose/joint_positions, root-height).
        self._state_epoch = 0
        self._contacts_epoch = -1
        self.num_envs = int(num_envs)
        self.device = torch.device(handles["device"])
        self._substeps = int(handles["substeps"])
        self._sim_dt = float(handles["sim_dt"])
        # Which object variant each world was BUILT from — recorded by the parser's assembly loop, not
        # recomputed here (see runtime/backend.py ObjectVariant).
        self._obj_variant = torch.tensor(handles["object_variant"], dtype=torch.long, device=self.device)
        self._obj_variant_n = int(handles["object_variant_count"])
        self._viewer = handles.get("viewer")             # --viz GL viewer OR headless record viewer (None = neither)
        self._rerun = handles.get("rerun_viewer")        # record path + rrd_path: second viewer, records the .rrd
        # The .rrd is timestamped on its OWN clock: frame k at exactly k * one control period, NOT at
        # ``_sim_time``. Two reasons. (1) Uniformity: the reset settle advances _sim_time with rendering off,
        # so real sim time is not affine in the frame index and anything mapping frames<->series rows would
        # drift after the first mid-rollout reset. (2) Playback: rerun plays a temporal timeline at 1x wall
        # clock, and one control period per frame is exactly what makes the replay run at the policy rate.
        self._rerun_step = 0
        self._rerun_dt = float(spec.physics.dt) * int(spec.physics.decimation)
        self._rerun_offsets = None                       # (N,3) per-world tile offsets, resolved on first use
        self._sim_time = 0.0

        # CUDA-graph capture of the physics loop (GPU-only, no per-kernel host launch overhead).
        # Valid only headless on GPU: a --viz viewer's mouse-force writes are host-side and can't be captured,
        # so viewer runs fall back to the eager loop. External-wrench DR is graph-resident (_write_body_wrench
        # kernel over fixed buffers — events only rewrite values), so it never disables the graph. Captured
        # lazily on the first step. The graph spans the WHOLE control step (n=decimation physics steps ×
        # substeps solver calls) — one capture_launch per policy step, isaaclab-newton parity.
        self._graph = None
        self._graph_n = 0                                # physics steps baked into the captured graph
        self._graphable = (self._viewer is None and self.device.type == "cuda")

        m = self.model
        wc = int(m.world_count)
        assert wc == self.num_envs, f"world_count({wc}) != num_envs({self.num_envs})"
        self._coords_pw = m.joint_coord_count // wc      # per-world joint coords
        self._dofs_pw = m.joint_dof_count // wc          # per-world joint dofs
        self._bodies_pw = m.body_count // wc             # per-world bodies
        self._joints_pw = m.joint_count // wc            # per-world joints

        # for per-world local resolution (scan world-0 range only)
        self._q_start = m.joint_q_start.numpy()
        self._qd_start = m.joint_qd_start.numpy()
        self._labels = list(m.joint_label)
        self._body_labels = list(m.body_label)
        self._world_off = (torch.arange(wc, device=self.device) * self._coords_pw).unsqueeze(1)   # (N,1)
        self._world_off_dof = (torch.arange(wc, device=self.device) * self._dofs_pw).unsqueeze(1)
        self._world_off_body = torch.arange(wc, device=self.device) * self._bodies_pw             # (N,)

        # caches (avoid rebuilding index tensors every read/write)
        self._coord_cache: dict[tuple, torch.Tensor] = {}
        self._dof_cache: dict[tuple, torch.Tensor] = {}
        self._body_cache: dict[str, torch.Tensor] = {}
        self._sensors: dict[tuple, object] = {}          # names → SensorContact (net, total_force)
        self._cp_sensors: dict[tuple, object] = {}       # (names, target) → SensorContact (counterpart force_matrix)
        self._jt_mjc_dof_cache: dict[tuple, torch.Tensor] = {}   # names → MuJoCo dof idx (qfrc_actuator readout)
        # per-step READ cache (mirrors the genesis backend's _rcache): obs + reward + terminate each read
        # the same quantities within one policy step, and every read here is a gather COPY off a warp
        # buffer. Keyed by (kind, names) and dropped whenever _state_epoch bumps, so a hit is always
        # fresh. Measured: `_body_pose` alone was 12% of the newton sim-server's python time.
        self._rcache: dict = {}
        self._rcache_epoch = -1
        # torch views of the FIXED newton/mjw buffers, wrapped ONCE. wp.to_torch builds a fresh wrapper
        # object per call (16% of server python time in a py-spy profile) while the underlying storage
        # never moves — the arrays are allocated at build and mutated in place (CUDA-graph requirement).
        self._t_joint_q = wp.to_torch(self.state_0.joint_q)
        self._t_joint_qd = wp.to_torch(self.state_0.joint_qd)
        self._t_body_q = wp.to_torch(self.state_0.body_q)
        self._t_body_qd = wp.to_torch(self.state_0.body_qd)

        # object(s) = child body of each FREE joint (robot is fixed base). Scene-only standalone tasks may have
        # ZERO objects (robot only), so guard every object index/pose cache; object read/DR methods just aren't
        # called then. When objects exist the single-object RL API (object_pos/vel, set_object_pose, mass/wrench
        # DR) uses the FIRST free object, mirroring the genesis backend (objects[0]).
        jtype = m.joint_type.numpy()
        jchild = m.joint_child.numpy()
        free_local = [j for j in range(self._joints_pw) if int(jtype[j]) == int(JointType.FREE)]
        self._has_object = len(free_local) >= 1
        self._obj_body_idx = (self._world_off_body + int(jchild[free_local[0]])) if self._has_object else None  # (N,) object 0
        # ALL free-object bodies (every world); empty when no objects. Classifies shapes as 'object' below so that
        # objects 1..N-1 are NOT misclassified as robot (which would wrongly pick up robot_friction).
        _obj_locals = torch.tensor([int(jchild[fj]) for fj in free_local], dtype=torch.long, device=self.device)  # (n_obj,)
        self._all_obj_body_idx = (self._world_off_body.view(-1, 1) + _obj_locals.view(1, -1)).flatten()  # (N*n_obj,)
        # Viewer overlay "Show Origin" (left panel → MetaLab), off by default, drawn in the render calls
        # below: each object's body-ORIGIN frame as RGB axes — the frame object_pos(), the goal keypoint cage
        # and every task bar are expressed in, and the spot the hand grasps (the asset pipeline aligns them).
        # Object bodies only; the table is a static shape, not a body.
        self._origin_axes = add_origin_axes(self._viewer, self._all_obj_body_idx if self._has_object else None)
        # NOT on the rrd-recording viewer: the origin/grasp axes are a --viz debugging aid, and baking them
        # into every recording just clutters the scene. Re-enable by passing this viewer to add_origin_axes
        # with enabled=True (it has no checkbox — see viewer_origin).
        self._rerun_axes = None

        # for set_object_pose: FREE joint coords [px py pz, quat xyzw] are in the parent-anchor frame —
        # inverting a world target pose gives X_j = X_pj⁻¹ · X_target · X_cj (FK convention, newton articulation.py).
        # Pre-capture X_pj⁻¹·X_cj per-world (safe even if anchors differ per variant). quat kept internally as wxyz.
        if self._has_object:
            self._obj_q0 = int(self._q_start[free_local[0]])            # per-world local coord start
            self._obj_qd0 = int(self._qd_start[free_local[0]])
            obj_j_global = (torch.arange(wc, device=self.device) * self._joints_pw + free_local[0]).long()
            jXp = wp.to_torch(m.joint_X_p)[obj_j_global]                 # (N,7) pos+quat(xyzw)
            jXc = wp.to_torch(m.joint_X_c)[obj_j_global]
            pq = xyzw_to_wxyz(jXp[:, 3:7])
            self._obj_Xp_inv_quat = quat_conj(pq).contiguous()           # X_pj⁻¹
            self._obj_Xp_inv_pos = (-quat_rotate(self._obj_Xp_inv_quat, jXp[:, :3])).contiguous()
            self._obj_Xc_pos = jXc[:, :3].clone()                        # X_cj
            self._obj_Xc_quat = xyzw_to_wxyz(jXc[:, 3:7]).contiguous()

        # init joint_q (= model default, init_pose baked) — reference for restoring PD target on reset
        self._init_jq = wp.to_torch(m.joint_q).view(self.num_envs, self._coords_pw).clone()

        # ---- G5 write-surface caches (DR event write) ----
        # (a) friction: target→(collision shape idx, per-shape world). shape_material_mu is flat across all worlds.
        self._body_shape_masks: dict[tuple, torch.Tensor] = {}        # body names → (n_shapes,) bool
        sb = wp.to_torch(m.shape_body).long()                         # (n_shapes,) global body idx (-1=static)
        assert hasattr(m, "shape_world"), "newton model has no shape_world — cannot map friction DR per-world"
        sw = wp.to_torch(m.shape_world).long()                        # (n_shapes,) world idx (-1=global)
        coll = (wp.to_torch(m.shape_flags) & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        obj_mask = torch.isin(sb, self._all_obj_body_idx) & coll      # object(s) (all FREE child bodies) collision shapes
        robot_mask = (sb >= 0) & ~torch.isin(sb, self._all_obj_body_idx) & coll   # robot (articulated non-object)
        table_mask = (sb == -1) & (sw >= 0) & coll                    # per-world static box (table); ground=global(-1)
        # "nail" is a SUBSET of robot, not a sibling: the pin below writes it once, and a friction DR keeps
        # randomizing it unless the contract's event says `exclude="@bodies.nail"`. Overlapping on purpose —
        # what a reset overwrites is the contract's call to make, not this map's.
        nail_mask = torch.zeros_like(robot_mask)
        nail = spec.robot.nail_friction
        if nail is not None:
            nail_mask = self._body_shape_mask(nail.bodies) & coll
            assert bool(nail_mask.any()), (
                f"robot.nail_friction names {nail.bodies}, none of which carries a collision shape — nothing "
                f"to pin (a body whose collision mask is 0 has none)")
        self._fric_shapes = {t: (mk.nonzero(as_tuple=True)[0], sw[mk])
                             for t, mk in (("object", obj_mask), ("robot", robot_mask),
                                           ("table", table_mask), ("nail", nail_mask))}
        # Who owns the contact list: mjwarp broad/narrowphase (True) or newton-native collide() (False).
        # contact_penetration needs the former — mjwarp's contact.dist is the only signed distance exposed.
        self._mjw_contacts = bool(spec.overrides.get("newton", {}).get("use_mujoco_contacts", True))
        self._pen_luts: dict = {}                 # (links, target) → per-geom (tip column, counterpart flag)
        # The object's MuJoCo body ids (set_object_gravity writes their body_gravcomp). Resolved by INDEX
        # through mjc_body_to_newton rather than by name: an object's body name is an asset detail, while
        # _all_obj_body_idx is what every other object read here already uses. mj_model is one world, so the
        # world-0 slice is the whole mapping.
        b2n = wp.to_torch(self.solver.mjc_body_to_newton)[0].long()      # (nbody,) newton body idx, -1 = none
        obj_w0 = self._all_obj_body_idx[self._all_obj_body_idx < self._bodies_pw] if self._has_object else None
        self._obj_mjc_bodies = ([] if obj_w0 is None
                                else torch.nonzero(torch.isin(b2n, obj_w0)).flatten().tolist())
        # Contact/constraint buffer caps. mjwarp pre-sizes both (GPU: no arena) and DROPS the excess in
        # SILENCE — an unconstrained contact exerts no force, so bodies interpenetrate instead of erroring.
        # genesis pre-sizes too but halts on the same condition ("Exceeding it at runtime halts the
        # simulation with an error"), and MuJoCo's own CPU arena raises. _check_contact_budget restores that:
        # the peak accumulates on the GPU (no sync), and the compare fails loud. naconmax is a GLOBAL pool
        # (nconmax x worlds, one world may borrow another's slack); njmax is PER WORLD and cannot be borrowed.
        self._naconmax = int(self.solver.mjw_data.naconmax)     # contact slots, shared by ALL worlds
        self._njmax = int(self.solver.mjw_data.njmax)            # constraint rows, PER world, hard
        # [nacon, ncollision, worst world's nefc], each in the unit its own cap is judged in. TWO counters
        # share naconmax because two stages allocate from the same pool: the broadphase writes candidate
        # PAIRS (ncollision) and the narrowphase writes CONTACTS (nacon), each with its own atomic and its
        # own `>= naconmax` bail-out. Candidates run several times the contacts, so watching nacon alone
        # misses the gate that trips first.
        self._budget_peak = torch.zeros(3, dtype=torch.int32, device=self.device)
        self._budget_now = torch.zeros(3, dtype=torch.int32, device=self.device)    # same three, THIS step
        # The caps as a tensor. Fixed for the process (put_data runs once, and the captured CUDA graph holds
        # the buffer pointers), so a channel can plot them as the flat line the counters must stay under.
        self._budget_cap = torch.tensor([self._naconmax, self._njmax],
                                        dtype=torch.int32, device=self.device)
        # (b) object mass/inertia default (scale baseline; each reset relative to default → no accumulation)
        self._scale_syncs = []                       # viewer instance-scale pushes; filled below when there is an object
        if self._has_object:
            self._obj_m0 = wp.to_torch(m.body_mass)[self._obj_body_idx].clone()          # (N,)
            self._obj_I0 = wp.to_torch(m.body_inertia)[self._obj_body_idx].clone()       # (N,3,3)
            self._obj_invI0 = wp.to_torch(m.body_inv_inertia)[self._obj_body_idx].clone()
            # (b2) object GEOMETRY scale baseline (set_object_scale). VISUAL shapes included too, so the render
            # tracks the physics size instead of showing one fixed size for every env.
            obj_all = torch.isin(sb, self._all_obj_body_idx)
            self._obj_shapes = obj_all.nonzero(as_tuple=True)[0]                         # (S,) visual+collision
            self._obj_shape_world = sw[obj_all]                                          # (S,) world per shape
            self._obj_scale0 = wp.to_torch(m.shape_scale)[self._obj_shapes].clone()      # (S,3)
            self._obj_radius0 = wp.to_torch(m.shape_collision_radius)[self._obj_shapes].clone()   # (S,)
            assert m.shape_collision_aabb_lower is not None, \
                "model has no shape_collision_aabb — newton's builder did not bake local AABBs"
            self._obj_aabb0 = (wp.to_torch(m.shape_collision_aabb_lower)[self._obj_shapes].clone(),
                               wp.to_torch(m.shape_collision_aabb_upper)[self._obj_shapes].clone())  # (S,3) x2
            # (b3) mjwarp contacts read a GLOBAL vertex array, so shape_scale alone resizes the render and
            # nothing else. Give each world its own copy of the object's collision vertices instead — see
            # mjw_object_scale. Native contacts need none of this (their narrowphase reads shape_scale).
            # A GHOST object (ObjectSpec.collision=False) carries no collision shape, so there is no colliding
            # hull to copy per world and shape_scale reaches the render alone — as on the native path.
            self._mjw_scale = None
            if self._mjw_contacts and self._fric_shapes["object"][0].numel():
                g2s = wp.to_torch(self.solver.mjc_geom_to_newton_shape)[0]        # (ngeom,) world-0 shape
                obj_w0 = self._fric_shapes["object"]
                w0 = set(obj_w0[0][obj_w0[1] == 0].tolist())                      # object collision shapes, world 0
                obj_geoms = [g for g in range(int(g2s.numel())) if int(g2s[g]) in w0]
                assert len(obj_geoms) == len(w0), \
                    f"{len(w0)} object collision shapes but {len(obj_geoms)} MuJoCo geoms — cannot map size DR"
                self._mjw_scale = _install_mjw_object_scale(self.solver, obj_geoms, self.num_envs, self.device)
                print(f"mjw per-env object size: {len(obj_geoms)} geoms x {self._mjw_scale.n_vert} verts "
                      f"x {self.num_envs} worlds = +{self._mjw_scale.bytes / 1e6:.1f} MB", flush=True)
            # (b4) …and the RENDER: newton freezes each instance's scale at set_model, so without this every
            # env draws the build-time size in the viewer and in the .rrd (see viewer_scale). Empty headless.
            self._scale_syncs = make_scale_syncs((self._viewer, self._rerun), m, self._obj_shapes)
        # (c) fixed-base root joint (FIXED & parent==-1; object FREE joint is the only other parent==-1) → joint_X_p z
        jparent = m.joint_parent.numpy()
        root_local = [j for j in range(self._joints_pw)
                      if int(jtype[j]) == int(JointType.FIXED) and int(jparent[j]) == -1]
        assert len(root_local) == 1, f"exactly one fixed-base root joint required — {len(root_local)}"
        self._root_j = (torch.arange(wc, device=self.device) * self._joints_pw + root_local[0]).long()
        self._root_z0 = wp.to_torch(m.joint_X_p)[self._root_j, 2].clone()            # (N,)
        # (d) external-wrench impulse buffer — reapplied to body_f after clear_forces each substep via the
        # _write_body_wrench kernel (graph-resident; lasts 1 control-step). Events write the torch view; the
        # wp views alias the same memory, so the captured graph always reads current values.
        self._ext_wrench = torch.zeros(self.num_envs, 6, device=self.device)         # [force(3), torque(3)] world@COM
        self._ext_wrench_wp = wp.from_torch(self._ext_wrench, dtype=wp.spatial_vector)
        self._obj_body_idx_wp = None                                                 # no object → no wrench scatter
        if self._has_object:
            self._obj_body_idx_i32 = self._obj_body_idx.to(torch.int32).contiguous() # kernel-side index (kept alive)
            self._obj_body_idx_wp = wp.from_torch(self._obj_body_idx_i32)

        # Build-time robot contact friction override (absolute μ) — standalone/eval knob; None = keep MJCF.
        # DR friction events override this at reset when active; standalone (no DR) keeps it for the whole run.
        if spec.robot_friction is not None:
            wp.to_torch(self.model.shape_material_mu)[self._fric_shapes["robot"][0]] = float(spec.robot_friction)
            self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

        # Pinned bodies (robot.nail_friction): written once, never again — they are not in the "robot" set the
        # friction DR writes. mu=0 reaches the solver as FRICTION_EPS (the geometric-mean mixing floors it).
        if spec.robot.nail_friction is not None:
            wp.to_torch(self.model.shape_material_mu)[self._fric_shapes["nail"][0]] = \
                float(spec.robot.nail_friction.mu)
            self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

        # Torque limit (mirror real HW): wire the spec `effort` overrides into the NEWTON SOURCE
        # `model.joint_effort_limit` [Nm] — the JOINT_DOF_PROPERTIES re-sync derives mjw.jnt_actfrcrange = (-e, +e)
        # from it (and re-derives on every re-sync, so it persists; writing mjw.jnt_actfrcrange directly would be
        # clobbered). import_mjcf already seeds joint_effort_limit from the MJCF actuatorfrcrange (autolimits), so
        # this matters when the spec effort DIVERGES from the MJCF — spec is the SSoT. MuJoCo clamps each joint's
        # total actuator force (PD, and PD+gravcomp on actuator-gravcomp joints) to ±e → the joint yields past its
        # max torque. NOTE: jnt_actfrcrange is symmetric, so an asymmetric effort [lo, hi] is applied as
        # ±max(|lo|, |hi|). Applied before the first step_n graph capture.
        jel = wp.to_torch(self.model.joint_effort_limit)         # (joint_dof_count,) newton source, per dof
        n_eff = 0
        for jname, ov in spec.robot.joint_mode_param.items():
            if ov.effort == "default":
                continue
            e = float(ov.effort) if isinstance(ov.effort, (int, float)) else max(abs(ov.effort[0]), abs(ov.effort[1]))
            jel[self._dof_idx([jname]).reshape(-1)] = e          # all worlds' dof of this joint (symmetric ±e)
            n_eff += 1

        # Gravity compensation (mirror real HW): body_gravcomp generates the per-joint gravcomp force
        # (qfrc_gravcomp). For ACTUATOR joints, jnt_actgravcomp routes it through the force-limited actuator —
        # MuJoCo adds it to qfrc_actuator and clamps PD+gravcomp to jnt_actfrcrange (so the joint yields past max
        # torque, and joint_torque reads it directly from qfrc_actuator). PASSIVE (waist/neck) stays an external
        # passive force → uncapped, and NOT in qfrc_actuator. Applied before the first step_n graph capture so
        # ngravcomp>0 bakes the gravcomp kernel into the CUDA graph. METALAB_GRAVCOMP sets the INITIAL on/off.
        # COUPLED joints own their gravcomp share instead: the coupled-PD kernel folds it into the motor
        # torque (τ_m += G⁻ᵀ·τ_g, clamped with the PD) and cancels MuJoCo's application (joint_f −= τ_g) —
        # so they are EXCLUDED from actuator routing and their share stays passive (unclamped → the
        # kernel's cancel is exact).
        self._gc_on = False
        self._act_gc_js: set = set()      # non-coupled joints whose gravcomp rides in qfrc_actuator
        gc = spec.robot.gravcomp
        coupled_js = ({j for g in spec.robot.coupled_groups() for j in g.joints}
                      if handles.get("motor_coupling_on") else set())
        if gc is not None:
            body_ids, _ = _gravcomp.resolve(self.solver, gc.joints())
            body_ids = sorted(set(body_ids) | set(_gravcomp.resolve_bodies(self.solver, gc.passive_bodies)))
            self._gc_on = os.environ.get("METALAB_GRAVCOMP", "1") != "0"
            _gravcomp.set_body_gravcomp(self.solver, body_ids, 1.0 if self._gc_on else 0.0)
            act_js = [j for j in gc.actuator_joints if j not in coupled_js]
            if self._gc_on:
                _gravcomp.set_jnt_actgravcomp(self.solver, act_js)
            # Which NON-coupled joints carry gravcomp inside qfrc_actuator — the component split below
            # subtracts their share out of it (coupled joints get their components from the kernel).
            self._act_gc_js = set(act_js) if self._gc_on else set()
            # JOINT_DOF_PROPERTIES (not JOINT_PROPERTIES) runs _update_joint_dof_properties, which re-derives
            # mjw.jnt_actfrcrange from joint_effort_limit; jnt_actgravcomp needs no notify (direct device write,
            # read per-step, never re-synced).
            self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        elif n_eff:
            self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)   # → recompute jnt_actfrcrange

        # Motor-to-joint coupled PD (parser zeroed the coupled joints' native PD). Built here, BEFORE
        # the first step/graph-capture, over fixed buffers. Launched every substep in _run_substeps,
        # writing control.joint_f (→ qfrc_applied). Coord/dof indices reuse the name→flat-index caches.
        # Owners split by kernel shape: 'hand' (3-DOF triangular, finger/thumb) + 'shoulder' (3-DOF
        # diagonal, arm-gain slice) → one MotorCoupledPDHand; 'arm' (2-DOF 2×2, elbow/wrist) →
        # MotorCoupledPDArm. Each launches every substep and writes disjoint dof columns of
        # control.joint_f. Empty list = coupling off.
        self._coupled_owners: list = []
        self._coupled_col_cache: dict[tuple, tuple[list, list]] = {}
        if handles.get("motor_coupling_on"):
            cg = spec.robot.coupled_groups()
            hand = [g for g in cg if g.kind in ("hand", "shoulder")]
            arm = [g for g in cg if g.kind == "arm"]
            # gravcomp fold input: MuJoCo's qfrc_gravcomp buffer (nworld, nv) + per-joint mjc dof idx.
            # Only when gravcomp is ON — off → None → every group's fold flag is forced 0.
            gc_buf = self.solver.mjw_data.qfrc_gravcomp if self._gc_on else None
            if hand:
                gh = [load_hand_group(g.params_key, g.joints, model_file=g.model_file,
                                      gain_slice=g.arm_slice) for g in hand]
                nh = [j for grp in gh for j in grp["joints"]]
                self._coupled_owners.append(
                    MotorCoupledPDHand(gh, self._coord_idx(nh), self._dof_idx(nh), self.num_envs, self.device,
                                       gravcomp=gc_buf,
                                       gc_dof=self._jt_mjc_dofs(nh) if gc_buf is not None else None))
            if arm:
                ga = [load_arm_group(g.params_key, g.joints, model_file=g.model_file, arm_slice=g.arm_slice)
                      for g in arm]
                na = [j for grp in ga for j in grp["joints"]]
                self._coupled_owners.append(
                    MotorCoupledPDArm(ga, self._coord_idx(na), self._dof_idx(na), self.num_envs, self.device,
                                      gravcomp=gc_buf,
                                      gc_dof=self._jt_mjc_dofs(na) if gc_buf is not None else None))

        # reset to init pose right after model load (init viewer/PD target immediately)
        self.reset_idx(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    # --- name → all-world flat index tensor (cached) ---
    def _local_joint(self, name: str) -> int:
        for i in range(self._joints_pw):
            lbl = self._labels[i]
            if lbl == name or lbl.endswith(f"/{name}"):
                return i
        raise ValueError(f"joint '{name}' not found")

    def _coord_idx(self, names) -> torch.Tensor:
        key = tuple(names)
        t = self._coord_cache.get(key)
        if t is None:
            base = torch.tensor([int(self._q_start[self._local_joint(n)]) for n in names], device=self.device)
            t = (self._world_off + base).long()          # (N, k)
            self._coord_cache[key] = t
        return t

    def _dof_idx(self, names) -> torch.Tensor:
        key = tuple(names)
        t = self._dof_cache.get(key)
        if t is None:
            base = torch.tensor([int(self._qd_start[self._local_joint(n)]) for n in names], device=self.device)
            t = (self._world_off_dof + base).long()       # (N, k)
            self._dof_cache[key] = t
        return t

    def _body_idx(self, name: str) -> torch.Tensor:
        t = self._body_cache.get(name)
        if t is None:
            local = next((i for i in range(self._bodies_pw)
                          if self._body_labels[i] == name or self._body_labels[i].endswith(f"/{name}")), None)
            assert local is not None, f"body '{name}' not found"
            t = (self._world_off_body + local).long()      # (N,)
            self._body_cache[name] = t
        return t

    # --- read (canonical: wxyz·world·SI, (N,...)) ---
    def _cached(self, key, fn):
        """Per-policy-step read cache: recompute only when the physics state changed (_state_epoch)."""
        if self._rcache_epoch != self._state_epoch:
            self._rcache.clear()
            self._rcache_epoch = self._state_epoch
        v = self._rcache.get(key)
        if v is None:
            v = fn()
            self._rcache[key] = v
        return v

    def joint_pos(self, names):
        return self._cached(("jp", tuple(names)), lambda: self._t_joint_q[self._coord_idx(names)])

    def joint_vel(self, names):
        return self._cached(("jv", tuple(names)), lambda: self._t_joint_qd[self._dof_idx(names)])

    def joint_torque(self, names):
        # Actual actuator torque the solver applied = MuJoCo ``qfrc_actuator`` (nworld, nv). This is already
        # clamped to the joint torque limit (jnt_actfrcrange, set from spec effort) and, for actuator-gravcomp
        # joints, INCLUDES the gravcomp force — jnt_actgravcomp routes it through the force-limited actuator, so
        # PD+gravcomp is capped together and the joint yields past max torque (mirrors real HW). Passive-channel
        # gravcomp is applied as an external force and is NOT in qfrc_actuator (correctly excluded). Read each
        # joint's MuJoCo dof column (name → jnt_dofadr), cached per names-tuple.
        return self._cached(("jt", tuple(names)), lambda: self._joint_torque(names))

    def _joint_torque(self, names):
        mjc = self._jt_mjc_dofs(names)
        t = wp.to_torch(self.solver.mjw_data.qfrc_actuator)[:, mjc]        # (N, k) — advanced idx → copy
        # coupled joints' native PD is off (qfrc_actuator ≈ 0 there); overlay the coupled-PD torque each
        # owner produced (tau_torch, last substep) so the readout reflects real applied torque.
        for oi, o in enumerate(self._coupled_owners):
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                t[:, cols] = o.tau_torch[:, tcols]
        return t

    def joint_torque_pd(self, names):
        """PD component of the applied actuator torque, PRE-clamp (see api/state.py)."""
        return self._cached(("jtpd", tuple(names)), lambda: self._joint_torque_components(names)[0])

    def joint_torque_gravcomp(self, names):
        """Gravity-feedforward component of the applied actuator torque, PRE-clamp (see api/state.py)."""
        return self._cached(("jtgc", tuple(names)), lambda: self._joint_torque_components(names)[1])

    def _joint_torque_components(self, names):
        """(pd, gravcomp) split of :meth:`joint_torque`, per joint — computed together (one read of the
        shared buffers) and cached separately.

        * COUPLED joints: straight from the coupled-PD kernel, which forms Gᵀ·τ_m^PD and τ_g before its
          clamp — exact, and they sum to the applied torque whenever no motor saturates.
        * NON-coupled ACTUATOR joints: MuJoCo hands back only the post-clamp sum (``qfrc_actuator``), so
          the gravity share is ``qfrc_gravcomp`` where jnt_actgravcomp routed it (``_act_gc_js``) and the
          PD share is the remainder. Here the clamp gap is therefore attributed to PD, not split.
        * PASSIVE-channel gravcomp joints (waist/neck) apply an EXTERNAL force, not motor torque: excluded
          from both components, exactly as they are excluded from ``joint_torque``.
        """
        mjc = self._jt_mjc_dofs(names)
        gc = torch.zeros(self.num_envs, len(names), device=self.device)
        if self._act_gc_js:
            act_cols = [i for i, n in enumerate(names) if n in self._act_gc_js]
            if act_cols:
                qg = wp.to_torch(self.solver.mjw_data.qfrc_gravcomp)[:, mjc[act_cols]]
                gc[:, act_cols] = qg.to(gc.dtype)
        pd = self.joint_torque(names) - gc           # non-coupled: post-clamp sum minus the gravity share
        for oi, o in enumerate(self._coupled_owners):
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                pd[:, cols] = o.tau_pd_torch[:, tcols]
                gc[:, cols] = o.tau_gc_torch[:, tcols]
        return pd, gc

    def _coupled_cols(self, oi, owner, names):
        """names → (cols into `names`, cols into owner.tau_torch) for that owner's coupled joints present."""
        key = (oi, tuple(names))
        r = self._coupled_col_cache.get(key)
        if r is None:
            jmap = {j: i for i, j in enumerate(owner.joints)}
            cols, tcols = [], []
            for i, n in enumerate(names):
                if n in jmap:
                    cols.append(i)
                    tcols.append(jmap[n])
            r = (cols, tcols)
            self._coupled_col_cache[key] = r
        return r

    def set_coupled_float_damping(self, off: bool):
        """Torque/float mode support: zero the coupled D gain on the gravcomp-fold groups (arm) so the
        float is gravity feedforward + joint friction only — the real gravity-comp mode sends the
        feedforward alone, with dissipation from physical friction. ``off=False`` restores Position-mode
        damping. Hand groups keep their D (they still position-hold in float). Buffer-content write →
        CUDA-graph safe."""
        for o in self._coupled_owners:
            o.set_fold_kd_zero(off)

    def _jt_mjc_dofs(self, names):
        key = tuple(names)
        t = self._jt_mjc_dof_cache.get(key)
        if t is None:
            t = torch.tensor(_gravcomp.resolve_dofs(self.solver, list(names)), dtype=torch.long, device=self.device)
            self._jt_mjc_dof_cache[key] = t
        return t

    def joint_limits(self, names):
        # world-0 dofs only → (J,), (J,), matching genesis (per-joint, broadcast over envs). Limits are
        # identical across worlds, and returning (N,J) would broadcast-corrupt event clamps (e.g.
        # reset_joints_by_offset: pos (k,J) clamped against (N,J) inflates to (N,J), mismatching env_ids).
        didx = self._dof_idx(names)[0]
        lo = wp.to_torch(self.model.joint_limit_lower)[didx]
        hi = wp.to_torch(self.model.joint_limit_upper)[didx]
        return lo, hi

    def _body_pose(self, name):
        """(pos, quat wxyz) for one body — ONE gather off body_q per (body, step), shared by
        body_pos/body_quat (obs and reward both ask, often for the same body)."""
        def _read():
            bq = self._t_body_q[self._body_idx(name)]                  # (N,7) pos + quat(xyzw)
            return bq[:, :3], bq[:, [6, 3, 4, 5]]                      # pos, quat(wxyz)
        return self._cached(("bp", name), _read)

    def body_pos(self, name):
        return self._body_pose(name)[0]

    def body_quat(self, name):
        return self._body_pose(name)[1]

    def _object_pose(self):
        def _read():
            bq = self._t_body_q[self._obj_body_idx]
            return bq[:, :3], bq[:, [6, 3, 4, 5]]                      # pos, quat(wxyz)
        return self._cached("op", _read)

    def object_pos(self):
        return self._object_pose()[0]

    def object_quat(self):
        return self._object_pose()[1]

    # body_qd: first 3 = linear vel [m/s], last 3 = angular vel [rad/s], world (newton State convention).
    def _object_vel(self):
        return self._cached("ov", lambda: self._t_body_qd[self._obj_body_idx])

    def object_lin_vel(self):
        return self._object_vel()[:, :3]

    def object_ang_vel(self):
        return self._object_vel()[:, 3:6]

    def _body_vel(self, name):
        return self._cached(("bv", name), lambda: self._t_body_qd[self._body_idx(name)])

    def body_lin_vel(self, name):
        return self._body_vel(name)[:, :3]

    def body_ang_vel(self, name):
        return self._body_vel(name)[:, 3:6]

    def _ensure_contacts(self):
        """Refresh self._contacts from the current physics state — but only ONCE per state change.
        Within a policy step every contact reader (obs terms, reward gate, terminate) shares this single
        update_contacts pass; a stale-epoch guard skips the redundant re-solve. Lazy-alloc is kept AFTER
        sensor creation so the requested force/force_matrix attributes are included (callers create their
        sensor first, then call this)."""
        if self._contacts is None:
            self._contacts = newton.Contacts(
                self.solver.get_max_contact_count(), 0,
                requested_attributes=self.model.get_requested_contact_attributes())
        if self._contacts_epoch != self._state_epoch:
            self.solver.update_contacts(self._contacts, self.state_0)
            self._contacts_epoch = self._state_epoch

    def contact_force(self, link_names):
        """Net external contact force per link (N, K, 3), world, column k = ``link_names[k]``. SensorContact
        (net per body)."""
        sensor = self._contact_sensor(link_names)        # sensor requests the force attr on the model
        self._ensure_contacts()                          # cached: one update_contacts per state change
        sensor.update(self.state_0, self._contacts)
        return wp.to_torch(sensor.total_force).view(self.num_envs, len(link_names), 3)

    def _local_bodies(self, link_names) -> list[int]:
        """link names → per-world body indices, IN THE CALLER'S ORDER.

        NOT sorted. The sensors below lay their output out as one row per entry of ``sensing_bodies``, so
        that order IS the column order of the (N, K, 3) read — sorting here silently permuted every
        per-column consumer. On allex_right the fingertip list [Index, Middle, Ring, Little, Thumb] maps to
        bodies [60, 65, 70, 75, 55], so sorting moved THUMB into column 0 while the contract (and the
        dashboard's dim labels) still called that column Index: a thumb-only touch was reported, plotted and
        fed to the critic as an index-finger touch. Order-insensitive consumers (grip COUNT, amax, mean)
        were unaffected, which is why it survived — genesis, which builds its columns in caller order, has
        always been right, so this was also a silent sim2sim divergence."""
        return [next(i for i in range(self._bodies_pw)
                     if self._body_labels[i] == n or self._body_labels[i].endswith(f"/{n}"))
                for n in link_names]

    def _body_shape_mask(self, body_names) -> torch.Tensor:
        """(n_shapes,) bool — shapes belonging to these bodies, in every world. Cached per name tuple."""
        key = tuple(body_names)
        mk = self._body_shape_masks.get(key)
        if mk is None:
            gidx = torch.tensor([w * self._bodies_pw + lb for w in range(self.num_envs)
                                 for lb in self._local_bodies(list(key))], device=self.device)
            mk = torch.isin(wp.to_torch(self.model.shape_body).long(), gidx)
            self._body_shape_masks[key] = mk
        return mk

    def _contact_sensor(self, link_names):
        key = tuple(link_names)
        s = self._sensors.get(key)
        if s is None:
            locals_ = self._local_bodies(link_names)
            bodies = [w * self._bodies_pw + lb for w in range(self.num_envs) for lb in locals_]
            s = newton.sensors.SensorContact(self.model, sensing_bodies=bodies, measure_total=True)
            self._sensors[key] = s
        return s

    def contact_force_with(self, link_names, target):
        """Contact force per link against a **specific counterpart (target)** only (N, K, 3), world, column
        k = ``link_names[k]``. A counterpart-scoped version of net contact_force (hammer vs table).
        target={object|robot|table} — uses G5 _fric_shapes collision shapes as the counterpart. Sum
        force_matrix (per-counterpart) over the counterpart axis → net vs target."""
        s = self._counterpart_sensor(link_names, target)
        self._ensure_contacts()                          # cached: one update_contacts per state change
        s.update(self.state_0, self._contacts)
        fm = wp.to_torch(s.force_matrix)                  # (N*K, max_cp, 3) — sum over cp axis = net against all target shapes
        return fm.sum(dim=1).view(self.num_envs, len(link_names), 3)

    def contact_penetration(self, link_names, target):
        """Deepest overlap per link against **target**, (N, K) [m], positive = overlap, column k =
        ``link_names[k]``.

        Read off mjwarp's raw contact list, NOT through a SensorContact: the sensors report force, friction
        and position, and ``update_contacts`` drops ``dist`` on its way into newton's ``Contacts`` (only
        ``requires_grad`` allocates a distance array), so ``mjw_data.contact.dist`` is the one place a signed
        distance survives. Everything else — the sign flip, the per-link max — is in _scatter_penetration.

        Works on BOTH contact paths, and that is measured, not assumed. It used to assert
        ``use_mujoco_contacts=True`` on the grounds that newton-native collide() exposes no signed distance —
        true of newton's own ``Contacts``, but irrelevant here, because what this reads is mjwarp's buffer and
        the native bridge fills it: ``dist`` is written by both the full and the fast conversion path
        (newton solvers/mujoco/kernels.py, write_contact). Measured on 4hammers, 64 envs, hammer resting on
        the table: use_mujoco_contacts=False gives nacon=440 with dist populated 440/440, and this method
        reports overlap in 63 of 64 envs (max 0.394 mm) against 62 of 64 (max 0.032 mm) on the mjwarp path.
        The depth DIFFERS between the paths — native runs gap=0 with one collide() per control step — but
        both report a real signed distance, so nothing here needs a path gate."""

        def _read():
            geom_tip, geom_cp = self._pen_geom_lut(link_names, target)
            out = torch.zeros(self.num_envs, len(link_names), device=self.device)
            d = self.solver.mjw_data
            wp.launch(_scatter_penetration, dim=d.naconmax,
                      inputs=[d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid, geom_tip, geom_cp],
                      outputs=[wp.from_torch(out, dtype=wp.float32)], device=d.contact.dist.device)
            return out
        return self._cached(("cpen", tuple(link_names), target), _read)

    def _pen_geom_lut(self, link_names, target):
        """Per-GEOM lookup tables for :meth:`contact_penetration`: ``(geom → output column, geom → is
        counterpart)``. Built once per (links, target) and reused — the mapping is build-time geometry.

        Indexed by mjwarp geom because that is what a contact carries, and the index is PER-WORLD, so one
        table serves every world: which link a geom belongs to is world-invariant (only the newton shape
        behind it differs, which is why world 0's row of ``mjc_geom_to_newton_shape`` is enough).

        The counterpart side is matched by SHAPE and not by body, unlike the fingertip side: the table is
        static, so its shapes carry ``shape_body == -1`` and there is no body index to compare against."""
        key = (tuple(link_names), target)
        lut = self._pen_luts.get(key)
        if lut is None:
            g2s = wp.to_torch(self.solver.mjc_geom_to_newton_shape)[0].long()   # (ngeom,) world-0 shape, -1 = no shape
            sb = wp.to_torch(self.model.shape_body).long()                      # (n_shapes,) body idx, -1 = static
            live = g2s >= 0
            body_of = torch.where(live, sb[g2s.clamp(min=0)], torch.full_like(g2s, -2))   # -2 ≠ any real body/static
            tip = torch.full_like(g2s, -1, dtype=torch.int32)
            for k, b in enumerate(self._local_bodies(link_names)):   # world 0: local body idx == global (caller order)
                tip[body_of == b] = k
            cp_shapes, cp_worlds = self._fric_shapes[target]
            cp = torch.zeros_like(tip)
            cp[live & torch.isin(g2s, cp_shapes[cp_worlds == 0])] = 1
            missing = [n for k, n in enumerate(link_names) if not (tip == k).any()]
            assert not missing, f"contact_penetration: no MuJoCo geom for {missing} — nothing to measure overlap on"
            assert cp.any(), f"contact_penetration: no MuJoCo geom for target '{target}'"
            lut = (wp.from_torch(tip.contiguous()), wp.from_torch(cp.contiguous()))
            self._pen_luts[key] = lut
        return lut

    #: [N] below this a fingertip is not touching — same floor the grip predicate and the contact obs use.
    _ARROW_FORCE_MIN = 1.0e-3

    def _rerun_world_offsets(self):
        """(N,3) tile offsets the rerun viewer draws each world at. Cached: they are computed once when the
        model is set and nothing in the recording path moves them."""
        if self._rerun_offsets is None:
            o = getattr(self._rerun, "world_offsets", None)
            arr = (np.zeros((self.num_envs, 3), np.float32) if o is None
                   else np.asarray(o.numpy()).reshape(-1, 3)[:self.num_envs])
            self._rerun_offsets = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        return self._rerun_offsets

    def _fingertip_contact_arrows(self):
        """Fingertip↔object contact NORMALS for the rerun overlay: ``(origins, forces, tip_idx)``.

        The normal component is ``force_matrix - force_matrix_friction`` (total force minus its tangential
        part), summed over the object's collision shapes; the origin is those shapes' contact points averaged
        with the same weights. So a fingertip pressing two faces of the hammer yields ONE arrow at the
        resultant — ``position_matrix`` is a force-weighted mean of contact midpoints, not a per-contact
        point, which is the resolution limit of reading contacts off one collision hull per distal link.

        World offsets are added: the worlds physically overlap at the origin (separate_worlds, spacing 0)
        while the recording tiles them, so without this every env's arrows land on env 0's tile.

        Returns three arrays of length M = fingertips touching this step (M may be 0), or None when the task
        has no fingertips or no object."""
        tips = self.spec.robot.fingertips
        if not tips or not self._has_object:
            return None
        s = self._counterpart_sensor(tips, "object")
        self._ensure_contacts()
        s.update(self.state_0, self._contacts)               # cheap re-accumulation; obs may already have run it
        nrm = wp.to_torch(s.force_matrix) - wp.to_torch(s.force_matrix_friction)   # (N*K, C, 3) normal only
        w = nrm.norm(dim=-1, keepdim=True)                   # weight the point average by |normal|
        tot = w.sum(dim=1)                                   # (N*K, 1)
        vec = nrm.sum(dim=1)                                 # (N*K, 3) resultant normal
        org = (wp.to_torch(s.position_matrix) * w).sum(dim=1) / tot.clamp(min=1e-12)
        # rows are env-major ([env0 × tips, env1 × tips, …] — see _counterpart_sensor), so the tile offset
        # repeats per env.
        org = org + self._rerun_world_offsets().repeat_interleave(len(tips), dim=0)
        keep = tot.squeeze(-1) > self._ARROW_FORCE_MIN
        idx = torch.arange(len(tips), device=self.device).repeat(self.num_envs)[keep]
        return (org[keep].detach().cpu().numpy(), vec[keep].detach().cpu().numpy(),
                idx.detach().cpu().numpy())

    def _counterpart_sensor(self, link_names, target):
        key = (tuple(link_names), target)
        s = self._cp_sensors.get(key)
        if s is None:
            locals_ = self._local_bodies(link_names)     # caller order — see _local_bodies
            bodies = [w * self._bodies_pw + lb for w in range(self.num_envs) for lb in locals_]
            cp_shapes = self._fric_shapes[target][0].tolist()   # target collision shapes (all worlds) — sensor resolves per-world
            assert cp_shapes, f"contact_force_with: no collision shapes for target '{target}'"
            s = newton.sensors.SensorContact(self.model, sensing_bodies=bodies,
                                             counterpart_shapes=cp_shapes, measure_total=False)
            self._cp_sensors[key] = s
        return s

    # --- control / step / reset ---
    def set_joint_targets(self, names, targets):
        wp.to_torch(self.control.joint_target_q)[self._coord_idx(names)] = targets

    def step(self, render: bool = True):
        self.step_n(1, render=render)

    def step_n(self, n: int, render: bool = True):
        # render=False advances physics but emits no viewer frame — used for the off-screen settle at reset.
        # n physics steps = one policy step when n == decimation (env_driver calls step_n(decimation) when
        # available, else step() n times). CUDA-graph path (newton-idiomatic: examples capture their own
        # loop, no core helper): a headless control step is pure warp kernels (incl. the wrench scatter) →
        # capture the WHOLE n×substeps loop once, replay it with a single capture_launch per policy step so
        # the hundreds of physics kernels launch from the GPU with zero per-kernel host overhead. First
        # graphed step warms up lazy allocs + JIT (a real advance) then records the graph (capture mode
        # records without executing → no double advance); later steps replay it. Viewer runs stay eager.
        use_graph = self._graphable
        if use_graph and self._graph is not None and self._graph_n != n:
            self._graph = None                             # span changed (e.g. step() after step_n) → recapture
        if use_graph and self._graph is None:
            for _ in range(n):
                self._run_substeps()                       # warm up + advance this step
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as cap:
                for _ in range(n):
                    self._run_substeps()                   # capture mode: record kernels, no execution
            self._graph = cap.graph
            self._graph_n = n
        elif use_graph:
            wp.capture_launch(self._graph)                 # GPU-only replay (advances one control step)
        else:
            for _ in range(n):
                self._run_substeps(host_ops=True)          # eager: viewer/external-wrench per-substep host ops
        self._state_epoch += 1                             # physics advanced → contact cache stale
        self._check_contact_budget()                       # fail loud on a silent buffer overflow
        self._sim_time += self._sim_dt * self._substeps * n
        if render:                                         # --viz / record: one frame per control step (skipped during off-screen settle)
            self._emit_viewer_frames(advance=True)

    def _check_contact_budget(self):
        """Track per-world contact/constraint use and FAIL if either buffer overflowed this step.

        Both counters keep counting past their cap — that is how mjwarp reports an overflow (``nacon`` is
        documented as "larger than naconmax means an overflow occurred", and ``nefc`` is an atomic_add whose
        writers bail out with ``if efcid >= njmax: return``) — so a peak above the cap is the signal. What it
        costs to miss it: the dropped contacts generate NO constraint, so the solver never pushes those bodies
        apart and they sink into each other, silently, while the force reads say 0 — which also disarms any
        force-threshold termination watching for exactly that.

        WHICH CAP BOUNDS WHAT — the two are not symmetric, so each is measured in its own unit. ``njmax`` is
        per world and cannot be borrowed, so the WORST world's ``nefc`` is compared to it. Contacts are one
        flat pool of ``naconmax`` slots that every world draws from through a single global atomic
        (``pairid = atomic_add(ncollision, 0, 1)``, then ``if pairid >= naconmax: return``), so the slot a
        contact gets depends only on when its thread arrived — no world has a reserved share. A world may
        therefore sit far ABOVE ``naconmax / nworld`` legally, and once the pool is full the contacts that get
        dropped are simply the late ones, which may belong to a world that generated very few. That is why the
        TOTAL is what gets judged: a per-world count could not even identify who lost a contact.

        TWO GATES SHARE naconmax. The broadphase counts candidate PAIRS into ``ncollision`` and bails with
        ``if pairid >= naconmax: return`` — BEFORE the narrowphase ever runs — while the narrowphase counts
        CONTACTS into ``nacon`` and only writes ``if cid < naconmax``. Candidates outnumber contacts several
        times over, so ``ncollision`` is the gate that trips first and ``nacon`` would still look small when
        it did. Both are watched.

        Only the final compare syncs, which the driver was going to do here anyway (``dones.any()``)."""
        d = self.solver.mjw_data
        cur = torch.stack([wp.to_torch(d.nacon)[0], wp.to_torch(d.ncollision)[0],
                           wp.to_torch(d.nefc).amax()])
        self._budget_now.copy_(cur)
        torch.maximum(self._budget_peak, cur, out=self._budget_peak)
        nacon, ncollision, nefc = self._budget_peak.tolist()
        assert max(nacon, ncollision) <= self._naconmax, (
            f"mjwarp CONTACT buffer overflow: {nacon} contacts / {ncollision} broadphase candidates across "
            f"all {self.num_envs} worlds vs naconmax {self._naconmax} (= nconmax "
            f"{self._naconmax // max(1, self.num_envs)}/world x {self.num_envs}). Whichever counter is over "
            f"is the stage that dropped work: candidates die in the broadphase before narrowphase sees them, "
            f"contacts die on write. The pool is shared, so a few busy worlds can starve the rest — and a "
            f"dropped contact exerts no force → interpenetration. "
            f"Raise overrides.newton.nconmax (or contact_scale).")
        assert nefc <= self._njmax, (
            f"mjwarp CONSTRAINT buffer overflow: {nefc} rows in one world vs njmax {self._njmax}. This cap is "
            f"PER WORLD and cannot borrow from another world's slack. A contact costs 3 rows (elliptic cone) "
            f"or 4 (pyramidal), plus equality and joint-limit rows; rows past the cap are dropped and those "
            f"contacts exert no force → interpenetration. Raise overrides.newton.njmax (or contact_scale).")

    def contact_budget(self) -> dict:
        """Peak buffer use so far vs the caps — ``{nacon, ncollision, naconmax, nefc, njmax}``.

        ``nacon`` (contacts) and ``ncollision`` (broadphase candidates) are both summed over every world and
        both bounded by ``naconmax``; ``nefc`` is the worst single world, bounded by ``njmax``. Peaks are
        monotonic, so this answers "how close did we ever get"."""
        nacon, ncollision, nefc = self._budget_peak.tolist()
        return {"nacon": int(nacon), "ncollision": int(ncollision), "naconmax": self._naconmax,
                "nefc": int(nefc), "njmax": self._njmax}

    def contact_budget_t(self) -> torch.Tensor:
        """``[nacon, ncollision, nefc]`` this step, then the caps ``[naconmax, njmax]`` → (5,), no host sync.

        Values and caps together, because the cap is the only thing that makes a value readable: a channel
        plotting both draws the overflow line, and a counter crossing it IS the overflow. (The running peak is
        kept separately for the assert; a plot does not need it — the line does that job.)"""
        return torch.cat([self._budget_now, self._budget_cap])

    def coupled_kq(self):
        """Joint-space stiffness K_q of every coupled group at the CURRENT pose (env 0), for the dashboard's
        Joint Kp channel: ``[{name, joints, K}]`` in N*m/rad. Host-side (see motor_coupling._kq) — the hot
        loop and the captured graph are untouched."""
        return [e for o in self._coupled_owners
                for e in o.kq(self.joint_pos(o.joints)[0].detach().cpu().numpy())]

    def coupled_tau_lim(self):
        """Joint-space torque limit of every coupled group at the CURRENT pose and speed (env 0), for the
        dashboard's Joint Torque Limit channel: ``[{name, joints, part, hi, lo}]`` in N*m. Host-side (see
        motor_coupling._tau_lim) — the hot loop and the captured graph are untouched."""
        return [e for o in self._coupled_owners
                for e in o.tau_lim(self.joint_pos(o.joints)[0].detach().cpu().numpy(),
                                   self.joint_vel(o.joints)[0].detach().cpu().numpy())]

    def motor_gain_warnings(self):
        """Gain-consistency warnings for the gains the coupled-PD buffers hold now (dashboard + log)."""
        return [w for o in self._coupled_owners for w in o.gain_warnings()]

    def reload_motor_gains(self):
        """Re-read robot_model.json gains into the coupled-PD buffers (contents only → the captured CUDA
        graph keeps working). No coupled groups (control_mode != motor) → nothing to reload."""
        return [n for o in self._coupled_owners for n in o.reload_gains()]

    def render_frame(self):
        """One viewer frame at the CURRENT sim time, no substeps (standalone Pause). Same calls step_n makes
        after advancing — newton's GL viewer only pumps window input inside a frame, so a paused run must keep
        calling this or the window stops responding."""
        self._emit_viewer_frames()

    def _emit_viewer_frames(self, advance: bool = False):
        """Emit ONE frame on every attached viewer at the current sim time: the ``--viz`` viewer, the .rrd
        recording viewer, or both when an interactive run also records. Each has its own origin-axes overlay
        because those buffers are per-viewer.

        A recording run has NO interactive viewer, so ``_viewer`` is None and the CUDA graph stays enabled
        (``_graphable``) — the graph captures ``_run_substeps`` only, and this call sits outside it.

        ``advance`` = physics moved since the last frame, so this is a NEW sample; the rerun clock ticks only
        then. A repaint that did not advance (standalone Pause) re-sends the same sample rather than inventing
        one, which is what keeps frame k of the .rrd equal to policy step k.
        """
        for v, axes in ((self._viewer, self._origin_axes), (self._rerun, self._rerun_axes)):
            if v is None:
                continue
            v.begin_frame(self._rerun_step * self._rerun_dt if v is self._rerun else self._sim_time)
            if v is self._rerun:
                mark_step(self._rerun_step)     # the axis the report and the viewer both key to
            v.log_state(self.state_0)
            if v is self._rerun:                # fingertip contact normals, recording only
                log_contact_arrows(self._fingertip_contact_arrows())
            if axes is not None:                           # left panel → MetaLab → Show Origin / Show Grasp
                axes.draw(self.state_0)
            v.end_frame()
        if advance and self._rerun is not None:
            self._rerun_step += 1

    def _run_substeps(self, host_ops: bool = False):
        """One control step = ``self._substeps`` solver substeps. Ping-pongs local refs over the two fixed
        state buffers and leaves the latest state in ``self.state_0`` (natural for even substeps, e.g.
        hammer=2; an explicit mirror covers odd counts). Pure warp kernels when ``host_ops`` is False
        (graph-capturable); ``host_ops`` adds the viewer / external-wrench torch writes for the eager path.
        self.state_0/self.state_1 are never reassigned, so the captured graph and every getter/reset that
        reads self.state_0 share one canonical buffer across replays."""
        a, b = self.state_0, self.state_1
        for _ in range(self._substeps):
            a.clear_forces()
            if host_ops and self._viewer is not None:
                self._viewer.apply_forces(a)               # viewer external forces (e.g. mouse grab)
            if self._has_object:                               # DR external-wrench impulse (graph-resident;
                wp.launch(_write_body_wrench, dim=self.num_envs,   # unfired envs write 0 = no-op). No object → skip.
                          inputs=[self._obj_body_idx_wp, self._ext_wrench_wp, a.body_f],
                          device=self.model.device)
            for o in self._coupled_owners:                 # motor-space coupled PD (hand + wrist) → control.joint_f
                o.launch(a.joint_q, a.joint_qd, self.control.joint_target_q, self.control.joint_f)
            if self._contacts_native is not None:
                # Newton-native collision, EVERY substep, against the state just integrated (upstream
                # anymal_c_walk / selection_articulations pattern). The rate is therefore physics.substeps:
                # detecting less often leaves the contact geometry stale, which the contract's solref
                # stiffness turns into a divergence feedback loop. Measured at 4096 envs (hull 32/32):
                # one collide() is ~0.9 ms with the hull caps on, ~3.64 ms with uncapped hulls — the cost
                # is the support-function vertex scan, so the caps are what make this cadence affordable.
                self.model.collide(a, self._contacts_native)
            self.solver.step(a, b, self.control, self._contacts_native, self._sim_dt)
            a, b = b, a
        if a is not self.state_0:                          # odd substeps → latest ended in state_1; mirror to state_0
            wp.copy(self.state_0.joint_q, a.joint_q)
            wp.copy(self.state_0.joint_qd, a.joint_qd)
            wp.copy(self.state_0.body_q, a.body_q)
            wp.copy(self.state_0.body_qd, a.body_qd)

    def set_object_pose(self, env_idx: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor):
        """Write object pose (K env, world pos (K,3) + quat (K,4) wxyz) + zero velocity.

        Inverts to FREE joint coords (joint_q 7-coords, parent-anchor frame), writes them, then eval_fk updates body_q.
        SolverMuJoCo has update_data_interval=1 (default), so the next step syncs state_0 to mjwarp — write guaranteed.
        """
        if int(env_idx.numel()) == 0:
            return
        # X_j = (X_pj⁻¹ · X_target) · X_cj — transform composition (p1+q1·p2, q1⊗q2)
        p1 = self._obj_Xp_inv_pos[env_idx] + quat_rotate(self._obj_Xp_inv_quat[env_idx], pos)
        q1 = quat_mul(self._obj_Xp_inv_quat[env_idx], quat)
        pj = p1 + quat_rotate(q1, self._obj_Xc_pos[env_idx])
        qj = quat_mul(q1, self._obj_Xc_quat[env_idx])
        jq = wp.to_torch(self.state_0.joint_q).view(self.num_envs, self._coords_pw)
        jqd = wp.to_torch(self.state_0.joint_qd).view(self.num_envs, self._dofs_pw)
        q0, d0 = self._obj_q0, self._obj_qd0
        jq[env_idx, q0:q0 + 3] = pj
        jq[env_idx, q0 + 3:q0 + 7] = wxyz_to_xyzw(qj)
        jqd[env_idx, d0:d0 + 6] = 0.0
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._state_epoch += 1                             # state_0 mutated → contact cache stale

    # ---- G5 event write-surface (called by DR terms via EnvDriver; symmetric across both engines) ----
    def set_joint_positions(self, names, env_idx, pos, vel):
        """Teleport ``names`` joints to pos/vel for env_idx + hold PD target at pos + eval_fk.
        Writes state (not model) so no notify needed — same path as set_object_pose."""
        if int(env_idx.numel()) == 0:
            return
        cidx = self._coord_idx(names)[env_idx]              # (K, J)
        didx = self._dof_idx(names)[env_idx]
        wp.to_torch(self.state_0.joint_q)[cidx] = pos
        wp.to_torch(self.state_0.joint_qd)[didx] = vel
        wp.to_torch(self.control.joint_target_q)[cidx] = pos   # prevent next-step pullback (mirrors reset_idx)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._state_epoch += 1                             # state_0 mutated → contact cache stale

    def set_object_friction(self, target, env_idx, mu, exclude=()):
        """Set friction μ (absolute) of ``target`` ({object|robot|table}) collision shapes per-world + notify.
        shape_material_mu is flat across worlds → scatter per-env μ to each shape's world.

        ``exclude`` = body names whose shapes this write skips, so a pinned surface (robot.nail_friction)
        keeps the value the build gave it while the rest of the target is randomized."""
        if int(env_idx.numel()) == 0:
            return
        shapes, worlds = self._fric_shapes[target]
        if exclude:
            keep_shape = ~self._body_shape_mask(exclude)[shapes]
            shapes, worlds = shapes[keep_shape], worlds[keep_shape]
        if shapes.numel() == 0:
            return
        mu_w = torch.zeros(self.num_envs, device=self.device)
        mu_w[env_idx] = mu
        keep = torch.isin(worlds, env_idx)
        wp.to_torch(self.model.shape_material_mu)[shapes[keep]] = mu_w[worlds[keep]]
        self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    def set_object_mass(self, env_idx, scale):
        """Scale object body mass·inertia(+inv) by per-env scale relative to default + notify. Rigid-body uniform
        scale → inertia scales identically (genesis is mass-only → asymmetric rotational dynamics sim2sim, documented)."""
        if int(env_idx.numel()) == 0:
            return
        bi = self._obj_body_idx[env_idx]
        s = scale.clamp(min=1e-6)
        wp.to_torch(self.model.body_mass)[bi] = self._obj_m0[env_idx] * s
        wp.to_torch(self.model.body_inv_mass)[bi] = 1.0 / (self._obj_m0[env_idx] * s)
        wp.to_torch(self.model.body_inertia)[bi] = self._obj_I0[env_idx] * s[:, None, None]
        wp.to_torch(self.model.body_inv_inertia)[bi] = self._obj_invI0[env_idx] / s[:, None, None]
        self.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

    def set_object_scale(self, env_idx, scale):
        """Scale the object's geometry (collision + visual) by per-env ``scale`` relative to the asset default.

        Writes ``shape_scale`` — the per-shape geometry scale newton's OWN narrowphase reads every step (the
        GJK support function multiplies each mesh vertex by it) — and ``shape_collision_radius``, the
        broadphase culling radius derived from it at build time. Both are listed as runtime-mutable under
        ``ModelFlags.SHAPE_PROPERTIES``, so one notify publishes them.

        Mass and inertia are deliberately NOT touched: ``set_object_mass`` owns body_mass/body_inertia and a
        second writer would clobber it. A scaled object therefore keeps the UNSCALED shape's mass and
        rotational inertia (constant density would want m ∝ s³, I ∝ s⁵) — fold the geometry scale into the
        mass event before training on this, not after.

        Under ``use_mujoco_contacts=True`` shape_scale reaches the RENDER only — mjwarp collides against a
        global vertex array. The colliding hull is resized by the per-world vertex copies installed in
        __init__ (see mjw_object_scale), which is what the ``_mjw_scale.apply`` call below drives.

        ``shape_collision_aabb_lower/upper`` is the OTHER half, and it is the newton-native path's whole
        broadphase: ``compute_shape_aabbs`` reads it for MESH/CONVEX_MESH/HFIELD and applies only the
        rotation, because newton's builder bakes the scale into it at finalize (newton
        sim/collide.py + sim/builder.py). It is NOT in ``ModelFlags.SHAPE_PROPERTIES``, so no notify
        refreshes it — a grown mesh keeps its build-time box and the broadphase culls the pair before the
        (correctly scaled) narrowphase ever runs. Measured on 4hammers with native contacts: at scale 1.2,
        24 of 32 worlds had ZERO object contacts and 0/32 hammers came to rest; writing this array made it
        29/32 at rest, matching the mjwarp path. Scales BELOW the build size were unaffected, since the
        stale box still encloses the shrunk mesh.

        The RENDER needs a push of its own: newton bakes each instance's scale at ``set_model`` and never
        re-reads it, so an attached viewer keeps drawing the build-time size until the sync writes the new
        one into its instance batch (see viewer_scale)."""
        if int(env_idx.numel()) == 0:
            return
        s_w = torch.ones(self.num_envs, device=self.device)
        s_w[env_idx] = scale.clamp(min=1e-6)
        keep = torch.isin(self._obj_shape_world, env_idx)
        s = s_w[self._obj_shape_world[keep]]                                   # (S_sel,) per-shape scale
        idx = self._obj_shapes[keep]
        wp.to_torch(self.model.shape_scale)[idx] = self._obj_scale0[keep] * s[:, None]
        wp.to_torch(self.model.shape_collision_radius)[idx] = self._obj_radius0[keep] * s
        lo0, hi0 = self._obj_aabb0                                             # native broadphase box
        wp.to_torch(self.model.shape_collision_aabb_lower)[idx] = lo0[keep] * s[:, None]
        wp.to_torch(self.model.shape_collision_aabb_upper)[idx] = hi0[keep] * s[:, None]
        if self._mjw_scale is not None:            # mjwarp colliding hull (per-world vertex copies)
            self._mjw_scale.apply(env_idx, s_w[env_idx])
        # Publishes shape_scale/radius AND re-derives per-world geom_pos from mesh_pos[geom_dataid[world]],
        # which is how the scaled mesh frame reaches the colliding geom.
        self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        for sync in self._scale_syncs:             # viewer/.rrd instance scales — newton never re-reads them
            sync.refresh()

    def object_variant_id(self):
        """Which asset variant each world spawned → (N,) long (SimBackend ObjectVariant capability)."""
        assert self._obj_variant.numel() == self.num_envs, (
            f"object_variant_id: the scene has no movable object to have a variant "
            f"({self._obj_variant.numel()} recorded for {self.num_envs} envs)")
        return self._obj_variant

    def object_variant_count(self) -> int:
        """How many variants the movable object was built with (SimBackend ObjectVariant capability)."""
        return self._obj_variant_n

    def set_root_height(self, env_idx, dz):
        """Offset the fixed-base root joint's joint_X_p z by per-env dz relative to default + notify + eval_fk.
        (legacy randomize_fixed_base_root_height_newton — root identified exactly via FIXED&parent==-1, not xy-nearest.)"""
        if int(env_idx.numel()) == 0:
            return
        j = self._root_j[env_idx]
        wp.to_torch(self.model.joint_X_p)[j, 2] = self._root_z0[env_idx] + dz
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._state_epoch += 1                             # state_0 mutated → contact cache stale

    def set_gravity(self, gz: float) -> None:
        """World gravity along z [m/s^2, signed] — ALL worlds at once (curriculum write-surface, not per-env).

        newton's supported path: write ``model.gravity`` (a 1-element vec3 array) in place, then
        ``notify_model_changed(MODEL_PROPERTIES)`` runs update_model_properties_kernel over nworld and pushes
        it into ``mjw_model.opt.gravity``. In place = the captured CUDA graph keeps working (contents change,
        no reallocation), and the notify itself is a host-side launch between control steps, outside the graph.
        MODEL_PROPERTIES touches gravity only (plus a contact fast-path invalidation) — the authored
        contact/equality stiffness written by _apply_contact_params is not re-synced away.
        MuJoCo derives qfrc_gravcomp from opt.gravity, so gravity compensation follows automatically here
        (genesis has to refresh its own cached anti-gravity constants — see its set_gravity).

        The notify alone is NOT enough (measured): update_model_properties_kernel launches over nworld but
        reads ``gravity_src[world_idx]`` from a ONE-element array, so warp's bounds check drops every world
        but 0 — worlds 1..N-1 keep the gravity baked at build time. So broadcast the whole per-world
        ``opt.gravity`` here too. Both writes are in place; ``model.gravity`` stays the newton-side source of
        truth (other newton paths read it) and the notify keeps the contact fast path honest."""
        self.model.gravity.assign(np.array([[0.0, 0.0, float(gz)]], dtype=np.float32))
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        g = wp.to_torch(self.solver.mjw_model.opt.gravity)   # (nworld, 3) — notify only wrote world 0
        g[:, 0] = 0.0
        g[:, 1] = 0.0
        g[:, 2] = float(gz)

    def set_object_gravity(self, gz: float) -> None:
        """Effective gravity on the OBJECT alone [m/s^2, signed] — the WORLD's gravity is left untouched.

        MuJoCo per-body gravity compensation on the object's bodies::

            body_gravcomp = 1 - |gz| / |g_world|      1 = fully cancelled (weightless), 0 = full world gravity

        WHY NOT :meth:`set_gravity` for a lift curriculum: that one moves the whole world, so a near-weightless
        early level ALSO unloads the robot — the arm stops having to hold its own weight and the policy never
        learns to, which is the opposite of what sim2real needs. Here the robot keeps exactly the gravity the
        model was built with and only the object floats.

        Applied through the same write surface the robot's gravcomp uses, so the Newton source is updated too
        and the BODY_INERTIAL re-sync cannot undo it. The notify is not free (it re-syncs inertial properties),
        so the caller should only write on a CHANGE — the curriculum ramp does."""
        gw = abs(float(self.spec.physics.gravity[2]))
        assert gw > 0.0, "set_object_gravity needs a non-zero physics.gravity to scale against"
        assert self._obj_mjc_bodies, "set_object_gravity: the task declares no movable object body"
        scale = 1.0 - min(abs(float(gz)), gw) / gw
        _gravcomp.set_body_gravcomp(self.solver, self._obj_mjc_bodies, scale)

    def apply_object_force(self, env_idx, force):
        """Set the object's world-frame external FORCE — step() reapplies to body_f after clear_forces each
        substep (impulse lasting 1 control-step). The interval event rewrites every env before each step
        (unfired=0). Writes only the force half, so a torque term can own the other half independently."""
        if int(env_idx.numel()) == 0:
            return
        self._ext_wrench[env_idx, 0:3] = force

    def apply_object_torque(self, env_idx, torque):
        """Set the object's world-frame external TORQUE — same lifetime/rewrite contract as
        :meth:`apply_object_force`, and leaves the force half untouched."""
        if int(env_idx.numel()) == 0:
            return
        self._ext_wrench[env_idx, 3:6] = torque

    def viewer_step_allowed(self) -> bool:
        """May physics advance? newton's viewer owns Pause/Step natively (checkbox, Space, and `.`), and
        ``should_step()`` CONSUMES a pending single-step request — so call this once per wait iteration and
        step exactly when it returns True. True headless (nothing to pause)."""
        return True if self._viewer is None else bool(self._viewer.should_step())

    def pump_viewer(self) -> None:
        """Keep the window responsive while the sim is held. newton's GL viewer only pumps input inside a
        frame, so a paused run must keep drawing or the window stops answering (incl. un-pausing)."""
        self.render_frame()

    def focus_env(self, env_idx: int):
        """Move the --viz viewer camera to frame env ``env_idx`` = contract camera pose + that env's tile
        offset (``viewer.world_offsets`` from env_spacing) — the engine-agnostic hook the RL dashboard's env
        tabs call, same meaning as the genesis one: EVERY env stays visible, only the camera moves.

        The worlds overlap in physics (separate_worlds) and the viewer is what spreads them, on the contract
        grid (``parser.tile_worlds``), so world_offsets is the newton counterpart of genesis ``envs_offset``.
        ``env_idx < 0`` frames the grid center. No-op headless, without a contract camera, or on a viewer that
        has no camera to move (rerun)."""
        cam = self.spec.camera
        if self._viewer is None or cam is None:
            return
        viewer_cam = getattr(self._viewer, "camera", None)      # rerun has none — nothing to move
        if viewer_cam is None:
            return
        offs = getattr(self._viewer, "world_offsets", None)
        off = (np.zeros(3, dtype=np.float32) if offs is None or int(env_idx) < 0
               else np.asarray(offs.numpy()).reshape(-1, 3)[int(env_idx)])
        eye = np.asarray(cam.eye, dtype=np.float32) + off
        lookat = np.asarray(cam.lookat, dtype=np.float32) + off
        # Assign through the camera's own vector type (pyglet Vec3) — a plain tuple breaks its vector math.
        viewer_cam.pos = type(viewer_cam.pos)(*(float(v) for v in eye))
        viewer_cam.look_at([float(v) for v in lookat])    # orientation AND orbit pivot → mouse-orbit circles this env
        print(f"[telemetry] focus env {env_idx}: eye={eye.round(2).tolist()} "
              f"lookat={lookat.round(2).tolist()}", flush=True)

    def nan_world_detected(self) -> torch.Tensor:
        """Diverged (NaN/Inf) worlds → (N,) bool. Checks reduced joint_q only (NaN propagates to body_q via FK)."""
        jq = wp.to_torch(self.state_0.joint_q).view(self.num_envs, self._coords_pw)
        return (~torch.isfinite(jq)).any(dim=1)

    def reset_idx(self, env_mask: torch.Tensor):
        if not bool(env_mask.any()):
            return
        mask = env_mask.to(torch.bool)
        # SolverMuJoCo.reset: masked worlds → model default (init_pose baked) + clear warm-start/act/ctrl.
        self.solver.reset(self.state_0, world_mask=wp.from_torch(mask.contiguous(), dtype=wp.bool))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._state_epoch += 1                             # state_0 mutated → contact cache stale
        # hold PD target at init too (else actuators pull to the old target and wreck the spawn pose)
        tgt = wp.to_torch(self.control.joint_target_q).view(self.num_envs, self._coords_pw)
        tgt[mask] = self._init_jq[mask]
        self._ext_wrench[mask] = 0.0   # clear leftover external wrench from prior episode (interval overwrites anyway, but defensive)
        for o in self._coupled_owners:
            o.tau_torch[mask] = 0.0   # clear stale coupled-torque readout (rewritten next substep)
