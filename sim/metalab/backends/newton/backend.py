from __future__ import annotations

import os

import newton
from newton import JointType
import numpy as np
import torch
import warp as wp

from sim.metalab.actuation.coupled_pd import CoupledPDMixin
from sim.metalab.actuation.motor_coupling import (
    MotorCoupledPDArm,
    MotorCoupledPDHand,
    load_arm_group,
    load_hand_group,
)
from sim.metalab.api.transforms import quat_conj, quat_mul, quat_rotate, wxyz_to_xyzw, xyzw_to_wxyz
from sim.metalab.backends.newton import gravcomp as _gravcomp
from sim.metalab.backends.newton.mjw_object_scale import install as _install_mjw_object_scale
from sim.metalab.backends.newton.viewer import NewtonViewer


@wp.kernel
def _write_body_wrench(body_idx: wp.array(dtype=wp.int32), wrench: wp.array(dtype=wp.spatial_vector),
                       body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    wp.atomic_add(body_f, body_idx[i], wrench[i])


@wp.kernel(enable_backward=False)
def _scatter_penetration(
    nacon: wp.array(dtype=wp.int32),
    contact_dist: wp.array(dtype=float),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_worldid: wp.array(dtype=wp.int32),
    geom_tip: wp.array(dtype=wp.int32),
    geom_counterpart: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=float),
):
    tid = wp.tid()
    if tid >= nacon[0]:
        return
    pen = -contact_dist[tid]
    if pen <= 0.0:
        return
    g = contact_geom[tid]
    w = contact_worldid[tid]
    k0 = geom_tip[g[0]]
    if k0 >= 0 and geom_counterpart[g[1]] == 1:
        wp.atomic_max(out, w, k0, pen)
    k1 = geom_tip[g[1]]
    if k1 >= 0 and geom_counterpart[g[0]] == 1:
        wp.atomic_max(out, w, k1, pen)


class NewtonBackend(CoupledPDMixin):
    _ARROW_FORCE_MIN = 1.0e-3

    def __init__(self, spec, handles: dict, num_envs: int):
        self.spec = spec
        self.num_envs = int(num_envs)
        self.device = torch.device(handles["device"])
        self._bind_handles(handles)
        self._init_layout()
        self._init_object(spec)
        self._init_shape_groups(spec)
        self._init_contact_budget()
        self._init_object_dr()
        self._init_root(spec)
        self._init_wrench()
        self._apply_material_overrides(spec)
        self._init_effort_and_gravcomp(spec, handles)
        self._init_coupled_pd(spec, handles)
        self.viewer = NewtonViewer(handles.get("viewer"), handles.get("rerun_viewer"),
                                   float(spec.physics.dt) * int(spec.physics.decimation),
                                   self.num_envs, self.device, self.model,
                                   self._all_obj_body_idx if self._has_object else None,
                                   self._obj_shapes)
        self._graphable = (self.viewer.gl is None and self.device.type == "cuda")
        self.reset_idx(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    def _bind_handles(self, handles: dict):
        self.model = handles["model"]
        self.solver = handles["solver"]
        self.state_0 = handles["state_0"]
        self.state_1 = handles["state_1"]
        self.control = handles["control"]
        self._contacts = None
        self._contacts_native = handles.get("contacts_native")
        self._state_epoch = 0
        self._contacts_epoch = -1
        self._substeps = int(handles["substeps"])
        self._sim_dt = float(handles["sim_dt"])
        self._obj_variant = torch.tensor(handles["object_variant"], dtype=torch.long, device=self.device)
        self._obj_variant_n = int(handles["object_variant_count"])
        self._sim_time = 0.0
        self._graph = None
        self._graph_n = 0

    def _init_layout(self):
        m = self.model
        wc = int(m.world_count)
        assert wc == self.num_envs, f"world_count({wc}) != num_envs({self.num_envs})"
        self._coords_pw = m.joint_coord_count // wc
        self._dofs_pw = m.joint_dof_count // wc
        self._bodies_pw = m.body_count // wc
        self._joints_pw = m.joint_count // wc
        self._q_start = m.joint_q_start.numpy()
        self._qd_start = m.joint_qd_start.numpy()
        self._labels = list(m.joint_label)
        self._body_labels = list(m.body_label)
        self._world_off = (torch.arange(wc, device=self.device) * self._coords_pw).unsqueeze(1)
        self._world_off_dof = (torch.arange(wc, device=self.device) * self._dofs_pw).unsqueeze(1)
        self._world_off_body = torch.arange(wc, device=self.device) * self._bodies_pw
        self._coord_cache: dict[tuple, torch.Tensor] = {}
        self._dof_cache: dict[tuple, torch.Tensor] = {}
        self._body_cache: dict[str, torch.Tensor] = {}
        self._body_shape_masks: dict[tuple, torch.Tensor] = {}
        self._sensors: dict[tuple, object] = {}
        self._cp_sensors: dict[tuple, object] = {}
        self._pen_luts: dict = {}
        self._jt_mjc_dof_cache: dict[tuple, torch.Tensor] = {}
        self._rcache: dict = {}
        self._rcache_epoch = -1
        self._t_joint_q = wp.to_torch(self.state_0.joint_q)
        self._t_joint_qd = wp.to_torch(self.state_0.joint_qd)
        self._t_body_q = wp.to_torch(self.state_0.body_q)
        self._t_body_qd = wp.to_torch(self.state_0.body_qd)

    def _init_object(self, spec):
        m = self.model
        jtype = m.joint_type.numpy()
        jchild = m.joint_child.numpy()
        free_local = [j for j in range(self._joints_pw) if int(jtype[j]) == int(JointType.FREE)]
        if not spec.robot.fixed_base:
            assert free_local, "robot.fixed_base=False but the model has no FREE joint"
            free_local = free_local[1:]
        assert len(free_local) == len(spec.movable_objects), (
            f"{len(spec.movable_objects)} movable objects in the contract but {len(free_local)} free "
            f"joints in the model (after the floating-robot root) — object classification would be wrong")
        self._has_object = len(free_local) >= 1
        self._obj_body_idx = (self._world_off_body + int(jchild[free_local[0]])) if self._has_object else None
        obj_locals = torch.tensor([int(jchild[fj]) for fj in free_local], dtype=torch.long, device=self.device)
        self._all_obj_body_idx = (self._world_off_body.view(-1, 1) + obj_locals.view(1, -1)).flatten()
        if self._has_object:
            self._obj_q0 = int(self._q_start[free_local[0]])
            self._obj_qd0 = int(self._qd_start[free_local[0]])
            obj_j_global = (torch.arange(self.num_envs, device=self.device) * self._joints_pw + free_local[0]).long()
            jXp = wp.to_torch(m.joint_X_p)[obj_j_global]
            jXc = wp.to_torch(m.joint_X_c)[obj_j_global]
            pq = xyzw_to_wxyz(jXp[:, 3:7])
            self._obj_Xp_inv_quat = quat_conj(pq).contiguous()
            self._obj_Xp_inv_pos = (-quat_rotate(self._obj_Xp_inv_quat, jXp[:, :3])).contiguous()
            self._obj_Xc_pos = jXc[:, :3].clone()
            self._obj_Xc_quat = xyzw_to_wxyz(jXc[:, 3:7]).contiguous()
        self._init_jq = wp.to_torch(m.joint_q).view(self.num_envs, self._coords_pw).clone()

    def _init_shape_groups(self, spec):
        m = self.model
        sb = wp.to_torch(m.shape_body).long()
        assert hasattr(m, "shape_world"), "newton model has no shape_world — cannot map friction DR per-world"
        sw = wp.to_torch(m.shape_world).long()
        coll = (wp.to_torch(m.shape_flags) & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        obj_mask = torch.isin(sb, self._all_obj_body_idx) & coll
        robot_mask = (sb >= 0) & ~torch.isin(sb, self._all_obj_body_idx) & coll
        table_mask = (sb == -1) & (sw >= 0) & coll
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
        self._mjw_contacts = bool(spec.overrides.get("newton", {}).get("use_mujoco_contacts", True))
        b2n = wp.to_torch(self.solver.mjc_body_to_newton)[0].long()
        obj_w0 = self._all_obj_body_idx[self._all_obj_body_idx < self._bodies_pw] if self._has_object else None
        self._obj_mjc_bodies = ([] if obj_w0 is None
                                else torch.nonzero(torch.isin(b2n, obj_w0)).flatten().tolist())

    def _init_contact_budget(self):
        self._naconmax = int(self.solver.mjw_data.naconmax)
        self._njmax = int(self.solver.mjw_data.njmax)
        self._budget_peak = torch.zeros(3, dtype=torch.int32, device=self.device)
        self._budget_now = torch.zeros(3, dtype=torch.int32, device=self.device)
        self._budget_cap = torch.tensor([self._naconmax, self._njmax], dtype=torch.int32, device=self.device)

    def _init_object_dr(self):
        m = self.model
        self._mjw_scale = None
        self._obj_shapes = None
        if not self._has_object:
            return
        sb = wp.to_torch(m.shape_body).long()
        sw = wp.to_torch(m.shape_world).long()
        self._obj_m0 = wp.to_torch(m.body_mass)[self._obj_body_idx].clone()
        self._obj_I0 = wp.to_torch(m.body_inertia)[self._obj_body_idx].clone()
        self._obj_invI0 = wp.to_torch(m.body_inv_inertia)[self._obj_body_idx].clone()
        obj_all = torch.isin(sb, self._all_obj_body_idx)
        self._obj_shapes = obj_all.nonzero(as_tuple=True)[0]
        self._obj_shape_world = sw[obj_all]
        self._obj_scale0 = wp.to_torch(m.shape_scale)[self._obj_shapes].clone()
        self._obj_radius0 = wp.to_torch(m.shape_collision_radius)[self._obj_shapes].clone()
        assert m.shape_collision_aabb_lower is not None, \
            "model has no shape_collision_aabb — newton's builder did not bake local AABBs"
        self._obj_aabb0 = (wp.to_torch(m.shape_collision_aabb_lower)[self._obj_shapes].clone(),
                           wp.to_torch(m.shape_collision_aabb_upper)[self._obj_shapes].clone())
        if self._mjw_contacts and self._fric_shapes["object"][0].numel():
            g2s = wp.to_torch(self.solver.mjc_geom_to_newton_shape)[0]
            obj_w0 = self._fric_shapes["object"]
            w0 = set(obj_w0[0][obj_w0[1] == 0].tolist())
            obj_geoms = [g for g in range(int(g2s.numel())) if int(g2s[g]) in w0]
            assert len(obj_geoms) == len(w0), \
                f"{len(w0)} object collision shapes but {len(obj_geoms)} MuJoCo geoms — cannot map size DR"
            self._mjw_scale = _install_mjw_object_scale(self.solver, obj_geoms, self.num_envs, self.device)
            print(f"mjw per-env object size: {len(obj_geoms)} geoms x {self._mjw_scale.n_vert} verts "
                  f"x {self.num_envs} worlds = +{self._mjw_scale.bytes / 1e6:.1f} MB", flush=True)

    def _init_root(self, spec):
        m = self.model
        self._root_j = None
        self._root_z0 = None
        if not spec.robot.fixed_base:
            return
        jtype = m.joint_type.numpy()
        jparent = m.joint_parent.numpy()
        root_local = [j for j in range(self._joints_pw)
                      if int(jtype[j]) == int(JointType.FIXED) and int(jparent[j]) == -1]
        assert len(root_local) == 1, f"exactly one fixed-base root joint required — {len(root_local)}"
        self._root_j = (torch.arange(self.num_envs, device=self.device) * self._joints_pw + root_local[0]).long()
        self._root_z0 = wp.to_torch(m.joint_X_p)[self._root_j, 2].clone()

    def _init_wrench(self):
        self._ext_wrench = torch.zeros(self.num_envs, 6, device=self.device)
        self._ext_wrench_wp = wp.from_torch(self._ext_wrench, dtype=wp.spatial_vector)
        self._obj_body_idx_wp = None
        if self._has_object:
            self._obj_body_idx_i32 = self._obj_body_idx.to(torch.int32).contiguous()
            self._obj_body_idx_wp = wp.from_torch(self._obj_body_idx_i32)

    def _apply_material_overrides(self, spec):
        if spec.robot_friction is not None:
            wp.to_torch(self.model.shape_material_mu)[self._fric_shapes["robot"][0]] = float(spec.robot_friction)
            self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        if spec.robot.nail_friction is not None:
            wp.to_torch(self.model.shape_material_mu)[self._fric_shapes["nail"][0]] = \
                float(spec.robot.nail_friction.mu)
            self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    def _init_effort_and_gravcomp(self, spec, handles: dict):
        jel = wp.to_torch(self.model.joint_effort_limit)
        n_eff = 0
        for jname, ov in spec.robot.joint_mode_param.items():
            if ov.effort == "default":
                continue
            e = float(ov.effort) if isinstance(ov.effort, (int, float)) else max(abs(ov.effort[0]), abs(ov.effort[1]))
            jel[self._dof_idx([jname]).reshape(-1)] = e
            n_eff += 1
        self._gc_on = False
        self._act_gc_js: set = set()
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
            self._act_gc_js = set(act_js) if self._gc_on else set()
            self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        elif n_eff:
            self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def _init_coupled_pd(self, spec, handles: dict):
        self._coupled_owners: list = []
        self._coupled_col_cache: dict[tuple, tuple[list, list]] = {}
        if not handles.get("motor_coupling_on"):
            return
        cg = spec.robot.coupled_groups()
        hand = [g for g in cg if g.kind in ("hand", "shoulder")]
        arm = [g for g in cg if g.kind == "arm"]
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
            t = (self._world_off + base).long()
            self._coord_cache[key] = t
        return t

    def _dof_idx(self, names) -> torch.Tensor:
        key = tuple(names)
        t = self._dof_cache.get(key)
        if t is None:
            base = torch.tensor([int(self._qd_start[self._local_joint(n)]) for n in names], device=self.device)
            t = (self._world_off_dof + base).long()
            self._dof_cache[key] = t
        return t

    def _body_idx(self, name: str) -> torch.Tensor:
        t = self._body_cache.get(name)
        if t is None:
            local = next((i for i in range(self._bodies_pw)
                          if self._body_labels[i] == name or self._body_labels[i].endswith(f"/{name}")), None)
            assert local is not None, f"body '{name}' not found"
            t = (self._world_off_body + local).long()
            self._body_cache[name] = t
        return t

    def _cached(self, key, fn):
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
        return self._cached(("jt", tuple(names)), lambda: self._joint_torque(names))

    def _joint_torque(self, names):
        mjc = self._jt_mjc_dofs(names)
        t = wp.to_torch(self.solver.mjw_data.qfrc_actuator)[:, mjc]
        for oi, o in enumerate(self._coupled_owners):
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                t[:, cols] = o.tau_torch[:, tcols]
        return t

    def _actuator_gravcomp(self, names):
        mjc = self._jt_mjc_dofs(names)
        gc = torch.zeros(self.num_envs, len(names), device=self.device)
        if self._act_gc_js:
            act_cols = [i for i, n in enumerate(names) if n in self._act_gc_js]
            if act_cols:
                qg = wp.to_torch(self.solver.mjw_data.qfrc_gravcomp)[:, mjc[act_cols]]
                gc[:, act_cols] = qg.to(gc.dtype)
        return gc

    def _jt_mjc_dofs(self, names):
        key = tuple(names)
        t = self._jt_mjc_dof_cache.get(key)
        if t is None:
            t = torch.tensor(_gravcomp.resolve_dofs(self.solver, list(names)), dtype=torch.long, device=self.device)
            self._jt_mjc_dof_cache[key] = t
        return t

    def joint_limits(self, names):
        didx = self._dof_idx(names)[0]
        lo = wp.to_torch(self.model.joint_limit_lower)[didx]
        hi = wp.to_torch(self.model.joint_limit_upper)[didx]
        return lo, hi

    def _body_pose(self, name):
        def _read():
            bq = self._t_body_q[self._body_idx(name)]
            return bq[:, :3], bq[:, [6, 3, 4, 5]]
        return self._cached(("bp", name), _read)

    def body_pos(self, name):
        return self._body_pose(name)[0]

    def body_quat(self, name):
        return self._body_pose(name)[1]

    def _object_pose(self):
        def _read():
            bq = self._t_body_q[self._obj_body_idx]
            return bq[:, :3], bq[:, [6, 3, 4, 5]]
        return self._cached("op", _read)

    def object_pos(self):
        return self._object_pose()[0]

    def object_quat(self):
        return self._object_pose()[1]

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
        if self._contacts is None:
            self._contacts = newton.Contacts(
                self.solver.get_max_contact_count(), 0,
                requested_attributes=self.model.get_requested_contact_attributes())
        if self._contacts_epoch != self._state_epoch:
            self.solver.update_contacts(self._contacts, self.state_0)
            self._contacts_epoch = self._state_epoch

    def contact_force(self, link_names):
        sensor = self._contact_sensor(link_names)
        self._ensure_contacts()
        sensor.update(self.state_0, self._contacts)
        return wp.to_torch(sensor.total_force).view(self.num_envs, len(link_names), 3)

    def _local_bodies(self, link_names) -> list[int]:
        return [next(i for i in range(self._bodies_pw)
                     if self._body_labels[i] == n or self._body_labels[i].endswith(f"/{n}"))
                for n in link_names]

    def _body_shape_mask(self, body_names) -> torch.Tensor:
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
        s = self._counterpart_sensor(link_names, target)
        self._ensure_contacts()
        s.update(self.state_0, self._contacts)
        fm = wp.to_torch(s.force_matrix)
        return fm.sum(dim=1).view(self.num_envs, len(link_names), 3)

    def contact_penetration(self, link_names, target):
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
        key = (tuple(link_names), target)
        lut = self._pen_luts.get(key)
        if lut is None:
            g2s = wp.to_torch(self.solver.mjc_geom_to_newton_shape)[0].long()
            sb = wp.to_torch(self.model.shape_body).long()
            live = g2s >= 0
            body_of = torch.where(live, sb[g2s.clamp(min=0)], torch.full_like(g2s, -2))
            tip = torch.full_like(g2s, -1, dtype=torch.int32)
            for k, b in enumerate(self._local_bodies(link_names)):
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

    def _fingertip_contact_arrows(self):
        tips = self.spec.robot.fingertips
        if not tips or not self._has_object:
            return None
        s = self._counterpart_sensor(tips, "object")
        self._ensure_contacts()
        s.update(self.state_0, self._contacts)
        nrm = wp.to_torch(s.force_matrix) - wp.to_torch(s.force_matrix_friction)
        w = nrm.norm(dim=-1, keepdim=True)
        tot = w.sum(dim=1)
        vec = nrm.sum(dim=1)
        org = (wp.to_torch(s.position_matrix) * w).sum(dim=1) / tot.clamp(min=1e-12)
        org = org + self.viewer.rerun_world_offsets().repeat_interleave(len(tips), dim=0)
        keep = tot.squeeze(-1) > self._ARROW_FORCE_MIN
        idx = torch.arange(len(tips), device=self.device).repeat(self.num_envs)[keep]
        return (org[keep].detach().cpu().numpy(), vec[keep].detach().cpu().numpy(),
                idx.detach().cpu().numpy())

    def _counterpart_sensor(self, link_names, target):
        key = (tuple(link_names), target)
        s = self._cp_sensors.get(key)
        if s is None:
            locals_ = self._local_bodies(link_names)
            bodies = [w * self._bodies_pw + lb for w in range(self.num_envs) for lb in locals_]
            cp_shapes = self._fric_shapes[target][0].tolist()
            assert cp_shapes, f"contact_force_with: no collision shapes for target '{target}'"
            s = newton.sensors.SensorContact(self.model, sensing_bodies=bodies,
                                             counterpart_shapes=cp_shapes, measure_total=False)
            self._cp_sensors[key] = s
        return s

    def set_joint_targets(self, names, targets):
        wp.to_torch(self.control.joint_target_q)[self._coord_idx(names)] = targets

    def step(self, render: bool = True):
        self.step_n(1, render=render)

    def step_n(self, n: int, render: bool = True):
        use_graph = self._graphable
        if use_graph and self._graph is not None and self._graph_n != n:
            self._graph = None
        if use_graph and self._graph is None:
            for _ in range(n):
                self._run_substeps()
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as cap:
                for _ in range(n):
                    self._run_substeps()
            self._graph = cap.graph
            self._graph_n = n
        elif use_graph:
            wp.capture_launch(self._graph)
        else:
            for _ in range(n):
                self._run_substeps(host_ops=True)
        self._state_epoch += 1
        self._check_contact_budget()
        self._sim_time += self._sim_dt * self._substeps * n
        if render:
            self.viewer.emit(self.state_0, self._sim_time, self._fingertip_contact_arrows, advance=True)

    def _check_contact_budget(self):
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

    def contact_budget_t(self) -> torch.Tensor:
        return torch.cat([self._budget_now, self._budget_cap])

    def render_frame(self):
        self.viewer.emit(self.state_0, self._sim_time, self._fingertip_contact_arrows)

    def _run_substeps(self, host_ops: bool = False):
        a, b = self.state_0, self.state_1
        for _ in range(self._substeps):
            a.clear_forces()
            if host_ops:
                self.viewer.apply_pick_forces(a)
            if self._has_object:
                wp.launch(_write_body_wrench, dim=self.num_envs,
                          inputs=[self._obj_body_idx_wp, self._ext_wrench_wp, a.body_f],
                          device=self.model.device)
            for o in self._coupled_owners:
                o.launch(a.joint_q, a.joint_qd, self.control.joint_target_q, self.control.joint_f)
            if self._contacts_native is not None:
                self.model.collide(a, self._contacts_native)
            self.solver.step(a, b, self.control, self._contacts_native, self._sim_dt)
            a, b = b, a
        if a is not self.state_0:
            wp.copy(self.state_0.joint_q, a.joint_q)
            wp.copy(self.state_0.joint_qd, a.joint_qd)
            wp.copy(self.state_0.body_q, a.body_q)
            wp.copy(self.state_0.body_qd, a.body_qd)

    def set_object_pose(self, env_idx: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor):
        if int(env_idx.numel()) == 0:
            return
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
        self._state_epoch += 1

    def set_joint_positions(self, names, env_idx, pos, vel):
        if int(env_idx.numel()) == 0:
            return
        cidx = self._coord_idx(names)[env_idx]
        didx = self._dof_idx(names)[env_idx]
        wp.to_torch(self.state_0.joint_q)[cidx] = pos
        wp.to_torch(self.state_0.joint_qd)[didx] = vel
        wp.to_torch(self.control.joint_target_q)[cidx] = pos
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._state_epoch += 1

    def set_object_friction(self, target, env_idx, mu, exclude=()):
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
        if int(env_idx.numel()) == 0:
            return
        s_w = torch.ones(self.num_envs, device=self.device)
        s_w[env_idx] = scale.clamp(min=1e-6)
        keep = torch.isin(self._obj_shape_world, env_idx)
        s = s_w[self._obj_shape_world[keep]]
        idx = self._obj_shapes[keep]
        wp.to_torch(self.model.shape_scale)[idx] = self._obj_scale0[keep] * s[:, None]
        wp.to_torch(self.model.shape_collision_radius)[idx] = self._obj_radius0[keep] * s
        lo0, hi0 = self._obj_aabb0
        wp.to_torch(self.model.shape_collision_aabb_lower)[idx] = lo0[keep] * s[:, None]
        wp.to_torch(self.model.shape_collision_aabb_upper)[idx] = hi0[keep] * s[:, None]
        if self._mjw_scale is not None:
            self._mjw_scale.apply(env_idx, s_w[env_idx])
        self.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        self.viewer.refresh_scales()

    def object_variant_id(self):
        assert self._obj_variant.numel() == self.num_envs, (
            f"object_variant_id: the scene has no movable object to have a variant "
            f"({self._obj_variant.numel()} recorded for {self.num_envs} envs)")
        return self._obj_variant

    def object_variant_count(self) -> int:
        return self._obj_variant_n

    def set_root_height(self, env_idx, dz):
        if int(env_idx.numel()) == 0:
            return
        assert self._root_j is not None, "set_root_height needs a fixed-base robot (robot.fixed_base=False)"
        j = self._root_j[env_idx]
        wp.to_torch(self.model.joint_X_p)[j, 2] = self._root_z0[env_idx] + dz
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._state_epoch += 1

    def set_gravity(self, gz: float) -> None:
        self.model.gravity.assign(np.array([[0.0, 0.0, float(gz)]], dtype=np.float32))
        self.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        g = wp.to_torch(self.solver.mjw_model.opt.gravity)
        g[:, 0] = 0.0
        g[:, 1] = 0.0
        g[:, 2] = float(gz)

    def set_object_gravity(self, gz: float) -> None:
        gw = abs(float(self.spec.physics.gravity[2]))
        assert gw > 0.0, "set_object_gravity needs a non-zero physics.gravity to scale against"
        assert self._obj_mjc_bodies, "set_object_gravity: the task declares no movable object body"
        scale = 1.0 - min(abs(float(gz)), gw) / gw
        _gravcomp.set_body_gravcomp(self.solver, self._obj_mjc_bodies, scale)

    def apply_object_force(self, env_idx, force):
        if int(env_idx.numel()) == 0:
            return
        self._ext_wrench[env_idx, 0:3] = force

    def viewer_step_allowed(self) -> bool:
        return self.viewer.step_allowed()

    def pump_viewer(self) -> None:
        self.render_frame()

    def focus_env(self, env_idx: int):
        self.viewer.focus_env(self.spec.camera, env_idx)

    def nan_world_detected(self) -> torch.Tensor:
        jq = wp.to_torch(self.state_0.joint_q).view(self.num_envs, self._coords_pw)
        return (~torch.isfinite(jq)).any(dim=1)

    def reset_idx(self, env_mask: torch.Tensor):
        if not bool(env_mask.any()):
            return
        mask = env_mask.to(torch.bool)
        self.solver.reset(self.state_0, world_mask=wp.from_torch(mask.contiguous(), dtype=wp.bool))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._state_epoch += 1
        tgt = wp.to_torch(self.control.joint_target_q).view(self.num_envs, self._coords_pw)
        tgt[mask] = self._init_jq[mask]
        self._ext_wrench[mask] = 0.0
        for o in self._coupled_owners:
            o.tau_torch[mask] = 0.0
