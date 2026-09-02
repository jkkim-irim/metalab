from __future__ import annotations

import os

import genesis as gs
import torch

from sim.metalab.backends.genesis.viewer import GenesisViewer
from sim.metalab.control.coupled_pd import CoupledPDMixin, TorchCoupledPD
from sim.metalab.control.loaders import load_coupled_groups


class GenesisBackend(CoupledPDMixin):
    def __init__(self, spec, handles: dict, num_envs: int):
        self.spec = spec
        self.scene = handles["scene"]
        self.robot = handles["robot"]
        self.objects = handles["objects"]
        self.fixtures = handles.get("fixtures", {})
        self.num_envs = int(num_envs)
        self.device = gs.device
        self._substeps = int(handles.get("substeps", 1))
        self._init_caches()
        self._init_init_pose(spec)
        self._init_objects(spec, handles)
        self._apply_material_overrides(spec)
        self._init_coupled_pd(spec)
        self._init_gravcomp(spec)
        self.viewer = GenesisViewer(self.scene)
        self.reset_idx(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    def _init_caches(self):
        self._dof_cache: dict[str, int] = {}
        self._dof_tensor_cache: dict[tuple, torch.Tensor] = {}
        self._link_cache: dict[str, object] = {}
        self._contact_idx_cache: dict[tuple, list] = {}
        self._fric_base: dict = {}
        self._rcache: dict = {}

    def _init_init_pose(self, spec):
        self._act_names = [n for _, g in spec.action.items() for n in g.joints]
        self._act_dofs = self._dofs(self._act_names) if self._act_names else None
        self._init_dofs = None
        if self._act_dofs is not None:
            ipos = spec.robot.init_pose
            row = torch.tensor([ipos.get(n, 0.0) for n in self._act_names],
                               device=self.device, dtype=torch.float32)
            self._init_dofs = row.unsqueeze(0).expand(self.num_envs, -1).contiguous()

    def _init_objects(self, spec, handles: dict):
        self._obj_init = []
        for obj_cfg, ent in zip(handles.get("object_specs", spec.objects), self.objects):
            pos = torch.tensor(obj_cfg.init_pos or (0.0, 0.0, 0.0), device=self.device, dtype=torch.float32)
            quat = torch.tensor(obj_cfg.init_quat, device=self.device, dtype=torch.float32)
            self._obj_init.append((ent, pos, quat))
        self._obj_default_mass = self.objects[0].get_links_inertial_mass() if self.objects else None
        self._base_pos0 = self.robot.get_pos(relative=False).clone()
        self._obj_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._has_force = False

    def _apply_material_overrides(self, spec):
        if spec.robot_friction is not None:
            self.set_object_friction("robot", torch.arange(self.num_envs, device=self.device),
                                     torch.full((self.num_envs,), float(spec.robot_friction), device=self.device))
        self._pin_nail_friction()

    def _init_coupled_pd(self, spec):
        self._coupled_owners: list = []
        self._coupled_dofs: list = []
        self._coupled_tgt: list = []
        self._coupled_tgt0: list = []
        self._coupled_col_cache: dict = {}
        if not (spec.robot.control_mode == "motor" and os.environ.get("METALAB_MOTOR_COUPLING", "1") != "0"):
            return
        for groups in load_coupled_groups(spec.robot):
            if not groups:
                continue
            owner = TorchCoupledPD(groups, self.num_envs, self.device)
            names = owner.joints
            dofs = self._dofs(names)
            tgt0 = torch.tensor([spec.robot.init_pose.get(n, 0.0) for n in names],
                                device=self.device, dtype=torch.float32)
            self._coupled_owners.append(owner)
            self._coupled_dofs.append(dofs)
            self._coupled_tgt0.append(tgt0)
            self._coupled_tgt.append(tgt0.unsqueeze(0).repeat(self.num_envs, 1))
            self.robot.set_dofs_kp([0.0] * len(names), dofs)
            self.robot.set_dofs_kv([0.0] * len(names), dofs)
            lo, hi = self.robot.get_dofs_force_range(dofs)
            assert bool(torch.isinf(lo).all() and torch.isinf(hi).all()), (
                f"coupled joints must be force-range UNLIMITED in motor mode (the motor-space envelope "
                f"is the only clamp) — got lo={lo.tolist()} hi={hi.tolist()} for {names}")

    def _init_gravcomp(self, spec):
        self._gc_link_idx = None
        self._gc_force = None
        self._gc_act_links: list = []
        self._gc_act_dofs = None
        self._gc_act_bias12 = None
        self._gc_act_js: set = set()
        self._gc_on = False
        gc = spec.robot.gravcomp
        if gc is None:
            return
        coupled_js = {j for o in self._coupled_owners for j in o.joints}
        self._gc_on = os.environ.get("METALAB_GRAVCOMP", "1") != "0"
        self._gc_g_up = -torch.tensor(spec.physics.gravity, device=self.device, dtype=torch.float32)
        p_link_idx, p_masses = [], []
        for name in gc.passive_joints:
            lk = self.robot.get_joint(name).link
            p_link_idx.append(int(lk.idx)); p_masses.append(float(lk.get_mass()))
        for bname in gc.passive_bodies:
            lk = self.robot.get_link(bname)
            p_link_idx.append(int(lk.idx)); p_masses.append(float(lk.get_mass()))
        if p_link_idx:
            self._gc_link_idx = torch.tensor(p_link_idx, device=self.device, dtype=torch.long)
            self._gc_mass = torch.tensor(p_masses, device=self.device, dtype=torch.float32)
            self._gc_force = (self._gc_mass[:, None] * self._gc_g_up[None, :]).unsqueeze(0) \
                .expand(self.num_envs, -1, 3).contiguous()
        if gc.actuator_joints:
            self._gc_act_links = [(lk, float(lk.get_mass()), lk.inertial_pos)
                                  for lk in (self.robot.get_joint(n).link for n in gc.actuator_joints)]
            act_js = [j for j in gc.actuator_joints if j not in coupled_js]
            self._gc_act_js = set(act_js)
            if act_js:
                self._gc_act_dofs = self._dofs(act_js)
                _, b1, b2 = self.robot.get_dofs_act_bias(self._gc_act_dofs)
                self._gc_act_bias12 = (b1, b2)

    def _dofs(self, names: list[str]) -> torch.Tensor:
        key = tuple(names)
        t = self._dof_tensor_cache.get(key)
        if t is None:
            for n in names:
                if n not in self._dof_cache:
                    self._dof_cache[n] = int(self.robot.get_joint(n).dofs_idx_local[0])
            t = torch.tensor([self._dof_cache[n] for n in names], dtype=torch.long, device=self.device)
            self._dof_tensor_cache[key] = t
        return t

    def _link(self, name: str):
        link = self._link_cache.get(name)
        if link is None:
            link = self.robot.get_link(name)
            self._link_cache[name] = link
        return link

    def _cached(self, key, fn):
        v = self._rcache.get(key)
        if v is None:
            v = fn()
            self._rcache[key] = v
        return v

    def _entity(self, target: str):
        return self.objects[0] if target == "object" else (self.robot if target == "robot" else self.fixtures[target])

    def joint_pos(self, names):
        return self._cached(("jp", tuple(names)), lambda: self.robot.get_dofs_position(self._dofs(names)))

    def joint_vel(self, names):
        return self._cached(("jv", tuple(names)), lambda: self.robot.get_dofs_velocity(self._dofs(names)))

    def joint_torque(self, names):
        return self._cached(("jt", tuple(names)), lambda: self._joint_torque(names))

    def _joint_torque(self, names):
        t = self.robot.get_dofs_control_force(self._dofs(names))
        for oi, o in enumerate(self._coupled_owners):
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                t[:, cols] = o.tau_torch[:, tcols]
        return t

    def _actuator_gravcomp(self, names):
        gc = torch.zeros(self.num_envs, len(names), device=self.device)
        if self._gc_act_js and self._gc_on:
            act_cols = [i for i, n in enumerate(names) if n in self._gc_act_js]
            if act_cols:
                act_names = [names[i] for i in act_cols]
                gc[:, act_cols] = self._gravcomp_torque()[:, self._dofs(act_names)].to(gc.dtype)
        return gc

    def _gravcomp_torque(self):
        g = self._rcache.get("gc_tau")
        if g is None:
            g = torch.zeros(self.num_envs, self.robot.n_dofs, device=self.device)
            for link, mass, com in self._gc_act_links:
                J = self.robot.get_jacobian(link, local_point=com)
                g = g + mass * torch.einsum("nij,i->nj", J[:, :3, :], self._gc_g_up)
            self._rcache["gc_tau"] = g
        return g

    def _apply_gravcomp_actbias(self):
        if self._gc_act_dofs is None or not self._gc_on:
            return
        g = self._gravcomp_torque()[:, self._gc_act_dofs]
        b1, b2 = self._gc_act_bias12
        self.robot.set_dofs_act_bias(g, b1, b2, self._gc_act_dofs)

    def joint_limits(self, names):
        lo, hi = self.robot.get_dofs_limit(self._dofs(names))
        if lo.ndim == 2:
            lo, hi = lo[0], hi[0]
        return lo, hi

    def body_pos(self, name):
        return self._cached(("bp", name), lambda: self._link(name).get_pos(relative=False))

    def body_quat(self, name):
        return self._cached(("bq", name), lambda: self._link(name).get_quat(relative=False))

    def body_lin_vel(self, name):
        def _read():
            link = self._link(name)
            return link.entity.get_links_vel([link.idx_local], ref="link_com")[:, 0, :]
        return self._cached(("blv", name), _read)

    def body_ang_vel(self, name):
        return self._cached(("bav", name), lambda: self._link(name).get_ang())

    def object_pos(self):
        return self._cached("op", lambda: self.objects[0].get_pos())

    def object_quat(self):
        return self._cached("oq", lambda: self.objects[0].get_quat())

    def object_lin_vel(self):
        return self._cached("olv", lambda: self.objects[0].get_vel())

    def object_ang_vel(self):
        return self._cached("oav", lambda: self.objects[0].get_ang())

    def contact_force(self, link_names):
        def _read():
            key = tuple(link_names)
            idx = self._contact_idx_cache.get(key)
            if idx is None:
                idx = [self._link(n).idx_local for n in link_names]
                self._contact_idx_cache[key] = idx
            return self.robot.get_links_net_contact_force()[:, idx, :]
        return self._cached(("cf", tuple(link_names)), _read)

    def contact_force_with(self, link_names, target):
        ent = self._entity(target)

        def _read():
            c = self.robot.get_contacts(with_entity=ent)
            la, lb = c["link_a"], c["link_b"]
            fa, fb = c["force_a"], c["force_b"]
            vm = c["valid_mask"]
            out = torch.zeros(self.num_envs, len(link_names), 3, device=self.device)
            for k, name in enumerate(link_names):
                gl = self._link(name).idx
                ma = (vm & (la == gl)).unsqueeze(-1)
                mb = (vm & (lb == gl)).unsqueeze(-1)
                out[:, k] = (fa * ma).sum(dim=1) + (fb * mb).sum(dim=1)
            return out
        return self._cached(("cfw", tuple(link_names), target), _read)

    def contact_penetration(self, link_names, target):
        ent = self._entity(target)

        def _read():
            c = self.robot.get_contacts(with_entity=ent)
            out = torch.zeros(self.num_envs, len(link_names), device=self.device)
            pen = c["penetration"]
            if pen.shape[1] == 0:
                return out
            la, lb = c["link_a"], c["link_b"]
            vm = c["valid_mask"]
            for k, name in enumerate(link_names):
                gl = self._link(name).idx
                m = vm & ((la == gl) | (lb == gl))
                out[:, k] = (pen * m).amax(dim=1)
            return out.clamp_(min=0.0)
        return self._cached(("cpen", tuple(link_names), target), _read)

    def set_joint_targets(self, names, targets):
        self.robot.control_dofs_position(targets, self._dofs(names))
        for oi, o in enumerate(self._coupled_owners):
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                self._coupled_tgt[oi][:, tcols] = targets[..., cols].to(self._coupled_tgt[oi].dtype)

    def step(self, render: bool = True):
        self._apply_gravcomp_actbias()
        gc_tau = (self._gravcomp_torque()
                  if (self._coupled_owners and self._gc_on and self._gc_act_links) else None)
        for sub in range(self._substeps):
            if self._has_force:
                self.objects[0].solver.apply_links_external_force(
                    self._obj_force.unsqueeze(1), [self.objects[0].base_link_idx], ref="link_com")
            if self._gc_link_idx is not None and self._gc_on:
                self.robot.solver.apply_links_external_force(self._gc_force, self._gc_link_idx, ref="link_com")
            for o, dofs, tgt in zip(self._coupled_owners, self._coupled_dofs, self._coupled_tgt):
                tau = o.compute(self.robot.get_dofs_position(dofs), self.robot.get_dofs_velocity(dofs),
                                tgt, tau_g=(gc_tau[:, dofs] if gc_tau is not None else None))
                self.robot.control_dofs_force(tau, dofs)
            self.scene.step(update_visualizer=render and sub == self._substeps - 1)
        self._rcache.clear()

    def render_frame(self):
        pass

    def viewer_step_allowed(self) -> bool:
        return self.viewer.step_allowed()

    def pump_viewer(self) -> None:
        self.viewer.pump()

    def focus_env(self, env_idx: int):
        self.viewer.focus_env(self.spec.camera, env_idx)

    def set_object_pose(self, env_idx: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor):
        ent = self.objects[0]
        ent.set_pos(pos, envs_idx=env_idx)
        ent.set_quat(quat, envs_idx=env_idx)
        self._rcache.clear()

    def set_joint_positions(self, names, env_idx, pos, vel):
        if int(env_idx.numel()) == 0:
            return
        dofs = self._dofs(names)
        self.robot.set_dofs_position(pos, dofs, envs_idx=env_idx, zero_velocity=False)
        self.robot.set_dofs_velocity(vel, dofs, envs_idx=env_idx)
        self.robot.control_dofs_position(pos, dofs, envs_idx=env_idx)
        for oi, o in enumerate(self._coupled_owners):
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                self._coupled_tgt[oi][env_idx[:, None], torch.tensor(tcols, device=self.device)] = \
                    pos[..., cols].to(self._coupled_tgt[oi].dtype)
        self._rcache.clear()

    def _friction_base(self, target, ent):
        b = self._fric_base.get(target)
        if b is None:
            b = float(ent.geoms[0].friction)
            self._fric_base[target] = b
        return b

    def set_object_friction(self, target, env_idx, mu, exclude=()):
        if int(env_idx.numel()) == 0:
            return
        ent = self._entity(target)
        base = self._friction_base(target, ent)
        links = self._links_except(ent, exclude)
        ratio = (mu / base).unsqueeze(1).expand(-1, len(links) if links is not None else ent.n_links).contiguous()
        ent.set_friction_ratio(ratio, links_idx_local=links, envs_idx=env_idx)

    def _links_except(self, ent, exclude) -> list | None:
        if not exclude:
            return None
        drop = {self._link(n).idx_local for n in exclude}
        return [i for i in range(ent.n_links) if i not in drop]

    def _pin_nail_friction(self) -> None:
        nail = self.spec.robot.nail_friction
        if nail is None:
            return
        base = self._friction_base("robot", self.robot)
        idx = [self._link(n).idx_local for n in nail.bodies]
        ratio = torch.full((self.num_envs, len(idx)), nail.mu / base, device=self.device)
        self.robot.set_friction_ratio(ratio, links_idx_local=idx, envs_idx=None)

    def set_object_mass(self, env_idx, scale):
        if int(env_idx.numel()) == 0:
            return
        dm = self._obj_default_mass
        dm = dm[env_idx] if dm.dim() == 2 else dm.unsqueeze(0)
        shift = (scale.unsqueeze(1) - 1.0) * dm
        self.objects[0].set_mass_shift(shift, links_idx_local=None, envs_idx=env_idx)

    def set_root_height(self, env_idx, dz):
        if int(env_idx.numel()) == 0:
            return
        pos = self._base_pos0[env_idx].clone()
        pos[:, 2] += dz
        self.robot.set_pos(pos, envs_idx=env_idx, relative=False)
        self._rcache.clear()

    def set_gravity(self, gz: float) -> None:
        self.scene.sim.set_gravity((0.0, 0.0, float(gz)))
        if getattr(self, "_gc_g_up", None) is not None:
            self._gc_g_up = -torch.tensor([0.0, 0.0, float(gz)], device=self.device, dtype=torch.float32)
            if self._gc_force is not None:
                self._gc_force.copy_(self._gc_mass[:, None] * self._gc_g_up[None, :])
        self._rcache.clear()

    def set_object_gravity(self, gz: float) -> None:
        gw = abs(float(self.spec.physics.gravity[2]))
        assert gw > 0.0, "set_object_gravity needs a non-zero physics.gravity to scale against"
        assert self.objects, "set_object_gravity: the task declares no movable object body"
        comp = 1.0 - min(abs(float(gz)), gw) / gw
        info = self.objects[0].solver.entities_info
        for ent in self.objects:
            info.gravity_compensation[ent._idx_in_solver] = comp

    def apply_object_force(self, env_idx, force):
        if int(env_idx.numel()) == 0:
            return
        self._obj_force[env_idx] = force
        self._has_force = True

    def nan_world_detected(self) -> torch.Tensor:
        bad = (~torch.isfinite(self.robot.get_dofs_position(self._act_dofs))).any(dim=1)
        if self.objects:
            bad = bad | (~torch.isfinite(self.objects[0].get_pos())).any(dim=1)
        return bad

    def reset_idx(self, env_mask: torch.Tensor):
        idx = env_mask.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        n = int(idx.numel())
        if self._init_dofs is not None:
            self.robot.set_dofs_position(self._init_dofs[idx], self._act_dofs, envs_idx=idx, zero_velocity=True)
            self.robot.control_dofs_position(self._init_dofs[idx], self._act_dofs, envs_idx=idx)
        for ent, pos, quat in self._obj_init:
            ent.set_pos(pos.unsqueeze(0).expand(n, 3), envs_idx=idx)
            ent.set_quat(quat.unsqueeze(0).expand(n, 4), envs_idx=idx)
        self._obj_force[idx] = 0.0
        for oi, o in enumerate(self._coupled_owners):
            self._coupled_tgt[oi][idx] = self._coupled_tgt0[oi]
            o.tau_torch[idx] = 0.0
        self._rcache.clear()
