"""GenesisBackend — wraps a Genesis scene (parser) as a :class:`sim.metalab.runtime.backend.SimBackend`.

env_driver talks to Genesis only through this adapter (read+control+step+reset). Joints resolve
name→DOF index (ByBasename); returns follow the canonical convention (wxyz, world, SI, ``(N, ...)``).
reset currently uses the init dofs captured at build + the contract object init pose (event randomization TBD).
"""
from __future__ import annotations

import os

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
import warp as wp

from sim.metalab.drive.motor2joint.loaders import load_arm_group, load_hand_group

# control_mode:motor reuses the NEWTON warp kernels here (one kernel per owner per sub-step). warp is
# a declared dependency of the genesis env (setup/genesis/pyproject.toml) precisely so this holds: a
# pure-torch evaluation of the same law needs padded per-term tensors and a ~90-op launch chain per
# step, which measured ~5 ms/step worse at 2048 envs — so warp is the only path, no fallback.
from sim.metalab.drive.motor2joint.motor_coupling import (
    MotorCoupledPDArm as _WarpPDArm,
)
from sim.metalab.drive.motor2joint.motor_coupling import (
    MotorCoupledPDHand as _WarpPDHand,
)


class _GenesisWarpOwner:
    """Newton warp coupled-PD owner driven from genesis torch tensors: ``compute(q, qd, q*, τ_g)``
    → τ_q, plus the ``tau_torch`` readout and the ``set_fold_kd_zero`` float-mode toggle.

    The warp kernel's ``joint_f`` output carries newton's −τ_g cancellation (MuJoCo re-applies the
    gravcomp share passively) and is DISCARDED here; genesis applies :attr:`tau_torch` = τ_q, which
    is exactly what its force channel needs."""

    def __init__(self, groups: list[dict], num_envs: int, device):
        wp.init()
        d = len(groups[0]["joints"])
        k = sum(len(g["joints"]) for g in groups)
        idx = torch.arange(num_envs * k, dtype=torch.int32, device=device).reshape(num_envs, k)
        self._gc_buf = torch.zeros(num_envs, k, dtype=torch.float32, device=device)   # τ_g staging
        cls = _WarpPDHand if d == 3 else _WarpPDArm
        self._inner = cls(groups, idx, idx, num_envs, device,
                          gravcomp=wp.from_torch(self._gc_buf),
                          gc_dof=torch.arange(k, dtype=torch.int32, device=device))
        self._jf = torch.zeros(num_envs * k, dtype=torch.float32, device=device)      # discarded
        self.joints = self._inner.joints
        self.n_groups = self._inner.n_groups
        self.tau_torch = self._inner.tau_torch           # (N, k) τ_q — readout AND applied force
        self.tau_pd_torch = self._inner.tau_pd_torch     # (N, k) PRE-clamp PD component (read-only)
        self.tau_gc_torch = self._inner.tau_gc_torch     # (N, k) PRE-clamp gravity component (read-only)
        self.set_fold_kd_zero = self._inner.set_fold_kd_zero
        self.reload_gains = self._inner.reload_gains     # robot_model.json gain re-read (same kernels)
        self.gain_warnings = self._inner.gain_warnings   # differential-group gain consistency
        self.kq = self._inner.kq                         # host-side K_q readout (Joint Kp channel)
        self.tau_lim = self._inner.tau_lim               # host-side τ_q limit readout (Joint Torque Limit)

    def compute(self, q, qd, q_tgt, tau_g=None):
        if tau_g is not None:
            self._gc_buf.copy_(tau_g)                    # fixed buffer (kernel aliases it)
        # Run the warp kernel ON torch's current stream: torch/taichi tensors are only ordered (and
        # their memory only lifetime-safe under torch's stream-aware allocator) within that stream —
        # warp's own stream would race the genesis getters/appliers.
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            self._inner.launch(wp.from_torch(q.reshape(-1).contiguous()),
                               wp.from_torch(qd.reshape(-1).contiguous()),
                               wp.from_torch(q_tgt.reshape(-1).contiguous()),
                               wp.from_torch(self._jf))
        return self.tau_torch


class GenesisBackend:
    def __init__(self, spec, handles: dict, num_envs: int):
        self.spec = spec
        self.scene = handles["scene"]
        self._viewer_paused = False        # genesis has no native pause — this flag is the sim hold
        self._viewer_step_once = False     # one-shot single step (consumed by viewer_step_allowed)
        self.robot = handles["robot"]        # ByBasename
        self.objects = handles["objects"]
        self.num_envs = int(num_envs)
        self._substeps = int(handles.get("substeps", 1))   # scene is built at dt/substeps (see parser)
        self.device = gs.device
        # hot-path caches (resolve indices/links once — avoid per-step rebuild/linear scan)
        self._dof_cache: dict[str, int] = {}                 # joint name → dofs_idx_local[0] (int)
        self._dof_tensor_cache: dict[tuple, "torch.Tensor"] = {}  # names tuple → dof index tensor
        self._link_cache: dict[str, object] = {}             # link name → RigidLink (avoid ByBasename scan)
        self._contact_idx_cache: dict[tuple, list] = {}      # link-name tuple → idx_local list
        # per-step read cache: collapse duplicate reads of the same quantity within a step
        # (obs/reward/terminate each read). Self-invalidates on state change (step/reset_idx) → always fresh.
        self._rcache: dict = {}

        # all joints controlled by the action (reset targets)
        self._act_names = [n for _, g in spec.action.items() for n in g.joints]
        self._act_dofs = self._dofs(self._act_names) if self._act_names else None
        # initial joint pose = **task** contract (EnvSpec.init_pose); unlisted joints = 0. (Not robot YAML — per-task.)
        if self._act_dofs is not None:
            ipos = spec.robot.init_pose
            row = torch.tensor([ipos.get(n, 0.0) for n in self._act_names],
                               device=self.device, dtype=torch.float32)
            self._init_dofs = row.unsqueeze(0).expand(self.num_envs, -1).contiguous()
        else:
            self._init_dofs = None

        # object init pose (reset targets)
        self._obj_init = []
        for obj_cfg, ent in zip(handles.get("object_specs", spec.objects), self.objects):
            pos = torch.tensor(obj_cfg.init_pos or (0.0, 0.0, 0.0), device=self.device, dtype=torch.float32)
            quat = torch.tensor(obj_cfg.init_quat, device=self.device, dtype=torch.float32)
            self._obj_init.append((ent, pos, quat))

        # ---- G5 write-surface caches (DR event write; symmetric across engines) ----
        self.fixtures = handles.get("fixtures", {})                  # name → fixed RigidEntity (table, etc.)
        self._obj_default_mass = self.objects[0].get_links_inertial_mass() if self.objects else None   # (n_links,)/(N,n_links); None if no object
        self._base_pos0 = self.robot.get_pos(relative=False).clone()         # (N,3) fixed-base default world pos
        self._fric_base: dict = {}                                   # target → per-target base friction (lazy)
        # external-force impulse buffer — applied once by step() before scene.step (genesis auto-clears after step → 1-step impulse)
        self._obj_wrench = torch.zeros(self.num_envs, 6, device=self.device)   # [force(3), torque(3)] world@COM
        self._has_wrench = False

        # Build-time robot contact friction override (absolute μ) — standalone/eval knob; None = keep MJCF.
        # DR friction events override this at reset when active; standalone (no DR) keeps it for the whole run.
        if spec.robot_friction is not None:
            self.set_object_friction("robot", torch.arange(self.num_envs, device=self.device),
                                     torch.full((self.num_envs,), float(spec.robot_friction), device=self.device))
        self._pin_nail_friction()          # after the robot-wide write, so the pin wins where they overlap

        # Motor-to-joint coupled PD (control_mode: motor) — the SAME newton warp kernels, driven from
        # genesis tensors (same loaders/maps/gains; parity pinned in tests/test_motor_coupling.py).
        # Native PD (kp/kv) is zeroed on the coupled dofs and step() drives them with
        # control_dofs_force(τ_q) at SUB-STEP rate (the scene is built at dt/substeps and step() loops —
        # newton's _run_substeps cadence). Their gravcomp share is folded INSIDE the motor clamp
        # (τ_m += G⁻ᵀ·τ_g) — they are excluded from the act-bias route below, and the passive channel
        # never touches arm dofs, so unlike newton no cancellation term is needed.
        self._coupled_owners: list = []
        self._coupled_dofs: list = []        # per-owner entity-local dof index tensor
        self._coupled_tgt: list = []         # per-owner PD target buffer (N, k) — genesis has no target getter
        self._coupled_tgt0: list = []        # per-owner init-pose target row (k,) for reset re-sync
        self._coupled_col_cache: dict = {}
        coupled_js: set = set()
        if spec.robot.control_mode == "motor" and os.environ.get("METALAB_MOTOR_COUPLING", "1") != "0":
            cg = spec.robot.coupled_groups()
            # One owner per KERNEL FAMILY (3-DOF hand/thumb/shoulder, 2-DOF elbow/wrist) — the warp
            # kernels loop a group's own terms per thread, so mixing maps with different term counts
            # in one owner costs nothing.
            buckets: dict = {"hand3": [], "arm2": []}
            for g in cg:
                if g.kind in ("hand", "shoulder"):
                    buckets["hand3"].append(load_hand_group(g.params_key, g.joints,
                                                            model_file=g.model_file,
                                                            gain_slice=g.arm_slice))
                else:
                    buckets["arm2"].append(load_arm_group(g.params_key, g.joints,
                                                          model_file=g.model_file,
                                                          arm_slice=g.arm_slice))
            for groups in (g for g in buckets.values() if g):
                owner = _GenesisWarpOwner(groups, self.num_envs, self.device)
                names = owner.joints
                dofs = self._dofs(names)
                tgt0 = torch.tensor([spec.robot.init_pose.get(n, 0.0) for n in names],
                                    device=self.device, dtype=torch.float32)
                self._coupled_owners.append(owner)
                self._coupled_dofs.append(dofs)
                self._coupled_tgt0.append(tgt0)
                self._coupled_tgt.append(tgt0.unsqueeze(0).repeat(self.num_envs, 1))   # real copy (no tgt0 alias)
                coupled_js.update(names)
                self.robot.set_dofs_kp([0.0] * len(names), dofs)   # native PD off — coupled law replaces it
                self.robot.set_dofs_kv([0.0] * len(names), dofs)
                # INVARIANT the τ readout below depends on: a coupled dof must have NO joint-level torque
                # clamp. genesis clamps every ctrl mode — `qf_applied = clamp(ctrl_force, force_range)` — so a
                # finite range here would (a) silently re-limit torque the motor-space envelope already
                # bounded, diverging from newton (whose coupled τ rides qfrc_applied, untouched by
                # jnt_actfrcrange), and (b) make `_joint_torque`'s tau_torch overlay OVER-REPORT, since
                # tau_torch is what the kernel asked for, not what the solver kept. parser._open_coupled_
                # force_range opens it; assert rather than trust, because both effects are SILENT — a finite
                # range would quietly cap the torque and quietly inflate the readout, with nothing to see.
                lo, hi = self.robot.get_dofs_force_range(dofs)
                assert bool(torch.isinf(lo).all() and torch.isinf(hi).all()), (
                    f"coupled joints must be force-range UNLIMITED in motor mode (the motor-space envelope "
                    f"is the only clamp) — got lo={lo.tolist()} hi={hi.tolist()} for {names}")

        # Gravity compensation (mirror real HW). Two channels, physically distinct:
        #  - ACTUATOR joints: the motor supplies gravcomp, so it must count against the motor torque limit. Each
        #    step we compute the per-joint gravcomp torque g(q)=Σ Jᵀ(m·g_up) and write it into act_bias[0] (a
        #    constant term added to the PD force BEFORE genesis clamps to force_range). Native position control
        #    then applies clamp(PD + g(q), force_range) — the joint yields past max torque, and joint_torque
        #    reads it straight from get_dofs_control_force. Needs batch_dofs_info (per-env, config-dependent g).
        #  - PASSIVE joints/bodies (waist/neck spring): external anti-gravity body force (m_i·(-g) at COM),
        #    uncapped and NOT motor torque → excluded from the readout. Re-applied every step (genesis clears
        #    external forces each step). METALAB_GRAVCOMP sets the INITIAL on/off state.
        self._gc_link_idx = None            # (n_passive,) passive-channel link idx receiving the anti-gravity force
        self._gc_force = None               # (N, n_passive, 3) constant anti-gravity force = m_i·(-gravity)
        self._gc_act_links: list = []       # [(RigidLink, mass, COM)] actuator child links — gravcomp Jacobian sum
        self._gc_act_dofs = None            # (k_act,) entity-local dof idx of actuator-gravcomp joints (act_bias write)
        self._gc_act_bias12 = None          # (bias1, bias2) cached for those dofs (preserved on act_bias[0] write)
        self._gc_act_js: set = set()        # those joints by name — the component split subtracts their share
        self._gc_on = False
        gc = spec.robot.gravcomp
        if gc is not None:
            self._gc_on = os.environ.get("METALAB_GRAVCOMP", "1") != "0"
            self._gc_g_up = -torch.tensor(spec.physics.gravity, device=self.device, dtype=torch.float32)  # (3,)
            # PASSIVE channel → external anti-gravity body force (uncapped, not in the readout)
            p_link_idx, p_masses = [], []
            for name in gc.passive_joints:
                lk = self.robot.get_joint(name).link                     # child link of the passive joint
                p_link_idx.append(int(lk.idx)); p_masses.append(float(lk.get_mass()))
            for bname in gc.passive_bodies:                              # trunk/neck mass above the passive joints
                lk = self.robot.get_link(bname)
                p_link_idx.append(int(lk.idx)); p_masses.append(float(lk.get_mass()))
            if p_link_idx:
                self._gc_link_idx = torch.tensor(p_link_idx, device=self.device, dtype=torch.long)
                # kept as state (not a local): set_gravity rebuilds _gc_force from it when the curriculum
                # ramps gravity, so the anti-gravity force tracks the world instead of the build-time value.
                self._gc_mass = torch.tensor(p_masses, device=self.device, dtype=torch.float32)           # (n_passive,)
                self._gc_force = (self._gc_mass[:, None] * self._gc_g_up[None, :]).unsqueeze(0) \
                    .expand(self.num_envs, -1, 3).contiguous()
            # ACTUATOR channel → gravcomp routed through the force-limited actuator via act_bias[0] (per step).
            # (link, mass, COM-in-link-frame) — the anti-gravity force acts at the COM, so the gravcomp Jacobian
            # must use the COM Jacobian (link-origin Jacobian under-reports; arm links have large COM offset).
            if gc.actuator_joints:
                # FULL actuator link list — _gravcomp_torque's Jacobian sum needs every compensated
                # mass regardless of which dofs consume the share (act-bias vs coupled fold).
                self._gc_act_links = [(lk, float(lk.get_mass()), lk.inertial_pos)
                                      for lk in (self.robot.get_joint(n).link for n in gc.actuator_joints)]
                # act-bias route: only NON-coupled actuator joints — coupled dofs fold their share
                # into the motor-torque clamp instead (see the coupled-PD block above).
                act_js = [j for j in gc.actuator_joints if j not in coupled_js]
                # Non-coupled joints whose gravcomp rides inside get_dofs_control_force via act_bias[0]
                # — the component split subtracts their share back out (coupled joints use the kernel's).
                self._gc_act_js = set(act_js)
                if act_js:
                    self._gc_act_dofs = self._dofs(act_js)               # entity-local dof idx (batch-safe setter)
                    # Cache PD coeffs (bias1=-kp, bias2=-kv) ONCE — valid because kp/kv overrides are applied in
                    # build_scene (before this backend is constructed) and nothing changes them at runtime. If a
                    # runtime kp/kv setter is ever added, it must refresh this cache or the per-step act_bias write
                    # below would silently revert kp/kv on the actuator-gravcomp joints.
                    _, b1, b2 = self.robot.get_dofs_act_bias(self._gc_act_dofs)
                    self._gc_act_bias12 = (b1, b2)

        # spawn at init pose right after model load (viewer shows init immediately + PD target holds at init).
        self._install_pause_keybind()       # Space / '.' on the viewer, same keys newton binds natively
        self.reset_idx(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    def _dofs(self, names: list[str]) -> torch.Tensor:
        # cache names tuple → dof index tensor (avoid re-creating torch.tensor = CPU→GPU transfer per read/write)
        key = tuple(names)
        t = self._dof_tensor_cache.get(key)
        if t is None:
            for n in names:
                if n not in self._dof_cache:
                    # entity-local DOF index; get_dofs_* takes local.
                    self._dof_cache[n] = int(self._link_or_joint(n, joint=True).dofs_idx_local[0])
            t = torch.tensor([self._dof_cache[n] for n in names], dtype=torch.long, device=self.device)
            self._dof_tensor_cache[key] = t
        return t

    def _link_or_joint(self, name: str, joint: bool = False):
        return self.robot.get_joint(name) if joint else self._link(name)

    def _link(self, name: str):
        # link-resolution cache (ByBasename.get_link linear-scans all links per call → do it once)
        link = self._link_cache.get(name)
        if link is None:
            link = self.robot.get_link(name)
            self._link_cache[name] = link
        return link

    # --- read (via per-step cache: duplicate reads in a step collapse to one; step/reset_idx invalidates) ---
    def _cached(self, key, fn):
        v = self._rcache.get(key)
        if v is None:
            v = fn(); self._rcache[key] = v
        return v

    def joint_pos(self, names): return self._cached(("jp", tuple(names)), lambda: self.robot.get_dofs_position(self._dofs(names)))
    def joint_vel(self, names): return self._cached(("jv", tuple(names)), lambda: self.robot.get_dofs_velocity(self._dofs(names)))
    def joint_torque(self, names): return self._cached(("jt", tuple(names)), lambda: self._joint_torque(names))

    def _joint_torque(self, names):
        # Actual motor torque the actuator applied = get_dofs_control_force = clamp(PD + act_bias[0], force_range),
        # where act_bias[0] carries the actuator-routed gravcomp feedforward g(q) written each step (see step()).
        # So for actuator-gravcomp joints this is clamp(PD + gravcomp, force_range) — capped at the torque limit,
        # the joint yields past it. Fingers/non-gravcomp joints: act_bias[0]=0 → clamp(PD). Excludes qf_passive
        # (joint spring/damping), gravity/Coriolis bias, and the PASSIVE-channel gravcomp (external) — none of
        # which is motor torque. (get_dofs_force would return the net M·q̈, wrong for a joint-torque sensor.)
        t = self.robot.get_dofs_control_force(self._dofs(names))      # (N, k)
        # Coupled joints: overlay the coupled-PD τ_q (motor-delivered torque incl. the gravity
        # feedforward — same readout semantics as newton's overlay), robust to control-mode churn.
        # EXACT only because a coupled dof's force_range is ±inf, so the solver keeps ctrl_force as written
        # (asserted at build, see __init__). Re-limit it and this overlay reports the kernel's REQUEST while
        # the solver applied a clamped value — the readout would exceed the real torque.
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
        """(pd, gravcomp) split of :meth:`joint_torque`, per joint — same contract as the newton backend
        (both engines run the same coupled-PD kernel, so the coupled columns are the same quantity).

        * COUPLED joints: the kernel's PRE-clamp Gᵀ·τ_m^PD and τ_g.
        * NON-coupled ACTUATOR-gravcomp joints: genesis clamps ``PD + act_bias[0]`` together, so the
          gravity share is the ``_gravcomp_torque()`` feedforward written to those dofs and PD is the
          remainder (clamp gap attributed to PD).
        * PASSIVE-channel gravcomp is an external link force, not motor torque → excluded from both,
          matching ``joint_torque``.
        """
        gc = torch.zeros(self.num_envs, len(names), device=self.device)
        if self._gc_act_js and self._gc_on:
            act_cols = [i for i, n in enumerate(names) if n in self._gc_act_js]
            if act_cols:
                act_names = [names[i] for i in act_cols]
                gc[:, act_cols] = self._gravcomp_torque()[:, self._dofs(act_names)].to(gc.dtype)
        pd = self.joint_torque(names) - gc
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
        """Torque/float mode: zero the coupled D gain on the gravcomp-fold groups (arm) — feedforward
        only, dissipation from joint friction; restore for Position mode. Same API as newton."""
        for o in self._coupled_owners:
            o.set_fold_kd_zero(off)

    def _gravcomp_torque(self):
        """Actuator gravcomp generalized force Σ_actlinks Jᵀ(m·g_up) → (N, entity n_dofs). Cached per step."""
        g = self._rcache.get("gc_tau")
        if g is None:
            g = torch.zeros(self.num_envs, self.robot.n_dofs, device=self.device)
            for link, mass, com in self._gc_act_links:
                J = self.robot.get_jacobian(link, local_point=com)   # COM Jacobian (N, 6, n_dofs); rows 0:3 = linear
                g = g + mass * torch.einsum("nij,i->nj", J[:, :3, :], self._gc_g_up)
            self._rcache["gc_tau"] = g
        return g

    def _apply_gravcomp_actbias(self):
        """Write the actuator-routed gravcomp feedforward into act_bias[0] so native position control applies
        clamp(PD + g(q), force_range). Per-env (batch_dofs_info), config-dependent → recomputed each step."""
        if self._gc_act_dofs is None or not self._gc_on:
            return
        g = self._gravcomp_torque()[:, self._gc_act_dofs]            # (N, k_act) gravcomp torque at actuator dofs
        b1, b2 = self._gc_act_bias12
        self.robot.set_dofs_act_bias(g, b1, b2, self._gc_act_dofs)   # bias0=g(q); preserve bias1(-kp)/bias2(-kv)

    def joint_limits(self, names):
        lo, hi = self.robot.get_dofs_limit(self._dofs(names))  # (k,) or (N,k) under batch_dofs_info
        if lo.ndim == 2:                                       # per-env identical → world-0 slice (keep (k,) contract)
            lo, hi = lo[0], hi[0]
        return lo, hi
    def body_pos(self, name): return self._cached(("bp", name), lambda: self._link(name).get_pos(relative=False))    # world
    def body_quat(self, name): return self._cached(("bq", name), lambda: self._link(name).get_quat(relative=False))  # world, wxyz
    def object_pos(self): return self._cached("op", lambda: self.objects[0].get_pos())
    def object_quat(self): return self._cached("oq", lambda: self.objects[0].get_quat())
    def object_lin_vel(self): return self._cached("olv", lambda: self.objects[0].get_vel())   # world (N,3)
    def object_ang_vel(self): return self._cached("oav", lambda: self.objects[0].get_ang())   # world (N,3)
    def body_lin_vel(self, name):
        """Link LINEAR velocity at the link's CENTER OF MASS → (N,3) world.

        ``ref="link_com"``, not genesis' ``get_vel()``: that one defaults to ``ref="link_origin"`` (the velocity
        of the link FRAME), while newton's ``State.body_qd`` is documented as the velocity at the body's centre
        of mass. Same name, two different physical quantities — they differ by ω × r_com, measured at 6.5 mm/s
        per rad/s on the finger distal links (r_com 6.5mm), i.e. 1-3% of the ``fingertip_rel_vel`` /
        ``palm_lin_vel`` obs channels at ordinary hand speeds and more when the wrist swings. Newton's read is
        native, genesis' is a keyword, so genesis is the side that moves."""
        return self._cached(("blv", name), lambda: self._link_com_vel(name))

    def _link_com_vel(self, name):
        link = self._link(name)
        return link.entity.get_links_vel([link.idx_local], ref="link_com")[:, 0, :]
    def body_ang_vel(self, name): return self._cached(("bav", name), lambda: self._link(name).get_ang())   # world (N,3)

    def contact_force(self, link_names):
        """Net external contact force per link (N, len(link_names), 3), world. link.idx_local = entity-local."""
        def _read():
            key = tuple(link_names)
            idx = self._contact_idx_cache.get(key)
            if idx is None:
                idx = [self._link(n).idx_local for n in link_names]
                self._contact_idx_cache[key] = idx
            return self.robot.get_links_net_contact_force()[:, idx, :]   # (N, n_links, 3) → select
        return self._cached(("cf", tuple(link_names)), _read)

    def viewer_step_allowed(self) -> bool:
        """May physics advance? genesis' viewer has NO native pause (``keybindings.PAUSE`` exists but nothing
        consumes it), so the flag is ours — set by the dashboard toggle or by the Space keybind registered in
        :meth:`_install_pause_keybind`. ``_step_once`` is consumed here, mirroring newton's Step."""
        if self._viewer_paused and self._viewer_step_once:
            self._viewer_step_once = False
            return True
        return not self._viewer_paused

    def pump_viewer(self) -> None:
        """Keep the window responsive while the sim is held. On linux/win the viewer runs in its own thread
        (``run_in_thread`` resolves True), so it redraws itself and there is nothing to do; only a main-thread
        viewer (macOS) needs us to refresh, since genesis then refreshes only from inside a sim step."""
        viewer = getattr(self.scene, "viewer", None)
        pv = getattr(viewer, "_pyrender_viewer", None) if viewer is not None else None
        if pv is None or getattr(pv, "run_in_thread", True):
            return
        with viewer.lock:
            pv.refresh()

    def _install_pause_keybind(self) -> None:
        """Space = pause/play, `.` = single step — the keys newton's viewer already binds, so the two engines
        feel the same. genesis exposes ``register_keybinds`` for exactly this (no vendored edit)."""
        viewer = getattr(self.scene, "viewer", None)
        if viewer is None or not hasattr(viewer, "register_keybinds"):
            return
        from genesis.vis.keybindings import KeyAction, Keybind
        import pyglet

        def _toggle():
            self._viewer_paused = not self._viewer_paused

        def _step():
            self._viewer_step_once = True

        viewer.register_keybinds(
            Keybind(name="allex_pause", key=pyglet.window.key.SPACE, key_action=KeyAction.PRESS,
                    key_mods=None, callback=_toggle),
            Keybind(name="allex_step", key=pyglet.window.key.PERIOD, key_action=KeyAction.PRESS,
                    key_mods=None, callback=_step),
            overwrite=True)

    def focus_env(self, env_idx: int):
        """Move the --viz viewer camera to frame env ``env_idx`` = contract camera pose + that env's tile
        offset (scene.envs_offset from env_spacing). ``env_idx < 0`` frames the grid center (the tiles are
        centered on the origin). No-op when headless (no viewer) or no contract camera."""
        viewer = getattr(self.scene, "viewer", None)
        cam = self.spec.camera
        if viewer is None or cam is None:
            return
        # (3,) per-env visualization offset (np.ndarray). Guarded: a bare negative index would silently frame
        # the LAST env instead of the whole grid — same <0 contract as the newton hook.
        off = (np.zeros(3, dtype=np.float32) if int(env_idx) < 0
               else self.scene.envs_offset[int(env_idx)])
        eye = np.asarray(cam.eye, dtype=np.float32) + off      # translate the contract camera onto this env's tile
        lookat = np.asarray(cam.lookat, dtype=np.float32) + off
        # Build the pose with a FIXED world-up so the framing stays upright (matches the record camera).
        # Passing pos/lookat instead makes set_camera_pose reuse+drift its internal _camera_up → rolled/tilted view.
        pose = gu.pos_lookat_up_to_T(eye, lookat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        with viewer.lock:                                 # pyrender render lock — apply on the live viewer safely
            viewer.set_camera_pose(pose=pose)
        print(f"[telemetry] focus env {env_idx}: eye={eye.round(2).tolist()} "
              f"lookat={lookat.round(2).tolist()}", flush=True)

    def contact_force_with(self, link_names, target):
        """Per-link contact force with **a specific counterpart (target)** only, (N, K, 3), world.
        get_contacts(with_entity) filters to caller(robot)↔target contacts. target={object|robot|table} →
        that RigidEntity. link_a/link_b are global link idx, so match via self._link(name).idx (global) and
        sum force_a(=−f)/force_b(=+f) under valid_mask."""
        ent = (self.objects[0] if target == "object"
               else self.robot if target == "robot" else self.fixtures[target])

        def _read():
            c = self.robot.get_contacts(with_entity=ent)
            la, lb = c["link_a"], c["link_b"]                # (N, C)
            fa, fb = c["force_a"], c["force_b"]              # (N, C, 3)
            vm = c["valid_mask"]                             # (N, C) bool
            out = torch.zeros(self.num_envs, len(link_names), 3, device=self.device)
            for k, name in enumerate(link_names):
                gl = self._link(name).idx
                ma = (vm & (la == gl)).unsqueeze(-1)
                mb = (vm & (lb == gl)).unsqueeze(-1)
                out[:, k] = (fa * ma).sum(dim=1) + (fb * mb).sum(dim=1)
            return out
        return self._cached(("cfw", tuple(link_names), target), _read)

    def contact_penetration(self, link_names, target):
        """Deepest overlap per link against **target**, (N, K) [m], positive = overlap. Same get_contacts
        filter as contact_force_with; genesis' ``penetration`` is already positive-is-in (its solver feeds
        ``-penetration`` where MuJoCo puts ``dist``), so only the per-link MAX is ours."""
        ent = (self.objects[0] if target == "object"
               else self.robot if target == "robot" else self.fixtures[target])

        def _read():
            c = self.robot.get_contacts(with_entity=ent)
            out = torch.zeros(self.num_envs, len(link_names), device=self.device)
            pen = c["penetration"]                            # (N, C) [m]
            if pen.shape[1] == 0:                             # nothing collided anywhere this step
                return out
            la, lb = c["link_a"], c["link_b"]                 # (N, C)
            vm = c["valid_mask"]                              # (N, C) bool
            for k, name in enumerate(link_names):
                gl = self._link(name).idx
                m = vm & ((la == gl) | (lb == gl))            # either side of the pair is this link
                out[:, k] = (pen * m).amax(dim=1)             # masked-out slots are 0 → no contact reads 0
            return out.clamp_(min=0.0)
        return self._cached(("cpen", tuple(link_names), target), _read)

    # --- control / step / reset ---
    def set_joint_targets(self, names, targets):
        self.robot.control_dofs_position(targets, self._dofs(names))
        for oi, o in enumerate(self._coupled_owners):      # coupled dofs: PD target lives in our buffer
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                self._coupled_tgt[oi][:, tcols] = targets[..., cols].to(self._coupled_tgt[oi].dtype)

    def step(self, render: bool = True):
        # render=False advances physics but does not update the viewer — used for the off-screen settle at reset.
        # One control step = self._substeps scene sub-steps (the scene is built at dt/substeps — parser).
        # Whatever genesis auto-clears per scene.step (external forces) and whatever must track the freshest
        # state (the motor-space coupled PD) is re-applied INSIDE the loop — newton's _run_substeps cadence.
        # (Coupled PD at control rate is discretely unstable: stiff motor-space gains limit-cycle the
        # low-inertia axes; τ_g is config-slow, so one gravcomp evaluation per control step suffices.)
        self._apply_gravcomp_actbias()                # actuator-channel gravcomp → act_bias[0] (persists; capped by force_range)
        gc_tau = (self._gravcomp_torque()
                  if (self._coupled_owners and self._gc_on and self._gc_act_links) else None)
        for sub in range(self._substeps):
            if self._has_wrench:                      # DR external-force impulse (cleared per sub-step → reapply = one full control step)
                solver = self.objects[0].solver
                li = [self.objects[0].base_link_idx]
                solver.apply_links_external_force(self._obj_wrench[:, :3].unsqueeze(1), li, ref="link_com")
                solver.apply_links_external_torque(self._obj_wrench[:, 3:].unsqueeze(1), li, ref="link_com")
            if self._gc_link_idx is not None and self._gc_on:   # passive-channel gravcomp: external anti-gravity force (uncapped, cleared per sub-step)
                self.robot.solver.apply_links_external_force(self._gc_force, self._gc_link_idx, ref="link_com")
            if self._coupled_owners:                  # motor-space coupled PD at SUB-STEP rate → control_dofs_force(τ_q)
                for o, dofs, tgt in zip(self._coupled_owners, self._coupled_dofs, self._coupled_tgt):
                    tau = o.compute(self.robot.get_dofs_position(dofs), self.robot.get_dofs_velocity(dofs),
                                    tgt, tau_g=(gc_tau[:, dofs] if gc_tau is not None else None))
                    self.robot.control_dofs_force(tau, dofs)
            self.scene.step(update_visualizer=render and sub == self._substeps - 1)
        self._rcache.clear()   # state changed → invalidate per-step read cache

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
        """Re-read robot_model.json gains into the coupled-PD buffers. Native genesis PD stays off for
        these joints (zeroed at build) — the reloaded gains are the ones the warp kernels use."""
        return [n for o in self._coupled_owners for n in o.reload_gains()]

    def render_frame(self):
        """No-op (standalone Pause): genesis runs its viewer on its own thread (pyrender — see focus_camera's
        ``viewer.lock``), so the window keeps repainting itself while a paused run holds off ``scene.step``.
        Nothing to pump from here; the counterpart on newton, whose viewer only draws when asked, does work."""

    def set_object_pose(self, env_idx: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor):
        """Write object pose (K envs, world pos (K,3) + quat (K,4) wxyz). RigidEntity.set_pos/quat default
        zero_velocity=True → also zeroes velocity. Same call convention as reset_idx (relative default =
        heterogeneous-variant offset correction path)."""
        ent = self.objects[0]
        ent.set_pos(pos, envs_idx=env_idx)
        ent.set_quat(quat, envs_idx=env_idx)
        self._rcache.clear()   # state changed → invalidate per-step read cache

    # ---- G5 event write-surface (called by DR terms via EnvDriver; symmetric across engines) ----
    def set_joint_positions(self, names, env_idx, pos, vel):
        """Teleport ``names`` joints to pos/vel for env_idx + hold PD target at pos. Same API as reset_idx.
        zero_velocity=False + explicit set_dofs_velocity to touch only named joints surgically (avoid zeroing all dofs)."""
        if int(env_idx.numel()) == 0:
            return
        dofs = self._dofs(names)
        self.robot.set_dofs_position(pos, dofs, envs_idx=env_idx, zero_velocity=False)
        self.robot.set_dofs_velocity(vel, dofs, envs_idx=env_idx)
        self.robot.control_dofs_position(pos, dofs, envs_idx=env_idx)   # prevent next-step pullback
        for oi, o in enumerate(self._coupled_owners):      # coupled dofs: sync our PD target buffer too
            cols, tcols = self._coupled_cols(oi, o, names)
            if cols:
                self._coupled_tgt[oi][env_idx[:, None], torch.tensor(tcols, device=self.device)] = \
                    pos[..., cols].to(self._coupled_tgt[oi].dtype)
        self._rcache.clear()

    def _friction_base(self, target, ent):
        """Lazily capture target entity's base friction (first geom) — genesis has no absolute-μ setter and
        effective μ = base * friction_ratio, so ratio = μ/base reproduces an absolute μ."""
        b = self._fric_base.get(target)
        if b is None:
            b = float(ent.geoms[0].friction)
            self._fric_base[target] = b
        return b

    def set_object_friction(self, target, env_idx, mu, exclude=()):
        """Set per-env collision-shape friction μ (absolute) for ``target`` ({object|robot|table}).
        genesis only supports ratio multiplication → ratio = μ/base (base = asset default friction). μ (K,).

        ``exclude`` names bodies this write SKIPS, so a pinned surface (robot.nail_friction) keeps the value
        the build gave it. LINK granularity is all this engine has (``links_idx_local``), which is exact for
        the case it exists for: the pinned nail shells sit on their own links."""
        if int(env_idx.numel()) == 0:
            return
        ent = self.objects[0] if target == "object" else (self.robot if target == "robot" else self.fixtures[target])
        base = self._friction_base(target, ent)
        links = self._links_except(ent, exclude)
        ratio = (mu / base).unsqueeze(1).expand(-1, len(links) if links is not None else ent.n_links).contiguous()
        ent.set_friction_ratio(ratio, links_idx_local=links, envs_idx=env_idx)

    def _links_except(self, ent, exclude) -> list | None:
        """Entity-local link indices minus the bodies named in ``exclude`` — ``None`` when nothing is excluded
        (genesis reads that as "every link", and building the full list would say the same thing slower)."""
        if not exclude:
            return None
        drop = {self._link(n).idx_local for n in exclude}
        return [i for i in range(ent.n_links) if i not in drop]

    def _pin_nail_friction(self) -> None:
        """Write ``robot.nail_friction`` onto its links, once, at build. The friction DR then leaves them
        alone as long as the contract's event excludes them (``exclude="@bodies.nail"``) — which is the
        contract's call to make, not this spoke's."""
        nail = self.spec.robot.nail_friction
        if nail is None:
            return
        base = self._friction_base("robot", self.robot)
        idx = [self._link(n).idx_local for n in nail.bodies]
        ratio = torch.full((self.num_envs, len(idx)), nail.mu / base, device=self.device)
        self.robot.set_friction_ratio(ratio, links_idx_local=idx, envs_idx=None)

    def set_object_mass(self, env_idx, scale):
        """Scale object mass per-env by `scale` vs default — set_mass_shift(additive delta=(scale-1)*default).
        ⚠️ genesis mass_shift affects mass only (inertia tensor unchanged) → rotational sim2sim asymmetry vs newton (which also scales inertia)."""
        if int(env_idx.numel()) == 0:
            return
        dm = self._obj_default_mass
        dm = dm[env_idx] if dm.dim() == 2 else dm.unsqueeze(0)          # (K, n_links) broadcast
        shift = (scale.unsqueeze(1) - 1.0) * dm
        self.objects[0].set_mass_shift(shift, links_idx_local=None, envs_idx=env_idx)

    # No set_object_scale: geometry scale is a morph property consumed when the entity is added to the scene,
    # and the collision vertices (`verts_info.init_pos`) carry no env axis, so there is nothing to rescale
    # per env afterwards. The method's ABSENCE is what drops the `object_scale` capability (runtime/backend.py)
    # and with it any Event that declares `requires="object_scale"`.

    def set_root_height(self, env_idx, dz):
        """Move fixed-base robot root per-env to default world pos + [0,0,dz]. base_quat=identity so dz is the
        same in world/user frames. ⚠️ fixed-base + collision geom requires parser to load with
        batch_fixed_verts=True or the per-env set raises."""
        if int(env_idx.numel()) == 0:
            return
        pos = self._base_pos0[env_idx].clone()
        pos[:, 2] += dz
        self.robot.set_pos(pos, envs_idx=env_idx, relative=False)
        self._rcache.clear()

    def set_gravity(self, gz: float) -> None:
        """World gravity along z [m/s^2, signed] — ALL envs at once (curriculum write-surface, not per-env;
        newton's supported path is global, so the spokes stay symmetric even though genesis could take
        ``envs_idx``).

        Also refreshes the gravcomp constants. Unlike newton — where MuJoCo derives qfrc_gravcomp from
        opt.gravity itself — our genesis gravcomp is an anti-gravity force WE compute, cached at build from
        ``spec.physics.gravity``: ``_gc_g_up`` feeds the actuator-channel Jacobian sum and ``_gc_force`` is the
        precomputed passive-channel body force. Leaving them stale would hold the arm up against 9.81 while
        the world pulled at 0.1, i.e. the robot would float."""
        self.scene.sim.set_gravity((0.0, 0.0, float(gz)))
        if getattr(self, "_gc_g_up", None) is not None:
            self._gc_g_up = -torch.tensor([0.0, 0.0, float(gz)], device=self.device, dtype=torch.float32)
            if self._gc_force is not None:                 # passive channel: m_i * (-g), rebuilt in place
                self._gc_force.copy_(self._gc_mass[:, None] * self._gc_g_up[None, :])
        self._rcache.clear()

    def set_object_gravity(self, gz: float) -> None:
        """Effective gravity z on the OBJECT alone [m/s^2, signed] — the WORLD's gravity is left untouched.

        genesis carries a per-ENTITY gravity compensation that forward dynamics reads every step
        (``cdd_vel[root] = -gravity * (1 - gravity_compensation)``), so the scale written here is the one
        newton puts in MuJoCo's ``body_gravcomp``::

            gravity_compensation = 1 - |gz| / |g_world|   1 = fully cancelled (weightless), 0 = world gravity

        The field is per-entity and NOT per-env, so this moves ALL envs at once (same as :meth:`set_gravity`).
        Gravity is scaled, not force applied, so a per-env object-mass DR keeps its own scale. genesis exposes
        no runtime setter (``gravity_compensation`` is a build-time material property), hence the write into
        the solver's ``entities_info`` at the entity's own index."""
        gw = abs(float(self.spec.physics.gravity[2]))
        assert gw > 0.0, "set_object_gravity needs a non-zero physics.gravity to scale against"
        assert self.objects, "set_object_gravity: the task declares no movable object body"
        comp = 1.0 - min(abs(float(gz)), gw) / gw
        info = self.objects[0].solver.entities_info
        for ent in self.objects:
            info.gravity_compensation[ent._idx_in_solver] = comp

    def apply_object_force(self, env_idx, force):
        """Set the object's world-frame external FORCE — step() applies it once before scene.step (genesis
        auto-clears after step → 1-step impulse). The interval event rewrites every env before each step
        (unfired=0). Writes only the force half, so a torque term can own the other half independently."""
        if int(env_idx.numel()) == 0:
            return
        self._obj_wrench[env_idx, 0:3] = force
        self._has_wrench = True

    def apply_object_torque(self, env_idx, torque):
        """Set the object's world-frame external TORQUE — same lifetime/rewrite contract as
        :meth:`apply_object_force`, and leaves the force half untouched."""
        if int(env_idx.numel()) == 0:
            return
        self._obj_wrench[env_idx, 3:6] = torque
        self._has_wrench = True

    def nan_world_detected(self) -> torch.Tensor:
        """Physics-diverged (NaN/Inf) worlds → (N,) bool. Checks robot dof positions + object positions (reset on either)."""
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
            # PD control target to init too — else the actuator pulls back to 0 (default target) and the spawn pose collapses.
            self.robot.control_dofs_position(self._init_dofs[idx], self._act_dofs, envs_idx=idx)
        for ent, pos, quat in self._obj_init:
            ent.set_pos(pos.unsqueeze(0).expand(n, 3), envs_idx=idx)
            ent.set_quat(quat.unsqueeze(0).expand(n, 4), envs_idx=idx)
        self._obj_wrench[idx] = 0.0   # clear leftover external force from the previous episode (interval overwrites anyway; defensive)
        for oi, o in enumerate(self._coupled_owners):   # coupled PD: target back to init pose, readout cleared
            self._coupled_tgt[oi][idx] = self._coupled_tgt0[oi]
            o.tau_torch[idx] = 0.0
        self._rcache.clear()   # reset changed state → invalidate per-step read cache
