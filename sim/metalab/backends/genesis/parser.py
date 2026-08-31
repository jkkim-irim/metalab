"""Genesis parser (sim.metalab.backends.genesis.parser) — EnvSpec (engine-agnostic contract) → gs.Scene (Genesis spoke).

Loads **both robot and objects from MJCF (source of truth)** (native equality/gains/armature/palm/chest
frame/mass/contact properties). Mask-0 joints load from a ``mjcf_prep`` copy that makes them 0°-FIXED.
Objects spawn the MJCF-variant list as a **heterogeneous entity** (per-env variant; genesis maps them). Asset
paths resolve through ``contract.asset_path`` (repo-root-relative, or ``s3://`` fetched into a cache).

Scene construction only (no obs/reward/action logic). Applies env_spacing and robot overrides (kp/kv).
"""
from __future__ import annotations

import genesis as gs

from sim.metalab.backends.genesis import _patches
from sim.metalab.backends.genesis import friction as _friction
from sim.metalab.backends.genesis.by_basename import ByBasename
from sim.metalab.contract.asset_path import resolve_asset
from sim.metalab.contract.mjcf_prep import prepare_mjcf, prepare_object_mjcf
from sim.metalab.contract.spec import EnvSpec, RobotSpec


def _apply_overrides(robot: ByBasename, r: RobotSpec) -> None:
    """Apply active-joint MJCF-value overrides (kp/kv/armature/effort) — **after scene.build()**.

    frictionloss is NOT here: it is Coulomb friction, a physical joint property read from the MJCF only
    (genesis imports it into ``dofs_info.frictionloss`` and solves it as a constraint, same as newton).
    ``"default"`` fields are skipped (keep XML value). Assumes 1-DOF joints (single dofs_idx_local).

    A MOTOR-COUPLED joint is left force-range UNLIMITED — see :func:`_open_coupled_force_range`."""
    coupled = _coupled_joints(r)
    for jname, ov in r.joint_mode_param.items():
        idx = list(robot.get_joint(jname).dofs_idx_local)
        if ov.kp != "default":           robot.set_dofs_kp([ov.kp], idx)
        if ov.kv != "default":           robot.set_dofs_kv([ov.kv], idx)
        if ov.armature != "default":     robot.set_dofs_armature([ov.armature], idx)
        # torque limit [Nm] → dof force range. scalar e → symmetric (-e, +e); [lower, upper] → asymmetric.
        # NOT for a coupled joint: its limit lives in motor space (see _open_coupled_force_range).
        if ov.effort != "default" and jname not in coupled:
            lo, hi = (-ov.effort, ov.effort) if isinstance(ov.effort, (int, float)) else ov.effort
            robot.set_dofs_force_range([lo], [hi], idx)
    _open_coupled_force_range(robot, coupled)


def _coupled_joints(r: RobotSpec) -> set[str]:
    """Joints driven by motor-space coupled PD, or empty. Gated on ``RobotSpec.motor_coupling_on`` — the one
    place that owns ``control_mode`` AND the build toggle, so this can never disagree with the backend about
    which joints the kernel owns (they would silently double-clamp or double-open)."""
    return {j for g in r.coupled_groups() for j in g.joints} if r.motor_coupling_on() else set()


def _open_coupled_force_range(robot: ByBasename, coupled: set[str]) -> None:
    """Remove the JOINT-level torque clamp on motor-coupled joints (force_range → ±inf).

    In motor mode a coupled joint's torque comes from its group's motors, and the ONE clamp that belongs
    there is in MOTOR space, inside the coupled-PD kernel: ``τ_m = clamp(τ_m, envelope(φ̇) ∩ ±rated)``, then
    ``τ_q = Gᵀ·τ_m``. A joint-level ``force_range`` on top of that is a SECOND clamp, and a wrong one:

    * It has no counterpart on newton — there the coupled τ is written to ``control.joint_f`` (→
      ``qfrc_applied``), which ``jnt_actfrcrange`` does not touch. genesis's solver, by contrast, clamps
      EVERY ctrl mode including ``FORCE`` (``qf_applied = clamp(force, force_range)``), so leaving it makes
      the two engines deliver DIFFERENT torque from the same contract.
    * No fixed scalar can even express the real bound. The joint-space limit is ``Gᵀ·(envelope ∩ rated)``,
      which moves with the POSE (leverage) and the SPEED (per-motor torque-speed envelope) — that is a
      READOUT, published as the standalone ``Joint Torque Limit`` channel (drive/monitor._tau_lim_channel),
      not a constant. Re-clamping τ_q against it would anyway be a no-op: for any ``τ_m`` inside the motor
      box, ``(Gᵀτ_m)_j ≤ Σᵢ max(G_ij·loᵢ, G_ij·hiᵢ)`` holds identically, so the projected bound can never bite.

    Skipping the YAML override alone would in fact be enough TODAY — genesis's importer reads the ACTUATOR's
    ``forcerange``, while ALLEX declares its limit as the JOINT attribute ``actuatorfrcrange`` (MuJoCo
    ``jnt_actfrcrange``), which genesis ignores, so a coupled dof arrives at genesis's ±inf default. Writing
    ±inf anyway makes the invariant hold no matter where a finite range comes from — an MJCF that starts
    setting the actuator's own ``forcerange``, or any other write — instead of resting on that coincidence.
    The backend asserts the result (see ``GenesisBackend.__init__``), so a regression fails loud."""
    if not coupled:
        return
    inf = float("inf")
    for jname in sorted(coupled):
        idx = list(robot.get_joint(jname).dofs_idx_local)
        robot.set_dofs_force_range([-inf], [inf], idx)


def _apply_contact_params(spec: EnvSpec, handles: dict) -> None:
    """Write the contract's ``contact_params`` onto each group's collision geoms — **after scene.build()**.

    genesis carries MuJoCo's contact parameters per geom as ``sol_params`` = (timeconst, dampratio, dmin,
    dmax, width, midpoint, power), i.e. solref ++ solimp in that exact order, and consumes them in
    ``gu.imp_aref`` inside the contact constraint (same formula as MuJoCo: b = 2/(dmax·timeconst),
    k = 1/(dmax·timeconst·dampratio)²). So the contract's numbers mean the same thing on both engines.

    Group key resolves to an entity: ``"robot"``, a fixture name, or an object name. Only the fields the
    contract sets are touched (the current value is read back and patched), so ``solref`` alone cannot
    silently reset ``solimp``. Two genesis limits, both loud rather than silent:

    * ``solmix`` has no genesis equivalent — its contact mixing is a fixed 0.5/0.5 average of the two geoms,
      not solmix-weighted, so a declared solmix is reported instead of quietly ignored;
    * genesis clamps timeconst to ``2·substep_dt`` (``_sanitize_sol_params``) and warns when it bites.
    """
    cp = spec.contact_params or {}
    if not cp:
        return
    ents = {"robot": handles["robot"], **handles["fixtures"],
            **{o.name: e for o, e in zip(handles["object_specs"], handles["objects"])}}
    for key, p in cp.items():
        ent = ents.get(key)
        assert ent is not None, \
            f"contact_params key '{key}' matches no entity (have: {'|'.join(ents)})"
        if "solmix" in p:
            gs.logger.warning(f"contact_params['{key}'].solmix={p['solmix']} ignored — genesis mixes the two "
                              "geoms 0.5/0.5 and has no solmix input (newton honors it; this is a sim2sim gap)")
        if "solref" not in p and "solimp" not in p:
            continue
        for geom in ent.geoms:
            sp = [float(v) for v in geom.sol_params]          # (7,) timeconst, dampratio, dmin..power
            assert len(sp) == 7, f"contact_params['{key}']: unexpected sol_params length {len(sp)}"
            if "solref" in p:
                sp[0:2] = [float(v) for v in p["solref"]]
            if "solimp" in p:
                sp[2:7] = [float(v) for v in p["solimp"]]
            geom.set_sol_params(sp)


def build_scene(spec: EnvSpec, num_envs: int | None = None, viz: bool = False, backend=None) -> dict:
    """EnvSpec → built gs.Scene. Returns handles: {scene, robot, objects, fixtures}.

    ``gs.init`` is once-per-process — build only one scene per process. Rasterizer path (RT needs a
    LuisaRender build).
    """
    gs.init(backend=backend if backend is not None else gs.gpu, logging_level="warning")
    _patches.apply()   # remove the O(n_envs) GPU-scalar indexing bottleneck in invweight init (required pre-build)
    _patches.apply_round_robin_variants()   # env i -> variant i % N, matching newton (see the patch)

    ph = spec.physics
    scene = gs.Scene(
        # Substeps are driven by the BACKEND loop (scene dt = one sub-step, substeps=1): backend.step()
        # calls scene.step() ph.substeps times per control step, re-applying external forces and the
        # motor-space coupled PD each sub-step — newton's _run_substeps cadence. (Coupled PD at control
        # rate is discretely unstable: the stiff motor-space gains limit-cycle low-inertia axes.)
        sim_options=gs.options.SimOptions(dt=ph.dt / ph.substeps, substeps=1),
        # self-collision off (default) → skip full-ALLEX self-pair computation, ~10x faster scene.build (go2 approach).
        # robot↔object/fixture collisions stay on. opt-in via contract physics.self_collision.
        # batch_dofs_info: per-env dof info (act_bias/force_range/…) so the backend can write a per-env, config-
        # dependent gravity-compensation feedforward into act_bias[0] (actuator-routed gravcomp, capped by
        # force_range) — see GenesisBackend.step / _joint_torque.
        rigid_options=gs.options.RigidOptions(enable_self_collision=ph.self_collision, batch_dofs_info=True),
        vis_options=gs.options.VisOptions(),
        show_viewer=viz,
    )
    handles: dict = {"scene": scene, "fixtures": {}, "objects": [], "substeps": ph.substeps,
                     # spec entries matching handles["objects"] 1:1 (fixed objects are scenery, excluded)
                     "object_specs": [o for o in spec.objects if not o.fixed]}

    # --- the stage: ground plane (spec.scene, not a fixture) ---
    if spec.scene.ground:
        handles["fixtures"][spec.scene.ground_name] = scene.add_entity(gs.morphs.Plane())

    # --- fixtures (props the task places: table, ...) ---
    for fx in spec.fixtures:
        if fx.kind == "box":
            assert fx.size is not None, f"box fixture '{fx.name}' requires size"
            handles["fixtures"][fx.name] = scene.add_entity(
                gs.morphs.Box(size=tuple(fx.size), pos=tuple(fx.pos), fixed=True)
            )
        else:  # pragma: no cover
            raise ValueError(f"unsupported fixture kind: {fx.kind}")

    # --- objects (MJCF assets; variant list → heterogeneous entity, one variant per env) ---
    # env i → variant i % N, matching newton. genesis' own map is contiguous BLOCKS, which ALIGNS with SAPG's
    # contiguous env blocks and leaves a block training on one shape — _patches.apply_round_robin_variants()
    # rebinds it (applied above, pre-build).
    # A FIXED object (procedural parts, e.g. a table) is welded scenery: same spawn as a box fixture, and it
    # is registered under its own name so contact/DR selectors find it — but it stays out of the "object"
    # selector, which means the first movable object.
    for obj in spec.objects:
        if obj.fixed:
            assert obj.parts, f"fixed object '{obj.name}': procedural parts required"
            for part in obj.parts:
                assert part.shape == "box", f"object '{obj.name}': fixed parts support shape 'box' for now"
                pos = tuple(a + b for a, b in zip(obj.init_pos or (0.0, 0.0, 0.0), part.pos))
                ent = scene.add_entity(gs.morphs.Box(size=tuple(part.size), pos=pos, fixed=True))
            handles["fixtures"][obj.name] = ent      # last part is the selector handle (single-part today)
            continue                                 # NOT in handles["objects"]: that list is the movable set
        mjcfs = obj.asset.get("mjcf") if obj.asset else None
        assert mjcfs, f"object '{obj.name}': MJCF asset required (asset.mjcf)"
        variants = mjcfs if isinstance(mjcfs, list) else [mjcfs]
        kw: dict = {"quat": tuple(obj.init_quat)}
        if obj.init_pos is not None:
            kw["pos"] = tuple(obj.init_pos)
        # contract mass (+consistent inertia) is written into a prepared copy — see prepare_object_mjcf
        morphs = [gs.morphs.MJCF(file=prepare_object_mjcf(str(resolve_asset(v)), obj.mass, obj.collision),
                                 default_armature=None, **kw)          # see the robot morph below
                  for v in variants]
        # 2+ variants = heterogeneous (per-env geometry); 1 = single morph.
        ent = scene.add_entity(morphs if len(morphs) > 1 else morphs[0])
        handles["objects"].append(ent)

    # --- robot (MJCF source of truth; mask-0 joints made 0°-FIXED by mjcf_prep; native equality/gains/armature) ---
    r = spec.robot
    assert "mjcf" in r.asset, "robot.asset.mjcf required (MJCF source of truth)"
    # MASK-0 joints listed in init_pose are welded AT that angle (posed-then-welded; baked into the MJCF
    # copy — same copy on both engines = parity). Active init_pose entries are applied by the backend.
    fixed_pose = {n: v for n, v in spec.robot.init_pose.items() if r.joints.get(n) == 0}
    mjcf = prepare_mjcf(str(resolve_asset(r.asset["mjcf"])), r.joints, r.collision, fixed_pose=fixed_pose)
    # batch_fixed_verts=True: allow per-env root pose writes on the fixed-base (+collision geom) robot
    # (set_root_height DR — else env-distinct set_pos raises). More memory, but a precondition for root-height DR.
    # requires_jac_and_IK=True: the actuator-routed gravcomp feedforward g(q)=Σ Jᵀ(m·g) (written into
    # act_bias[0] each step, see GenesisBackend._apply_gravcomp_actbias) needs get_jacobian.
    # default_armature=None: genesis' MJCF morph otherwise DEFAULTS armature to 0.1 for every non-free joint
    # that does not declare one (utils/mjcf.build_model: attrib.setdefault("armature", default_armature)).
    # MuJoCo and newton use 0 there, so the ALLEX joints that omit the attribute — the five equality FOLLOWERS
    # R_{Index,Middle,Ring,Little}_DIP and R_Thumb_IP — were silently given 0.1 kg*m^2 of rotor inertia here,
    # about 5 orders of magnitude above their own link inertia. Reflected through the mimic coupling
    # (dDIP/dPIP ~ 0.5) that is ~0.025 kg*m^2 added at the driving PIP against its authored 0.00154 — a finger
    # roughly 16x heavier than newton's, i.e. a different plant for the same motor-space PD. The MJCF is the
    # single source of truth for armature; None makes genesis read exactly what it says. (Measured: with the
    # default on, the two engines' fingers settle 0.02-0.04 rad apart under an identical command.)
    robot = ByBasename(scene.add_entity(gs.morphs.MJCF(
        file=mjcf, pos=tuple(r.base_pos), quat=tuple(r.base_quat), batch_fixed_verts=True,
        requires_jac_and_IK=True, default_armature=None,
    )))
    handles["robot"] = robot

    # viewer plugin to grab a body with the mouse and apply an external spring force (viz only; force only while grabbing).
    if viz:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(use_force=True, color=(0.1, 0.6, 0.8, 0.6))
        )

    # env_spacing: visualization-only grid spacing (no physics effect) — pass contract value (else envs overlap at origin).
    scene.build(
        n_envs=num_envs if num_envs is not None else spec.num_envs,
        env_spacing=(spec.env_spacing, spec.env_spacing),
    )
    _apply_overrides(robot, r)   # kp/kv + effort(torque-limit) overrides from robot YAML (post-build)
    _apply_contact_params(spec, handles)   # contract-authored contact softness (robot / fixtures / objects)
    _friction.install(scene.sim.rigid_solver)   # geometric-mean contact friction (runtime/physics/friction.py)

    return handles
