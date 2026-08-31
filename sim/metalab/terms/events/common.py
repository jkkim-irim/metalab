"""Common event/DR terms — backend writes (engine-agnostic).

Term contract — a FLAT function, the same shape obs and reward terms use:
  ``fn(env, env_ids, **knobs)``
  - ``env``     = env_driver's :class:`EnvDriver` — backend read/**write** delegation (set_object_pose etc.)
                  + driver context (``.lifted``). Same ``__getattr__`` delegation as EnvDriver.
  - ``env_ids`` = ``(K,)`` long tensor. mode="reset" -> **envs reset this step**;
                  mode="interval" -> **all envs, called every policy step**. How often an interval term
                  actually ACTS is the term's own knob (see ``apply_object_external_force``'s
                  ``interval_range_s``): the driver owns no cadence, because only the term knows whether its
                  interval counts wall time, lifted time, or something else.
  - ``**knobs`` = the contract's ``Event(...)`` keywords, resolved into ``EventTerm.params`` at load and
                  passed on every call. All NAMED — there is no factory left to take positional args.
  - No return (side effect = sim write). Order: backend.reset_idx -> reset events ->
    driver init-state capture (object_init_z, default) — terms may rely on this order.
  - Knobs the CURRICULUM tunes (``mu_scale``, ``mass_scale``, an external load's ``z_range``) are ordinary
    parameters with a neutral default; the curriculum writes ``env.event_terms[name].params[knob] = …`` and
    the next call picks it up. This is why no term here needs to be a class with a mutable attribute.
  - Per-env episode state is DRIVER-owned — ``env.buffer(key, shape, fill)`` (:meth:`EnvDriver.buffer`), cleared to ``fill`` for the reset envs BEFORE the reset events run. That is
    why no term needs ``.init``/``.reset`` hooks: a term that must remember something across steps (an
    interval countdown, say) still stays a flat function.

Reference: sim/isaaclab/envs/{hammer_lift,perceptive_dexdeepmimic}/mdp/events.py,
hammer_lift/physics_material.py (set_shape_friction). DR value source: legacy 60a5e58 experiment.py.
"""
from __future__ import annotations

import torch

from sim.metalab.api.frames import quat_mul

# ============================================================================
# WRITE-API — the write methods event terms reach through env (EnvDriver -> backend). All but the last are
# implemented SYMMETRICALLY on both engines (genesis/newton):
#     env.set_object_pose(env_ids, pos, quat)             — teleport object pose (reset_object_pose)
#     env.set_joint_positions(names, env_ids, pos, vel)   — teleport joint state (reset_joints_by_offset)
#     env.set_object_friction(target, env_ids, mu)        — target asset collision-shape friction mu (set_shape_friction)
#     env.set_object_mass(env_ids, scale)                 — object mass (+inertia) scale vs default (randomize_rigid_body_mass)
#     env.set_root_height(env_ids, dz)                    — fixed-base root z offset vs default (randomize_fixed_base_root_height)
#     env.set_object_scale(env_ids, scale)                — object GEOMETRY scale vs default (randomize_object_scale)
#       ^ NEWTON-ONLY (native contacts): genesis bakes morph scale at build time and raises. The one
#         asymmetric entry — see the term's own docstring for what mjwarp contacts do and do not honor.
#     env.apply_object_force(env_ids, force)              — object world-frame force at its COM  (apply_object_external_force)
#     env.apply_object_torque(env_ids, torque)            — object world-frame torque about its COM (apply_object_external_torque)
#   Those last two write OPPOSITE HALVES of one (N,6) external-wrench buffer, which is why both terms can be
#   wired at once. Either half is HELD FOR ONE control step and then auto-cleared, so an interval term must
#   rewrite every env every step (unfired envs = 0) or a stale load would linger.
# ============================================================================


def reset_object_pose(env, env_ids, active_position, x_range, y_range, yaw_range,
                      base_quat=(1.0, 0.0, 0.0, 0.0)):
    """Place reset envs' object at ``active_position`` + random (x, y, yaw) offset (zero velocity).

    ``base_quat`` (wxyz) is a fixed base rotation applied before yaw (e.g. laying a composite flat) —
    final orientation = ``q_yaw ⊗ base_quat``. Default identity. Same semantics as original
    ``reset_object_pose`` (perceptive, xyzw), ported to the wxyz convention.

The sampled position describes the object's ORIGIN, which the asset pipeline puts on the grasp point,
    so the spawn envelope means the same thing for every shape without the caller correcting for where a
    mesh happens to be authored."""
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    pos = torch.as_tensor(active_position, dtype=torch.float32, device=dev).unsqueeze(0).expand(k, 3).clone()
    pos[:, 0] += torch.empty(k, device=dev).uniform_(*x_range)
    pos[:, 1] += torch.empty(k, device=dev).uniform_(*y_range)
    yaw = torch.empty(k, device=dev).uniform_(*yaw_range)
    half = yaw * 0.5
    q_yaw = torch.zeros(k, 4, device=dev)
    q_yaw[:, 0] = torch.cos(half)                      # w
    q_yaw[:, 3] = torch.sin(half)                      # z
    base_rot = torch.as_tensor(base_quat, dtype=torch.float32, device=dev)
    quat = quat_mul(q_yaw, base_rot.unsqueeze(0).expand(k, 4))
    env.set_object_pose(env_ids, pos, quat)


def reset_joints_by_offset(env, env_ids, joints, position_range, velocity_range=(0.0, 0.0)):
    """Teleport reset envs' ``joints`` to the (reset-restored) default pose + random offset.

    Ports upstream ``reset_joints_by_offset``: pos = default + U(position_range), vel = U(velocity_range).
    Runs **after** reset_idx already restored init state, so default = current joint_pos, vel ~ 0.
    Clamps pos to ``joint_limits`` (per-joint (J,)) on engines that provide them. _post_reset re-captures
    the written pos as the action-decode default -> the offset pose becomes the new neutral.
    """
    k = int(env_ids.numel())
    if k == 0:
        return
    base = env.joint_pos(joints)[env_ids]                      # (k, J) restored init
    pos = base + torch.empty_like(base).uniform_(*position_range)
    lo, hi = env.joint_limits(joints)                          # (J,), (J,) — broadcast over envs
    pos = torch.clamp(pos, lo, hi)
    vel = torch.empty_like(base).uniform_(*velocity_range)     # vel ~ 0 after reset -> target vel
    env.set_joint_positions(joints, env_ids, pos, vel)


def set_shape_friction(env, env_ids, target, mu=1.0, mu_range=None, mu_scale=1.0, exclude=()):
    """Set ``target`` asset's collision-shape friction mu on each reset.

    With ``mu_range``, per-env U(mu_range) random; else fixed ``mu``. Same semantics as original
    physics_material.set_shape_friction (all shapes in an env share one mu). ``target`` = asset selector
    (e.g. "object"/"robot"/"table") — resolved to collision shapes by the G5 backend.

    ``exclude`` names BODIES this write skips (``"@bodies.nail"`` → the robot's pinned nail shells). Their
    friction then stays whatever the build wrote — ``RobotSpec.nail_friction`` for the nails — while every
    other shape of the target keeps being randomized. Stated at the call site rather than hidden in the
    target set, so a contract that WANTS the nails randomized just drops the knob.

    Two separate factors, meaning different things — the reason the ramp is its OWN knob rather than the
    curriculum rewriting ``mu_range``:

    * ``mu_range`` / ``mu`` — the contract's authored DR, "how uncertain this surface's friction is". Fixed.
    * ``mu_scale`` — a CURRICULUM-owned multiplier on whatever was drawn, "how grippy the world is at this
      level" (1.0 = the authored value). The curriculum writes it into ``EventTerm.params``; left at its
      default a task with no friction ramp is unaffected.

    Kept apart so the ramp never rewrites the authored range and the DR spread stays PROPORTIONAL at every
    level. The engines combine the two sides of a contact as ``sqrt(mu_a * mu_b)``, so scaling both the hand
    and the object by s scales the pair's effective mu by s too.

    Publishes the drawn mu (post ``mu_scale``) as the DR channel ``"<target>_friction"``, read by
    ``obs.dr_params``.
    """
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    if mu_range is not None:
        vals = torch.empty(k, device=dev).uniform_(*mu_range)
    else:
        vals = torch.full((k,), float(mu), device=dev)
    mu_eff = vals * mu_scale
    env.set_dr_value(f"{target}_friction", env_ids, mu_eff)
    env.set_object_friction(target, env_ids, mu_eff, exclude=tuple(exclude))


def randomize_rigid_body_mass(env, env_ids, scale_range, mass_scale=1.0, operation="scale"):
    """Scale object mass (+inertia) by per-env U(scale_range) vs default on each reset.

    Ports only the operation="scale" path of original ``randomize_rigid_body_mass_newton`` (hammer uses
    scale only). Scaling is relative to default (USD/native) mass, so it doesn't accumulate across resets
    (caching is the G5 backend's job).

    Two separate factors, for the same reason as ``set_shape_friction``'s pair:

    * ``scale_range`` — the contract's DR spread, "how uncertain the object's mass is". Authored, fixed.
    * ``mass_scale`` — a CURRICULUM-owned multiplier, "how heavy the object is at this level" (1.0 = the
      contract's own mass). Left at its default a task with no mass ramp is unaffected.

    Keeping them apart means the ramp never rewrites the authored DR range and the DR spread stays
    proportional at every level.

    Publishes the drawn scale (post ``mass_scale``) as the DR channel ``"object_mass_scale"``, read by
    ``obs.dr_params``.
    """
    assert operation == "scale", \
        f"randomize_rigid_body_mass: only operation='scale' supported (got {operation!r})"
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    scale = torch.empty(k, device=dev).uniform_(*scale_range) * mass_scale
    env.set_dr_value("object_mass_scale", env_ids, scale)
    env.set_object_mass(env_ids, scale)


def randomize_object_scale(env, env_ids, scale_range):
    """Scale the object's GEOMETRY by per-env U(scale_range) vs the asset default on each reset.

    Relative to the asset default, so it does not accumulate across resets (the backend caches the baseline).
    Uniform scale on collision + visual shapes; mass and inertia stay with ``randomize_rigid_body_mass``, so
    a 2x hammer is currently a 2x-large hammer of unchanged weight and rotational inertia.

    ENGINE SUPPORT (fails loud where absent rather than silently spawning one size):
    * newton + ``use_mujoco_contacts=False`` — honored, the native narrowphase scales mesh vertices per shape.
    * newton + ``use_mujoco_contacts=True`` — the COLLIDING hull does NOT change (mjwarp reads a global
      ``mesh_vert``; the scale is baked at build time from the template world). Visual size does.
    * genesis — absent: morph scale is build-time only, and the collision vertices carry no env axis. The
      contract states this with ``Event(..., requires="object_scale")``, which drops the term on genesis.

    Publishes the drawn scale as the DR channel ``"object_scale"``, read by ``obs.dr_params`` — so on an
    engine that drops this term the channel is absent and a critic asking for it fails loud rather than
    reading a constant 1.0 as if it were a live draw.
    """
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    assert scale_range[0] > 0.0, f"randomize_object_scale: scale_range must be positive (got {scale_range})"
    scale = torch.empty(k, device=dev).uniform_(*scale_range)
    env.set_dr_value("object_scale", env_ids, scale)
    env.set_object_scale(env_ids, scale)


def _external_wrench_vec(env, env_ids, x_range, y_range, z_range, interval_range_s, eligible):
    """Shared body of the two external-load terms → ``(K,3)`` world-frame vector, already gated to 0 for the
    envs that do not act this step.

    Per axis, independently per env: ``U(range)`` in the term's unit, each range defaulting to ``(0,0)`` so an
    unmentioned axis simply contributes nothing. Non-firing envs get 0, which is not merely a no-op — the
    backend HOLDS its half of the external wrench for one control step and auto-clears it, so an interval term
    must rewrite EVERY env every step or a stale load would linger.

    ``interval_range_s`` — the TERM's firing cadence, not the driver's. The driver calls every interval term
    once per policy step; how often the term ACTS is its own business:

    * ``(0.0, 0.0)`` (default) — act on every step: a sustained load, magnitude redrawn each step.
    * ``(lo, hi)`` with ``0 < lo <= hi`` — fire once, then wait ``U(lo, hi)`` seconds before the next one. Only
      ELIGIBLE steps tick the countdown (see ``eligible``), so a gated wrapper's wait measures time spent in
      its gate rather than wall time. The countdown lives in a driver-owned ``EnvDriver.buffer``, which is what
      keeps these flat functions with no init/reset hook; it starts at 0 = due, so the first eligible step
      always fires. The wait is stated in seconds but counted in whole POLICY STEPS
      (``round(s / step_dt)``, at least 1) — an integer countdown is exact, where repeatedly subtracting
      ``step_dt`` from a float drifts (measured: it stretched a 10-step interval to 12).

    ``eligible`` is NOT a contract knob — leave it unset. It is how a task-specific wrapper term narrows one of
    these to a condition it owns: a ``(K,)`` bool mask over ``env_ids`` (see
    ``events/hammer_lift.apply_object_external_force_when_lifted``, which passes its lift z-window).
    """
    k, dev = int(env_ids.numel()), env_ids.device
    ok = torch.ones(k, dtype=torch.bool, device=dev) if eligible is None else eligible
    lo, hi = float(interval_range_s[0]), float(interval_range_s[1])
    if (lo, hi) == (0.0, 0.0):
        fire = ok
    else:
        assert 0.0 < lo <= hi, \
            f"interval_range_s={interval_range_s} must be (0,0)=every step or 0 < lo <= hi [s]"
        wait = env.buffer("next_fire_steps", dtype=torch.long)     # (N,) eligible STEPS until this env fires
        # Tick BEFORE testing, so a drawn wait of n steps puts exactly n steps between fires (testing first
        # would cost one extra step). Ticks only on eligible steps; otherwise the count is untouched.
        left = wait[env_ids] - ok.long()
        fire = ok & (left <= 0)
        draw = (torch.empty(k, device=dev).uniform_(lo, hi) / env.step_dt).round().long().clamp(min=1)
        wait[env_ids] = torch.where(fire, draw, left)
    bounds = torch.tensor([tuple(x_range), tuple(y_range), tuple(z_range)],
                          dtype=torch.float32, device=dev)         # (3,2) per-axis [lo, hi]
    assert (bounds[:, 0] <= bounds[:, 1]).all(), \
        f"each range needs lo <= hi — got x={tuple(x_range)} y={tuple(y_range)} z={tuple(z_range)}"
    vec = torch.rand(k, 3, device=dev) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return vec * fire.float().unsqueeze(-1)                        # (k,1) branchless gate


def apply_object_external_force(env, env_ids, x_range=(0.0, 0.0), y_range=(0.0, 0.0), z_range=(0.0, 0.0),
                                interval_range_s=(0.0, 0.0), eligible=None):
    """Apply an ABSOLUTE world-frame external FORCE at the object COM — interval event.

    One range per world axis, in NEWTONS, drawn independently per env; an axis left at ``(0,0)`` contributes
    nothing (so a downward pull is just ``z_range=[-5, -3]``). Torque is a separate term,
    :func:`apply_object_external_torque` — they own opposite halves of the backend's external-wrench buffer, so
    wiring both is safe and neither zeroes the other.

    Absolute newtons ON PURPOSE. The force enters the dynamics as an external load and the solver resolves it
    against whatever constraints hold the object — a firm grasp's friction cone carries it and the object does
    not move, a weak one slips. (That is what a VELOCITY write cannot express: overwriting the object's
    velocity bypasses the contact solve, so the object moves regardless of how good the grip is.) Because the
    load is absolute, its size RELATIVE to the object's weight varies with any mass DR.

    Cadence (``interval_range_s``) and ``eligible`` semantics: :func:`_external_wrench_vec`. Any of the axis
    ranges is designed to be **curriculum-tuned** — a ramp retunes it by writing
    ``env.event_terms[name].params["z_range"]``, which the next call reads.
    """
    if int(env_ids.numel()) == 0:
        return
    force = _external_wrench_vec(env, env_ids, x_range, y_range, z_range, interval_range_s, eligible)
    env.apply_object_force(env_ids, force)


def apply_object_external_torque(env, env_ids, x_range=(0.0, 0.0), y_range=(0.0, 0.0), z_range=(0.0, 0.0),
                                 interval_range_s=(0.0, 0.0), eligible=None):
    """Apply an ABSOLUTE world-frame external TORQUE about the object COM — interval event.

    The rotational twin of :func:`apply_object_external_force`: one range per world axis, in N·m, drawn
    independently per env, an axis left at ``(0,0)`` contributing nothing. Same absolute-load reasoning — the
    solver resolves the twist against the grasp, so a firm grip resists it and a sloppy one lets the object
    rotate in the hand. Owns the TORQUE half of the backend's external-wrench buffer, so it composes with the
    force term instead of competing with it.

    Cadence (``interval_range_s``) and ``eligible`` semantics: :func:`_external_wrench_vec`.
    """
    if int(env_ids.numel()) == 0:
        return
    torque = _external_wrench_vec(env, env_ids, x_range, y_range, z_range, interval_range_s, eligible)
    env.apply_object_torque(env_ids, torque)


def randomize_fixed_base_root_height(env, env_ids, z_offset_range):
    """Offset fixed-base robot root height by per-env U(z_offset_range) vs default on each reset.

    Ports original ``randomize_fixed_base_root_height_newton`` (edits Newton model.joint_X_p z) —
    the G5 backend does engine-specific root-anchor manipulation; the HUB term only samples dz.

    Publishes the drawn dz as the DR channel ``"root_height"``, read by ``obs.dr_params``.
    """
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    dz = torch.empty(k, device=dev).uniform_(*z_offset_range)
    env.set_dr_value("root_height", env_ids, dz)
    env.set_root_height(env_ids, dz)

