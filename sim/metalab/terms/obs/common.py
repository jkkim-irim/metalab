"""Observation terms — one flat function per obs (engine-agnostic).

    def <obs_name>(env, <knob>=<default>, ...) -> (N, d)

The driver calls ``t.fn(env, **t.params)`` once per obs GROUP per policy step and assembles

    group_frame = cat([ noise(value) * scale  for each term in the group ])

Noise and scale belong to the ENTRY, not to the function: the same term object is shared by the actor and
the critic (critic = ACTOR + CRITIC blocks), and only the groups in ``TaskSpec.obs_noise_groups`` get
corrupted — so the actor reads a noisy sensor while the critic reads that very same term clean.

THREE RULES:

1. **A term never scales or corrupts itself.** It returns the physical quantity; ``scale`` and ``noise``
   live in the contract's ``Obs(...)`` entry.
2. **Every knob is named.** Entries are kwargs-only (``Obs(obs.object_pose, chest_body="@frames.chest_origin")``),
   so a contract line says what each value IS without opening this file. The loader checks the names against
   the signature at LOAD, so a typo fails at launch, not on step 1.
3. **A term is stateless.** Anything that must be counted, latched or differenced across steps lives in the
   DRIVER and is read here — because a term is evaluated once per GROUP, a counter incremented in here would
   tick twice per step for any term the actor and critic share. That is why ``fingertip_contact_steps``,
   ``joint_accelerations`` read driver buffers instead of keeping their own.

``env`` (EnvDriver) gives every backend read (``object_pos()``, ``body_pos(name)``, ``joint_pos(names)``,
``contact_force_with(...)``) plus the driver-owned context the backend cannot supply: ``last_action``,
``prev_action_targets``, ``episode_step``, ``last_reward``, ``joint_acc(names)``, ``contact_steps(...)``,
``lifted``, ``goal_pos``/``goal_quat``/``goal_half_extent``,
``gate_near``/``curriculum_near``, ``curriculum_values``, ``reward_best(term)``. Those are DATA and
episode state; a metric built on them (the goal distance, the grasp point) is computed by the term.

Quaternions are exposed as **wxyz** — the pipeline's canonical order (``envs.conventions.CANONICAL_QUAT``)
AND the real robot's FK order, so a pose term hands the state adapter's quaternion straight through and no
obs slice needs reordering at deployment. (Was xyzw per the perceptive convention, with
:func:`palm_pose_in_chest` as the sole exception; one order everywhere beats one exception.)
Shared math lives in ``sim/metalab/api`` (``frames``, ``keypoints``, ``kinematics``).
"""
from __future__ import annotations

import torch

from sim.metalab.api import frames, keypoints, kinematics
from sim.metalab.api.contact import contact_mask


def joint_state(env, names: list[str]) -> torch.Tensor:
    """Joint pos+vel concat → (N, 2*len(names))."""
    return torch.cat([env.joint_pos(names), env.joint_vel(names)], dim=-1)


def object_state_world(env) -> torch.Tensor:
    """Object world pose (pos3 + quat wxyz4) → (N, 7)."""
    return torch.cat([env.object_pos(), env.object_quat()], dim=-1)


def object_pose(env, chest_body: str, offset=(0.0, 0.0, 0.0), seen: bool = False) -> torch.Tensor:
    """Object pose relative to chest-origin frame (pos3 + quat wxyz4) → (N, 7). Invariant to robot translation.

    ``offset`` shifts the chest REFERENCE frame (see kinematics.to_chest). The object's own origin IS its
    grasp point — the asset pipeline puts it there — so this channel already reads "where the graspable spot
    is" with nothing to configure.

    ``seen`` picks WHICH pose: False = live (privileged — the critic's channel), True = the driver's
    perception latch, i.e. the object as the actor last SAW it. The latch tracks the object for the opening
    steps of the episode and freezes after (window = ``_advance_object_seen``, ramped by the curriculum), so
    a 1-step window makes this the spawn pose and a full-episode window makes it the live pose. That is the
    real-robot-producible channel: on hardware the hand covers the object as it closes."""
    p, q = ((env.object_seen_pose_w[:, :3], env.object_seen_pose_w[:, 3:7]) if seen
            else (env.object_pos(), env.object_quat()))
    cp, cq = env.body_pos(chest_body), env.body_quat(chest_body)
    op = p
    rp, rq = kinematics.to_chest(op, q, cp, cq, offset)
    return torch.cat([rp, rq], dim=-1)


def body_pose_in_chest(env, chest_body: str, target_body: str, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Target body pose relative to chest-origin frame (pos3 + quat wxyz4) → (N, 7). Invariant to robot translation.
    (For palm relative pose — body version of object_pose.)"""
    cp, cq = env.body_pos(chest_body), env.body_quat(chest_body)
    rp, rq = kinematics.to_chest(env.body_pos(target_body), env.body_quat(target_body), cp, cq, offset)
    return torch.cat([rp, rq], dim=-1)


# The real robot's palm frame is our MJCF palm body rotated 180° about its own X axis (its Y and Z point the
# other way), and its FK reports the quaternion as (qw, qx, qy, qz). Measured on the right arm at
# q = [-15.341, -9.684, -5.829, -84.220, -1.307, 1.554, -4.675] deg: real (w,x,y,z) =
# (0.056494, 0.611168, 0.037405, 0.788595) vs sim (0.612612, -0.055801, -0.787515, 0.037601) — a pure
# body-side 180° X rotation, residual 0.25°. (Both engines agree with each other; this is a model/convention
# difference, not physics.) Position was NOT affected by the flip and still differs by ~6 mm — a separate
# reference-point question, deliberately NOT compensated here.
_PALM_TO_REAL = torch.tensor([0.0, 1.0, 0.0, 0.0])   # wxyz: 180° about the palm's local X


def palm_pose_in_chest(env, chest_body: str, palm_body: str, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Palm pose relative to chest-origin, **in the real robot's convention** → (N, 7) = pos3 + quat4.

    Same transform as :func:`body_pose_in_chest` plus the palm frame alignment above, so a policy trained on
    this term can be fed the real robot's FK output verbatim at deployment — no conversion step on the robot
    side, which is the whole point of putting it here rather than in the dashboard.

    ONE deliberate deviation from the usual obs convention, to match the robot: the frame is rotated 180°
    about the palm's local X (``_PALM_TO_REAL``). Quaternion order is no longer one of them — every pose
    term emits the real FK's ``qw, qx, qy, qz`` now (see the module header).

    Changing a policy's obs to/from this term invalidates existing checkpoints — the quaternion slots carry
    different numbers. Verified against the real arm on the right palm; the left is assumed symmetric.
    """
    cp, cq = env.body_pos(chest_body), env.body_quat(chest_body)
    rp, rq = kinematics.to_chest(env.body_pos(palm_body), env.body_quat(palm_body), cp, cq, offset)
    rq = frames.quat_mul(rq, _PALM_TO_REAL.to(rq))   # body side: ours → real
    return torch.cat([rp, rq], dim=-1)               # wxyz (real order)


# --- privileged(teacher) terms: EnvDriver context + backend read compositions ---
def last_action(env) -> torch.Tensor:
    """Last policy action (raw, pre-decode) → (N, num_actions)."""
    return env.last_action


def joint_positions(env, names: list[str]) -> torch.Tensor:
    """Joint positions → (N, len(names))."""
    return env.joint_pos(names)


def joint_velocities(env, names: list[str]) -> torch.Tensor:
    """Joint velocities → (N, len(names))."""
    return env.joint_vel(names)


def joint_accelerations(env, names: list[str]) -> torch.Tensor:
    """Joint accelerations (finite difference) → (N, len(names))."""
    return env.joint_acc(names)


def joint_torque_obs(env, names: list[str]) -> torch.Tensor:
    """Applied **actuator** torque per joint → (N, len(names)) [Nm]. Name-based so arm/hand can be split.
    Per the read contract this is motor-space PD + the actuator-routed gravity feedforward, after the
    driver's clamp, mapped back to joint space — what the real robot's joint-torque sensor reads
    (see :meth:`sim.metalab.api.state.StateAdapter.joint_torque`)."""
    return env.joint_torque(names)


def joint_pd_torque_obs(env, names: list[str]) -> torch.Tensor:
    """**PD component** of the applied actuator torque → (N, len(names)) [Nm], PRE-clamp. The motor-space
    PD torque mapped back through ``Gᵀ`` before the torque-speed/rated envelope clamp — which the driver
    applies to the SUM (PD + gravity). So ``pd + gravcomp == joint_torque_obs`` only while unsaturated.
    Needs a backend with the motor-level control path (optional read)."""
    return env.joint_torque_pd(names)


def joint_gravcomp_torque_obs(env, names: list[str]) -> torch.Tensor:
    """**Gravity-feedforward component** of the applied actuator torque → (N, len(names)) [Nm], PRE-clamp.
    Only the ACTUATOR-routed gravcomp (it consumes motor-torque budget inside the clamp, like the real
    motor); passive-channel gravcomp is an external force, not motor torque, so it is excluded — the same
    boundary as :func:`joint_torque_obs`. Needs a backend with gravcomp + the motor-level path (optional)."""
    return env.joint_torque_gravcomp(names)


def body_contact_flags(env, bodies: list[str], threshold: float = 1.0) -> torch.Tensor:
    """Per-body contact flag (|NET contact force| > threshold) → (N, len(bodies)) {0,1}.

    Net, so it answers "is this link touching ANYTHING" — the table, the object, the ground or another link
    alike. Scope it to one counterpart with ``hand_object_force_magnitude`` instead when the question is
    "is it touching THAT"."""
    f = env.contact_force(bodies)                    # (N, K, 3)
    return (f.norm(dim=-1) > threshold).float()      # (N, K)


def fingertip_contact_steps(env, force_threshold: float = 0.1,
                            target: str | None = "object") -> torch.Tensor:
    """Per-fingertip **consecutive PAD-press duration** in policy steps → (N, K) float, K = the robot's
    fingertips.

        +1 every policy step ``contact_mask`` holds for that tip, 0 the step it breaks, 0 at episode reset

        force_threshold  0.1 [N]   pad-normal force that counts as pressing
        target          "object"   counterpart the force is scoped to; ``None`` = NET over everything, which
                                   on a table-top task is dominated by the table during the approach

    Same predicate as the grip reward and the GATE (:func:`sim.metalab.api.contact.contact_mask`), so "5
    steps of contact" here means what the reward pays for rather than "touching something with any surface".
    WHICH bodies is not a knob: the tips are pad shells, a robot fact (``RobotSpec.fingertips``).

    Worth more than a boolean to a blind policy: a firm hold and a chattering touch look identical as flags,
    and contact flicker is exactly what breaks a carry.

    The counter lives in the DRIVER (``EnvDriver._contact_steps``) because an obs term is evaluated once per
    obs GROUP — a counter incremented here would tick twice per step for a term the actor and critic share.
    Raw counts (0 → episode length) are a wide range on purpose; the runner's obs normalization rescales it."""
    return env.contact_steps(force_threshold, target)   # (N, K) driver-owned counter


# --- D1 contact richness (teacher) — counterpart-scoped (A5 contact_force_with). Richer than boolean. ---
def hand_object_force_magnitude(env, bodies: list[str], target: str = "object") -> torch.Tensor:
    """Per-fingertip **object(target)** contact force magnitude → (N, K) [N]. Continuous version of the boolean contact (fingertip_contact_obs).
    counterpart-scoped so table contacts aren't mixed in. (perceptive hand_object_force_magnitude.)"""
    return env.contact_force_with(bodies, target).norm(dim=-1)   # (N, K)


def joint_pose_error(env, joint_pose: dict, tolerance_key: str | None = None) -> torch.Tensor:
    """Distance from the target POSTURE, beside the bar it is judged against → (N, 1) or (N, 2) [rad].

        err       = max_j |q_j - joint_pose[j]|
        returns     [err]  — or  [err, curriculum_values[tolerance_key]]  when that key is given

        joint_pose     {joint: rad}  the target posture (pass GATE's own, via the reward term's variable)
        tolerance_key  the curriculum log key holding the CURRENT bound (None = value only, 1 dim)

    Both dims in ONE term because the pair is what carries the meaning: the bar moves every level, so the
    error alone cannot say whether the posture condition is met. Their crossing IS the condition."""
    q = env.joint_pos(list(joint_pose))
    err = (q - frames.const(tuple(joint_pose.values()), q)).abs().amax(dim=-1)
    return _against_bound(err, env, tolerance_key, "joint_pose_error")


def _against_bound(value: torch.Tensor, env, bound_key: str | None, who: str) -> torch.Tensor:
    """(N,) live value → (N, 1), or (N, 2) with the curriculum's CURRENT bound beside it.

    The pair in ONE term because that is what carries the meaning: every one of these bars moves per level,
    so a value alone cannot say whether its condition is met — the crossing IS the condition, and on the
    dashboard it reads directly as two curves in one panel. Asking for a key the curriculum does not publish
    fails loud rather than plotting a silent zero (a ramp that is switched off publishes nothing)."""
    v = value.unsqueeze(-1)
    if bound_key is None:
        return v
    vals = env.curriculum_values
    assert bound_key in vals, (
        f"{who}: the curriculum publishes no {bound_key!r} — it reports {sorted(vals)}. "
        f"(A ramp that is off publishes nothing.)")
    return torch.cat([v, vals[bound_key].reshape(1, 1).expand(v.shape[0], 1)], dim=-1)


def goal_dist_error(env, tolerance_key: str | None = None) -> torch.Tensor:
    """Object↔goal keypoint distance, beside the bar it is judged against → (N, 1) or (N, 2) [m].

    THE goal metric (``api.keypoints.object_goal_dist``) — the same number the gate predicate and the goal
    reward measure, so the panel shows how far the object is from counting as delivered, against a tolerance
    that tightens every level."""
    d = keypoints.object_goal_dist(env.object_pos(), env.object_quat(),
                                   env.goal_pos, env.goal_quat, env.goal_half_extent)
    return _against_bound(d, env, tolerance_key, "goal_dist_error")


def palm_distance_error(env, palm_body: str, tolerance_key: str | None = None) -> torch.Tensor:
    """Palm ↔ object grasp point distance, beside its bar → (N, 1) or (N, 2) [m]. POSITION only, no
    orientation — the same test the gate makes to refuse a delivered-but-released object."""
    grasp = env.object_pos()
    d = (env.body_pos(palm_body) - grasp).norm(dim=-1)
    return _against_bound(d, env, tolerance_key, "palm_distance_error")


def grip_count(env, force_threshold: float = 0.1, required_key: str | None = None) -> torch.Tensor:
    """Fingertips gripping the object RIGHT NOW, beside the count required → (N, 1|2).

        returns  #{tips whose PAD presses the object} — or that beside curriculum_values[required_key]

        force_threshold  0.1 [N]  pad-normal force that counts as gripping
        required_key              the curriculum log key holding the CURRENT bar (None = value only, 1 dim)

    The GATE's own grip test (:func:`sim.metalab.api.contact.contact_mask`, counterpart-scoped to the
    object), summed instead of thresholded — so this reads as how far the grip condition has come, where
    ``gate_success`` only says whether every condition together happened to hold.

    Like the other bars, the pair carries the meaning: the required count climbs each level, so the live
    count alone cannot say whether the condition is met."""
    tips = env.fingertips
    assert tips, "grip_count needs the robot's fingertip bodies, which it declares none of"
    grip = contact_mask(env.contact_force_with(tips, "object"), force_threshold)
    return _against_bound(grip.sum(dim=1).float(), env, required_key, "grip_count")


def curriculum_hold_progress(env, required_key: str | None = None) -> torch.Tensor:
    """Consecutive steps the success conditions have held, beside the count required → (N, 1|2) [steps].

    Counted at the CURRICULUM's bars, by the driver's own counter (``_advance_curriculum``) — the one that
    ends the episode and pays the bonus, so numerator and denominator move together. At the GATE's final bars
    this would read 0 for the whole curriculum and tell the policy nothing.

    The one pairing here that runs the other way: this value must climb ABOVE its companion, where every other
    bound is a ceiling — hence the labels ``held``/``required`` rather than error/tolerance."""
    return _against_bound(env.curriculum_hold.float(), env, required_key, "curriculum_hold_progress")


def fingertip_penetration_depth(env, bodies: list[str], target: str = "object") -> torch.Tensor:
    """Per-fingertip **overlap depth** into ``target`` → (N, K) [mm]. 0 = not overlapping.

    How far that tip's collision hull has sunk INTO the counterpart at its worst point — the interpenetration
    the contact solver is currently working off. Counterpart-scoped like the force reads, so leaning on the
    table and squeezing the object stay separate channels.

    Worth a channel of its own beside the force: the force says how hard the constraint pushes back, this
    says how far the surfaces had to overlap to make it do that, and the two come apart exactly where a grasp
    goes wrong (soft contact params, a heavy object, a spike the solver has not caught up with).

    **mm, not the pipeline's SI metres** — the one term that converts, because a millimetre IS this quantity's
    natural size and the runner cannot rescale it back: obs normalization is ``(x - mean) / (std + 1e-2)``, and
    a metre-valued overlap has std ~1e-3, so the eps floor sits 10× above the signal and the channel reaches
    the critic squashed flat. In mm the std is ~1 and the normalizer behaves. The read underneath stays SI
    (api/contact.py ``contact_penetration`` [m]); the conversion is here so the unit lives in ONE place, and
    the contract says ``unit="mm"`` with no scale.

    CRITIC ONLY — overlap is a solver artifact, not something a real hand can sense.

    Reads overlap, never separation: a tip 1 mm short of the object reads the same 0 as a tip across the
    table, because the engines' contact list IS the detection envelope (api/contact.py contact_penetration)."""
    return env.contact_penetration(bodies, target) * 1000.0   # (N, K) m → mm


def hand_contact_force(env, bodies: list[str], target: str = "object",
                       ref_body: str | None = None) -> torch.Tensor:
    """Per-fingertip **object(target)** contact force VECTOR → (N, K*3), counterpart-scoped. The direction
    is the grasp geometry — which way each finger is being pushed — that the magnitude alone throws away.

    ``ref_body`` picks the frame the vector is expressed in. Force is a free vector, so every choice is a
    ROTATION only, no translation:

    * ``"self"`` → EACH tip's OWN frame, i.e. column k is rotated by ``bodies[k]``'s quaternion.
      **Prefer this.** A contact is a fact about that fingertip's surface: the fingerprint face has a fixed
      direction in the tip's own frame, so "pressed on the pad" is one sign in one component no matter how
      the finger is flexed. Any shared frame mixes the finger's own joint angles into the reading — measured
      on this hand, a tip's local +x sits at ``+0.995`` of the palm's +x extended and at ``-0.872`` fully
      curled, so the same press flips sign as the finger closes.
    * a body name (e.g. the palm) → one shared frame for every tip.
    * ``None`` → world frame (the original behaviour, kept as the default).

    Redundant with :func:`hand_object_force_magnitude` by construction (mag = ‖vec‖ of this same read) —
    the magnitude is kept where both are declared because it is the rotation-invariant part, so the network
    gets "how hard" without having to compute a norm."""
    f = env.contact_force_with(bodies, target)       # (N, K, 3) world
    if ref_body == "self":
        q = torch.stack([env.body_quat(b) for b in bodies], dim=1)      # (N,K,4) per-tip
        f = frames.quat_rotate_inverse(q, f)         # world → each tip's own frame
    elif ref_body is not None:
        q = env.body_quat(ref_body).unsqueeze(1).expand(-1, f.shape[1], -1)
        f = frames.quat_rotate_inverse(q, f)         # world → ref_body frame
    return f.reshape(f.shape[0], -1)


def episode_step(env) -> torch.Tensor:
    """How far into the episode, as a fraction of the horizon → (N, 1) [0,1]."""
    return (env.episode_length_buf.float() / max(1, env.max_episode_length)).unsqueeze(-1)


def object_lifed(env, lift_threshold: float) -> torch.Tensor:
    """1 if the object is LIFTED → (N, 1) float {0,1}.

    Measured on the object's ORIGIN, which the asset pipeline puts on the grasp point — the same point the
    goal metric and the palm distance use, so one threshold means the same physical clearance everywhere.

    Absolute, so the number encodes the table: pick it a few mm above where the object RESTS. The
    spawn-relative alternative is :func:`object_lifed` (rise since this episode's own spawn)."""
    grasp_z = env.object_pos()[:, 2]
    return (grasp_z >= lift_threshold).float().unsqueeze(-1)


def instantaneous_reward(env) -> torch.Tensor:
    """This step's reward, TOTAL then per term → (N, 1 + len(REWARD)).

    Column 0 is the sum the trainer receives; the rest are the terms that add up to it, in the contract's
    REWARD order, each already weighted. So the critic reads WHICH term paid, not only how much — the split a
    dashboard would otherwise have to be opened to see."""
    return torch.cat([env.last_reward.unsqueeze(-1), env.last_reward_terms], dim=-1)


def object_linear_velocity(env) -> torch.Tensor:
    """Object world linear velocity → (N, 3)."""
    return env.object_lin_vel()


def object_angular_velocity(env) -> torch.Tensor:
    """Object world angular velocity → (N, 3)."""
    return env.object_ang_vel()


def body_linear_velocity(env, body: str) -> torch.Tensor:
    """Body world linear velocity → (N, 3). (For palm velocity.)"""
    return env.body_lin_vel(body)


def body_angular_velocity(env, body: str) -> torch.Tensor:
    """Body world angular velocity → (N, 3). (For palm velocity.)"""
    return env.body_ang_vel(body)


# ---------------------------------------------------------------------------
# G4 progress/goal obs (teacher) — chest-origin frame (robot-translation invariant; perceptive convention, positions so no quat).
# progress obs using keypoint cages (object·goal, shared goal half-extent), fingertip relative pose/vel + EnvDriver
# extensions (goal·latched lift·decoded target·reward best-so-far). Source: perceptive_dexdeepmimic/mdp/observations.py.
# ---------------------------------------------------------------------------
_BEST_DIST_CLAMP = 2.0   # clamp closest_* best-so-far initial inf/huge values [m] (obs normalization stability)


def _points_to_chest(pts_w, chest_pos, chest_quat, offset):
    """World points (N,K,3) → chest-origin frame (N,K,3). (Shared by keypoints/fingertip positions.)"""
    op, oq = kinematics.chest_origin_pose(chest_pos, chest_quat, offset)   # (N,3),(N,4)
    oq = oq.unsqueeze(1).expand(-1, pts_w.shape[1], -1)                    # (N,K,4)
    return frames.points_to_frame(pts_w, op.unsqueeze(1), oq)             # (N,K,3)


def object_keypoints(env, chest_body: str, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Object keypoint cage 4 corners → chest-origin (N, 12). cage half-extent = **same as goal** (env.goal_half_extent),
    so corners map 1:1 to goal_keypoints = raw material for keypoint_max_dist metric (same extent as reward
    object_goal_keypoint_progress). Unusable in goal-less contracts (teacher goal-task only). (perceptive object_keypoints.)"""
    he = env.goal_half_extent
    assert he is not None, "object_keypoints requires goal.keypoint_half_extent (not a goal contract)"

    # the obs copy and the distance the reward/GATE measure must be built around the same point.
    op = env.object_pos()
    kp = keypoints.cage(op, env.object_quat(), he)                          # (N,4,3) world
    rel = _points_to_chest(kp, env.body_pos(chest_body), env.body_quat(chest_body), offset)
    return rel.reshape(rel.shape[0], -1)


def goal_keypoints(env, chest_body: str, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """**Fixed goal** keypoint cage 4 corners → chest-origin (N, 12). Same half-extent as object_keypoints (goal-owned),
    so corners correspond. Fixed goal → same value every step, but chest-relative so it responds to robot motion
    (teacher perceives the goal in body frame). (perceptive goal_keypoints.)"""
    gp, gq, he = env.goal_pos, env.goal_quat, env.goal_half_extent
    assert gp is not None, "goal_keypoints requires a fixed goal (not a goal contract)"
    kp = keypoints.cage(gp, gq, he)                                         # (N,4,3) world
    rel = _points_to_chest(kp, env.body_pos(chest_body), env.body_quat(chest_body), offset)
    return rel.reshape(rel.shape[0], -1)


def object_goal_keypoint_success(env, tolerance: float = 0.01) -> torch.Tensor:
    """1 if the object has reached the goal pose → (N, 1) {0,1}. Success = **max** over the 4 cage corners
    of the object↔goal distance ≤ ``tolerance`` [m] (keypoint_max_dist, position+orientation fused — same
    metric as the object_goal_keypoint rewards). **Fixed at the true 1cm tolerance and curriculum-independent**
    (unlike reward/terminate which follow the curriculum's loosening tolerance), so it always reflects the real
    success. Goal-task only; frame-invariant.

    Distance ONLY — no grip and no palm condition. For the full success predicate (the one the GATE and the
    reach bonus share) use :func:`gate_success` / :func:`curriculum_success`."""
    d = keypoints.object_goal_dist(env.object_pos(), env.object_quat(),
                                   env.goal_pos, env.goal_quat, env.goal_half_extent)
    return (d <= tolerance).float().unsqueeze(-1)


def curriculum_state(env, keys: list[str]) -> torch.Tensor:
    """The curriculum's CURRENT difficulty, broadcast to every env → (N, len(keys)), each value in the unit
    the curriculum reports it in (``tolerance`` [m], ``palm_distance`` [m], ``gravity`` [m/s²], the rest
    counts/levels/flags).

    WHY the critic needs it: the curriculum retunes the reward terms and the success bar as the run
    progresses, so the value function's own definition moves under the critic — the same state is worth a
    1000-point bonus at level 0's 15 cm tolerance and nothing at level 30's 1 cm. Without the difficulty in
    the input the critic can only learn a blur across every level it has seen. Privileged by nature: the
    robot has no way to know which curriculum level it is being trained at.

    Population-wide (identical in every env) — difficulty is a property of the run, not of an env.

    ``keys`` are the curriculum term's OWN log keys, i.e. what shows up under ``Curriculum/*``; asking
    for one the curriculum does not publish fails loud instead of silently reading zero."""
    vals = env.curriculum_values
    missing = [k for k in keys if k not in vals]
    assert not missing, (
        f"curriculum_state asks for {missing}, which this task's curriculum does not publish — it "
        f"reports {sorted(vals)}. (A ramp that is off publishes nothing.)")
    return torch.stack([vals[k] for k in keys]).unsqueeze(0).expand(env.num_envs, len(keys))


def dr_params(env, keys: list[str]) -> torch.Tensor:
    """This episode's DOMAIN-RANDOMIZATION draws, per env → (N, len(keys)), each in its own event's unit.

        keys   the DR channels this contract's events publish — ``"<target>_friction"`` (mu),
               ``"object_mass_scale"`` / ``"object_scale"`` (x the asset default), ``"root_height"`` [m]

    WHY the critic needs it: each of these is drawn at reset, held for the whole episode, and decides whether
    the SAME hand pose holds the object or drops it. Two envs whose observation is identical therefore earn
    different returns, and a value function without the draw can only learn the average over it — the
    residual reaches the advantage as noise, which a sparse one-shot success bonus can least afford. It is
    the argument ``action_delay`` already makes for its own draw; these are the rest of the same set.

    Not recoverable from a single frame either: friction and mass show up only in how the contact responds
    over several steps, and the critic's history length is 1.

    CRITIC ONLY — the real robot is never told what it was handed, so an actor reading this does not deploy.

    A channel no event publishes fails loud (:meth:`EnvDriver.dr_value`), so a DR that is off — or one
    dropped for want of an engine capability — cannot read as a live draw."""
    return torch.stack([env.dr_value(k) for k in keys], dim=-1)


def object_variant(env) -> torch.Tensor:
    """Which asset VARIANT this env spawned, ONE-HOT → (N, V).

    One-hot, not the index: the variants are a CATEGORY (four differently shaped hammers), and an integer
    channel would tell the network that ``cylinder`` lies between ``ycb`` and ``edge``.

    Same argument as :func:`dr_params` — the shape is drawn per env, held for the whole run, and decides
    whether a grip holds — except that it is fixed at BUILD time rather than at reset, so no event publishes
    it and the spoke that laid the worlds out is asked instead (``SimBackend`` ``object_variant`` capability).

    CRITIC ONLY. A single-asset scene fails loud rather than emitting a constant column."""
    v = env.object_variant_count()
    assert v > 1, f"object_variant: this scene builds the object from {v} variant(s) — nothing to observe"
    return torch.nn.functional.one_hot(env.object_variant_id(), num_classes=v).float()


def gate_success(env) -> torch.Tensor:
    """1 while the task is solved AT THE GATE → (N, 1) {0,1}. The GATE bar is absolute and
    curriculum-independent (goal tolerance + palm on the object + grip count, all from the contract's GATE),
    so this is the same bar ``val/SR`` reports and the two cannot drift apart.

    INSTANTANEOUS — the gate's consecutive-hold requirement is deliberately NOT applied: the hold is the
    gate's own latch, and the episode ends the step it fires, so a latched channel would be 1 on the final
    frame and 0 everywhere else. Read straight off ``_advance_gate``'s predicate (no recompute). Gate-task only."""
    return env.gate_near.float().unsqueeze(-1)


def curriculum_success(env) -> torch.Tensor:
    """1 while the task is solved AT THE CURRENT CURRICULUM BAR → (N, 1) {0,1}.

    The SAME predicate as :func:`gate_success`, at the thresholds the curriculum has reached so far — loose
    tolerance, fewer fingers, looser palm early on — i.e. the bar that promotes a level and ends the episode.
    Early in a run this is 1 long before ``gate_success`` ever is, and the gap between the two is how far the
    curriculum still has to go; the two meet only if the ramp's stated ends match the GATE.

    Latched by ``EnvDriver._advance_curriculum``; a contract with no curriculum term is judged at the GATE's
    own bars, so it then reads the same as ``gate_success``."""
    return env.curriculum_near.float().unsqueeze(-1)


def fingertip_relative_pos(env, chest_body: str, bodies: list[str], offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Fingertip body positions → chest-origin (N, K*3), bodies order (Index→Thumb). Provides precise grasp reference
    positions directly (otherwise policy must implicitly learn finger FK from joint_pos). (perceptive fingertip_relative_pos.)"""
    tips = torch.stack([env.body_pos(b) for b in bodies], dim=1)    # (N,K,3) world
    rel = _points_to_chest(tips, env.body_pos(chest_body), env.body_quat(chest_body), offset)
    return rel.reshape(rel.shape[0], -1)


def fingertip_relative_pose(env, ref_body: str, bodies: list[str], offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Fingertip body **poses** relative to a reference-body frame (per tip: pos3 + quat wxyz4) → (N, K*7),
    bodies order (Index→Thumb). Orientation-aware counterpart of ``fingertip_relative_pos`` (positions only) —
    the critic sees each fingertip's full pose (e.g. in the palm frame). (asymmetric student critic obs.)"""
    cp, cq = env.body_pos(ref_body), env.body_quat(ref_body)
    outs = []
    for b in bodies:
        rp, rq = kinematics.to_chest(env.body_pos(b), env.body_quat(b), cp, cq, offset)
        outs.append(torch.cat([rp, rq], dim=-1))                        # (N,7)
    return torch.cat(outs, dim=-1)                                      # (N, K*7)


def fingertip_relative_vel(env, chest_body: str, bodies: list[str]) -> torch.Tensor:
    """Fingertip world linear velocity → rotated into chest-origin orientation (N, K*3), bodies order. Explicit
    first derivative of position (no history needed). chest is torso-locked → ignore frame translation, rotate orientation only (legacy fingertip_vel convention)."""
    v = torch.stack([env.body_lin_vel(b) for b in bodies], dim=1)   # (N,K,3) world
    cq = env.body_quat(chest_body).unsqueeze(1).expand(-1, v.shape[1], -1)
    return frames.quat_rotate_inverse(cq, v).reshape(v.shape[0], -1)


def prev_action_targets(env) -> torch.Tensor:
    """Previous step's decoded joint targets (post-EMA·clamp, pre-delay) → (N, num_actions). Unlike raw last_action,
    the **actual commanded value** with EMA/limit applied — exposed so policy can recover EMA state (Markov). (perceptive prev_action_targets.)"""
    return env.prev_action_targets


def action_delay(env) -> torch.Tensor:
    """This episode's command delay [policy steps] → (N, 1). ONE value per env, not per action group: the
    delay is the controller→robot link, so every group is written with the same lag.

    Resampled from the contract's ``min_delay``/``max_delay`` at each reset, which is what makes it worth
    giving the critic — the same policy step means something different depending on how stale the command
    the robot is acting on is, and that draw is otherwise unobservable. 0 when the DR is off.

    Pairs with ``prev_action_targets``, which is the PRE-delay value: overlay the two against joint_pos and
    this term says how many steps of the gap between them are transport lag rather than PD tracking."""
    lag = env.action_delay_lag
    if lag is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return lag.float().unsqueeze(-1)


def closest_keypoint_max_dist(env, reward_term: str = "object_goal_keypoint_progress",
                              clamp: float = _BEST_DIST_CLAMP) -> torch.Tensor:
    """**Episode best-so-far** of object↔goal keypoint_max_dist → (N, 1) [m]. Exposes the progress reward term's hidden
    state (_best) — so policy perceives "how close it got" (progress baseline). inf (initial) is clamped. (perceptive.)"""
    return env.reward_best(reward_term).clamp(max=clamp).unsqueeze(-1)


def closest_fingertip_dist(env, reward_term: str = "fingertip_object_distance_delta",
                           clamp: float = _BEST_DIST_CLAMP) -> torch.Tensor:
    """Per-finger **episode best-so-far** of fingertip↔object grasp point distance → (N, K) [m]. Exposes the fingertip
    distance progress reward term's hidden state (_best) (shares reward grasp point definition). inf is clamped. (perceptive.)"""
    return env.reward_best(reward_term).clamp(max=clamp)
