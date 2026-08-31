"""Gate predicates — what counts as SOLVED, as a flat term.

    def <predicate>(env, <knob>=<default>, ...) -> (N,) bool

The contract names one in its ``GATE`` block (``predicate = gate.object_at_goal``) and the driver calls it
once per policy step at each of the two bars it judges — the GATE's own (``val/SR``) and the curriculum
level's (which promotes, ends the episode and pays the success bonus) — then owns only the counting: the
hold counters and the episode latches. So the JUDGMENT lives here, beside the reward and termination
libraries that judge the same world, while the STATE stays where a step-exactly-once guarantee can be made.

ONE implementation at both bars, which is what keeps "solved at this level" and "solved" the same question
asked with different numbers rather than two blocks of code that have to agree.

``env`` (EnvDriver) gives backend reads plus the fixed goal (``goal_pos``/``goal_quat``/
``goal_half_extent``) and the robot facts a grip test needs (``fingertips``, ``palm_body``) — those are
properties of the hand and the scene, not per-caller knobs.
"""
from __future__ import annotations

import torch

from sim.metalab.api import keypoints, kinematics
from sim.metalab.api.contact import contact_mask


def object_at_goal(env, goal_dist_tol: float,
                   palm_distance: float = 0.0, contact_count: int = 0,
                   contact_fingers: tuple[str, ...] = (),
                   force_threshold: float = 1.0e-3, joint_pose: dict | None = None,
                   joint_pose_tolerance: float = 0.0) -> torch.Tensor:
    """Is the object AT the goal, held, right now? → (N,) bool.

        at = keypoint_max_dist(object, goal) <= goal_dist_tol
             AND ||palm - grasp point|| <= palm_distance          if palm_distance > 0  (position only)
             AND #{fingertips pressing the object} >= contact_count                  if contact_count > 0
             AND EVERY tip in contact_fingers pressing it                            if contact_fingers
             AND worst |q_j - joint_pose[j]| <= joint_pose_tolerance if that goal_dist_tol > 0 [rad]

        goal_dist_tol            [m]    keypoint cage distance to the goal — the only condition always applied
        palm_distance        [m]    palm ↔ grasp point bound (0 = off)
        contact_count               fingertips that must be gripping (0 = off)
        contact_fingers  (body names) the NAMED tips that must EACH be gripping (() = off)
        force_threshold      [N]    per tip: contact force above this counts as gripping
        joint_pose     {joint: rad} the target posture (0-length unless the bound below is set)
        joint_pose_tolerance [rad]  worst-joint bound to that posture (0 = off)

Measured at the object's ORIGIN, which the asset pipeline puts on the grasp point, so every bar here and
    every height threshold elsewhere share one reference. The cage corner distance fuses position and
    orientation error into one metre-valued number — a pure yaw error still shows as corner displacement — so
    no separate rotation condition is needed.

    The grip test is :func:`sim.metalab.api.contact.contact_mask`, the same call the
    ``fingertip_object_contact`` reward makes, and it is counterpart-scoped to the object: a fingertip resting
    on the TABLE is contact too, and on a table-top task the approach keeps the fingers there.

    No lift condition, on purpose: at a tight goal_dist_tol being at the goal already implies being off the table
    by a wide margin (hammer_lift measured: kp <= 0.01 needs z >= 0.99, i.e. 13.5x the 0.01 m lift height).
    ``GATE.lift_height`` drives the separate ``lifted`` PHASE gate, where the curriculum's loose early
    goal_dist_tol makes it load-bearing. A task whose goal sits near the table surface breaks that implication and
    has to add the term here."""
    assert env.goal_half_extent is not None, \
        "a goal gate needs a fixed goal with keypoint_half_extent (contract `goal` block)"
    at = keypoints.object_goal_dist(env.object_pos(), env.object_quat(),
                                    env.goal_pos, env.goal_quat, env.goal_half_extent) <= goal_dist_tol
    if palm_distance > 0.0:                              # the hand is still ON it (position only, no quat)
        assert env.palm_body is not None, \
            "a palm-distance condition needs a robot frame named 'palm'"
        at = at & ((env.body_pos(env.palm_body) - env.object_pos()).norm(dim=-1) <= palm_distance)
    if contact_count > 0 or contact_fingers:
        tips = env.fingertips
        assert tips, ("a grip condition (contact_count > 0 / contact_fingers) needs the robot's fingertip "
                      "bodies, which it declares none of")
        grip = contact_mask(env.contact_force_with(tips, "object"), force_threshold)
        if contact_count > 0:
            at = at & (grip.sum(dim=1) >= contact_count)
        if contact_fingers:
            unknown = [b for b in contact_fingers if b not in tips]
            assert not unknown, \
                f"contact_fingers names {unknown}, which the robot does not declare as fingertips: {tips}"
            at = at & grip[:, [tips.index(b) for b in contact_fingers]].all(dim=1)
    if joint_pose_tolerance > 0.0:
        assert joint_pose, "a joint-pose condition needs the posture too (joint_pose={joint: angle} [rad])"
        names = list(joint_pose)
        err = kinematics.joint_pose_error(env.joint_pos(names), [joint_pose[j] for j in names])
        at = at & (err <= joint_pose_tolerance)
    return at
