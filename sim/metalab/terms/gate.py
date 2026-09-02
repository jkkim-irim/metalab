from __future__ import annotations

import torch

from sim.metalab.api import transforms


def cage(env, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:   # [m]
    he = env.goal_half_extent
    assert he is not None, "a keypoint cage needs a fixed goal with keypoint_half_extent (contract `goal` block)"
    return transforms.body_points(pos, quat, transforms.TETRA_CORNERS.to(pos) * transforms.const(he, pos))


def object_goal_dist(env) -> torch.Tensor:   # [m]
    d = cage(env, env.object_pos(), env.object_quat()) - cage(env, env.goal_pos, env.goal_quat)
    return torch.linalg.norm(d, dim=-1).max(dim=-1).values


def object_at_goal(env, goal_dist_tol: float,   # [m]
                   palm_distance: float = 0.0,   # [m]
                   contact_count: int = 0,
                   contact_fingers: tuple[str, ...] = (),
                   force_threshold: float = 1.0e-3,   # [N]
                   joint_pose: dict | None = None,   # {joint: rad}
                   joint_pose_tolerance: float = 0.0) -> torch.Tensor:   # [rad]
    at = object_goal_dist(env) <= goal_dist_tol
    if palm_distance > 0.0:
        assert env.palm_body is not None, \
            "a palm-distance condition needs a robot frame named 'palm'"
        at = at & ((env.body_pos(env.palm_body) - env.object_pos()).norm(dim=-1) <= palm_distance)
    if contact_count > 0 or contact_fingers:
        tips = env.fingertips
        assert tips, ("a grip condition (contact_count > 0 / contact_fingers) needs the robot's fingertip "
                      "bodies, which it declares none of")
        grip = env.contact_force_with(tips, "object").norm(dim=-1) > force_threshold
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
        q = env.joint_pos(names)
        err = (q - transforms.const(tuple(joint_pose[j] for j in names), q)).abs().amax(dim=-1)
        at = at & (err <= joint_pose_tolerance)
    return at
