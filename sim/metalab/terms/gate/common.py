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


def body_goal_dist(env, body: str) -> torch.Tensor:   # [m]
    return torch.linalg.norm(env.body_pos(body) - env.goal_pos, dim=-1)


def object_at_goal(env, goal_dist_tol: float,   # [m]
                   palm_distance: float = 0.0,   # [m]
                   contact_count: int = 0,
                   contact_fingers: tuple[str, ...] = (),
                   force_threshold: float = 1.0e-3,   # [N]
                   joint_pose: dict | None = None,   # {joint: rad}
                   joint_pose_tolerance: float = 0.0) -> torch.Tensor:   # [rad]
    at = object_goal_dist(env) <= goal_dist_tol
    if palm_distance > 0.0:
        palm = env.spec.robot.frames.get("palm")
        assert palm is not None, (
            f"a palm-distance condition needs a robot frame named 'palm' — declared frames: {sorted(env.spec.robot.frames)}")
        at = at & ((env.body_pos(palm) - env.object_pos()).norm(dim=-1) <= palm_distance)
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


def body_at_goal(env, goal_dist_tol: float,   # [m]
                 palm_distance: float = 0.0,
                 contact_count: int = 0,
                 contact_fingers: tuple[str, ...] = (),
                 force_threshold: float = 1.0e-3,
                 joint_pose: dict | None = None,
                 joint_pose_tolerance: float = 0.0) -> torch.Tensor:
    palm = env.spec.robot.frames.get("palm")
    assert palm is not None, (
        f"body_at_goal needs a robot frame named 'palm' (the body that must reach the goal) — declared frames: "
        f"{sorted(env.spec.robot.frames)}")
    assert palm_distance == 0.0 and contact_count == 0 and not contact_fingers and joint_pose_tolerance == 0.0, (
        "body_at_goal judges the palm position only — GATE.palm_distance / contact_count / contact_fingers / "
        "joint_pose_tolerance are object-grasp conditions it does not evaluate")
    return body_goal_dist(env, palm) <= goal_dist_tol


def hold_count(count: torch.Tensor, ok: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cumulative":
        return count + ok.to(count.dtype)
    assert mode == "consecutive", f"hold_count: unknown mode {mode!r} (consecutive | cumulative)"
    return torch.where(ok, count + 1, torch.zeros_like(count))


def hold_and_pass(env, predicate, bars: dict, hold_steps: int,
                  hold_mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    near = predicate(env, **bars)
    hold = env.buffer("hold", dtype=torch.long)
    passed = env.buffer("passed", dtype=torch.bool)
    hold.copy_(hold_count(hold, near, hold_mode))
    passed |= hold >= hold_steps
    return near, passed, hold
