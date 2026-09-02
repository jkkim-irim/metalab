from __future__ import annotations

import torch

from sim.metalab.api import transforms
from sim.metalab.terms.gate import object_goal_dist


def _exp_kernel(err: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-err / sigma)


def _min_dist_progress_(best: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    progress = torch.where(torch.isinf(best), torch.zeros_like(cur), (best - cur).clamp_min(0.0))
    torch.minimum(best, cur, out=best)
    return progress


def lifting_reward(env, lift_height: float = 0.1) -> torch.Tensor:   # [m]
    assert lift_height > 0.0, f"lifting_reward: lift_height must be > 0 (it normalizes the ratio) — got {lift_height}"
    return ((env.object_pos()[:, 2] - env.object_init_z) / lift_height).clamp(0.0, 1.0)


def palm_object_proximity(env, palm_body: str, std: float = 0.05,   # [m]
                          palm_offset=(0.0, 0.0, 0.0)) -> torch.Tensor:   # [m]
    palm = transforms.local_point(env.body_pos(palm_body), env.body_quat(palm_body), palm_offset)
    grasp = env.object_pos()
    return _exp_kernel(torch.linalg.norm(palm - grasp, dim=-1), std)


def fingertip_object_proximity(env, std: float = 0.05) -> torch.Tensor:   # [m]
    tips = torch.stack([env.body_pos(b) for b in env.fingertips], dim=1)
    grasp = env.object_pos()
    d = (tips - grasp.unsqueeze(1)).norm(dim=-1)
    return _exp_kernel(d, std).mean(dim=-1)


def object_goal_keypoint_progress(env, lift_threshold: float = 0.0) -> torch.Tensor:   # [m]
    best = env.buffer("best", fill=float("inf"))
    d0 = env.buffer("d0")   # [m]
    cur = object_goal_dist(env)   # [m]
    d0[:] = torch.where(torch.isinf(best), cur, d0)
    r = _min_dist_progress_(best, cur) / d0.clamp(min=1.0e-6)
    if lift_threshold > 0.0:
        r = r * (env.object_pos()[:, 2] >= lift_threshold).float()
    return r


def object_goal_keypoint_tracking(env, std: float = 0.1, lift_threshold: float = 0.0,   # [m]
                                  lift_full: float = 0.0) -> torch.Tensor:   # [m]
    assert lift_full <= 0.0 or lift_full > lift_threshold, (
        f"lift_full={lift_full} must be above lift_threshold={lift_threshold} — it is the top of the ramp "
        f"the height factor climbs over (0 = no ramp)")
    r = _exp_kernel(object_goal_dist(env), std)
    if lift_threshold > 0.0:
        z = env.object_pos()[:, 2]
        r = r * (((z - lift_threshold) / (lift_full - lift_threshold)).clamp(0.0, 1.0) if lift_full > 0.0
                 else (z > lift_threshold).float())
    return r


def joint_pose_convergence(env, joint_pose: dict,   # {joint: rad}
                           std: float,   # [rad]
                           lift_threshold: float = 0.0) -> torch.Tensor:   # [m]
    q = env.joint_pos(list(joint_pose))
    err = (q - transforms.const(tuple(joint_pose.values()), q)).abs().mean(dim=-1)   # [rad]
    r = _exp_kernel(err, std)
    if lift_threshold > 0.0:
        grasp_z = env.object_pos()[:, 2]
        r = r * (grasp_z >= lift_threshold).float()
    return r


def object_goal_reach_bonus(env) -> torch.Tensor:
    paid = env.buffer("paid", dtype=torch.bool)
    newly = env.curriculum_passed & ~paid
    paid |= env.curriculum_passed
    return newly.float()


def joint_torque_penalty(env, names: list[str]) -> torch.Tensor:
    tau = env.joint_torque(names)
    return torch.sum(tau * tau, dim=-1)   # [(N*m)^2]


def joint_vel_l1(env, names: list[str]) -> torch.Tensor:
    return env.joint_vel(names).abs().sum(dim=-1)   # [rad/s]


def action_rate_l2(env) -> torch.Tensor:
    prev = env.buffer("prev", shape=(env.num_actions,))
    seen = env.buffer("seen", dtype=torch.bool)
    a = env.last_action
    r = ((a - prev) ** 2).sum(dim=-1) * seen
    prev[:] = a
    seen[:] = True
    return r


def fingertip_object_contact(env, target: str = "object",
                             force_threshold: float = 0.1,   # [N]
                             lift_threshold: float = 0.0) -> torch.Tensor:   # [m]
    f = env.contact_force_with(env.fingertips, target)
    r = (f.norm(dim=-1) > force_threshold).float().mean(dim=-1)
    if lift_threshold > 0.0:
        r = r * (env.object_pos()[:, 2] >= lift_threshold).float()
    return r


def nail_object_contact(env, bodies: list[str], target: str = "object",
                        force_threshold: float = 0.1) -> torch.Tensor:   # [N]
    f = env.contact_force_with(bodies, target)
    return (f.norm(dim=-1) > force_threshold).float().mean(dim=-1)


def fingertip_object_pinch_contact(env, fingers: list[str], target: str = "object",
                                   force_threshold: float = 0.1,   # [N]
                                   lift_max: float = 0.0) -> torch.Tensor:   # [m]
    f = env.contact_force_with(fingers, target)
    r = (f.norm(dim=-1) > force_threshold).float().mean(dim=-1)
    if lift_max > 0.0:
        r = r * (env.object_pos()[:, 2] < lift_max).float()
    return r
