from __future__ import annotations

import torch

from sim.metalab.api import transforms
from sim.metalab.terms.gate import cage, object_goal_dist


def joint_state(env, names: list[str]) -> torch.Tensor:
    return torch.cat([env.joint_pos(names), env.joint_vel(names)], dim=-1)


def object_state_world(env) -> torch.Tensor:
    return torch.cat([env.object_pos(), env.object_quat()], dim=-1)


def object_pose(env, chest_body: str, offset=(0.0, 0.0, 0.0), seen: bool = False) -> torch.Tensor:
    p, q = ((env.object_seen_pose_w[:, :3], env.object_seen_pose_w[:, 3:7]) if seen
            else (env.object_pos(), env.object_quat()))
    cp, cq = env.body_pos(chest_body), env.body_quat(chest_body)
    rp, rq = transforms.to_frame(p, q, transforms.local_point(cp, cq, offset), cq)
    return torch.cat([rp, rq], dim=-1)


def body_pose_in_chest(env, chest_body: str, target_body: str, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    cp, cq = env.body_pos(chest_body), env.body_quat(chest_body)
    rp, rq = transforms.to_frame(env.body_pos(target_body), env.body_quat(target_body), transforms.local_point(cp, cq, offset), cq)
    return torch.cat([rp, rq], dim=-1)


_PALM_TO_REAL = torch.tensor([0.0, 1.0, 0.0, 0.0])


def palm_pose_in_chest(env, chest_body: str, palm_body: str, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    cp, cq = env.body_pos(chest_body), env.body_quat(chest_body)
    rp, rq = transforms.to_frame(env.body_pos(palm_body), env.body_quat(palm_body), transforms.local_point(cp, cq, offset), cq)
    rq = transforms.quat_mul(rq, _PALM_TO_REAL.to(rq))
    return torch.cat([rp, rq], dim=-1)


def last_action(env) -> torch.Tensor:
    return env.last_action


def joint_positions(env, names: list[str]) -> torch.Tensor:
    return env.joint_pos(names)


def joint_velocities(env, names: list[str]) -> torch.Tensor:
    return env.joint_vel(names)


def joint_accelerations(env, names: list[str]) -> torch.Tensor:
    return env.joint_acc(names)


def joint_torque_obs(env, names: list[str]) -> torch.Tensor:
    return env.joint_torque(names)   # [Nm]


def joint_pd_torque_obs(env, names: list[str]) -> torch.Tensor:
    return env.joint_torque_pd(names)   # [Nm]


def joint_gravcomp_torque_obs(env, names: list[str]) -> torch.Tensor:
    return env.joint_torque_gravcomp(names)   # [Nm]


def body_contact_flags(env, bodies: list[str], threshold: float = 1.0) -> torch.Tensor:
    f = env.contact_force(bodies)
    return (f.norm(dim=-1) > threshold).float()


def fingertip_contact_steps(env, force_threshold: float = 0.1,   # [N]
                            target: str | None = "object") -> torch.Tensor:
    return env.contact_steps(force_threshold, target)   # [policy steps]


def hand_object_force_magnitude(env, bodies: list[str], target: str = "object") -> torch.Tensor:
    return env.contact_force_with(bodies, target).norm(dim=-1)   # [N]


def joint_pose_error(env, joint_pose: dict,   # {joint: rad}
                     tolerance_key: str | None = None) -> torch.Tensor:
    q = env.joint_pos(list(joint_pose))
    err = (q - transforms.const(tuple(joint_pose.values()), q)).abs().amax(dim=-1)
    return _against_bound(err, env, tolerance_key, "joint_pose_error")   # [rad]


def _against_bound(value: torch.Tensor, env, bound_key: str | None, who: str) -> torch.Tensor:
    v = value.unsqueeze(-1)
    if bound_key is None:
        return v
    vals = env.curriculum_values
    assert bound_key in vals, (
        f"{who}: the curriculum publishes no {bound_key!r} — it reports {sorted(vals)}. "
        f"(A ramp that is off publishes nothing.)")
    return torch.cat([v, vals[bound_key].reshape(1, 1).expand(v.shape[0], 1)], dim=-1)


def goal_dist_error(env, tolerance_key: str | None = None) -> torch.Tensor:
    d = object_goal_dist(env)
    return _against_bound(d, env, tolerance_key, "goal_dist_error")   # [m]


def palm_distance_error(env, palm_body: str, tolerance_key: str | None = None) -> torch.Tensor:
    grasp = env.object_pos()
    d = (env.body_pos(palm_body) - grasp).norm(dim=-1)
    return _against_bound(d, env, tolerance_key, "palm_distance_error")   # [m]


def curriculum_hold_progress(env, required_key: str | None = None) -> torch.Tensor:
    return _against_bound(env.curriculum_hold.float(), env, required_key, "curriculum_hold_progress")   # [steps]


def fingertip_penetration_depth(env, bodies: list[str], target: str = "object") -> torch.Tensor:
    return env.contact_penetration(bodies, target) * 1000.0   # m → mm


def hand_contact_force(env, bodies: list[str], target: str = "object",
                       ref_body: str | None = None) -> torch.Tensor:
    f = env.contact_force_with(bodies, target)
    if ref_body == "self":
        q = torch.stack([env.body_quat(b) for b in bodies], dim=1)
        f = transforms.quat_rotate_inverse(q, f)
    elif ref_body is not None:
        q = env.body_quat(ref_body).unsqueeze(1).expand(-1, f.shape[1], -1)
        f = transforms.quat_rotate_inverse(q, f)
    return f.reshape(f.shape[0], -1)


def episode_step(env) -> torch.Tensor:
    return (env.episode_length_buf.float() / max(1, env.max_episode_length)).unsqueeze(-1)


def object_lifed(env, lift_threshold: float) -> torch.Tensor:
    grasp_z = env.object_pos()[:, 2]
    return (grasp_z >= lift_threshold).float().unsqueeze(-1)


def instantaneous_reward(env) -> torch.Tensor:
    return torch.cat([env.last_reward.unsqueeze(-1), env.last_reward_terms], dim=-1)


def object_linear_velocity(env) -> torch.Tensor:
    return env.object_lin_vel()


def object_angular_velocity(env) -> torch.Tensor:
    return env.object_ang_vel()


def body_linear_velocity(env, body: str) -> torch.Tensor:
    return env.body_lin_vel(body)


def body_angular_velocity(env, body: str) -> torch.Tensor:
    return env.body_ang_vel(body)


_BEST_DIST_CLAMP = 2.0   # [m]


def _points_to_chest(pts_w, chest_pos, chest_quat, offset):
    op = transforms.local_point(chest_pos, chest_quat, offset).unsqueeze(1)
    oq = chest_quat.unsqueeze(1).expand(-1, pts_w.shape[1], -1)
    return transforms.points_to_frame(pts_w, op, oq)


def object_keypoints(env, chest_body: str, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    kp = cage(env, env.object_pos(), env.object_quat())
    rel = _points_to_chest(kp, env.body_pos(chest_body), env.body_quat(chest_body), offset)
    return rel.reshape(rel.shape[0], -1)


def goal_keypoints(env, chest_body: str, offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    gp, gq = env.goal_pos, env.goal_quat
    assert gp is not None, "goal_keypoints requires a fixed goal (not a goal contract)"
    kp = cage(env, gp, gq)
    rel = _points_to_chest(kp, env.body_pos(chest_body), env.body_quat(chest_body), offset)
    return rel.reshape(rel.shape[0], -1)


def object_goal_keypoint_success(env, tolerance: float = 0.01) -> torch.Tensor:   # [m]
    d = object_goal_dist(env)
    return (d <= tolerance).float().unsqueeze(-1)


def curriculum_state(env, keys: list[str]) -> torch.Tensor:
    vals = env.curriculum_values
    missing = [k for k in keys if k not in vals]
    assert not missing, (
        f"curriculum_state asks for {missing}, which this task's curriculum does not publish — it "
        f"reports {sorted(vals)}. (A ramp that is off publishes nothing.)")
    return torch.stack([vals[k] for k in keys]).unsqueeze(0).expand(env.num_envs, len(keys))


def dr_params(env, keys: list[str]) -> torch.Tensor:
    return torch.stack([env.dr_value(k) for k in keys], dim=-1)


def object_variant(env) -> torch.Tensor:
    v = env.object_variant_count()
    assert v > 1, f"object_variant: this scene builds the object from {v} variant(s) — nothing to observe"
    return torch.nn.functional.one_hot(env.object_variant_id(), num_classes=v).float()


def fingertip_relative_pos(env, chest_body: str, bodies: list[str], offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    tips = torch.stack([env.body_pos(b) for b in bodies], dim=1)
    rel = _points_to_chest(tips, env.body_pos(chest_body), env.body_quat(chest_body), offset)
    return rel.reshape(rel.shape[0], -1)


def fingertip_relative_pose(env, ref_body: str, bodies: list[str], offset=(0.0, 0.0, 0.0)) -> torch.Tensor:
    cp, cq = env.body_pos(ref_body), env.body_quat(ref_body)
    op = transforms.local_point(cp, cq, offset)
    outs = []
    for b in bodies:
        rp, rq = transforms.to_frame(env.body_pos(b), env.body_quat(b), op, cq)
        outs.append(torch.cat([rp, rq], dim=-1))
    return torch.cat(outs, dim=-1)


def fingertip_relative_vel(env, chest_body: str, bodies: list[str]) -> torch.Tensor:
    v = torch.stack([env.body_lin_vel(b) for b in bodies], dim=1)
    cq = env.body_quat(chest_body).unsqueeze(1).expand(-1, v.shape[1], -1)
    return transforms.quat_rotate_inverse(cq, v).reshape(v.shape[0], -1)


def prev_action_targets(env) -> torch.Tensor:
    return env.prev_action_targets


def action_delay(env) -> torch.Tensor:
    lag = env.action_delay_lag
    if lag is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return lag.float().unsqueeze(-1)   # [policy steps]


def closest_keypoint_max_dist(env, reward_term: str = "object_goal_keypoint_progress",
                              clamp: float = _BEST_DIST_CLAMP) -> torch.Tensor:
    return env.reward_best(reward_term).clamp(max=clamp).unsqueeze(-1)   # [m]
