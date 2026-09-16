from __future__ import annotations

import torch

from sim.metalab.api import transforms
from sim.metalab.terms.events.locomotion import COMMAND_KEYS


def _command(env) -> torch.Tensor:   # (N, 3) = (vx, vy, wz) from sample_velocity_command
    return torch.stack([env.dr_value(k) for k in COMMAND_KEYS], dim=-1)


def _local(env, body: str, vec: torch.Tensor) -> torch.Tensor:
    return transforms.quat_rotate_inverse(env.body_quat(body), vec)


def tracking_lin_vel_xy(env, body: str, std: float) -> torch.Tensor:   # std [m^2/s^2]
    """exp(-|cmd_xy - v_xy|^2 / std) with v in the body frame — Genesis ``_reward_tracking_lin_vel``."""
    assert std > 0.0, f"tracking_lin_vel_xy: std must be > 0 — got {std}"
    v = _local(env, body, env.body_lin_vel(body))
    err = torch.sum(torch.square(_command(env)[:, :2] - v[:, :2]), dim=-1)
    return torch.exp(-err / std)


def tracking_ang_vel_z(env, body: str, std: float) -> torch.Tensor:   # std [rad^2/s^2]
    """exp(-(cmd_wz - w_z)^2 / std) with w in the body frame — Genesis ``_reward_tracking_ang_vel``."""
    assert std > 0.0, f"tracking_ang_vel_z: std must be > 0 — got {std}"
    w = _local(env, body, env.body_ang_vel(body))
    err = torch.square(_command(env)[:, 2] - w[:, 2])
    return torch.exp(-err / std)


def body_lin_vel_z_l2(env, body: str) -> torch.Tensor:   # [(m/s)^2]
    """Squared vertical velocity of ``body`` in its own frame — Genesis ``_reward_lin_vel_z``."""
    return torch.square(_local(env, body, env.body_lin_vel(body))[:, 2])


def body_height_l2(env, body: str, target_height: float) -> torch.Tensor:   # [m^2]
    """Squared world-z distance of ``body`` from ``target_height`` — Genesis ``_reward_base_height``."""
    return torch.square(env.body_pos(body)[:, 2] - target_height)


def joint_pose_l1_from_init(env, names: list[str]) -> torch.Tensor:   # [rad]
    """Sum |q - init_pose| over ``names`` — Genesis ``_reward_similar_to_default``."""
    q = env.joint_pos(names)
    init = env.spec.robot.init_pose
    return (q - transforms.const(tuple(init.get(n, 0.0) for n in names), q)).abs().sum(dim=-1)
