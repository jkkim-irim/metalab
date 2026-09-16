from __future__ import annotations

import torch

from sim.metalab.api import transforms

_GRAVITY_DIR = (0.0, 0.0, -1.0)


def body_angular_velocity_local(env, body: str) -> torch.Tensor:   # [rad/s], in the body frame
    return transforms.quat_rotate_inverse(env.body_quat(body), env.body_ang_vel(body))


def body_linear_velocity_local(env, body: str) -> torch.Tensor:   # [m/s], in the body frame
    return transforms.quat_rotate_inverse(env.body_quat(body), env.body_lin_vel(body))


def projected_gravity(env, body: str) -> torch.Tensor:
    """Unit gravity direction expressed in ``body``'s frame — (0, 0, -1) when the body is level."""
    q = env.body_quat(body)
    return transforms.quat_rotate_inverse(q, transforms.const(_GRAVITY_DIR, q).expand_as(q[:, :3]))


def joint_positions_relative_to_init(env, names: list[str]) -> torch.Tensor:   # [rad]
    """Joint positions minus the contract's ``robot.init_pose`` (the standing pose the policy offsets from)."""
    q = env.joint_pos(names)
    init = env.spec.robot.init_pose
    return q - transforms.const(tuple(init.get(n, 0.0) for n in names), q)
