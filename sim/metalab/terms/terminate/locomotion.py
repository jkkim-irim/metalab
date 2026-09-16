from __future__ import annotations

import math

import torch

from sim.metalab.api import transforms

_GRAVITY_DIR = (0.0, 0.0, -1.0)


def body_tilt_exceeded(env, body: str, max_roll_deg: float, max_pitch_deg: float) -> torch.Tensor:
    """True when ``body`` rolls or pitches past the bound. Roll/pitch come from the gravity direction in the
    body frame (R = Rz·Ry·Rx convention), so yaw never enters."""
    q = env.body_quat(body)
    g = transforms.quat_rotate_inverse(q, transforms.const(_GRAVITY_DIR, q).expand_as(q[:, :3]))
    pitch = torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))
    roll = torch.atan2(-g[:, 1], -g[:, 2])
    return (roll.abs() > math.radians(max_roll_deg)) | (pitch.abs() > math.radians(max_pitch_deg))
