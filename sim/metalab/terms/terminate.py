from __future__ import annotations

import torch


def object_below_height(env, min_height: float) -> torch.Tensor:   # [m]
    return env.object_pos()[:, 2] < min_height


def object_far_from_body(env, body: str, max_distance: float) -> torch.Tensor:   # [m]
    d = torch.linalg.norm(env.object_pos() - env.body_pos(body), dim=-1)
    return d > max_distance


def object_velocity_exceeded(env, max_lin_vel: float = 15.0,   # [m/s]
                             max_ang_vel: float = 30.0) -> torch.Tensor:   # [rad/s]
    lin = torch.linalg.norm(env.object_lin_vel(), dim=-1)
    ang = torch.linalg.norm(env.object_ang_vel(), dim=-1)
    return (lin > max_lin_vel) | (ang > max_ang_vel)


def table_fingertip_contact_force_exceeded(env, fingertips: list[str],
                                           force_threshold_n: float = 100.0) -> torch.Tensor:   # [N]
    f = env.contact_force_with(fingertips, "table")
    return f.norm(dim=-1).amax(dim=-1) > force_threshold_n


def body_contact_detected(env, bodies: list[str], force_threshold: float = 1.0) -> torch.Tensor:   # [N]
    return (env.contact_force(bodies).norm(dim=-1) > force_threshold).any(dim=-1)


def curriculum_passed(env) -> torch.Tensor:
    return env.curriculum_passed
