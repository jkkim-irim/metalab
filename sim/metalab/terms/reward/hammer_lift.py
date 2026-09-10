from __future__ import annotations

import torch


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
