from __future__ import annotations

import torch


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qw = q[..., 0:1]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conj(q), v)


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([x, y, z, w], dim=-1)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.stack([w, x, y, z], dim=-1)


def to_frame(
    pos: torch.Tensor, quat: torch.Tensor,
    origin_pos: torch.Tensor, origin_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv = quat_conj(origin_quat)
    rel_pos = quat_rotate(inv, pos - origin_pos)
    rel_quat = quat_mul(inv, quat)
    return rel_pos, rel_quat


def points_to_frame(points: torch.Tensor, origin_pos: torch.Tensor, origin_quat: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conj(origin_quat), points - origin_pos)


_CONST: dict = {}


def const(v, ref: torch.Tensor) -> torch.Tensor:
    k = (tuple(v), ref.device, ref.dtype)
    t = _CONST.get(k)
    if t is None:
        t = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
        _CONST[k] = t
    return t


def local_point(pos: torch.Tensor, quat: torch.Tensor, offset) -> torch.Tensor:
    off = const(offset, pos).expand_as(pos)
    return pos + quat_rotate(quat, off)


TETRA_CORNERS = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])


def body_points(pos: torch.Tensor, quat: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    n, k = pos.shape[0], local.shape[0]
    q = quat.unsqueeze(1).expand(n, k, 4)
    lp = local.to(pos).unsqueeze(0).expand(n, k, 3)
    return quat_rotate(q, lp) + pos.unsqueeze(1)
