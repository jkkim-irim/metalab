"""Quaternion ops (wxyz) and frame transforms — pure torch primitives (engine-agnostic).

All quaternions are canonical **wxyz** (:data:`sim.metalab.conventions.CANONICAL_QUAT`).
Arbitrary batch dims: broadcasts over ``(..., 4)`` quaternions and ``(..., 3)`` vectors.
"""
from __future__ import annotations

import torch


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (= inverse for unit quaternions) (..., 4) wxyz."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a⊗b (..., 4) wxyz."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v(..., 3) by quaternion q(..., 4) wxyz."""
    qw = q[..., 0:1]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Inverse rotation by q."""
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
    """world pose (pos, quat) → relative to origin frame (rel_pos, rel_quat). quat wxyz.

    Given the origin frame's world pose (e.g. CHEST_ORIGIN), yields quantities invariant to robot motion.
    """
    inv = quat_conj(origin_quat)
    rel_pos = quat_rotate(inv, pos - origin_pos)
    rel_quat = quat_mul(inv, quat)
    return rel_pos, rel_quat


def points_to_frame(points: torch.Tensor, origin_pos: torch.Tensor, origin_quat: torch.Tensor) -> torch.Tensor:
    """world points (..., 3) → positions in origin frame."""
    return quat_rotate(quat_conj(origin_quat), points - origin_pos)


_CONST: dict = {}


def const(v, ref: torch.Tensor) -> torch.Tensor:
    """Cache a small constant tuple (an offset, a goal quat) as a tensor on ref's device/dtype.

    Terms are evaluated at policy rate, so building these per call is a host→device copy every step."""
    k = (tuple(v), ref.device, ref.dtype)
    t = _CONST.get(k)
    if t is None:
        t = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
        _CONST[k] = t
    return t


def local_point(pos: torch.Tensor, quat: torch.Tensor, offset) -> torch.Tensor:
    """A body-local offset expressed in world coords → same shape as ``pos``.

        point = pos + R(quat) @ offset

    ``offset`` may be a plain tuple (cached via :func:`const`). offset (0,0,0) returns ``pos`` itself.
    This is the grasp point of an object (object pose + handle offset) and the palm point of a hand."""
    off = const(offset, pos).expand_as(pos)
    return pos + quat_rotate(quat, off)
