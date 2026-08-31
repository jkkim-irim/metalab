"""Kinematics primitives — chest-origin relative pose etc. (composes frames, pure torch).

CHEST_ORIGIN frame = robot torso body (e.g. Waist_Upper_Pitch_Link) world pose plus a local offset.
Expressing hand/fingertip/object quantities in this frame makes them invariant to robot motion.
(:mod:`sim.metalab.conventions` Frame)
"""
from __future__ import annotations

from typing import Sequence

import torch

from sim.metalab.api.frames import const, quat_rotate, to_frame

Offset = torch.Tensor | Sequence[float]     # a contract writes [x, y, z]; a caller may pass a tensor


def joint_pose_error(q: torch.Tensor, target: Sequence[float]) -> torch.Tensor:
    """WORST per-joint deviation from a target posture → ``(N,)``, same unit as ``q`` (rad).

        returns  max_j |q[:, j] - target[j]|

    The MAX, not the mean: a mean lets one joint sit wide of the target as long as the others compensate,
    which is a different posture than the one asked for. ``q`` columns and ``target`` must be in the same
    order — the caller owns that pairing (it is the caller that named the joints).

    ``target`` is a plain sequence, cached per device/dtype (:func:`frames.const`), because a posture is a
    contract constant read at policy rate."""
    t = const(tuple(target), q)
    assert t.shape[-1] == q.shape[-1], \
        f"joint_pose_error: {q.shape[-1]} joints but {t.shape[-1]} target angles"
    return (q - t).abs().amax(dim=-1)


def chest_origin_pose(chest_pos: torch.Tensor, chest_quat: torch.Tensor,
                      offset: Offset) -> tuple[torch.Tensor, torch.Tensor]:
    """chest body world pose + local offset → chest-origin world pose (pos, quat wxyz).

    ``offset`` may be a plain ``[x, y, z]`` — obs terms are flat functions that receive the contract's list
    verbatim, and ``as_tensor`` on an already-matching tensor is a no-op, so nothing is copied twice."""
    off = torch.as_tensor(offset, dtype=chest_pos.dtype, device=chest_pos.device)
    if off.dim() == 1:
        off = off.unsqueeze(0).expand_as(chest_pos)
    return chest_pos + quat_rotate(chest_quat, off), chest_quat


def to_chest(pos: torch.Tensor, quat: torch.Tensor,
             chest_pos: torch.Tensor, chest_quat: torch.Tensor,
             offset: Offset) -> tuple[torch.Tensor, torch.Tensor]:
    """world pose (pos, quat) → relative to chest-origin frame (rel_pos, rel_quat wxyz)."""
    op, oq = chest_origin_pose(chest_pos, chest_quat, offset)
    return to_frame(pos, quat, op, oq)
