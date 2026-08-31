"""Keypoint cage and max-dist metric — pose-error primitive fusing position and rotation.

Measures pose difference as the max corresponding-point distance between keypoint cages on
object/goal (captures position and orientation at once).
Same concept as perceptive_dexdeepmimic's goal-tracking metric (``keypoint_max_dist``).
"""
from __future__ import annotations

import torch

from sim.metalab.api.frames import quat_rotate

#: Default 4 keypoints — (±1,±1,±1) diagonal set (perceptive ``KEYPOINT_CORNERS``).
DEFAULT_CORNERS = torch.tensor(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
)

Pose = tuple[torch.Tensor, torch.Tensor]  # (pos (N,3), quat (N,4) wxyz)


def cage(pos: torch.Tensor, quat: torch.Tensor, half_extent, corners: torch.Tensor | None = None) -> torch.Tensor:
    """pose + half_extent → keypoint cage (N, K, 3) world."""
    c = (DEFAULT_CORNERS if corners is None else torch.as_tensor(corners)).to(pos)  # (K,3)
    he = torch.as_tensor(half_extent, dtype=pos.dtype, device=pos.device)           # (3,)
    local = c * he                                                                   # (K,3)
    n, k = pos.shape[0], local.shape[0]
    q = quat.unsqueeze(1).expand(n, k, 4)
    lp = local.unsqueeze(0).expand(n, k, 3)
    return quat_rotate(q, lp) + pos.unsqueeze(1)                                     # (N,K,3)


def max_dist(pose_a: Pose, pose_b: Pose, half_extent, corners: torch.Tensor | None = None) -> torch.Tensor:
    """Max corresponding-point distance between the two poses' keypoint cages (N,). Fused position+rotation error."""
    a = cage(pose_a[0], pose_a[1], half_extent, corners)
    b = cage(pose_b[0], pose_b[1], half_extent, corners)
    return torch.linalg.norm(a - b, dim=-1).max(dim=-1).values


def object_goal_dist(object_pos: torch.Tensor, object_quat: torch.Tensor,
                     goal_pos: torch.Tensor, goal_quat: torch.Tensor, half_extent) -> torch.Tensor:
    """THE goal metric → (N,) [m]: cage distance from the object to a fixed goal pose.

    Measured at the object's own origin, which the asset pipeline puts ON the grasp point (the converter's
    per-variant ``translate``), so no caller carries an offset to correct for where a mesh happens to be
    authored. The cage fuses position and orientation into one metre-valued number, so a pure yaw error
    still registers as corner displacement.

    Here rather than in each caller: the gate predicate, the two goal reward terms and the success obs must
    all measure the goal the SAME way, and one function is how that stays true."""
    return max_dist((object_pos, object_quat), (goal_pos, goal_quat), half_extent)
