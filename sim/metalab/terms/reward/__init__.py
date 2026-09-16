from __future__ import annotations

from sim.metalab.terms.reward.common import (
    action_rate_l2,
    body_goal_proximity,
    fingertip_object_proximity,
    joint_pose_convergence,
    joint_torque_penalty,
    joint_vel_l1,
    lifting_reward,
    object_goal_keypoint_progress,
    object_goal_keypoint_tracking,
    object_goal_reach_bonus,
    palm_object_proximity,
)
from sim.metalab.terms.reward.hammer_lift import (
    fingertip_object_contact,
    fingertip_object_pinch_contact,
    nail_object_contact,
)
from sim.metalab.terms.reward.locomotion import (
    body_height_l2,
    body_lin_vel_z_l2,
    joint_pose_l1_from_init,
    tracking_ang_vel_z,
    tracking_lin_vel_xy,
)

__all__ = [
    "action_rate_l2",
    "body_goal_proximity",
    "body_height_l2",
    "body_lin_vel_z_l2",
    "fingertip_object_contact",
    "fingertip_object_pinch_contact",
    "fingertip_object_proximity",
    "joint_pose_convergence",
    "joint_pose_l1_from_init",
    "joint_torque_penalty",
    "joint_vel_l1",
    "lifting_reward",
    "nail_object_contact",
    "object_goal_keypoint_progress",
    "object_goal_keypoint_tracking",
    "object_goal_reach_bonus",
    "palm_object_proximity",
    "tracking_ang_vel_z",
    "tracking_lin_vel_xy",
]
