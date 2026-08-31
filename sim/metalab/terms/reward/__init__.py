"""Reward term library — one flat function per reward (imported by the task contract as symbols).

Every term is ``def <name>(env, <knob>=<default>, ...) -> (N,)``, called by env_driver once per policy step as
``fn(env, **params)`` where ``params`` is the contract's ``kwargs``. A term never scales itself — magnitude
lives only in the contract's ``weight``. Details and per-term number tables: :mod:`.common`. No engine import.
"""
from sim.metalab.terms.reward.common import (
    action_rate_l2,
    fingertip_object_proximity,
    joint_pose_convergence,
    joint_torque_penalty,
    joint_vel_l1,
    lifting_reward,
    object_goal_keypoint_progress,
    object_goal_keypoint_tracking,
    object_goal_reach_bonus,
    object_lift_bonus,
    object_lift_ramp,
    palm_object_proximity,
)
from sim.metalab.terms.reward.hammer_lift import (
    fingertip_object_contact,
    fingertip_object_pinch_contact,
    nail_object_contact,
)

__all__ = [
    "lifting_reward", "object_lift_ramp", "object_lift_bonus",           # lift    → [0,1] ramp / 0-1 one-shot
    "fingertip_object_contact",                                          # contact → [0,1] pad-grip fraction
    "fingertip_object_pinch_contact",                                    # pinch   → [0,1] named pair, height-capped
    "nail_object_contact",                                               # PENALTY → [0,1] nail-touch fraction
    "palm_object_proximity", "fingertip_object_proximity",                 # reach   → (0,1]
    "joint_pose_convergence",                                            # posture → (0,1], lift-gated
    "object_goal_keypoint_progress", "object_goal_keypoint_tracking",     # carry   → metres / (0,1]
    "object_goal_reach_bonus",                                           # success → 0/1 one-shot
    "joint_vel_l1", "joint_torque_penalty", "action_rate_l2",             # penalty → positive, negate by weight
]
