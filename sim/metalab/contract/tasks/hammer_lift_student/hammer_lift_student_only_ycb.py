"""hammer_lift_student RECIPE — the single-YCB-hammer default ("only_ycb").

The base scene carries ONE hammer asset (ycb_048_hammer_grasp_offset, variants=1) instead of the multi-asset
round-robin, so what this recipe supplies is the tunable trio for that scene: ``reward`` / ``gate`` /
``curriculum``. The shared contract (obs / scene / robot / sim / action / events / termination) lives in
``_base.py`` beside this file.
"""
from __future__ import annotations

from sim.metalab.contract.spec import Curr, Rew
from sim.metalab.terms import curriculum, gate, reward

from . import _base as base


# --- reward terms ------------------------------------------------------------------------------------
# A weight is what ONE step of this term paying its maximum is worth. A term paid every step therefore
# totals weight * 300 (5.0 s at 60 Hz) over a full episode; the ONE-SHOT reach bonus totals its weight.
# The wandb Reward/<term> series divides by those 300 steps, so the log stays in per-step units.
class REWARD:
    fingertip_object_proximity      = Rew(reward.fingertip_object_proximity, weight=0.5, std=0.1)
    object_goal_keypoint_tracking   = Rew(reward.object_goal_keypoint_tracking, weight=25.0, std=0.2,
                                          lift_threshold=base._LIFT_BAR, lift_full=base._LIFT_TOP)
    # fingertip_object_pinch_contact  = Rew(reward.fingertip_object_pinch_contact, weight=1.0,
    #                                       fingers=["R_Thumb_Fingertip", "R_Index_Fingertip"],
    #                                       lift_max=base._LIFT_TOP)
    nail_object_contact             = Rew(reward.nail_object_contact, weight=-0.5,
                                          bodies="@bodies.nail", force_threshold=0.1)
    # joint_pose_convergence          = Rew(reward.joint_pose_convergence, weight=1.0, joint_pose=base._JOINT_FINAL_POSE,
    #                                       std=0.25, lift_threshold=base._LIFT_TOP)
    fingertip_object_contact        = Rew(reward.fingertip_object_contact, weight=1.0, force_threshold=0.1,
                                          lift_threshold=base._LIFT_BAR)
    object_goal_reach_bonus         = Rew(reward.object_goal_reach_bonus, weight=2000.0)   # one-shot = 1.0 x 300 steps
    arm_joint_vel                   = Rew(reward.joint_vel_l1,  weight=-0.0005,  names="@joints.arm")


# --- gate: WHAT COUNTS AS SOLVED ----------------------------------------------------------------------
class GATE:
    predicate = gate.object_at_goal
    goal_dist_tol = 0.02     # [m] keypoint_max_dist to the goal
    palm_distance = 0.04         # [m] palm ↔ grasp point
    hold_steps = 20              # steps
    hold_mode = "consecutive"
    joint_pose_tolerance = 0.3   # [rad] worst commanded joint vs the posture below
    joint_final_pose = base._JOINT_FINAL_POSE   # no reward term shapes toward it, so the GATE states the shape

# --- curriculum --------------------------------------------------------------------------------------
class CURRICULUM:
    success_curriculum = Curr(
        curriculum.hammer_lift_success_curriculum,
        levels=30,
        goal_dist_tol_start=0.1,
        goal_dist_tol_end=0.01,                 # [m]
        steps_start=1,
        steps_end=120,                           # steps
        palm_start=0.1,
        palm_end=0.03,                          # [m]
        contact_start=1,                        # 1 tip -> 5, +contact_step per level: L4 2, L11 3, L17 4, L24 5
        contact_step=0.15,                      # stated, not derived: the hardest grips land after tol/grav/friction ease in
        contact_end=5,
        joint_pose_start=1.2,
        joint_pose_end=0.2,                     # [rad]
        joint_final_pose=base._JOINT_FINAL_POSE,
        grav_start=1.0,                         # [m/s²]
        friction_scale_start=1.0,               # mu multiplier
        friction_events=("robot_friction",),
        level_up_threshold=0.4,                 # SR at the current level that promotes to the next
        demote_threshold=0.1,                   # SR below this DEMOTES one level (two-sided)
        eval_interval_iterations=100,
        dense_reward_off_level=10,
        dense_reward_terms=(
                "object_goal_keypoint_tracking",
                "fingertip_object_proximity",
                            ),
    )



TASK = base.build_task("hammer_lift_student_only_ycb", reward=REWARD, gate=GATE, curriculum=CURRICULUM)
