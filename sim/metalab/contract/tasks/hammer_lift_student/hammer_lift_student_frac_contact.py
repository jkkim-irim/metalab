"""hammer_lift_student RECIPE — grasp-gated, two-sided curriculum ("frac_contact", ported from PR #108).

Same scene and shared core as ``only_ycb``; what differs is what counts as SOLVED and how the ramp
behaves when a level stops working:

  * the GATE demands a FIVE-fingertip grasp (``contact_count``) while the hammer is at the goal, so
    carrying it there with the hand splayed open no longer scores by val/SR. That is the GATE's own bar —
    the curriculum's grip ramp (``contact_start``/``step``/``end``) is a separate condition on the level-up
    bar, and ``only_ycb`` runs the same ramp without asking val/SR for a grip.
  * the ramp starts at ONE tip rather than zero — level 0 must still be touching the hammer — and rises
    ``contact_step=0.15`` tips per level, so the 4th and 5th land near the end (cc4 ~L20, cc5 ~L27),
    after tolerance / gravity / friction have eased in.
  * ``demote_threshold`` makes the curriculum TWO-SIDED: a level whose success collapses falls back to
    one the policy can still solve instead of dwelling at a zero-reward bar.
  * ``fingertip_object_contact`` is weighted up (1.0 -> 2.0) and stays OUT of ``dense_reward_terms`` —
    it is the per-fingertip gradient that recruits the 4th and 5th fingers as the ramp rises, so cutting
    it at the dense cutoff would leave a zero-gradient cliff at every contact bump.

Weights are OUR semantics (a weight is what ONE step paying its maximum is worth), not PR #108's, so
its numbers are not transplanted — only the relative decision to raise the grip term is.
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
    fingertip_object_proximity      = Rew(reward.fingertip_object_proximity, weight=0.2, std=0.3)
    fingertip_object_contact        = Rew(reward.fingertip_object_contact, weight=2.0, force_threshold=0.1)
    object_goal_keypoint_tracking   = Rew(reward.object_goal_keypoint_tracking, weight=20.0, lift_threshold=base._LIFT_BAR)
    object_goal_reach_bonus         = Rew(reward.object_goal_reach_bonus, weight=600.0)   # one-shot = 1.0 x 300 steps
    arm_joint_vel                   = Rew(reward.joint_vel_l1,  weight=-0.002,  names="@joints.arm")


# --- gate: WHAT COUNTS AS SOLVED ----------------------------------------------------------------------
class GATE:
    predicate = gate.object_at_goal
    goal_dist_tol = 0.03         # [m] keypoint_max_dist to the goal
    hold_steps = 30              # steps
    hold_mode = "consecutive"
    contact_count = 5            # fingertips gripping the hammer while it is held at the goal
    palm_distance = 0.03         # [m] palm ↔ grasp point
    joint_pose_tolerance = 0.2   # [rad]
    joint_final_pose = base._JOINT_FINAL_POSE   # no reward term shapes toward it, so the GATE states the shape


# --- curriculum --------------------------------------------------------------------------------------
class CURRICULUM:
    success_curriculum = Curr(
        curriculum.hammer_lift_success_curriculum,
        levels=30,
        goal_dist_tol_start=0.2,
        goal_dist_tol_end=0.03,   # [m]
        steps_start=1,
        steps_end=30,             # steps
        contact_start=1,        # 1 tip -> 5 tips, +contact_step per level
        contact_step=0.15,      # cc4 ~L20, cc5 ~L27 (PR #108: at +1/level the full grasp came at L4)
        contact_end=5,
        palm_start=0.1,
        palm_end=0.03,          # [m]
        joint_pose_start=1.7,
        joint_pose_end=0.2,     # [rad]
        joint_final_pose=base._JOINT_FINAL_POSE,
        grav_start=1.0,         # [m/s²]
        friction_scale_start=3.0,   # mu multiplier
        friction_events=("robot_friction",),
        level_up_threshold=0.5,   # SR at the current level that promotes to the next
        demote_threshold=0.1,     # SR below this DEMOTES one level (two-sided)
        eval_interval_iterations=50,
        dense_reward_off_level=30,
        dense_reward_terms=(
                "object_goal_keypoint_tracking",
                "fingertip_object_proximity",
                            ),
    )


TASK = base.build_task("hammer_lift_student_frac_contact", reward=REWARD, gate=GATE, curriculum=CURRICULUM)
