"""hammer_lift_teacher RECIPE — the privileged-state reference ("privileged").

The teacher's actor reads the SAME full state as its critic (obs_groups actor/privileged = "all" in
``_base.py``), so this trio is tuned for a fast-learning reference rather than for deployment: a
TELESCOPING keypoint-progress reward carries the object to the goal, and a one-sided 20-level success
curriculum ramps tolerance / hold / grip up to the GATE. The shared contract (obs / scene / robot / sim
/ action / events / termination) lives in ``_base.py`` beside this file.
"""
from __future__ import annotations

from sim.metalab.contract.spec import Curr, Rew
from sim.metalab.terms import curriculum, gate, reward

from . import _base as base


# --- reward terms ------------------------------------------------------------------------------------
# A weight is what ONE step of this term paying its maximum is worth. A term paid every step therefore
# totals weight * 600 (10.0 s at 60 Hz) over a full episode; the ONE-SHOT reach bonus and the TELESCOPING
# keypoint progress total their weight. The wandb Reward/<term> series divides by those 600 steps.
class REWARD:
    lifting_reward                  = Rew(reward.lifting_reward,                 weight=20.0,
                                          lift_height=0.1)
    fingertip_object_proximity      = Rew(reward.fingertip_object_proximity,      weight=5.0,
                                          std=0.05)
    object_goal_keypoint_progress   = Rew(reward.object_goal_keypoint_progress,   weight=378.0)  # telescoping = 0.63 x 600
    object_goal_reach_bonus         = Rew(reward.object_goal_reach_bonus, weight=6000.0)  # one-shot = 10.0 x 600
    arm_joint_vel                   = Rew(reward.joint_vel_l1,  weight=-0.1,   names="@joints.arm")
    hand_joint_vel                  = Rew(reward.joint_vel_l1,  weight=-0.01,  names="@joints.hand")


# --- gate: WHAT COUNTS AS SOLVED ----------------------------------------------------------------------
class GATE:
    predicate = gate.object_at_goal
    goal_dist_tol = 0.01     # [m] keypoint_max_dist to the goal — true 1cm
    hold_steps = 20              # consecutive steps inside tolerance (counter resets the moment it breaks)
    contact_count = 5            # full 5-fingertip grasp while holding
    force_threshold = 0.1


# --- curriculum --------------------------------------------------------------------------------------
# The ramp UP TO the gate. ONLY the starting difficulty and the ramp shape live here — every destination is
# read from GATE by the term itself, so each knob is written exactly once (start here, end in GATE).
class CURRICULUM:
    success_curriculum = Curr(
        curriculum.hammer_lift_success_curriculum,
        levels=20,
        goal_dist_tol_start=0.10,          # 10cm → 1cm
        goal_dist_tol_end=0.01,
        steps_start=1,           # 1 step → 20 steps
        steps_end=20,
        contact_start=0,         # grip off → 5 tips
        contact_step=0.25,       # 5 tips over 20 levels — the rate this ramp already ran at
        contact_end=5,
        force_start=0.1,         # [N] no end stated → the bar never moves
        level_up_threshold=0.5,  # SR at the current level that promotes to the next
        eval_interval_iterations=50,
    )

TASK = base.build_task("hammer_lift_teacher_privileged",
                       reward=REWARD, gate=GATE, curriculum=CURRICULUM)
