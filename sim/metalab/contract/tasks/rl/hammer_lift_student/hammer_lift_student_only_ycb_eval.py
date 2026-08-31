"""hammer_lift_student RECIPE — the single-YCB-hammer EVAL contract ("only_ycb_eval").

The training recipe's world, judged at ONE fixed bar: no curriculum (nothing ramps, so level 0 already is the
END criteria whether or not the run passes ``--curriculum_end``), and the ACTOR obs group alone — an eval
rolls the deployed policy, so the critic's privileged set is never read. Everything else (obs terms, scene,
robot, sim, action) comes from ``_base.py`` beside this file.
"""
from __future__ import annotations

from sim.metalab.contract.spec import Done, Event, Rew, values
from sim.metalab.terms import events, gate, reward, terminate

from . import _base as base

# The success bar, written ONCE in the GATE below. With no curriculum term to relax it, the driver judges
# the level-up bar (what ends the episode and what the reach bonus is paid for) at the GATE itself, so this
# one bar is every bar this contract has.
_TOL = 0.05     # [m]   keypoint_max_dist to the goal
_HOLD = 20       # steps
_PALM = 0.05    # [m]   palm ↔ grasp point
_JTOL = 2.0     # [rad] worst-joint bound to the final posture
_EPISODE_S = 5.0  # [s]   episode horizon

# Hammer contact. False spawns it VISUAL-only (the asset's collision geoms are stripped): the hand and the
# table pass straight through it, so a rollout shows the trajectory the policy runs with nothing reacting to
# it. The actor never sees the object after the spawn frame anyway (object_init_pose, seen=True, and no
# curriculum widens that window), so its input is the same either way.
_HAMMER_COLLISION = True

_HAMMER, _TABLE = base.SCENE.objects
OBJECTS = [{**_HAMMER, "collision": _HAMMER_COLLISION}, _TABLE]


# --- reward terms ------------------------------------------------------------------------------------
# A weight is what ONE step of this term paying its maximum is worth. A term paid every step therefore
# totals weight * 300 (5.0 s at 60 Hz) over a full episode; the ONE-SHOT reach bonus totals its weight.
# The wandb Reward/<term> series divides by those 300 steps, so the log stays in per-step units.
class REWARD:
    fingertip_object_proximity      = Rew(reward.fingertip_object_proximity, weight=0.2, std=0.3)
    fingertip_object_contact        = Rew(reward.fingertip_object_contact, weight=1.0, force_threshold=0.1)
    object_goal_keypoint_tracking   = Rew(reward.object_goal_keypoint_tracking, weight=20.0, lift_threshold=base._LIFT_BAR)
    object_goal_reach_bonus         = Rew(reward.object_goal_reach_bonus, weight=600.0)
    arm_joint_vel                   = Rew(reward.joint_vel_l1,  weight=-0.003,  names="@joints.arm")

# --- events (domain randomization) -------------------------------------------------------------------
class EVENTS:
    reset_object_pose  = Event(events.reset_object_pose, "reset", active_position=[0.6, 0.0, base._LIFT_BAR], x_range=[-0.1, 0.05], y_range=[-0.2, 0.0], yaw_range=[-1.0, 1.0])
    reset_joints_by_offset = Event(events.reset_joints_by_offset, "reset", joints="@joints.ctrl", position_range=[-0.1, 0.1])
    object_friction    = Event(events.set_shape_friction, "reset", target="object", mu_range=[0.4, 0.8])
    robot_friction     = Event(events.set_shape_friction, "reset", target="robot",  mu_range=[0.8, 1.6],
                               exclude="@bodies.nail")   # as in _base.py: the pinned nails stay pinned
    table_friction     = Event(events.set_shape_friction, "reset", target="table",  mu_range=[0.4, 0.8])
    randomize_rigid_body_mass = Event(events.randomize_rigid_body_mass, "reset", scale_range=[0.5, 1.5])
    randomize_object_scale = Event(events.randomize_object_scale, "reset", scale_range=[0.8, 1.2], requires="object_scale")
    randomize_fixed_base_root_height = Event(events.randomize_fixed_base_root_height, "reset", z_offset_range=[-0.02, 0.02])

# --- gate: WHAT COUNTS AS SOLVED ----------------------------------------------------------------------
class GATE:
    predicate = gate.object_at_goal
    goal_dist_tol = _TOL
    hold_steps = _HOLD
    hold_mode = "consecutive"
    palm_distance = _PALM
    # joint_pose_tolerance = _JTOL
    # joint_final_pose = base._JOINT_FINAL_POSE   # no reward term shapes toward it, so the GATE states the shape

# --- termination / truncation ------------------------------------------------------------------------
class TERMINATE:
    # object_below_height = Done(terminate.object_below_height, min_height=0.85)
    curriculum_passed   = Done(terminate.curriculum_passed, truncation=True)

if not _HAMMER_COLLISION:
    # Every term that needs a contact SURFACE on the hammer — there is none to resize, to hand a mu to, or
    # to read a fingertip force off. The pose-driven terms (proximity, keypoint tracking, reach bonus) stay.
    del REWARD.fingertip_object_contact, EVENTS.randomize_object_scale, EVENTS.object_friction

# --- curriculum --------------------------------------------------------------------------------------
class CURRICULUM:
    """EMPTY on purpose — an eval judges at the GATE from step 0, so there is nothing to ramp."""


# The goal block is the base's, with the tolerance this recipe judges at (the goal block's own copy is
# scene metadata — what the bars are judged at is the GATE's).
GOAL = {**values(base.SCENE.goal), "goal_dist_tol": _TOL}

TASK = base.build_task("hammer_lift_student_only_ycb_eval", reward=REWARD, gate=GATE, curriculum=CURRICULUM,
                       events=EVENTS, terminate=TERMINATE, goal=GOAL, objects=OBJECTS,
                       episode_length_s=_EPISODE_S, obs_groups={"actor": base.ACTOR_OBS})
