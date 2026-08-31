"""hammer_lift_student RECIPE — four hammer shapes, evenly spread over the envs ("4hammers").

Same tunable trio as ``only_ycb`` (``reward`` / ``gate`` / ``curriculum`` are that recipe's, verbatim); the
ONE difference is the scene's hammer entry — a FOUR-variant asset list instead of a single MJCF, so env i
trains on variant i % 4 (both engines round-robin: newton's parser does ``w % N``, genesis is patched to
match). The object is renamed ``hammer`` because no single asset name describes four, which moves the
hammer's contact-params key with it — hence the ``contact`` block below. The shared contract (obs / scene /
robot / sim / action / events / termination) lives in ``_base.py`` beside this file.
"""
from __future__ import annotations

from sim.metalab.contract.spec import Curr, Event, Obs, Rew
from sim.metalab.terms import curriculum, events, gate, obs, reward

from .. import _assets as assets
from . import _base as base


# --- scene: the ONLY thing this recipe changes about the world ----------------------------------------
_HAMMERS = ["hammer_ycb", "hammer_cylinder", "hammer_edge", "hammer_rect"]
_TABLE = next(o for o in base.SCENE.objects if o["name"] == "table")

OBJECTS = [
    {
        "name": "hammer",
        "asset": {"mjcf": [assets.s3_mjcf(a) for a in _HAMMERS]},
        "mass": 0.55, "variants": len(_HAMMERS), "init_pos": [0.7, -0.2, 0.874],
    },
    _TABLE,
]


class CONTACT:
    robot = base.SCENE.contact_params.robot
    table = base.SCENE.contact_params.table
    hammer = base.SCENE.contact_params.ycb_048_hammer_grasp_offset


# --- physics: only what differs from _base.PHYSICS (build_task MERGES) --------------------------------
class PHYSICS:
    hz = 120            # control rate — physics dt = 1/hz
    substeps = 4        # physics substeps per control step
    decimation = 3      # physics steps per POLICY step


# --- per-engine corrections: only what differs from _base.OVERRIDES (build_task MERGES per engine) -----
class OVERRIDES:
    class newton:
        use_mujoco_contacts = False
        hull_maxvert = 32
        object_hull_maxvert = 64
        object_gap = 0.01
        cone = "pyramidal"
        impratio = 1
        njmax = 640

# --- action: the base's groups with the EMA time constants retuned -----------------------------------
_ARM_EMA_TAU = 3.0    # [s]
_HAND_EMA_TAU = 0.1  # None = no EMA (0 is invalid: alpha = 1 - exp(-dt/tau) divides by it)

class ACTION:
    min_delay = base.ACTION.min_delay
    max_delay = base.ACTION.max_delay

    class arm:
        joints = base.ACTION.arm.joints
        scale = base.ACTION.arm.scale
        ema_tau = _ARM_EMA_TAU

    class hand:
        joints = base.ACTION.hand.joints
        scale = base.ACTION.hand.scale
        ema_tau = _HAND_EMA_TAU


# --- events: the base's DR block with ONE entry retuned -----------------------------------------------
# Narrower size range than the base's [0.8, 1.2]: the four variants already supply the shape diversity, so
# size only has to cover build tolerance. NOT a workaround for anything measured — forcing the scale in 16
# envs per cell, resting height stays exactly proportional to it (h = d*s to two decimals) and every env
# keeps its contacts from 0.6 through 2.0, so the range is a deliberate choice, not a ceiling.
_SCALE_RANGE = [0.9, 1.1]
class EVENTS:
    reset_object_pose                = base.EVENTS.reset_object_pose
    reset_joints_by_offset           = base.EVENTS.reset_joints_by_offset
    object_friction                  = base.EVENTS.object_friction
    robot_friction                   = base.EVENTS.robot_friction
    table_friction                   = base.EVENTS.table_friction
    randomize_rigid_body_mass        = base.EVENTS.randomize_rigid_body_mass
    randomize_object_scale           = Event(events.randomize_object_scale, "reset", scale_range=_SCALE_RANGE, requires="object_scale")
    randomize_fixed_base_root_height = base.EVENTS.randomize_fixed_base_root_height
    object_pull_down                 = base.EVENTS.object_pull_down

# --- reward terms ------------------------------------------------------------------------------------
class REWARD:
    palm_object_proximity           = Rew(reward.palm_object_proximity, weight=0.5, std=0.1, palm_body="@frames.palm")
    # object_goal_keypoint_tracking   = Rew(reward.object_goal_keypoint_tracking, weight=15.0, std=0.2, lift_threshold=base._LIFT_BAR, lift_full=base._LIFT_TOP)
    # nail_object_contact             = Rew(reward.nail_object_contact, weight=-0.5, bodies="@bodies.nail", force_threshold=0.1)
    fingertip_object_contact        = Rew(reward.fingertip_object_contact, weight=1.0, force_threshold=0.1, lift_threshold=0.0)
    object_goal_reach_bonus         = Rew(reward.object_goal_reach_bonus, weight=2000.0)
    # arm_joint_torque                = Rew(reward.joint_torque_penalty, weight=-1.0e-4, names="@joints.arm")
    # action_rate                     = Rew(reward.action_rate_l2, weight=-1.0e-6)


# --- gate: WHAT COUNTS AS SOLVED ----------------------------------------------------------------------
class GATE:
    predicate = gate.object_at_goal
    goal_dist_tol = 0.02     # [m] keypoint_max_dist to the goal
    palm_distance = 0.03         # [m] palm <-> grasp point
    hold_steps = 20              # steps
    hold_mode = "consecutive"
    # joint_pose_tolerance = 0.3   # [rad] worst commanded joint vs the posture below
    # joint_final_pose = base._JOINT_FINAL_POSE   # no reward term shapes toward it, so the GATE states the shape

# --- curriculum --------------------------------------------------------------------------------------
class CURRICULUM:
    success_curriculum = Curr(
        curriculum.hammer_lift_success_curriculum,
        levels=20,
        goal_dist_tol_start=0.15,
        goal_dist_tol_end=0.01,                 # [m]
        steps_start=1,
        steps_end=60,                           # steps
        palm_start=0.1,
        palm_end=0.02,                          # [m]
        contact_fingers=("R_Thumb_Fingertip", "R_Index_Fingertip", "R_Middle_Fingertip", "R_Ring_Fingertip", "R_Little_Fingertip"),
        contact_fingers_start=1,
        contact_fingers_step=0.3,
        level_up_threshold=0.4,                 # SR at the current level that promotes to the next
        demote_threshold=0.1,                   # SR below this DEMOTES one level (two-sided)
        eval_interval_iterations=50,
        dense_reward_off_level=10,
        dense_reward_terms=(
                "palm_object_proximity",
                            ),
    )


# --- obs: the base's groups, minus the critic term the (now disabled) joint-pose ramp fed, plus the one
# term only THIS recipe can carry — with four shapes in the env pool, which one an env got is a per-env
# latent the critic would otherwise have to average over (the base's single-asset recipes have no variant).
CRITIC_OBS = type("CRITIC_OBS", (), {
    **{k: v for k, v in vars(base.CRITIC_OBS).items() if not k.startswith("__") and k != "joint_pose_err"},
    "object_variant": Obs(obs.object_variant, labels=list(_HAMMERS)),
})


TASK = base.build_task("hammer_lift_student_4hammers", reward=REWARD, gate=GATE, curriculum=CURRICULUM,
                       objects=OBJECTS, contact=CONTACT, overrides=OVERRIDES, events=EVENTS, action=ACTION,
                       physics=PHYSICS, obs_groups={"actor": base.ACTOR_OBS, "critic": CRITIC_OBS})
