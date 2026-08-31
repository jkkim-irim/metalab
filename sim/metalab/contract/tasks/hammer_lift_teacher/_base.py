"""hammer_lift_teacher — SHARED CORE. Authoring rules → sim/metalab/contract/tasks/README.md.

SIM-OWNED, values inline (team decision 2026-07-23: every task knob — reward weights, thresholds,
action scales, DR ranges, curriculum — lives sim-side, written directly in the contract; the
trainer's experiment keeps only the RL config ``EXP``). The contract owns mechanism AND values: terms,
obs groups, action groups, physics, robot/assets, and the numbers plugged into them.

Everything here is the part every recipe shares; a recipe module (a sibling
``hammer_lift_teacher_<recipe>.py`` in this family folder) supplies the tunable trio (``reward`` /
``gate`` / ``curriculum``) — plus any of the optional overrides (``action`` / ``events`` / ``terminate``
/ ``objects``) where it differs — and calls :func:`build_task`.

The leading underscore keeps this out of the task list (it is a library, not a task): the launchpad
skips ``_*.py``, and the loader only imports the ``<task, recipe>`` pair a run asks for.
"""
from __future__ import annotations

from sim.metalab.contract.spec import Done, Event, Obs, TaskSpec, values
from sim.metalab.terms import events, obs, terminate

from .. import _assets as assets


# --- physics / solver --------------------------------------------------------------------------------
class PHYSICS:
    hz = 120            # control rate — physics dt = 1/hz
    substeps = 2        # physics substeps per control step
    decimation = 2      # physics steps per POLICY step → policy runs at 60 Hz

# --- per-engine corrections (NOT shared knobs) -------------------------------------------------------
class OVERRIDES:
    class newton:
        cone = "pyramidal"
        impratio = 1.0
        iterations = 30
        ls_iterations = 30
        tolerance = 1.0e-6
        ls_tolerance = 1.0e-6
        nconmax = 128
        ccd_iterations = 16
        eq_solref = [0.01, 1.0]
        eq_solimp = [0.95, 0.99, 0.001, 0.5, 1.0]


# --- the world: ground, robot(+placement/init pose), objects, contact params, camera, goal -----------
class SCENE:
    ground = True     # infinite ground plane at z=0

    class robot:
        name = "allex_right"
        base_pos = [0.0, 0.0, 0.6]
        base_quat = [1.0, 0.0, 0.0, 0.0]   # wxyz
        fixed_base = True
        # DEGREES (the loader converts to radians). A joint-name → angle TABLE, so it stays a dict: ACTIVE
        # (mask 1) = runtime spawn pose, MASK-0 = weld angle (posed-then-welded). The 12 equality FOLLOWERS
        # (waist dummy/upper, finger IP/DIP) are auto-computed by the loader from the MJCF <equality> polycoef.
        init_pose = {
            # waist + neck (4 — mask-0 on allex_right → welded at these angles)
            "Waist_Yaw_Joint": 0.0, "Waist_Lower_Pitch_Joint": 0.0,
            "Neck_Pitch_Joint": 0.0, "Neck_Yaw_Joint": 0.0,
            # right arm (7 — active)
            "R_Shoulder_Pitch_Joint": 20.0, "R_Shoulder_Roll_Joint": -20.0, "R_Shoulder_Yaw_Joint": 3.0,
            "R_Elbow_Joint": -110.0, "R_Wrist_Yaw_Joint": 180.0, "R_Wrist_Roll_Joint": 0.0,
            "R_Wrist_Pitch_Joint": 0.0,
            # right hand (15 — active; IP/DIP auto-follow)
            "R_Thumb_Yaw_Joint": 0.0, "R_Thumb_CMC_Joint": 0.0, "R_Thumb_MCP_Joint": 0.0,
            "R_Index_ABAD_Joint": 0.0, "R_Index_MCP_Joint": 0.0, "R_Index_PIP_Joint": 0.0,
            "R_Middle_ABAD_Joint": 0.0, "R_Middle_MCP_Joint": 0.0, "R_Middle_PIP_Joint": 0.0,
            "R_Ring_ABAD_Joint": 0.0, "R_Ring_MCP_Joint": 0.0, "R_Ring_PIP_Joint": 0.0,
            "R_Little_ABAD_Joint": 0.0, "R_Little_MCP_Joint": 0.0, "R_Little_PIP_Joint": 0.0,
            # left arm (7 — mask-0 → welded)
            "L_Shoulder_Pitch_Joint": 0.0, "L_Shoulder_Roll_Joint": 0.0, "L_Shoulder_Yaw_Joint": 0.0,
            "L_Elbow_Joint": 0.0, "L_Wrist_Yaw_Joint": 0.0, "L_Wrist_Roll_Joint": 0.0,
            "L_Wrist_Pitch_Joint": 0.0,
            # left hand (15 — mask-0 → welded; IP/DIP auto-follow)
            "L_Thumb_Yaw_Joint": 0.0, "L_Thumb_CMC_Joint": 0.0, "L_Thumb_MCP_Joint": 0.0,
            "L_Index_ABAD_Joint": 0.0, "L_Index_MCP_Joint": 0.0, "L_Index_PIP_Joint": 0.0,
            "L_Middle_ABAD_Joint": 0.0, "L_Middle_MCP_Joint": 0.0, "L_Middle_PIP_Joint": 0.0,
            "L_Ring_ABAD_Joint": 0.0, "L_Ring_MCP_Joint": 0.0, "L_Ring_PIP_Joint": 0.0,
            "L_Little_ABAD_Joint": 0.0, "L_Little_MCP_Joint": 0.0, "L_Little_PIP_Joint": 0.0,
        }

    # A LIST (order = variant round-robin), so it stays a list of entries rather than named blocks.
    objects = [
        {   # hammer — 3 MJCF variants, env i gets variant i % 3
            "name": "hammer",
            "asset": {"mjcf": [                       # from S3 — object assets are not in git
                assets.s3_mjcf("hammer_cylinder"),
                assets.s3_mjcf("hammer_rect"),
                assets.s3_mjcf("hammer_edge"),
            ]},
            "mass": 0.55, "variants": 3,   # friction: not authored here — the reset DR event owns it
            "init_pos": [0.5, -0.1, 0.91],
        },
        {   # table — scenery: welded, size stays a task knob (procedural, no asset)
            "name": "table", "fixed": True, "mass": 1.0,
            "parts": [{"shape": "box", "size": [0.6, 1.0, 0.2]}],
            "init_pos": [0.6, 0.0, 0.8],
        },
    ]

    # Contact softness per group (shared knob — genesis implements the same solref/solimp math; solmix is
    # newton-only, genesis averages the two geoms 0.5/0.5 and cannot weight them).
    class contact_params:
        robot = {"solref": [0.01, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solmix": 1.0}
        table = {"solref": [0.01, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0], "solmix": 2.0}
        hammer = {"solref": [0.01, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0]}   # was in the hammer MJCFs

    class camera:
        eye = [1.5, -1.1, 1.5]
        lookat = [0.4, -0.05, 1.0]
        fov = 30.0

    class goal:
        pos = [0.55, -0.1, 1.1]
        quat = [1.0, 0.0, 0.0, 0.0]
        keypoint_half_extent = [0.05, 0.05, 0.08]


# --- action groups: WHAT the policy commands (18 dims = arm 7 + hand 11) -----------------------------
# Listed explicitly so the action vector is readable here instead of only in robot/allex_right.yaml. The
# finger ABADs are deliberately NOT action dims — they stay PD-held at their init pose (the hand grasps with
# MCP/PIP + thumb), which is what keeps the dim at 18. Order IS the action order.
# Sim2Real: per-env random command delay (post-EMA ring buffer, resampled at reset) — the real controller has
# command latency, so training with a 0-2 policy-step lag makes the policy robust to it.
class ACTION:
    # Command latency [policy steps], per-env, redrawn at reset — ONE controller link, so not per group.
    min_delay = 0
    max_delay = 2

    class arm:
        joints = [
            "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
            "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
        ]
        scale = 0.1
        ema_tau = 0.825   # [s]

    class hand:
        joints = [
            "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
            "R_Index_MCP_Joint", "R_Index_PIP_Joint",
            "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
            "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
            "R_Little_MCP_Joint", "R_Little_PIP_Joint",
        ]
        scale = 1.0
        ema_tau = 0.325   # [s]


# --- obs terms ---------------------------------------------------------------------------------------
class OBS:
    """Full privileged state — the teacher perceives everything (attribute name = term name)."""
    prev_action_targets    = Obs(obs.prev_action_targets)
    joint_pos              = Obs(obs.joint_positions, names="@joints.ctrl")
    joint_vel              = Obs(obs.joint_velocities, names="@joints.ctrl")
    arm_torque             = Obs(obs.joint_torque_obs, names="@joints.arm")
    hand_torque            = Obs(obs.joint_torque_obs, names="@joints.hand")
    palm_pose              = Obs(obs.palm_pose_in_chest, chest_body="@frames.chest_origin", palm_body="@frames.palm")
    hammer_pose            = Obs(obs.object_pose, chest_body="@frames.chest_origin")
    object_lin_vel         = Obs(obs.object_linear_velocity)
    object_ang_vel         = Obs(obs.object_angular_velocity)
    palm_lin_vel           = Obs(obs.body_linear_velocity, body="@frames.palm")
    palm_ang_vel           = Obs(obs.body_angular_velocity, body="@frames.palm")
    fingertip_contact      = Obs(obs.body_contact_flags, bodies="@bodies.fingertips")
    hand_object_force_mag  = Obs(obs.hand_object_force_magnitude, bodies="@bodies.fingertips")
    hand_object_force_vec  = Obs(obs.hand_contact_force, bodies="@bodies.fingertips")
    object_keypoints       = Obs(obs.object_keypoints, chest_body="@frames.palm")
    goal_keypoints         = Obs(obs.goal_keypoints, chest_body="@frames.palm")
    fingertip_rel_pos      = Obs(obs.fingertip_relative_pos, chest_body="@frames.palm", bodies="@bodies.fingertips")
    fingertip_rel_vel      = Obs(obs.fingertip_relative_vel, chest_body="@frames.palm", bodies="@bodies.fingertips")
    closest_keypoint_dist  = Obs(obs.closest_keypoint_max_dist, reward_term="object_goal_keypoint_progress")
    episode_step           = Obs(obs.episode_step)
    object_lifed          = Obs(obs.object_lifed, lift_threshold=0.95)
    instantaneous_reward   = Obs(obs.instantaneous_reward)
    task_success           = Obs(obs.object_goal_keypoint_success, tolerance=0.01)


# --- events (domain randomization) -------------------------------------------------------------------
class EVENTS:
    reset_object_pose  = Event(events.reset_object_pose, "reset",
                               active_position=[0.6, 0.0, 0.926],
                               x_range=[-0.1, 0.0], y_range=[-0.2, -0.1], yaw_range=[-0.2, 0.2])
    reset_joints_by_offset = Event(events.reset_joints_by_offset, "reset", joints="@joints.ctrl",
                               position_range=[-0.1, 0.1])
    object_friction    = Event(events.set_shape_friction, "reset", target="object", mu_range=[0.5, 0.5])
    # `exclude` keeps the nail shells at the mu the robot yaml pinned them to (robot/allex_right.yaml
    # `nail_friction`); drop it and they are randomized with the rest of the hand.
    robot_friction     = Event(events.set_shape_friction, "reset", target="robot",  mu_range=[1.5, 1.5],
                               exclude="@bodies.nail")
    table_friction     = Event(events.set_shape_friction, "reset", target="table",  mu_range=[0.5, 0.5])
    randomize_rigid_body_mass = Event(events.randomize_rigid_body_mass, "reset", scale_range=[0.5, 1.0])
    randomize_fixed_base_root_height = Event(events.randomize_fixed_base_root_height, "reset", z_offset_range=[-0.01, 0.01])
    apply_object_external_force_when_lifted = Event(events.apply_object_external_force_when_lifted, "interval",
                               z_range=[-5.0, -3.0], lift_threshold=0.9)


# --- termination / truncation ------------------------------------------------------------------------
class TERMINATE:
    object_below_height = Done(terminate.object_below_height, min_height=0.85)
    object_far_from_palm = Done(terminate.object_far_from_body, body="@frames.palm", max_distance=1.0)
    table_fingertip_contact_force_exceeded = Done(terminate.table_fingertip_contact_force_exceeded,
                                                 fingertips="@bodies.fingertips", force_threshold_n=100.0)
    object_velocity_exceeded = Done(terminate.object_velocity_exceeded, max_lin_vel=30.0, max_ang_vel=60.0)
    curriculum_passed        = Done(terminate.curriculum_passed, truncation=True)



# --- assembly ------------------------------------------------------------------------------------
def build_task(name: str, *, reward, gate, curriculum,
               action=None, events=None, terminate=None, objects=None) -> TaskSpec:
    """Assemble the full contract from this shared core + a recipe's tunables.

    Required trio — every recipe states them: ``reward`` (what pays), ``gate`` (what counts as solved),
    ``curriculum`` (how difficulty ramps). Optional overrides — ``None`` takes this base's default, so a
    recipe states only what differs: ``action`` / ``events`` (DR) / ``terminate`` blocks, and ``objects``
    (the scene's object list). Values written inline (sim-owned single SoT; nothing arrives over the wire,
    no trainer import). The result is a ready ``TASK: TaskSpec`` — read by every consumer: train server,
    standalone, parity."""
    scene = values(SCENE)
    if objects is not None:
        scene = {**scene, "objects": objects}
    return TaskSpec(
        name=name,
        num_envs=4096,
        env_spacing=1.5,
        episode_length_s=10.0,
        physics=PHYSICS,
        overrides=OVERRIDES,
        scene=scene,
        action=action if action is not None else ACTION,
        obs=OBS,
        obs_groups={"actor": "all", "privileged": "all"},
        obs_history_length={"actor": 3, "privileged": 3},
        reward=reward,
        events=events if events is not None else EVENTS,
        gate=gate,
        terminate=terminate if terminate is not None else TERMINATE,
        curriculum=curriculum,
    )
