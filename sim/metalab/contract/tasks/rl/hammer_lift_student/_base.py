"""hammer_lift_student — SHARED CORE (physics / robot / scene / obs split / defaults for the rest).

The manipulation problem: an ASYMMETRIC actor-critic lifts a hammer. Everything here is the part every
recipe shares; a recipe module (a sibling ``hammer_lift_student_<recipe>.py`` in this family folder)
supplies the tunable trio (``reward`` / ``gate`` / ``curriculum``) — plus any of the optional overrides
(``action`` / ``events`` / ``terminate`` / ``objects``) where it differs — and calls :func:`build_task`.
Splitting it this way keeps every recipe on ONE core, so a fix to obs / scene / sim lands for all of them
at once instead of drifting per copy.

The leading underscore keeps this out of the task list (it is a library, not a task): the launchpad skips
``_*.py``, and the loader only imports the module a ``--task`` name asks for.

Observation split (SIM-OWNED, every knob inline — team decision 2026-07-23):
* **actor** (history_length=3): a compact proprioceptive set the real robot could produce — prev joint targets
  (post-EMA action), joint pos, joint torque, the object pose as last SEEN (chest-origin frame; the curriculum
  shrinks that window from the whole episode down to the spawn frame), and the palm pose (chest-origin frame).
  No realtime object perception by the end of the curriculum → deployable.
* **critic** (no history): everything — a CLEAN (noise-free) re-read of every actor term, then the full
  teacher state PLUS the realtime object pose (chest frame) and palm-frame fingertip poses (pos+quat). The two groups
  share no term name, so the console can plot the actor's corrupted copy and the critic's clean one side by
  side. The EXP wires actor<-"actor", critic<-"critic".

Authoring rules → sim/metalab/contract/tasks/README.md.
"""
from __future__ import annotations

import math

from sim.metalab.contract.spec import Done, Event, Obs, ObsNoise, TaskSpec, values
from sim.metalab.terms import events, obs, terminate

from ... import _assets as assets

# The posture the robot must end in — commanded joints only (ABAD is PD-held at 0 and not commandable, so
# a pose bound on it could not be recovered from). Authored in deg, consumed in rad; ONE variable, handed to
# both the GATE and the reward term that shapes toward it.
_JOINT_FINAL_POSE_DEG = {
    "R_Thumb_Yaw_Joint": -45.0, "R_Thumb_CMC_Joint": 40.0, "R_Thumb_MCP_Joint": 80.0,
    "R_Index_MCP_Joint": 80.0, "R_Index_PIP_Joint": 75.0,
    "R_Middle_MCP_Joint": 80.0, "R_Middle_PIP_Joint": 75.0,
    "R_Ring_MCP_Joint": 80.0, "R_Ring_PIP_Joint": 75.0,
    "R_Little_MCP_Joint": 80.0, "R_Little_PIP_Joint": 75.0,
}
_JOINT_FINAL_POSE = {j: math.radians(v) for j, v in _JOINT_FINAL_POSE_DEG.items()}

_LIFT_BAR = 0.875            # [m] world z of the grasp point that counts as OFF THE TABLE. Measured resting
                             # grasp z over 12288 spawns (newton, DR on): p50 0.8637, p99 0.8666, max 0.8710
                             # — so this bar leaks to <0.1% of envs, and 0.865 would pay 27% of them at rest.
_LIFT_TOP = 0.95             # [m] world z where the lift phase hands the signal to the carry terms

_NO_TOUCH_LINKS = ["R_Forearm_Link", "R_Upper_Arm_Link",
                   "Waist_Upper_Pitch_Link", "Waist_Lower_Pitch_Link"]
_NO_TOUCH_FORCE = 0.1         # [N] net force on a no-touch link that ends the episode

# --- obs, declared as the two sets the asymmetric actor-critic actually consumes -----------------------
_NOISE_ARM_JOINT_POS = 0.1    # [rad]   arm joint encoder  
_NOISE_HAND_JOINT_POS = 0.1   # [rad]   hand joint encoder 
_NOISE_ARM_TORQUE = 5.0       # [N·m]   arm PD torque read
_NOISE_HAND_TORQUE = 0.05     # [N·m]   hand PD torque read
_NOISE_PALM_POS = 0.01        # [m]     palm FK position (= legacy OBS_NOISE_HAND_REL_POS_STD)
_NOISE_PALM_ROT = 0.02        # [rad]   palm FK orientation, angular magnitude (≈ 1.15 deg; no legacy counterpart)
_NOISE_OBJ_INIT_POS = 0.01    # [m]     where the hammer was seen at episode start
_NOISE_OBJ_INIT_ROT = 0.02    # [rad]   ditto, angular magnitude

# --- physics / solver --------------------------------------------------------------------------------
class PHYSICS:
    hz = 120            # control rate — physics dt = 1/hz
    substeps = 4        # physics substeps per control step
    decimation = 1      # physics steps per POLICY step

# --- per-engine corrections (NOT shared knobs) -------------------------------------------------------
class OVERRIDES:
    class newton:
        use_mujoco_contacts = True    # False = newton-native collide(); True = mjwarp broad/narrowphase
        enable_multiccd = True        # up to 4 contact points per geom pair (newton default: 1)
        integrator = "implicitfast"   # euler | rk4 | implicit | implicitfast (= implicit minus Coriolis)
        solver = "newton"         # constraint solver: "newton" | "cg". PGS is NOT supported by mjwarp.
        cone = "pyramidal"        # friction cone: pyramidal (cheap) | elliptic (isotropic, needs impratio 100)
        impratio = 1.0            # frictional-to-normal impedance ratio (MuJoCo default 1.0)
        iterations = 100          # solver iterations (MuJoCo default 100)
        ls_iterations = 50        # line-search iterations (MuJoCo default 50)
        tolerance = 1.0e-8        # solver early-exit tolerance (MuJoCo default 1e-8)
        ls_tolerance = 0.01     # line-search tolerance (MuJoCo default 0.01)
        jacobian = "auto"       # "dense" | "sparse" | "auto" (auto = sparse iff nv>32; allex nv=33)
        nconmax = 128             # contact slots PER WORLD in the mjwarp buffer
        njmax = 256
        eq_solref = [0.01, 1.1]
        eq_solimp = [0.9999, 0.9999, 0.001, 0.5, 1.0]

class ACTOR_OBS:
    """Real-robot producible, history-stacked (H=3). Attribute name = term name."""
    object_init_pose    = Obs(obs.object_pose, chest_body="@frames.chest_origin", seen=True, noise=ObsNoise(pos=_NOISE_OBJ_INIT_POS, rot=_NOISE_OBJ_INIT_ROT))
    palm_pose           = Obs(obs.palm_pose_in_chest, chest_body="@frames.chest_origin", palm_body="@frames.palm", noise=ObsNoise(pos=_NOISE_PALM_POS, rot=_NOISE_PALM_ROT))
    prev_action_targets = Obs(obs.prev_action_targets, unit="rad")
    arm_joint_pos       = Obs(obs.joint_positions, unit="rad", names="@joints.arm",  noise=ObsNoise(std=_NOISE_ARM_JOINT_POS))
    hand_joint_pos      = Obs(obs.joint_positions, unit="rad", names="@joints.hand_all", noise=ObsNoise(std=_NOISE_HAND_JOINT_POS))
    arm_torque          = Obs(obs.joint_torque_obs, names="@joints.arm", unit="N*m", noise=ObsNoise(std=_NOISE_ARM_TORQUE))
    hand_torque         = Obs(obs.joint_torque_obs, names="@joints.hand_all", unit="N*m", noise=ObsNoise(std=_NOISE_HAND_TORQUE))

class CRITIC_OBS:

    hammer_pose               = Obs(obs.object_pose, chest_body="@frames.chest_origin")
    palm_pose_clean           = Obs(obs.palm_pose_in_chest, chest_body="@frames.chest_origin", palm_body="@frames.palm")
    prev_action_targets_clean = Obs(obs.prev_action_targets, unit="rad")

    joint_pos                 = Obs(obs.joint_positions, unit="rad", names="@joints.ctrl")
    joint_torque              = Obs(obs.joint_torque_obs, names="@joints.ctrl", unit="N*m")
    joint_vel                 = Obs(obs.joint_velocities, names="@joints.ctrl", unit="rad/s")
    joint_acc                 = Obs(obs.joint_accelerations, names="@joints.ctrl", unit="rad/s^2")    

    object_lin_vel            = Obs(obs.object_linear_velocity, unit="m/s")
    object_ang_vel            = Obs(obs.object_angular_velocity, unit="rad/s")
    palm_lin_vel              = Obs(obs.body_linear_velocity, body="@frames.palm", unit="m/s")
    palm_ang_vel              = Obs(obs.body_angular_velocity, body="@frames.palm", unit="rad/s")

    fingertip_rel_pose        = Obs(obs.fingertip_relative_pose, ref_body="@frames.palm", bodies="@bodies.fingertips")
    fingertip_rel_vel         = Obs(obs.fingertip_relative_vel, chest_body="@frames.palm", bodies="@bodies.fingertips")
    fingertip_contact_steps   = Obs(obs.fingertip_contact_steps, unit="steps", target="object", force_threshold=0.1)
    hand_object_force_vec     = Obs(obs.hand_contact_force, bodies="@bodies.fingertips", ref_body="self")
    hand_table_force_mag      = Obs(obs.hand_object_force_magnitude, bodies="@bodies.fingertips", target="table", unit="N")
    hand_object_penetration   = Obs(obs.fingertip_penetration_depth, bodies="@bodies.fingertips", target="object", unit="mm")
    hand_table_penetration    = Obs(obs.fingertip_penetration_depth, bodies="@bodies.fingertips", target="table", unit="mm")

    no_touch_link_contact     = Obs(obs.body_contact_flags, bodies=_NO_TOUCH_LINKS, threshold=_NO_TOUCH_FORCE)
    episode_step              = Obs(obs.episode_step)
    action_delay              = Obs(obs.action_delay)
    dr_params                 = Obs(obs.dr_params,
                                    keys=["object_friction", "robot_friction", "table_friction",
                                          "object_mass_scale", "object_scale", "root_height"],
                                    labels=["mu_object", "mu_robot", "mu_table", "mass_scale", "obj_scale",
                                            "root_dz"])
    lifted                    = Obs(obs.object_lifed, lift_threshold=_LIFT_TOP)
    instantaneous_reward      = Obs(obs.instantaneous_reward)

    curriculum_level          = Obs(obs.curriculum_state, keys=["level"])
    goal_dist_err             = Obs(obs.goal_dist_error, tolerance_key="goal_dist_tol", unit="m",
                                    labels=["error", "tolerance"])
    hold_progress             = Obs(obs.curriculum_hold_progress, required_key="hold_steps", unit="steps",
                                    labels=["held", "required"])
    palm_dist_err             = Obs(obs.palm_distance_error, palm_body="@frames.palm",
                                    tolerance_key="palm_distance", unit="m",
                                    labels=["error", "tolerance"])

# --- the world: ground, robot(+placement/init pose), objects, contact params, camera, goal -----------
class SCENE:
    ground = True     # infinite ground plane at z=0

    class robot:
        name = "allex_right"
        base_pos = [0.0, 0.0, 0.6]
        base_quat = [1.0, 0.0, 0.0, 0.0]   # wxyz
        fixed_base = True
        # DEGREES (the loader converts to radians). A joint-name → angle TABLE, so it stays a dict: ACTIVE
        # joints get the runtime spawn pose, mask-0 joints get their weld angle. The 12 equality FOLLOWERS
        # (waist dummy/upper, finger IP/DIP) are auto-computed by the loader from the MJCF <equality>.
        init_pose = {
            # waist + neck (4 — mask-0 on allex_right → welded at these angles)
            "Waist_Yaw_Joint": 0.0, "Waist_Lower_Pitch_Joint": 30.0,
            "Neck_Pitch_Joint": 40.0, "Neck_Yaw_Joint": 0.0,
            # right arm (7 — active)
            "R_Shoulder_Pitch_Joint": 38.28, "R_Shoulder_Roll_Joint": -25.0, "R_Shoulder_Yaw_Joint": -20.0,
            "R_Elbow_Joint": -125.5, "R_Wrist_Yaw_Joint": 180.0, "R_Wrist_Roll_Joint": 0.0,
            "R_Wrist_Pitch_Joint": 5.0,
            # right hand (15 — active; IP/DIP auto-follow)
            "R_Thumb_Yaw_Joint": 0.0, "R_Thumb_CMC_Joint": 10.0, "R_Thumb_MCP_Joint": 20.0,
            "R_Index_ABAD_Joint": 0.0, "R_Index_MCP_Joint": 10.0, "R_Index_PIP_Joint": 20.0,
            "R_Middle_ABAD_Joint": 0.0, "R_Middle_MCP_Joint": 10.0, "R_Middle_PIP_Joint": 20.0,
            "R_Ring_ABAD_Joint": 0.0, "R_Ring_MCP_Joint": 10.0, "R_Ring_PIP_Joint": 20.0,
            "R_Little_ABAD_Joint": 0.0, "R_Little_MCP_Joint": 10.0, "R_Little_PIP_Joint": 20.0,
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
        {   # hammer — the YCB 048_hammer scan re-framed onto the retired hammer_ycb.xml's convention
            # (origin = grasp point, handle on +y), so these numbers carry over unchanged. Fetched from
            # S3; the name must PREFIX the MJCF model, which newton uses to group contact params.
            # SIZE randomized per reset (randomize_object_scale).
            "name": "ycb_048_hammer_grasp_offset",
            "asset": {"mjcf": assets.object_mjcf("ycb_048_hammer_grasp_offset")},
            "mass": 0.55, "variants": 1, "init_pos": [0.7, -0.2, 0.874],
        },
        {   # table — scenery: welded, size stays a task knob (procedural, no asset)
            "name": "table", "fixed": True, "mass": 1.0,
            "parts": [{"shape": "box", "size": [0.4, 0.8, 0.1]}],
            "init_pos": [0.7, 0.0, 0.8],
        },
    ]

    class contact_params:            # per group — genesis implements the same solref/solimp math
        robot = {"solref": [0.01, 1.0], "solimp": [0.9, 0.9999, 0.001, 0.5, 2.0], "solmix": 2.0}
        table = {"solref": [0.01, 1.0], "solimp": [0.9, 0.99, 0.001, 0.5, 2.0], "solmix": 1.0}
        ycb_048_hammer_grasp_offset = {"solref": [0.01, 1.0], "solimp": [0.9, 0.99, 0.001, 0.5, 2.0], "solmix": 1.0}

    class camera:
        eye = [1.5, -1.1, 1.5]
        lookat = [0.4, -0.05, 1.0]
        fov = 30.0

    class goal:
        pos = [0.7, -0.3, 1.0]
        quat = [1.0, 0.0, 0.0, 0.0]
        keypoint_half_extent = [0.05, 0.05, 0.08]

# --- action groups -----------------------------------------------------------------------------------
class ACTION:
    min_delay = 0
    max_delay = 4

    class arm:
        joints = [
            "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
            "R_Elbow_Joint", "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
        ]
        scale = 0.1
        ema_tau = 3.0 #4.0=PPO #1.5=SAPG

    class hand:
        joints = [
            "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
            "R_Index_MCP_Joint", "R_Index_PIP_Joint",
            "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
            "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
            "R_Little_MCP_Joint", "R_Little_PIP_Joint",
        ]
        scale = 1.0
        ema_tau = 1.0   # [s]

# --- events (domain randomization) -------------------------------------------------------------------
class EVENTS:
    reset_object_pose  = Event(events.reset_object_pose, "reset", active_position=[0.7, -0.2, 0.874], x_range=[-0.1, 0.1], y_range=[-0.1, 0.1], yaw_range=[-0.5, 0.5])
    reset_joints_by_offset = Event(events.reset_joints_by_offset, "reset", joints="@joints.ctrl",position_range=[-0.1, 0.1])
    object_friction    = Event(events.set_shape_friction, "reset", target="object", mu_range=[0.3, 0.5])
    robot_friction     = Event(events.set_shape_friction, "reset", target="robot",  mu_range=[0.5, 1.0],
                               exclude="@bodies.nail")
    table_friction     = Event(events.set_shape_friction, "reset", target="table",  mu_range=[0.2, 0.4])
    randomize_rigid_body_mass = Event(events.randomize_rigid_body_mass, "reset", scale_range=[0.8, 1.2])
    randomize_object_scale = Event(events.randomize_object_scale, "reset", scale_range=[0.8, 1.2], requires="object_scale")
    randomize_fixed_base_root_height = Event(events.randomize_fixed_base_root_height, "reset", z_offset_range=[-0.05, 0.05])
    object_pull_down   = Event(events.apply_object_external_force_when_lifted, "interval",
                               z_range=[-5.0, -5.0], lift_threshold=0.0, z_max=_LIFT_BAR,
                               interval_range_s=[0.0, 0.0])

# --- termination / truncation ------------------------------------------------------------------------
class TERMINATE:
    object_below_height = Done(terminate.object_below_height, min_height=0.85)
    object_far_from_palm = Done(terminate.object_far_from_body, body="@frames.palm", max_distance=1.0)
    table_fingertip_contact_force_exceeded = Done(terminate.table_fingertip_contact_force_exceeded,
                                                 fingertips="@bodies.fingertips", force_threshold_n=50.0)
    object_velocity_exceeded = Done(terminate.object_velocity_exceeded, max_lin_vel=5.0, max_ang_vel=40.0)
    unexpected_contact       = Done(terminate.body_contact_detected, bodies=_NO_TOUCH_LINKS,
                                    force_threshold=_NO_TOUCH_FORCE)
    curriculum_passed        = Done(terminate.curriculum_passed, truncation=True)

# --- assembly ----------------------------------------------------------------------------------------
def build_task(name: str, *, reward, gate, curriculum,
               action=None, events=None, terminate=None, objects=None, contact=None, goal=None, robot=None,
               obs_groups=None, episode_length_s=None, overrides=None, physics=None) -> TaskSpec:
    """Assemble the full contract from this shared core + a recipe's tunables.

    Required trio — every recipe states them: ``reward`` (what pays), ``gate`` (what counts as solved),
    ``curriculum`` (how difficulty ramps; an EMPTY block = no curriculum, which an eval recipe wants since
    every bar it judges at is fixed). Optional overrides — ``None`` takes this base's default, so a recipe
    states only what differs: ``action`` / ``events`` (DR) / ``terminate`` blocks, ``objects`` (the scene's
    object list, e.g. a single-YCB recipe swaps the hammer entry), ``contact`` (the ``contact_params`` block —
    the hammer's key IS the object's name, so a recipe that renames the object restates it here), ``goal``
    (the goal block, whose
    ``goal_dist_tol`` is the bar the reach bonus judges at when no curriculum overwrites it), ``robot`` (the
    robot YAML NAME only — placement and init pose stay this core's, so a recipe can try another hand build
    without a second copy of everything around it),
    ``obs_groups`` (an eval recipe runs the deployed actor alone, so it drops the critic's privileged set;
    the history lengths follow the groups given), ``episode_length_s`` (the episode horizon in seconds), and
    ``overrides`` (per-engine solver corrections, MERGED per engine onto this core's block rather than
    replacing it — a recipe states only the knobs it flips, so later tuning of the rest still reaches it), and
    ``physics`` (MERGED onto PHYSICS the same way — engine-AGNOSTIC solver knobs, so a recipe that flips one
    flips it for every backend).
    Every shared value lives in ONE named section above, found
    by what it configures; the result is a ready ``TASK: TaskSpec`` — read by every consumer: train server,
    standalone, parity."""
    scene = values(SCENE)
    if objects is not None:
        scene = {**scene, "objects": objects}
    if contact is not None:
        scene = {**scene, "contact_params": values(contact)}
    if goal is not None:
        scene = {**scene, "goal": goal}
    if robot is not None:
        scene = {**scene, "robot": {**scene["robot"], "name": robot}}
    ov = values(OVERRIDES)
    for engine, knobs in values(overrides or {}).items():
        ov = {**ov, engine: {**ov.get(engine, {}), **knobs}}
    phys = {**values(PHYSICS), **values(physics or {})}
    groups = {"actor": ACTOR_OBS, "critic": CRITIC_OBS} if obs_groups is None else obs_groups
    return TaskSpec(
        name=name,
        num_envs=4096,
        env_spacing=1.5,
        episode_length_s=10.0 if episode_length_s is None else episode_length_s,
        physics=phys,
        overrides=ov,
        scene=scene,
        action=action if action is not None else ACTION,
        obs_groups=groups,
        obs_noise_groups=["actor"],   # only the actor's copies carry noise= at all
        obs_history_length={k: v for k, v in {"actor": 3, "critic": 1}.items() if k in groups},
        reward=reward,
        events=events if events is not None else EVENTS,
        gate=gate,
        terminate=terminate if terminate is not None else TERMINATE,
        curriculum=curriculum,
    )
