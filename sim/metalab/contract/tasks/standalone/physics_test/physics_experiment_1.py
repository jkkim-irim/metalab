"""physics_experiment_1 — STANDALONE scene contract (no learning).

Spawns the robot, a table, and a 10x6.5x4.5 cm box object, holding a hand-authored init pose
(right arm reaching, fingers slightly curled). Authoring rules → sim/metalab/contract/tasks/README.md.
"""
from __future__ import annotations

from sim.metalab.contract.spec import TaskSpec

# --- physics / solver --------------------------------------------------------------------------------
PHYSICS = {"hz": 120, "substeps": 2, "decimation": 2}

# --- per-engine corrections (NOT shared knobs) -------------------------------------------------------
# newton physics fidelity (robot/table contacts + finger-equality stiffness). genesis ignores its block.
OVERRIDES = {
    "newton": {
        "use_mujoco_contacts": True,
        "cone": "pyramidal",
        "impratio": 1.0,
        "iterations": 100,
        "ls_iterations": 30,
        "tolerance": 1.0e-6,
        "ls_tolerance": 1.0e-6,
        "nconmax": 256,     # contact-point buffer, per-world
        "njmax": 400,       # constraint-row (nefc) buffer, per-world — set independently of nconmax
        "ccd_iterations": 35,
        "eq_solref": [0.01, 1.2],
        "eq_solimp": [0.9999, 0.9999, 0.001, 0.5, 1.0],
    },
}

# --- the world: ground, robot(+placement/init pose), objects, contact params, camera, goal -----------
SCENE = {
    "ground": True,   # infinite ground plane at z=0
    "robot": {
        "name": "allex_right",
        "base_pos": [0.0, 0.0, 0.6],
        "base_quat": [1.0, 0.0, 0.0, 0.0],   # wxyz
        "fixed_base": True,
        # DEGREES (user-facing; the loader converts to radians). All 48 ACTUATED joints listed explicitly,
        # used or not; only the right elbow (-100°) + right fingers + waist pitch are posed, rest 0. The 12
        # equality FOLLOWERS (Pitch_Dummy/Upper, IP/DIP) are AUTO-COMPUTED by the loader from the MJCF polycoef
        # — waist 60° here gets its dummy/upper coupled angles for free.
        "init_pose": {
        # waist (2) + neck (2)
        "Waist_Yaw_Joint": 0.0, "Waist_Lower_Pitch_Joint": 0.0,
        "Neck_Pitch_Joint": 0.0, "Neck_Yaw_Joint": 0.0,
        # right arm (7)
        "R_Shoulder_Pitch_Joint": 0.0, "R_Shoulder_Roll_Joint": 0.0, "R_Shoulder_Yaw_Joint": 0.0,
        "R_Elbow_Joint": -105.0, "R_Wrist_Yaw_Joint": 180.0, "R_Wrist_Roll_Joint": 0.0, "R_Wrist_Pitch_Joint": -0.0,
        # left arm (7)
        "L_Shoulder_Pitch_Joint": 0.0, "L_Shoulder_Roll_Joint": 0.0, "L_Shoulder_Yaw_Joint": 0.0,
        "L_Elbow_Joint": 0.0, "L_Wrist_Yaw_Joint": 0.0, "L_Wrist_Roll_Joint": 0.0, "L_Wrist_Pitch_Joint": 0.0,
        # right hand (15 — IP/DIP auto-follow)
        "R_Thumb_Yaw_Joint": -70.0, "R_Thumb_CMC_Joint": 10.0, "R_Thumb_MCP_Joint": 15.0,
        "R_Index_ABAD_Joint": 0.0, "R_Index_MCP_Joint": 40.0, "R_Index_PIP_Joint": 15.0,
        "R_Middle_ABAD_Joint": 0.0, "R_Middle_MCP_Joint": 90.0, "R_Middle_PIP_Joint": 90.0,
        "R_Ring_ABAD_Joint": 0.0, "R_Ring_MCP_Joint": 90.0, "R_Ring_PIP_Joint": 90.0,
        "R_Little_ABAD_Joint": 0.0, "R_Little_MCP_Joint": 90.0, "R_Little_PIP_Joint": 90.0,
        # left hand (15 — IP/DIP auto-follow)
        "L_Thumb_Yaw_Joint": 0.0, "L_Thumb_CMC_Joint": 0.0, "L_Thumb_MCP_Joint": 0.0,
        "L_Index_ABAD_Joint": 0.0, "L_Index_MCP_Joint": 0.0, "L_Index_PIP_Joint": 0.0,
        "L_Middle_ABAD_Joint": 0.0, "L_Middle_MCP_Joint": 0.0, "L_Middle_PIP_Joint": 0.0,
        "L_Ring_ABAD_Joint": 0.0, "L_Ring_MCP_Joint": 0.0, "L_Ring_PIP_Joint": 0.0,
        "L_Little_ABAD_Joint": 0.0, "L_Little_MCP_Joint": 0.0, "L_Little_PIP_Joint": 0.0,
        },
    },
    # objects=[{                             # 10x6.5x4.5cm box (box.xml); mass/friction/pose authored here
    #     "name": "box",
    #     "asset": {"mjcf": [_assets.s3_mjcf("box")]},   # from .. import _assets
    #     "mass": 0.1377, "friction": 0.5, "restitution": 0.0,
    #     "init_pos": [0.6, -0.16, 0.855],    # on the table (top ≈ 0.86 + box half-height 0.0225)
    #     "init_rpy": [0.0, 90.0, 90.0],      # roll, pitch, yaw [deg] (or init_quat wxyz; either/or)
    # }],
    "objects": [
        {   # table — scenery: welded, size stays a task knob (procedural, no asset)
            "name": "table", "fixed": True, "mass": 1.0,
            "parts": [{"shape": "box", "size": [0.4, 0.8, 0.1]}],
            "init_pos": [0.65, -0.15, 0.1],
        },
    ],
    # Contact softness per group (shared knob — genesis implements the same solref/solimp math;
    # solmix is newton-only, genesis averages the two geoms 0.5/0.5 and cannot weight them).
    "contact_params": {
        "robot": {"solref": [0.01, 1.0], "solimp": [0.9, 0.99, 0.001, 0.5, 2.0], "solmix": 1.0},
        "table": {"solref": [0.01, 1.0], "solimp": [0.9, 0.99, 0.001, 0.5, 2.0], "solmix": 1.0},
        # "object": {"solref": [0.01, 1.0], "solimp": [0.9, 0.99, 0.001, 0.5, 2.0], "solmix": 1.0},
    },
    "robot_friction": 0.5,                # build-time robot collision μ (absolute); MJCF default = 1.0
}

# --- assembly ------------------------------------------------------------------------------------
# No `action`: the loader then takes the action groups the robot's YAML declares. Authoring a block here
# would pin nothing that standalone reads — a group's scale / ema_tau / delay are consumed by
# EnvDriver.step alone, and this runner writes backend.set_joint_targets directly.
TASK = TaskSpec(
    name="physics_experiment_1",
    num_envs=1,                        # standalone (the runner forces 1 via build_env anyway)
    physics=PHYSICS,
    overrides=OVERRIDES,
    scene=SCENE,
)
