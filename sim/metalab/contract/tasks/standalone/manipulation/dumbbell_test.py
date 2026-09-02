"""dumbbell_test — STANDALONE scene contract (no learning). Authoring rules → sim/metalab/contract/tasks/README.md.

Standalone-only smoke: ALLEX (full body, allex.yaml) at a hand-authored pose on a desk, with three
dumbbells (1 / 2 / 3 kg) placed on the desk. No reward / terminate / events / DR / goal / obs — the
Launchpad's Standalone mode steps the backend directly and resets only on the 'Reset Simulator' button.
Kept out of the train/eval task list (it lives under tasks/standalone/); NOT a training task.

The outlier of this group: a different robot (both arms), a deeper desk, a faster control rate and a
looser solver, so it replaces most of ``_base`` rather than inheriting it.
"""
from __future__ import annotations

from ... import _assets as assets
from . import _base as base


# --- robot: full-body ALLEX (both arms) — robot/allex/allex.yaml --------------------------------------------
class ROBOT:
    name = "allex"
    base_pos = [0.0, 0.0, 0.585]
    base_quat = [1.0, 0.0, 0.0, 0.0]   # wxyz
    fixed_base = True
    # Hand-authored standalone pose. DEGREES (user-facing; the loader converts to radians). All 48 ACTUATED
    # joints listed explicitly, used or not; the 12 equality FOLLOWERS (Pitch_Dummy/Upper, IP/DIP) are
    # AUTO-COMPUTED by the loader from the MJCF <equality> polycoef.
    init_pose = {
        # waist (2) + neck (2)
        "Waist_Yaw_Joint": 0.0, "Waist_Lower_Pitch_Joint": 0.0,
        "Neck_Pitch_Joint": 6.498, "Neck_Yaw_Joint": 0.394,
        # right arm (7)
        "R_Shoulder_Pitch_Joint": -6.564, "R_Shoulder_Roll_Joint": -26.962, "R_Shoulder_Yaw_Joint": 12.286,
        "R_Elbow_Joint": -55.357, "R_Wrist_Yaw_Joint": 131.716, "R_Wrist_Roll_Joint": -3.461,
        "R_Wrist_Pitch_Joint": 18.583,
        # left arm (7)
        "L_Shoulder_Pitch_Joint": -2.736, "L_Shoulder_Roll_Joint": 26.107, "L_Shoulder_Yaw_Joint": -16.242,
        "L_Elbow_Joint": -61.478, "L_Wrist_Yaw_Joint": -123.51, "L_Wrist_Roll_Joint": 2.855,
        "L_Wrist_Pitch_Joint": 25.431,
        # right hand (15 — IP/DIP auto-follow)
        "R_Thumb_Yaw_Joint": -30.049, "R_Thumb_CMC_Joint": 46.963, "R_Thumb_MCP_Joint": 29.502,
        "R_Index_ABAD_Joint": -5.007, "R_Index_MCP_Joint": 2.294, "R_Index_PIP_Joint": 5.22,
        "R_Middle_ABAD_Joint": -8.609, "R_Middle_MCP_Joint": 1.353, "R_Middle_PIP_Joint": 13.376,
        "R_Ring_ABAD_Joint": -11.647, "R_Ring_MCP_Joint": 0.682, "R_Ring_PIP_Joint": 20.017,
        "R_Little_ABAD_Joint": -18.642, "R_Little_MCP_Joint": 1.38, "R_Little_PIP_Joint": 10.357,
        # left hand (15 — IP/DIP auto-follow)
        "L_Thumb_Yaw_Joint": 46.589, "L_Thumb_CMC_Joint": 32.4, "L_Thumb_MCP_Joint": 26.495,
        "L_Index_ABAD_Joint": 5.718, "L_Index_MCP_Joint": 24.919, "L_Index_PIP_Joint": 5.946,
        "L_Middle_ABAD_Joint": 11.427, "L_Middle_MCP_Joint": 16.539, "L_Middle_PIP_Joint": 9.588,
        "L_Ring_ABAD_Joint": 14.584, "L_Ring_MCP_Joint": 10.385, "L_Ring_PIP_Joint": 9.533,
        "L_Little_ABAD_Joint": 18.391, "L_Little_MCP_Joint": -4.383, "L_Little_PIP_Joint": 15.845,
    }


# --- per-engine corrections: replaces _base.NEWTON wholesale (an unset key must STAY unset) -----------
# Newton-native collision pipeline (not mjwarp's): mjwarp GJK face-face contact can degenerate to a
# single point after a perturbation (tilt+penetration) and never recover — measured 2026-07-16 on
# physics_experiment_1 (1 contact/1.7mm sink/yaw-walk vs native 4/0.013mm).
class NEWTON:
    use_mujoco_contacts = True
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


TASK = base.build_task(
    "dumbbell_test",
    objects=[
        # dumbbells (usd_to_mjcf assets, served from the bucket); mass/friction/pose authored here
        {"name": "dumbbell_1kg", "asset": {"mjcf": [assets.object_mjcf("dumbbell_1kg")]},
         "mass": 1.0, "init_pos": [0.65, -0.18, 1.0]},
        {"name": "dumbbell_2kg", "asset": {"mjcf": [assets.object_mjcf("dumbbell_2kg")]},
         "mass": 2.0, "init_pos": [0.65, -0.01, 1.0]},
        {"name": "dumbbell_3kg", "asset": {"mjcf": [assets.object_mjcf("dumbbell_3kg")]},
         "mass": 3.0, "init_pos": [0.65, 0.145, 1.01]},
    ],
    physics={"hz": 200},
    newton=NEWTON,
    robot=ROBOT,
    desk={"parts": [{"shape": "box", "size": [0.6, 1.0, 0.2]}]},
    contact={"robot": {"solref": [0.01, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "solmix": 1.0},
             "table": {"solref": [0.01, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0], "solmix": 2.0}},
    camera=None,
    robot_friction=0.5,                # build-time robot collision μ (absolute); MJCF default = 1.0
)
