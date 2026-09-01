"""allex — zero-pose recipe: the full-body ALLEX (both arms) on its 0.6 m mount, all joints at 0.

Authoring rules → sim/metalab/contract/tasks/README.md.
"""
from __future__ import annotations

from . import _base as base

INIT_POSE = {
    "Waist_Yaw_Joint": 0.0, "Waist_Lower_Pitch_Joint": 0.0,
    "Neck_Pitch_Joint": 0.0, "Neck_Yaw_Joint": 0.0,
    "R_Shoulder_Pitch_Joint": 0.0, "R_Shoulder_Roll_Joint": 0.0, "R_Shoulder_Yaw_Joint": 0.0,
    "R_Elbow_Joint": 0.0, "R_Wrist_Yaw_Joint": 0.0, "R_Wrist_Roll_Joint": 0.0, "R_Wrist_Pitch_Joint": 0.0,
    "L_Shoulder_Pitch_Joint": 0.0, "L_Shoulder_Roll_Joint": 0.0, "L_Shoulder_Yaw_Joint": 0.0,
    "L_Elbow_Joint": 0.0, "L_Wrist_Yaw_Joint": 0.0, "L_Wrist_Roll_Joint": 0.0, "L_Wrist_Pitch_Joint": 0.0,
    "R_Thumb_Yaw_Joint": 0.0, "R_Thumb_CMC_Joint": 0.0, "R_Thumb_MCP_Joint": 0.0,
    "R_Index_ABAD_Joint": 0.0, "R_Index_MCP_Joint": 0.0, "R_Index_PIP_Joint": 0.0,
    "R_Middle_ABAD_Joint": 0.0, "R_Middle_MCP_Joint": 0.0, "R_Middle_PIP_Joint": 0.0,
    "R_Ring_ABAD_Joint": 0.0, "R_Ring_MCP_Joint": 0.0, "R_Ring_PIP_Joint": 0.0,
    "R_Little_ABAD_Joint": 0.0, "R_Little_MCP_Joint": 0.0, "R_Little_PIP_Joint": 0.0,
    "L_Thumb_Yaw_Joint": 0.0, "L_Thumb_CMC_Joint": 0.0, "L_Thumb_MCP_Joint": 0.0,
    "L_Index_ABAD_Joint": 0.0, "L_Index_MCP_Joint": 0.0, "L_Index_PIP_Joint": 0.0,
    "L_Middle_ABAD_Joint": 0.0, "L_Middle_MCP_Joint": 0.0, "L_Middle_PIP_Joint": 0.0,
    "L_Ring_ABAD_Joint": 0.0, "L_Ring_MCP_Joint": 0.0, "L_Ring_PIP_Joint": 0.0,
    "L_Little_ABAD_Joint": 0.0, "L_Little_MCP_Joint": 0.0, "L_Little_PIP_Joint": 0.0,
}

TASK = base.build_task("allex", robot="allex", init_pose=INIT_POSE, base_pos=[0.0, 0.0, 0.6])
