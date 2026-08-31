"""CSV basename (without ``.csv``) → ordered list of ALLEX joint names.

Each CSV holds columns ``duration, joint_1, ..., joint_N``; the k-th joint column
maps to the k-th name below. Names follow the MJCF convention
(``sim/metalab/assets/robots/proto_v4/mjcf/ALLEX.xml``) and are resolved against the loaded
robot's active joints at build time (``CsvTrajectory`` intersects this map with
``spec.robot.active_joints()``, so a right-only robot silently drops the left/waist/neck
groups it does not own).

The 48 names here are exactly the ALLEX **actuated** joints. Equality followers
(``*_IP_Joint``, ``*_DIP_Joint``, ``Waist_Upper_Pitch_Joint``, ``Waist_Pitch_Dummy_Joint``)
are intentionally omitted — they are driven by their master joints via the MJCF
``<equality>`` coupling, not by the CSV.

CSV basenames follow the current robot export convention (``left_arm``, ``right_index``,
``waist``, …); the legacy names (``Arm_L_theOne``, ``Hand_R_index_wir``, ``theOne_waist``, …)
are no longer recognized — rename old group dirs' files to play them.
"""
from __future__ import annotations

ALLEX_CSV_JOINT_NAMES: dict[str, list[str]] = {
    "waist": ["Waist_Yaw_Joint", "Waist_Lower_Pitch_Joint"],
    "neck": ["Neck_Pitch_Joint", "Neck_Yaw_Joint"],
    "left_arm": [
        "L_Shoulder_Pitch_Joint", "L_Shoulder_Roll_Joint", "L_Shoulder_Yaw_Joint",
        "L_Elbow_Joint",
        "L_Wrist_Yaw_Joint", "L_Wrist_Roll_Joint", "L_Wrist_Pitch_Joint",
    ],
    "right_arm": [
        "R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint",
        "R_Elbow_Joint",
        "R_Wrist_Yaw_Joint", "R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint",
    ],
    "left_thumb":   ["L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint"],
    "left_index":   ["L_Index_ABAD_Joint", "L_Index_MCP_Joint", "L_Index_PIP_Joint"],
    "left_middle":  ["L_Middle_ABAD_Joint", "L_Middle_MCP_Joint", "L_Middle_PIP_Joint"],
    "left_ring":    ["L_Ring_ABAD_Joint", "L_Ring_MCP_Joint", "L_Ring_PIP_Joint"],
    "left_little":  ["L_Little_ABAD_Joint", "L_Little_MCP_Joint", "L_Little_PIP_Joint"],
    "right_thumb":  ["R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint"],
    "right_index":  ["R_Index_ABAD_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint"],
    "right_middle": ["R_Middle_ABAD_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint"],
    "right_ring":   ["R_Ring_ABAD_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint"],
    "right_little": ["R_Little_ABAD_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint"],
}
