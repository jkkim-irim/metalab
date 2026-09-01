"""go2 — zero-pose recipe: a Unitree Go2 spawned free-floating at 0.45 m and dropped onto the ground.

The calf (knee) joints sit at -48 deg because their MJCF range is [-156, -48] deg and 0 is outside it.
Authoring rules → sim/metalab/contract/tasks/README.md.
"""
from __future__ import annotations

from . import _base as base

INIT_POSE = {
    "FL_hip_joint": 0.0, "FL_thigh_joint": 0.0, "FL_calf_joint": -48.0,
    "FR_hip_joint": 0.0, "FR_thigh_joint": 0.0, "FR_calf_joint": -48.0,
    "RL_hip_joint": 0.0, "RL_thigh_joint": 0.0, "RL_calf_joint": -48.0,
    "RR_hip_joint": 0.0, "RR_thigh_joint": 0.0, "RR_calf_joint": -48.0,
}

TASK = base.build_task("go2", robot="go2", init_pose=INIT_POSE,
                       base_pos=[0.0, 0.0, 0.45], fixed_base=False)
