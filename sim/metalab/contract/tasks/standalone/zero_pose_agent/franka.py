"""franka — zero-pose recipe: a single franka_panda on the ground.

panda0_joint4 sits at -25 deg because its MJCF range is [-176, -22.9] deg and 0 is outside it.
Authoring rules → sim/metalab/contract/tasks/README.md.
"""
from __future__ import annotations

from . import _base as base

INIT_POSE = {
    "panda0_joint1": 0.0,
    "panda0_joint2": 0.0,
    "panda0_joint3": 0.0,
    "panda0_joint4": -25.0,
    "panda0_joint5": 0.0,
    "panda0_joint6": 0.0,
    "panda0_joint7": 0.0,
    "panda0_finger_joint1": 0.0,
    "panda0_finger_joint2": 0.0,
}

TASK = base.build_task("franka", robot="franka", init_pose=INIT_POSE)
