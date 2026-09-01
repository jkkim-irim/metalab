"""bi_franka — zero-pose recipe: a bi-franka_panda (torso + two arms) on the ground.

panda0/panda1_joint4 sit at -25 deg because their MJCF range is [-176, -22.9] deg and 0 is outside it.
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
    "panda1_joint1": 0.0,
    "panda1_joint2": 0.0,
    "panda1_joint3": 0.0,
    "panda1_joint4": -25.0,
    "panda1_joint5": 0.0,
    "panda1_joint6": 0.0,
    "panda1_joint7": 0.0,
    "panda1_finger_joint1": 0.0,
    "panda1_finger_joint2": 0.0,
}

TASK = base.build_task("bi_franka", robot="bi_franka", init_pose=INIT_POSE)
