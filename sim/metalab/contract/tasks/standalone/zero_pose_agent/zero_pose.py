"""zero_pose — STANDALONE scene contract (no learning).

Spawns a single franka_panda on the ground plane, holding the zero pose. panda0_joint4 sits at -25 deg
because its MJCF range is [-176, -22.9] deg and 0 is outside it. Authoring rules →
sim/metalab/contract/tasks/README.md.
"""
from __future__ import annotations

from sim.metalab.contract.spec import TaskSpec

PHYSICS = {"hz": 120, "substeps": 2, "decimation": 2}

SCENE = {
    "ground": True,
    "robot": {
        "name": "franka",
        "base_pos": [0.0, 0.0, 0.0],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "fixed_base": True,
        "init_pose": {
            "panda0_joint1": 0.0,
            "panda0_joint2": 0.0,
            "panda0_joint3": 0.0,
            "panda0_joint4": -25.0,
            "panda0_joint5": 0.0,
            "panda0_joint6": 0.0,
            "panda0_joint7": 0.0,
            "panda0_finger_joint1": 0.0,
            "panda0_finger_joint2": 0.0,
        },
    },
}

TASK = TaskSpec(
    name="zero_pose",
    num_envs=1,
    physics=PHYSICS,
    scene=SCENE,
)
