from __future__ import annotations

from sim.metalab.contract.spec import TaskSpec, values


class PHYSICS:
    hz = 120
    substeps = 2
    decimation = 2


class CAMERA:
    eye = [1.9, -1.5, 1.4]
    lookat = [0.5, 0.0, 0.6]
    fov = 40.0


GRIPPER_BODIES = ["panda0_gripper", "panda0_leftfinger", "panda0_rightfinger"]

ARM_JOINTS = [f"panda0_joint{i}" for i in range(1, 8)]

INIT_POSE = {
    "panda0_joint1": 0.0,
    "panda0_joint2": 0.0,
    "panda0_joint3": 0.0,
    "panda0_joint4": -90.0,
    "panda0_joint5": 0.0,
    "panda0_joint6": 90.0,
    "panda0_joint7": 0.0,
    "panda0_finger_joint1": 0.0,
    "panda0_finger_joint2": 0.0,
}


def build_task(name: str, *, objects=(), contact_params=None, physics=None, **mdp) -> TaskSpec:
    scene = {
        "ground": True,
        "robot": {
            "name": "franka",
            "base_pos": [0.0, 0.0, 0.0],
            "base_quat": [1.0, 0.0, 0.0, 0.0],
            "fixed_base": True,
            "init_pose": dict(INIT_POSE),
        },
        "camera": values(CAMERA),
    }
    if objects:
        scene["objects"] = list(objects)
    if contact_params:
        scene["contact_params"] = dict(contact_params)
    return TaskSpec(name=name, num_envs=1, physics=values(physics if physics is not None else PHYSICS),
                    scene=scene, **mdp)
