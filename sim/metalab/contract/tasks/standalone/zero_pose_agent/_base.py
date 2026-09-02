"""Shared core for the zero_pose_agent standalone contracts — ONE robot on the ground, holding its pose.

Same parent/child split as ``tasks/standalone/manipulation/_base.py``: every value the recipes share
lives here once (physics, ground, base placement), and a recipe module (a sibling ``<robot>.py`` in this
family folder) passes only its robot YAML name and joint pose table to :func:`build_task`.

The leading underscore keeps this out of the task list (it is a library, not a task): the launchpad skips
``_*.py``, and the loader only imports the module a ``--task`` name asks for.
"""
from __future__ import annotations

from sim.metalab.contract.spec import TaskSpec, values


# --- physics / solver --------------------------------------------------------------------------------
class PHYSICS:
    hz = 120            # control rate — physics dt = 1/hz
    substeps = 2        # physics substeps per control step
    decimation = 2      # physics steps per CONTROL step


def build_task(name: str, *, robot: str, init_pose: dict[str, float],
               base_pos: list[float] = (0.0, 0.0, 0.0), fixed_base: bool = True) -> TaskSpec:
    """Assemble a zero-pose contract: the shared ground scene plus a recipe's robot and pose.

    ``robot`` is the robot YAML name (``contract/robot/<family>/<robot>.yaml``); ``init_pose`` is its joint-name →
    angle table in DEGREES (the loader converts to radians), held by the standalone runner's PD.
    ``base_pos`` defaults to the world origin (right for a fixed-base arm on the ground); a floating-base
    robot passes ``fixed_base=False`` and its standing height.
    """
    return TaskSpec(
        name=name,
        num_envs=1,                    # standalone (the runner forces 1 via build_env anyway)
        physics=values(PHYSICS),
        scene={
            "ground": True,
            "robot": {
                "name": robot,
                "base_pos": list(base_pos),
                "base_quat": [1.0, 0.0, 0.0, 0.0],
                "fixed_base": fixed_base,
                "init_pose": init_pose,
            },
        },
    )
