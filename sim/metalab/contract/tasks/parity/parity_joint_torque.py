from __future__ import annotations

from . import _base as base


class COMMAND:
    mode = "backend"
    joints = base.ARM_JOINTS
    bodies = base.GRIPPER_BODIES
    amp_deg = 5.0
    freq_hz = 0.25
    seconds = 8.0
    ramp_s = 1.0


TASK = base.build_task("parity_joint_torque")
