from __future__ import annotations

from .. import _assets as assets
from . import _base as base

TABLE_TOP = 0.70
DROP_HEIGHT = 0.10

OBJECTS = [
    {
        "name": "box",
        "asset": {"mjcf": assets.object_mjcf("ycb_009_gelatin_box")},
        "mass": 0.097,
        "init_pos": [0.80, 0.15, TABLE_TOP + DROP_HEIGHT],
        "init_rpy": [15.0, 25.0, 0.0],
    },
    {
        "name": "table",
        "fixed": True,
        "mass": 1.0,
        "parts": [{"shape": "box", "size": [0.2, 0.4, TABLE_TOP]}],
        "init_pos": [0.80, 0.0, TABLE_TOP / 2.0],
    },
]


class COMMAND:
    mode = "backend"
    joints = ["panda0_joint1"]
    bodies = base.GRIPPER_BODIES
    amp_deg = 0.0
    freq_hz = 0.5
    seconds = 8.0
    ramp_s = 1.0


TASK = base.build_task("parity_objects", objects=OBJECTS)
