from __future__ import annotations

from ... import _assets as assets
from . import _base as base

TABLE_TOP = 0.738

OBJECTS = [
    {
        "name": "box",
        "asset": {"mjcf": assets.object_mjcf("ycb_009_gelatin_box")},
        "mass": 0.097,
        "init_pos": [0.74, 0.0, TABLE_TOP],
    },
    {
        "name": "table",
        "fixed": True,
        "mass": 1.0,
        "parts": [{"shape": "box", "size": [0.16, 0.4, TABLE_TOP]}],
        "init_pos": [0.74, 0.0, TABLE_TOP / 2.0],
    },
]

TASK = base.build_task("parity_contact", objects=OBJECTS)
