"""hammer_lift — STANDALONE scene contract (no learning). Authoring rules → sim/metalab/contract/tasks/README.md.

Standalone version of the training task ``hammer_lift_student``: the SAME robot / object / desk / init
pose / physics / contact params, with ALL learning stripped — no reward, terminate, events (DR),
curriculum, obs, or goal. (``goal`` is deliberately absent even though the student declares one: the
loader requires goal and GATE as a pair, and a GATE is a learning concept.)
The Launchpad's Standalone mode steps the backend directly (holds the init pose) and resets only on the
'Reset Simulator' button. Kept out of the train/eval task list (it lives under tasks/standalone/);
distinct stem from the training contracts so they never collide. NOT a training task.

Everything but the hammer comes from ``_base``. Keep this file in sync with hammer_lift_student.py
whenever its scene changes — the point of this task is to inspect THAT environment by hand.
"""
from __future__ import annotations

from ... import _assets as assets
from . import _base as base

TASK = base.build_task(
    "hammer_lift",
    objects=[
        {   # hammer — the YCB 048_hammer scan re-framed onto the retired hammer_ycb.xml's convention
            # (origin = the grasp point on the handle, handle on +y), which is why this keeps that asset's
            # placement numbers. z = desk top + 20 mm: the origin sits that far above the scan's lowest
            # vertex, so the hammer rests exactly on the surface.
            # The name must PREFIX the MJCF model — newton groups contact params by the body-label prefix.
            "name": "ycb_048_hammer_grasp_offset",
            "asset": {"mjcf": assets.object_mjcf("ycb_048_hammer_grasp_offset")},
            # friction: comes from the asset MJCF (standalone runs no DR)
            "mass": 0.55, "init_pos": [0.7, -0.2, base.DESK_TOP],
        },

    ],
    contact={"ycb_048_hammer_grasp_offset": {"solref": [0.01, 1.0], "solimp": [0.9, 0.99, 0.001, 0.5, 2.0], "solmix": 1.0}},
)
