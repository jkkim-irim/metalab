"""master_chef_can — STANDALONE scene contract (no learning). Authoring rules → sim/metalab/contract/tasks/README.md.

The YCB 002_master_chef_can scan on the shared desk — ``object_mjcf`` turns the folder name into its
path under ``sim/metalab/assets/objects/``. Everything else — robot, init pose, desk, physics, contact
params, camera — is ``_base``'s. NOT a training task.

The object's name must PREFIX the MJCF's ``model=`` attribute: newton labels the imported body
``<model>/worldbody/object`` and classifies contact_params groups by that prefix, so a shorter alias
("can") would land the object's geoms in the robot group and assert. The converter writes
``model="<asset folder>"``, so naming the object after the folder keeps the three in step.
"""
from __future__ import annotations

from ... import _assets as assets
from . import _base as base

# z = the desk top exactly: a YCB scan is authored resting on z=0 (the objects were scanned standing on a
# turntable), unlike the hammer variants whose frames sit elsewhere in the mesh.
# mass: authored here. The scan carries none and the YCB download ships no mass table, so this is a task
# knob, not the real can's weight — replace it if you have the measured number.
TASK = base.build_task(
    "master_chef_can",
    objects=[
        {"name": "ycb_002_master_chef_can", "asset": {"mjcf": assets.object_mjcf("ycb_002_master_chef_can")},
         "mass": 0.4, "init_pos": [0.6, -0.1, base.DESK_TOP]},
    ],
    contact={"ycb_002_master_chef_can": {"solref": [0.01, 1.0], "solimp": [0.9, 0.99, 0.001, 0.5, 2.0], "solmix": 1.0}},
)
