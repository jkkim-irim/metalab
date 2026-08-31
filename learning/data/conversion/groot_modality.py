"""Generate the GR00T ``modality.json`` for an ALLEX LeRobot v2.1 export.

GR00T's ``LeRobotEpisodeLoader`` reads ``meta/modality.json`` — a declarative map that slices the
flat per-column arrays into named groups + maps the video/annotation columns. This emits ALLEX's
**canonical** layout, identical to the existing raw converter
(``allex_groot/scripts/preprocess/convert_groot_drill_raw_to_lerobot_v2.py``), chosen as the single
source of truth for the GR00T integration:

  * ``state``  q / dq / tau slices of the **132-D** ``observation.state`` (== our wir_v1 / v3.0 state,
    so no projection — the v3.0 state copies straight through),
  * ``target`` / ``residual`` the 44-D master-device columns (each its own column, sliced 0..44),
  * ``action`` r_arm_cmd / l_arm_cmd / r_hand_cmd / l_hand_cmd over the 44-D ``action``,
  * ``video``  each camera -> ``observation.images.<camera>``,
  * ``annotation`` human.action.task_description.

Only the camera list varies per dataset; the joint slices are fixed for the ALLEX 132-D/44-D layout.
"""
import json
from pathlib import Path

from learning.data import allex_modality

# Canonical ALLEX GR00T slices — derived from the single source of truth
# (``learning/data/allex_modality.py``) so the state/action layout is defined once and cannot drift.
# The GR00T modality.json uses ``*_cmd`` action keys; target/residual are each a full-44 span over
# their own separate column.
STATE_GROUPS = dict(allex_modality.STATE_GROUPS)                       # over observation.state (132-D)
TARGET_GROUPS = {"pos_target": allex_modality.ALL, "vel_target": allex_modality.ALL,
                 "trq_target": allex_modality.ALL}
RESIDUAL_GROUPS = {"tau_residual": allex_modality.ALL}
ACTION_GROUPS = {f"{k}_cmd": v for k, v in allex_modality.ACTION_GROUPS.items()}


def build_allex_modality(cameras):
    """Return the GR00T ``modality.json`` dict for ALLEX. ``cameras``: dataset camera names."""
    def _slices(groups):
        return {g: {"start": s, "end": e} for g, (s, e) in groups.items()}

    return {
        "state": _slices(STATE_GROUPS),
        "target": _slices(TARGET_GROUPS),
        "residual": _slices(RESIDUAL_GROUPS),
        "action": _slices(ACTION_GROUPS),
        "video": {c: {"original_key": f"observation.images.{c}"} for c in cameras},
        "annotation": {"human.action.task_description": {}},
    }


def write_modality_json(out_path, cameras):
    """Write ``modality.json`` for ALLEX. ``out_path`` may be the file or its ``meta/`` directory."""
    p = Path(out_path)
    if p.is_dir() or p.suffix != ".json":
        p = p / "modality.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_allex_modality(cameras), indent=2) + "\n")
    return str(p)
