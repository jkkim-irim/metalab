"""Scripted robot-driving sources for the standalone runner (no policy / no learning).

Engine-agnostic joint-target generators consumed by ``sim.metalab.runtime.standalone`` via
``backend.set_joint_targets``, organized by concern:

- ``trajectory/`` — spline trajectory generation: :class:`CsvTrajectory` (cubic-Hermite
  trajectory built from a directory of via-point CSVs) + the CSV↔joint-name map.
- ``motor2joint/`` — motor-to-joint coupled control: real-HW motor-space PD model (warp kernels +
  the firmware-derived transmission maps ``mj_mapping/{finger,thumb}.json`` + ``robot_model.json``
  gains). See ``motor2joint/README.md`` for how the maps were extracted.

Future driving modes (teleop, recorded replay, sine sweep, …) belong here too.
"""
from __future__ import annotations

from .trajectory.csv_trajectory import CsvTrajectory
from .trajectory.hermite_spline import (
    ViaCSVData,
    generate_trajectory,
    generate_trajectory_from_csv,
    parse_via_csv,
)
from .trajectory.joint_name_map import ALLEX_CSV_JOINT_NAMES

__all__ = [
    "CsvTrajectory",
    "ALLEX_CSV_JOINT_NAMES",
    "ViaCSVData",
    "parse_via_csv",
    "generate_trajectory",
    "generate_trajectory_from_csv",
]
