"""Cubic-Hermite CSV trajectory as a joint-target source for a sim backend.

A ``CsvTrajectory`` reads a directory of per-body-part via-point CSVs (the
``ALLEX_CSV_JOINT_NAMES`` groups), builds one dense, time-uniform joint-target
buffer [rad] for the joints the robot actually owns, and hands out one target row
per :meth:`step` call. It is **pure** (numpy only, no engine/backend/torch import):
the runner reads :attr:`joint_names` + :meth:`step` and writes the row into
``backend.set_joint_targets``.

Scope: **position playback only.** The CSV's optional per-via PD-gain / torque-limit
columns are parsed (available on :class:`ViaCSVData`) but not applied — the MetaLab
backend has no per-step gain-ramp surface, and gains are owned by the robot YAML
(``overrides``). Add a gain-scheduling source later if needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .hermite_spline import generate_trajectory, parse_via_csv
from .joint_name_map import ALLEX_CSV_JOINT_NAMES

_RAMP_EPS_RAD = 1e-4


def _prepend_ramp(seed: np.ndarray, dense: np.ndarray, hz: float, ramp_s: float) -> np.ndarray:
    """Prepend a quintic smoothstep (C2 ends → zero start/end velocity) from ``seed``
    to ``dense[0]`` so the robot eases onto the trajectory instead of being yanked."""
    first = dense[0]
    delta = first - seed
    if float(np.max(np.abs(delta))) < _RAMP_EPS_RAD:
        return dense
    n = max(1, int(round(ramp_s * hz)))
    u = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)[1:]
    s = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)          # 6u^5 - 15u^4 + 10u^3
    ramp = seed[None, :] + s[:, None] * delta[None, :]
    return np.concatenate([seed[None, :], ramp, dense[1:]], axis=0).astype(np.float32)


class CsvTrajectory:
    """Time-uniform cubic-Hermite trajectory over a CSV group directory.

    Args:
        csv_dir: directory holding the ``*.csv`` via-point files (a subset of
            :data:`ALLEX_CSV_JOINT_NAMES` — missing files are skipped).
        available_joints: joints the robot owns (e.g. ``spec.robot.active_joints()``);
            the CSV map is intersected with this so unowned groups are dropped.
        hz: rate at which :meth:`step` is consumed (the control rate =
            ``physics.hz / decimation``).
        seed_pose: current pose to ramp in from — ``{joint: rad}`` (extra joints
            ignored) or an array aligned with :attr:`joint_names`. Omit to start
            directly at the first via point.
        ramp_s: ramp-in duration [s]; ``0`` (or no ``seed_pose``) disables it.
    """

    def __init__(self, csv_dir, available_joints, hz: float, *,
                 seed_pose=None, ramp_s: float = 0.0):
        self._dir = Path(csv_dir)
        self._hz = float(hz)
        assert self._hz > 0, f"hz must be > 0 (got {hz})"
        assert self._dir.is_dir(), f"trajectory dir not found: {self._dir}"
        avail = set(available_joints)

        # 1) parse every present group; keep only columns whose joint the robot owns
        groups = []          # (csv_name, ViaCSVData, [(csv_col, joint_name), ...])
        durations: list[tuple[str, float]] = []
        for csv_name, jnames in ALLEX_CSV_JOINT_NAMES.items():
            path = self._dir / f"{csv_name}.csv"
            if not path.exists():
                continue
            data = parse_via_csv(path)
            n_cols = int(data.pos_via.shape[1])
            assert n_cols == len(jnames), (
                f"{csv_name}.csv has {n_cols} joint cols but the map expects "
                f"{len(jnames)} — CSV/joint-map mismatch")
            cols = [(c, jn) for c, jn in enumerate(jnames) if jn in avail]
            if not cols:
                continue
            groups.append((csv_name, data, cols))
            durations.append((csv_name, float(data.t_via[-1])))

        assert groups, f"no usable trajectory CSV groups found in {self._dir}"
        # 0 = every group has a single via-point (a static target pose): the dense buffer is one sample and the
        # motion comes from the ramp-in prefix (seed → pose). Multi-via groups give a positive duration.
        self.max_duration_s = max(d for _, d in durations)
        self.groups_used = [g[0] for g in groups]
        # groups shorter than the longest hold their final pose (Hermite clamps out-of-range)
        self._ragged = {n: d for n, d in durations if abs(d - self.max_duration_s) > 1e-6}

        # 2) build the dense (n_samples, n_target) rad buffer, one column per owned joint
        joint_names: list[str] = []
        columns: list[np.ndarray] = []
        for _, data, cols in groups:
            _, pos = generate_trajectory(data.t_via, data.pos_via,
                                         hz=self._hz, duration=self.max_duration_s)
            for csv_col, jname in cols:
                joint_names.append(jname)
                columns.append(pos[:, csv_col])
        self.joint_names: list[str] = joint_names
        dense = np.stack(columns, axis=1).astype(np.float32)   # (n_samples, n_target)

        # single-via-point trajectory (max_duration_s == 0): each group is a static commanded pose, so the CSV's
        # first-row duration IS the intended time to move there from the current pose → use it as the ramp-in time
        # (overrides the caller's ramp_s). Multi-via trajectories keep ramp_s + their own segment timing.
        if self.max_duration_s == 0.0:
            reach = max((data.reach0 for _, data, _ in groups), default=0.0)
            if reach > 0.0:
                ramp_s = reach

        # 3) optional ramp-in from the current pose
        if seed_pose is not None and ramp_s > 0.0:
            dense = _prepend_ramp(self._seed_vector(seed_pose), dense, self._hz, ramp_s)

        self._dense = dense
        self._i = 0

    def _seed_vector(self, seed_pose) -> np.ndarray:
        if isinstance(seed_pose, dict):
            return np.array([float(seed_pose.get(jn, 0.0)) for jn in self.joint_names],
                            dtype=np.float32)
        arr = np.asarray(seed_pose, dtype=np.float32).reshape(-1)
        assert arr.size == len(self.joint_names), (
            f"seed_pose size {arr.size} != driven joints {len(self.joint_names)}")
        return arr

    @property
    def n_samples(self) -> int:
        return int(self._dense.shape[0])

    @property
    def dense(self) -> np.ndarray:
        """Full ``(n_samples, len(joint_names))`` rad buffer. Upload once to the GPU so the runner's hot
        loop indexes it on-device (``dense_gpu[advance()]``) with no per-step numpy row / CPU→GPU copy."""
        return self._dense

    @property
    def duration_s(self) -> float:
        """Playback length [s], including any ramp-in prefix."""
        return float(self._dense.shape[0] - 1) / self._hz

    @property
    def elapsed_s(self) -> float:
        """Trajectory time [s] at the current sample (0 at start, ``duration_s`` when finished)."""
        return float(self._i) / self._hz

    @property
    def finished(self) -> bool:
        return self._i >= self._dense.shape[0] - 1

    def reset(self) -> None:
        """Rewind to the first sample (replays the ramp-in prefix)."""
        self._i = 0

    def step(self) -> np.ndarray:
        """Return the current target row (``(len(joint_names),)`` rad) and advance one
        sample. Holds the final row once finished (= hold the last pose)."""
        row = self._dense[self._i]
        if self._i < self._dense.shape[0] - 1:
            self._i += 1
        return row

    def advance(self) -> int:
        """Return the current sample index, then advance one (holds the last once finished). Index-only
        twin of :meth:`step` for the GPU path: the runner does ``dense_gpu[advance()]`` so no numpy row is
        built and nothing crosses CPU↔GPU. :attr:`elapsed_s` / :attr:`finished` track the same index."""
        i = self._i
        if self._i < self._dense.shape[0] - 1:
            self._i += 1
        return i
