"""Monotonic cubic Hermite spline trajectory generation (pure numpy, engine-agnostic).

Reads a via-point CSV (``duration, joint_1, ..., joint_N``) and produces a
time-uniform joint trajectory. Same tangent rule as the C++ ``monotonic_cubic_spline``
/ ``trajectory_manager`` so sim playback matches the real controller.

No engine / backend / torch import — this is a ``_runtime`` shared primitive
(MetaLab layering: the runtime never imports an engine). The runner turns the
returned rad arrays into ``backend.set_joint_targets`` tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EPS = 1e-9
_HOLD_TOL_RAD = 1e-4  # P1 ≈ P2 → hold (constant segment)


@dataclass
class ViaCSVData:
    """Parsed via-point CSV.

    Besides position, exposes the optional per-via torque-limit / PD-gain columns
    (when present) in the same ``(N, n_joints)`` layout; cells absent on a row are
    NaN. Units are kept as authored in the CSV — deg→rad is applied to position only.
    """

    t_via: np.ndarray                    # (N,) seconds
    pos_via: np.ndarray                  # (N, n_joints) rad
    reach0: float = 0.0                  # first data row's duration [s] — time to reach pos_via[0] from the current pose
    kps_via: np.ndarray | None = None    # (N, n_joints) CSV units
    kds_via: np.ndarray | None = None    # (N, n_joints) CSV units
    trq_via: np.ndarray | None = None    # (N, n_joints) CSV units


# ── Hermite basis functions ──
def _h00(u: np.ndarray) -> np.ndarray:
    return 2.0 * u**3 - 3.0 * u**2 + 1.0


def _h10(u: np.ndarray) -> np.ndarray:
    return u**3 - 2.0 * u**2 + u


def _h01(u: np.ndarray) -> np.ndarray:
    return -2.0 * u**3 + 3.0 * u**2


def _h11(u: np.ndarray) -> np.ndarray:
    return u**3 - u**2


def _hermite_1d(t_out: np.ndarray, t_via: np.ndarray, y_via: np.ndarray) -> np.ndarray:
    """1D monotonic cubic Hermite spline.

    Args:
        t_out: output time samples [s].
        t_via: via-point times [s], ascending.
        y_via: via-point values [rad].

    Returns:
        Interpolated values at ``t_out``. Out-of-range samples clamp to the endpoints.
    """
    n = len(t_via)
    out = np.empty_like(t_out)

    if n == 0:
        out[:] = 0.0
        return out
    if n == 1:
        out[:] = y_via[0]
        return out

    for i in range(n - 1):
        t0, t1 = t_via[i], t_via[i + 1]
        seg_dur = max(t1 - t0, _EPS)
        mask = (t_out >= t0) & (t_out <= t1)
        if not np.any(mask):
            continue

        P1 = float(y_via[i])
        P2 = float(y_via[i + 1])

        # P1 ≈ P2 → hold constant across the segment
        if abs(P2 - P1) <= _HOLD_TOL_RAD:
            out[mask] = P1
            continue

        u = (t_out[mask] - t0) / seg_dur

        # neighbour points (reflect at the ends)
        P0 = float(y_via[i - 1]) if i > 0 else P1
        P3 = float(y_via[i + 2]) if i + 2 < n else P2

        d01 = max(t_via[i] - t_via[i - 1], _EPS) if i > 0 else seg_dur
        d23 = max(t_via[i + 2] - t_via[i + 1], _EPS) if i + 2 < n else seg_dur

        s01 = (P1 - P0) / d01 if d01 > _EPS else 0.0
        s12 = (P2 - P1) / seg_dur
        s23 = (P3 - P2) / d23 if d23 > _EPS else 0.0

        # monotonic tangents: zero the slope at local extrema (no overshoot)
        m1 = 0.5 * (s01 + s12) if s01 * s12 > 0.0 else 0.0
        m2 = 0.5 * (s12 + s23) if s12 * s23 > 0.0 else 0.0

        T1, T2 = m1 * seg_dur, m2 * seg_dur
        pos = _h00(u) * P1 + _h10(u) * T1 + _h01(u) * P2 + _h11(u) * T2

        if not np.all(np.isfinite(pos)):
            pos = np.where(np.isfinite(pos), pos, P1 + (P2 - P1) * u)
        out[mask] = pos

    # out of range → clamp to endpoints (shorter groups hold their final value)
    out[t_out <= t_via[0]] = y_via[0]
    out[t_out >= t_via[-1]] = y_via[-1]
    return out


def _parse_float(s: str) -> float:
    """Parse one CSV cell. Empty/missing → NaN. Non-numeric → raises ValueError."""
    s = s.strip()
    if not s:
        return float("nan")
    return float(s)


def parse_via_csv(path: str | Path) -> ViaCSVData:
    """Parse a via-point CSV into :class:`ViaCSVData`.

    Header columns (case-sensitive, order-free):
        ``duration, joint_1..N, trq_lim_1..N, K_pos_1..N, K_vel_1..N``

    - **Multi-section headers**: a line starting with ``duration`` resets the section
      schema, so one file may mix a section that carries only ``trq_lim`` with one that
      carries ``K_pos``/``K_vel``. Categories not defined in the current section are NaN.
    - Rows with ``duration <= 0`` are comment / section markers and are ignored
      (e.g. ``-1, 아령 1KG 잡기 직전``).
    - C++ convention: row ``k``'s ``duration`` is the time of segment ``(k-1)→k``;
      ``time[0] = 0``, ``time[i] = sum(duration[1:i+1])`` (first row's duration is unused).
    """
    path = Path(path)

    rows_dur: list[float] = []
    rows_pos: list[list[float]] = []
    rows_trq: list[list[float]] = []
    rows_kp: list[list[float]] = []
    rows_kd: list[list[float]] = []

    # Section-local schema. Reset on every "duration,..." header line.
    joint_cols: list[int] = []
    trq_cols: list[int] = []
    kp_cols: list[int] = []
    kd_cols: list[int] = []
    n_joints: int | None = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue

            # header (or re-header for a new section)
            if parts[0] == "duration":
                joint_cols, trq_cols, kp_cols, kd_cols = [], [], [], []
                for i, name in enumerate(parts):
                    if name.startswith("joint_"):
                        joint_cols.append(i)
                    elif name.startswith("trq_lim_"):
                        trq_cols.append(i)
                    elif name.startswith("K_pos_"):
                        kp_cols.append(i)
                    elif name.startswith("K_vel_"):
                        kd_cols.append(i)
                if n_joints is None:
                    n_joints = len(joint_cols)
                continue

            if not joint_cols:
                continue                      # no header seen yet → cannot interpret

            try:
                d = float(parts[0])
            except ValueError:
                continue                      # non-numeric first cell → comment row
            if d <= 0:
                continue                      # section marker / comment

            def cell(idx: int) -> float:
                return _parse_float(parts[idx]) if idx < len(parts) else float("nan")

            pos_row = [cell(c) for c in joint_cols]
            if any(np.isnan(v) for v in pos_row):
                continue                      # position is required — skip incomplete rows

            n = n_joints or len(joint_cols)

            def cells(cols: list[int]) -> list[float]:
                return [cell(c) for c in cols] if cols else [float("nan")] * n

            rows_dur.append(d)
            rows_pos.append(pos_row)
            rows_trq.append(cells(trq_cols))
            rows_kp.append(cells(kp_cols))
            rows_kd.append(cells(kd_cols))

    assert rows_pos, f"no via-point rows parsed from {path}"

    durations = np.asarray(rows_dur, dtype=np.float64)
    t_via = np.concatenate([[0.0], np.cumsum(durations[1:])])
    pos_via = np.deg2rad(np.asarray(rows_pos, dtype=np.float32))

    def _arr_or_none(rows: list[list[float]]) -> "np.ndarray | None":
        arr = np.asarray(rows, dtype=np.float32)
        return None if arr.size == 0 or np.all(np.isnan(arr)) else arr

    return ViaCSVData(
        t_via=t_via,
        pos_via=pos_via,
        reach0=float(durations[0]),
        trq_via=_arr_or_none(rows_trq),
        kps_via=_arr_or_none(rows_kp),
        kds_via=_arr_or_none(rows_kd),
    )


def generate_trajectory(
    time_via: np.ndarray,
    pos_via_rad: np.ndarray,
    hz: float,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-uniform trajectory from via points.

    Args:
        time_via: via-point times [s], shape ``(N,)``.
        pos_via_rad: via-point positions [rad], shape ``(N, n_joints)``.
        hz: output sampling rate [Hz] (> 0). Pass the rate at which the caller will
            consume samples (e.g. the control rate = physics_hz / decimation).
        duration: output length [s]; ``None`` → to the last via point.

    Returns:
        ``(time_out [s], pos_out [rad])`` — ``pos_out`` shape ``(n_steps+1, n_joints)``.
    """
    assert hz > 0, f"hz must be > 0 (got {hz})"
    if duration is None:
        duration = float(time_via[-1])

    dt = 1.0 / hz
    n_steps = int(duration / dt)
    t_out = np.arange(n_steps + 1) * dt
    n_joints = pos_via_rad.shape[1]

    pos_out = np.zeros((n_steps + 1, n_joints), dtype=np.float32)
    for j in range(n_joints):
        pos_out[:, j] = _hermite_1d(t_out, time_via, pos_via_rad[:, j])

    return t_out.astype(np.float32), pos_out


def generate_trajectory_from_csv(
    csv_path: str | Path,
    hz: float,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: :func:`parse_via_csv` then :func:`generate_trajectory`."""
    data = parse_via_csv(csv_path)
    return generate_trajectory(data.t_via, data.pos_via, hz=hz, duration=duration)
