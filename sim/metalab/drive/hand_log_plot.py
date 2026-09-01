#!/usr/bin/env python3
"""hand_log_plot.py — overlay plot for a standalone hand-log run.

One portrait (2:3) high-DPI PNG with two shared-x subplots for a chosen joint:
  top    : joint_position (actual) vs target_position (command)  — both [deg], one y-axis
  bottom : joint_torque (right y-axis, [N·m]) vs the matching fingertip's contact_force
           (left y-axis, [N]) — dual y-axis so the two different-unit curves both fill the plot.

Reads the four CSVs written by ``sim/_runtime/standalone.py`` in a run dir:
  joint_position.csv / target_position.csv / joint_torque.csv / contact_force.csv
(first column = timestamp [s]; other columns = joints, or fingertips for contact_force).

The standalone runner auto-calls :func:`plot_hand_log` when a ``hand_<finger>_...`` trajectory finishes
(joint = the finger's tested joint — a ``cmc``/``mcp``/``pip`` group token picks ``R_<Finger>_<CMC|MCP|PIP>``,
no token → PIP; the thumb has no PIP, so thumb ``pip``/no-token maps to ``R_Thumb_MCP``; fingertip = default).
Also runnable manually:
  python -m sim.metalab.drive.hand_log_plot <run_dir> [--joint R_Index_PIP] [--fingertip ...] [--dpi 400] [--out PATH]
Stdlib + matplotlib only (no torch / engine import), so it runs in any matplotlib-capable python.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")               # headless: render to file, no display needed
import matplotlib.pyplot as plt

# palette (matches the MetaLab dashboards)
_ACTUAL, _TARGET = "#0f7d8c", "#b06a2c"
_FORCE, _TORQUE = "#3b7a57", "#8a4fbd"


def _load(csv_path: Path, col: str) -> tuple[list[float], list[float], str]:
    """Return (timestamps, values, matched_column) for the one column matching ``col`` (exact, else
    unique substring) in ``csv_path``. Fails loud on missing / ambiguous match."""
    rows = list(csv.reader(csv_path.open()))
    header = rows[0]
    cands = [c for c in header if c == col] or [c for c in header if col in c]
    assert len(cands) == 1, f"{col!r} in {csv_path.name}: matched {cands} (want exactly one)"
    ci, ti = header.index(cands[0]), header.index("timestamp")
    t = [float(r[ti]) for r in rows[1:]]
    v = [float(r[ci]) for r in rows[1:]]
    return t, v, cands[0]


def _padded(vals: list[float], frac: float = 0.08) -> tuple[float, float]:
    """min/max of ``vals`` with a small margin, so a curve fills its axis (min & max clearly visible)."""
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    m = (hi - lo) * frac
    return lo - m, hi + m


def plot_hand_log(run_dir, joint: str = "R_Index_PIP", fingertip: str | None = None,
                  dpi: int = 400, out=None) -> Path:
    """Render the overlay PNG for ``joint`` from the CSVs in ``run_dir``; return the saved path.

    ``fingertip`` defaults to the joint's finger (``R_Index_PIP`` -> ``R_Index_Fingertip``).
    ``out`` defaults to ``<run_dir>/<joint>_overlay.png``."""
    run = Path(run_dir)
    ftip = fingertip
    if ftip is None:                                # derive <side>_<finger>_Fingertip from the joint name
        p = joint.split("_")
        assert len(p) >= 2, f"cannot derive fingertip from joint {joint!r}; pass fingertip"
        ftip = f"{p[0]}_{p[1]}_Fingertip"

    tp, pos, jcol = _load(run / "joint_position.csv", joint)
    _,  tgt, _    = _load(run / "target_position.csv", joint)
    _,  trq, _    = _load(run / "joint_torque.csv", joint)
    tc, cf,  fcol = _load(run / "contact_force.csv", ftip)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 12), sharex=True)   # 2:3 portrait
    fig.suptitle(f"{jcol}   ·   {run.name}", fontsize=13, fontweight="bold")

    # --- top: actual vs target position (same [deg] axis) ---
    ax_top.plot(tp, pos, color=_ACTUAL, lw=1.6, label="joint_position (actual)")
    ax_top.plot(tp, tgt, color=_TARGET, lw=1.6, ls="--", label="target_position (command)")
    ax_top.set_ylabel("Joint angle [deg]")
    ax_top.set_title("Position: actual vs target", loc="left", fontsize=10, color="#666")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="best", fontsize=9)

    # --- bottom: torque (right axis) + contact force (left axis), each on its own scale ---
    ax_force = ax_bot                       # left y-axis = contact force [N]
    ax_torque = ax_bot.twinx()             # right y-axis = joint torque [N·m]
    l_force, = ax_force.plot(tc, cf, color=_FORCE, lw=1.6, label=f"contact_force · {fcol} [N]")
    l_torque, = ax_torque.plot(tp, trq, color=_TORQUE, lw=1.6, label=f"joint_torque · {jcol} [N·m]")
    ax_force.set_ylabel("Contact force [N]", color=_FORCE)
    ax_torque.set_ylabel("Joint torque [N·m]", color=_TORQUE)
    ax_force.tick_params(axis="y", labelcolor=_FORCE)
    ax_torque.tick_params(axis="y", labelcolor=_TORQUE)
    ax_force.set_ylim(*_padded(cf))         # each axis spans its own min/max → both curves fill the subplot
    ax_torque.set_ylim(*_padded(trq))
    ax_force.set_xlabel("timestamp [s]  (sim time)")
    ax_force.set_title("Joint torque (right axis) vs fingertip contact force (left axis)",
                       loc="left", fontsize=10, color="#666")
    ax_force.grid(True, alpha=0.3)
    ax_force.legend(handles=[l_torque, l_force], loc="best", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path = Path(out) if out else run / f"{joint}_overlay.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay plot for a standalone hand-log run.")
    ap.add_argument("run_dir", help="a _logs/standalone/<group>/<group>_<ts>_<engine>/ directory")
    ap.add_argument("--joint", default="R_Index_PIP",
                    help="joint column for position/target/torque (exact or unique substring)")
    ap.add_argument("--fingertip", default=None,
                    help="contact_force column; default = derived from the joint's finger (e.g. R_Index_Fingertip)")
    ap.add_argument("--dpi", type=int, default=400, help="PNG resolution (higher = sharper zoom)")
    ap.add_argument("--out", default=None, help="output path (default <run_dir>/<joint>_overlay.png)")
    args = ap.parse_args()
    png = plot_hand_log(args.run_dir, args.joint, args.fingertip, args.dpi, args.out)
    print(f"saved {png}")


if __name__ == "__main__":
    main()
