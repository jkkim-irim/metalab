"""MetaLab sim-service launcher — the spawn recipe learning/rl/service.py dispatches to.

Each sim owns HOW its server process starts (``sim/<name>/launch.py`` convention); the learning side
only knows the generic contract ``spawn(args, port_file) -> Popen``. This keeps
learning/ sim-agnostic: engine selection (genesis|newton) is an MetaLab-internal concern, read here
from ``SIM_ENGINE`` (set by the MetaLab launch scripts).

Pure stdlib — importing this module must never import an engine (it runs in the trainer's venv).
"""
from __future__ import annotations

import os
import subprocess
import sys

# repo root (sim/metalab/launch.py -> sim/metalab -> sim -> repo)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _engine() -> str:
    """MetaLab engine spoke for this run — 'genesis' | 'newton' (fail-loud otherwise)."""
    engine = os.environ.get("SIM_ENGINE", "").lower()
    assert engine in ("genesis", "newton"), \
        f"SIM=metalab needs SIM_ENGINE=genesis|newton (got {engine!r}) — set by the launch script"
    return engine


def spawn(args, port_file: str) -> subprocess.Popen:
    """Launch the MetaLab sim server (``python -m sim.metalab.backends.<engine>.server``) over the
    sim/service RPC + CUDA-IPC transport. SAME-VENV by default (the MetaLab single-venv rule — trainer +
    engine server share one env), i.e. the trainer's own interpreter; override with SIM_PYTHON for a
    split-env node. Task knobs are sim-owned (inline in the task contract), so nothing
    experiment-related is shipped."""
    fwd = ["--task", args.task, "--num_envs", str(args.num_envs), "--device", args.device,
           "--port", "0", "--port_file", port_file]
    recipe = getattr(args, "recipe", "") or ""
    if recipe:                          # a task FAMILY needs it; a single-file contract must not get one
        fwd += ["--recipe", recipe]
    if args.seed is not None:
        fwd += ["--seed", str(args.seed)]
    viz = getattr(args, "viz", None)
    if viz and viz != "none":
        fwd += ["--viz", viz]
    if getattr(args, "play", False):
        fwd += ["--play"]
    rrd = getattr(args, "rrd", "") or ""
    if rrd:                             # THE record path: .rrd + data.json + report.html, both spokes
        fwd += ["--rrd", rrd, "--record_envs", str(getattr(args, "record_envs", 1))]
    py = os.environ.get("SIM_PYTHON") or sys.executable
    return subprocess.Popen([py, "-m", f"sim.metalab.backends.{_engine()}.server", *fwd], cwd=_REPO)
