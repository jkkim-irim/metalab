# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Log-dir / wandb run-name generation for the RL trainer.

The wandb run name is the ``log_dir`` basename (``WandbSummaryWriter`` uses ``os.path.split(log_dir)[-1]``),
so building ``log_dir`` from ``build_run_name`` makes the wandb run name match. Pure-python (no Isaac Lab).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess


def short_git_sha(repo_root: Path | str) -> str:
    """Short git SHA of ``repo_root``. Deployed (rsynced) trees are not git repos, so the deploy
    stamps the source worktree's HEAD into a ``GIT_SHA`` file next to the tree (``-dirty`` suffixed
    when uncommitted changes shipped) — read that before surrendering to the ``nogit`` sentinel,
    which must never appear on a real run."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    stamp = Path(repo_root) / "GIT_SHA"
    try:
        sha = stamp.read_text().strip()
        if sha:
            return sha
    except OSError:
        pass
    return "nogit"


def task_slug(task_name: str) -> str:
    """Short lowercase label from a task name/id.

    e.g. ``Isaac-Dexblind-Newton-Allex-Dense-Lift-v0`` -> ``dense-lift``; ``hammer-lift`` -> ``hammer-lift``.
    """
    name = task_name.split(":")[-1]
    if name.startswith("Isaac-"):
        name = name[len("Isaac-"):]
    if name.endswith("-v0"):
        name = name[: -len("-v0")]
    lower = name.lower()
    marker = "allex-"
    if marker in lower:
        return lower.split(marker, 1)[1]
    return lower


def build_run_name(task_name: str, max_iterations: int, num_envs: int, run_name: str = "",
                   repo_root: Path | str | None = None) -> str:
    """log_dir basename / wandb run name.

    Format: ``{YYYY-MM-DD-HH-MM}_{name}_{max_it}it_{num_envs}envs-{sha}`` — ``name`` is an explicit
    ``run_name`` else ``task_slug(task_name)``; ``sha`` is the trainer repo's short SHA (``nogit`` off-repo).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    name = run_name or task_slug(task_name)
    sha = short_git_sha(repo_root or Path.cwd())
    return f"{timestamp}_{name}_{max_iterations}it_{num_envs}envs-{sha}"
