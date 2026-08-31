# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Experiment/task resolution by CONVENTION — no registry to maintain.

``learning.rl.<experiment>.<task>.experiment`` is the module: the ``--experiment`` and ``--task``
args ARE the package path. Adding a recipe or a sim env means creating the package — nothing to
edit here, no list to keep in sync (the mistake this replaces: a hand-kept mapping in
eval_service plus an if/else in rl_trainer plus argparse ``choices``, three registries for the
same fact). A wrong name fails loudly at import time with the attempted path in the error.

The module contract every experiment package exports: ``EXP`` (runner/trainer config) and
``POLICY`` (everything an eval needs to rebuild the actor). Task knobs are SIM-OWNED — written in
the sim's env/task files (``sim/isaaclab/envs/hammer_lift/knobs.py``, MetaLab task contracts);
the experiment ships nothing env-side. The sim side mirrors the task axis as
``sim/isaaclab/envs/<task>/`` (server-side task routing is the follow-up that completes it)."""
from __future__ import annotations

import importlib


def experiment_module(experiment: str, task: str = "hammer_lift"):
    """The experiment module for (experiment, task), by naming convention."""
    return importlib.import_module(f"learning.rl.{experiment}.{task}.experiment")
