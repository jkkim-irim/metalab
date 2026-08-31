# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Hammer-lift-teacher experiment (MetaLab spoke): the RL config + policy spec the trainer owns
(see ``experiment.py``). Task knobs are SIM-OWNED — they live inline in the task contract
``sim/metalab/contract/tasks/hammer_lift_teacher.py``."""

from learning.rl.dexblind.hammer_lift_teacher.experiment import POLICY  # noqa: F401
