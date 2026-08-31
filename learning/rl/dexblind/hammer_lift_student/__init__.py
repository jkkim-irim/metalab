# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Hammer-lift-student experiment (MetaLab spoke): the RL algorithm (EXP) + actor spec the
trainer owns (see ``experiment.py``). Asymmetric actor-critic (actor=compact proprioception, critic=privileged).
Task knobs are SIM-OWNED — they live inline in the task contract
``sim/metalab/contract/tasks/rl/hammer_lift_student/``; same manipulation problem as the teacher, differing
only in the obs split."""

from learning.rl.dexblind.hammer_lift_student.experiment import POLICY  # noqa: F401
