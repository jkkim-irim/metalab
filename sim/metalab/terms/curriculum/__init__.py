"""Curriculum term library — difficulty-progression term factories (imported by the task contract as symbols).

Factories return ``fn(env) -> dict[str, float]`` and are called at each done reset boundary.
env = EnvDriver (step counter, termination rate, reward/event term access). No engine imports.
Signature contract details = :mod:`sim.metalab.terms.curriculum.common` docstring.

The current contract (hammer_lift_teacher) is a dense teacher and wires no curriculum (no task_success reward/termination).
The terms below are **defined but unreferenced** — activated when B1 (task_success reward) / C3 (success termination) / G2 hammer_force are wired.
"""
# common.py = task-agnostic terms; <task>.py = task-specific (here: hammer_lift). __init__ re-exports both
# so the task contract imports terms from the package regardless of which file they live in.
from sim.metalab.terms.curriculum.common import (
    goal_tolerance_curriculum,
    task_success_difficulty,
)
from sim.metalab.terms.curriculum.hammer_lift import hammer_lift_success_curriculum

__all__ = ["goal_tolerance_curriculum", "hammer_lift_success_curriculum", "task_success_difficulty"]
