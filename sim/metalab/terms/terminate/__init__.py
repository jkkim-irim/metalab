"""Termination term library — StateAdapter-read flat terms.

Each term is a flat ``fn(env, **knobs) -> (N,) bool``; the contract's ``terminate`` list ORs them.
No engine imports (env is the backend).
"""
from sim.metalab.terms.terminate.common import (
    body_contact_detected,
    grasp_lost_after_lift,
    object_below_height,
    object_far_from_body,
    curriculum_passed,
    object_velocity_exceeded,
    table_fingertip_contact_force_exceeded,
)

__all__ = [
    "body_contact_detected",
    "grasp_lost_after_lift",
    "object_below_height",
    "object_far_from_body",
    "curriculum_passed",
    "object_velocity_exceeded",
    "table_fingertip_contact_force_exceeded",
]
