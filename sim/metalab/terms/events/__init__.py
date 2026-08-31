"""Event/DR term library — backend-write terms (imported by the task contract as symbols).

Terms are FLAT functions, ``fn(env, env_ids, **knobs)``; the contract's ``events`` list combines them with
mode (reset|interval) and their knobs. env = EnvDriver (backend read/write delegation). No engine imports.
Signature contract + backend WRITE-API (to be implemented by G5) = :mod:`sim.metalab.terms.events.common` docstring/header.
"""
# common.py = task-agnostic terms; <task>.py = task-specific (here: hammer_lift). __init__ re-exports both
# so the task contract imports terms from the package regardless of which file they live in.
from sim.metalab.terms.events.common import (
    apply_object_external_force,
    apply_object_external_torque,
    randomize_fixed_base_root_height,
    randomize_object_scale,
    randomize_rigid_body_mass,
    reset_joints_by_offset,
    reset_object_pose,
    set_shape_friction,
)
from sim.metalab.terms.events.hammer_lift import apply_object_external_force_when_lifted

__all__ = [
    "apply_object_external_force",
    "apply_object_external_force_when_lifted",
    "apply_object_external_torque",
    "randomize_fixed_base_root_height",
    "randomize_object_scale",
    "randomize_rigid_body_mass",
    "reset_joints_by_offset",
    "reset_object_pose",
    "set_shape_friction",
]
