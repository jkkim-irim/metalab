"""Gate predicate library — what counts as SOLVED, as flat terms the contract's ``GATE`` block names.

Each is a flat ``fn(env, **knobs) -> (N,) bool``. The driver calls the one the contract declares and keeps
only the hold counter and the episode latch; a reward term calls the same function at the curriculum's bar.
No engine imports (env is the backend).
"""
from sim.metalab.terms.gate.common import object_at_goal

__all__ = ["object_at_goal"]
