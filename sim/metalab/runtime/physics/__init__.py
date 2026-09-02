"""Physics-model corrections shared by the engine spokes (engine-agnostic definitions).

Each module here states ONE physics choice that both engines must make identically, keeps the reference
formula + a numpy oracle in pure python, and leaves the per-engine kernel to the spoke that can express it.
Unlike ``metalab/actuation`` (the motor↔joint transmission), which shares one warp kernel across engines
because both read joint state as torch, contact-level data has no such shared representation — newton's
lives in warp arrays, genesis' in ``quadrants`` fields — so here the FORMULA is shared and the kernels are
written twice, against the same oracle.
"""
from sim.metalab.runtime.physics.friction import FRICTION_EPS, geomean_friction

__all__ = ["FRICTION_EPS", "geomean_friction"]
