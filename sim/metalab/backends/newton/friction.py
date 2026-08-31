"""Geometric-mean contact friction for the newton spoke (warp kernel + one wrapped mujoco_warp call).

The rule is defined in :mod:`sim.metalab.runtime.physics.friction`; this is its warp implementation.

WHERE IT HOOKS. mujoco_warp mixes friction inside ``collision()`` and consumes it in ``make_constraint()``,
both called from ``fwd_position`` — so the only place to correct the value is between those two. Rather than
copy ``fwd_position``'s body (the legacy isaaclab SolverJK did that, and it carries sleep/island passes that
change between versions), we wrap the module-level ``make_constraint`` symbol: ``fwd_position`` resolves it
through the module at call time, so replacing the attribute puts our kernel in front of every call, on every
substep, for every solver — including inside a captured CUDA graph (a plain wp.launch over fixed buffers).

``install()`` is idempotent and fails loud if mujoco_warp's shape changes (missing symbol, or a
``geom_friction`` that is no longer (nworld, ngeom)) rather than silently leaving MuJoCo's max() in place.
"""
from __future__ import annotations

from mujoco_warp._src import constraint as _mw_constraint
from mujoco_warp._src.types import vec5f
import warp as wp

from sim.metalab.runtime.physics.friction import FRICTION_EPS

_EPS = wp.constant(float(FRICTION_EPS))
_installed = False


@wp.kernel(enable_backward=False)
def geomean_contact_friction(
    nacon: wp.array(dtype=wp.int32),            # (1,) active contact count this substep
    contact_geom: wp.array(dtype=wp.vec2i),     # (naconmax,) the two geom ids per contact
    contact_worldid: wp.array(dtype=wp.int32),  # (naconmax,) which world the contact is in
    geom_friction: wp.array2d(dtype=wp.vec3),   # (nworld, ngeom) per-geom (slide, torsional, rolling)
    contact_friction: wp.array(dtype=vec5f),    # OUT: (naconmax,) slide, slide, torsional, roll, roll
):
    """Overwrite the solver's mixed friction with the geometric mean of the two geoms' values."""
    tid = wp.tid()
    if tid >= nacon[0]:                         # inactive slot — leave it (make_constraint ignores it)
        return
    g = contact_geom[tid]
    w = contact_worldid[tid]
    fa = geom_friction[w, g[0]]
    fb = geom_friction[w, g[1]]
    slide = wp.max(wp.sqrt(fa[0] * fb[0]), _EPS)
    tors = wp.max(wp.sqrt(fa[1] * fb[1]), _EPS)
    roll = wp.max(wp.sqrt(fa[2] * fb[2]), _EPS)
    contact_friction[tid] = vec5f(slide, slide, tors, roll, roll)


def install() -> None:
    """Put the geometric-mean kernel in front of every ``make_constraint`` call. Idempotent."""
    global _installed
    if _installed:
        return
    assert hasattr(_mw_constraint, "make_constraint"), \
        "mujoco_warp._src.constraint.make_constraint missing — cannot install geometric-mean friction"
    original = _mw_constraint.make_constraint

    def make_constraint(m, d):
        assert m.geom_friction.ndim == 2, \
            f"geom_friction is {m.geom_friction.ndim}-D (expected (nworld, ngeom)) — friction mixing hook stale"
        wp.launch(geomean_contact_friction, dim=d.naconmax,
                  inputs=[d.nacon, d.contact.geom, d.contact.worldid, m.geom_friction],
                  outputs=[d.contact.friction], device=d.contact.friction.device)
        original(m, d)

    make_constraint.__wrapped__ = original          # keep the original reachable/inspectable
    _mw_constraint.make_constraint = make_constraint
    _installed = True
