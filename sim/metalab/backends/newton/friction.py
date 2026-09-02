from __future__ import annotations

from mujoco_warp._src import constraint as _mw_constraint
from mujoco_warp._src.types import vec5f
import warp as wp

from sim.metalab.runtime.physics.friction import FRICTION_EPS

_EPS = wp.constant(float(FRICTION_EPS))
_installed = False


@wp.kernel(enable_backward=False)
def geomean_contact_friction(
    nacon: wp.array(dtype=wp.int32),
    contact_geom: wp.array(dtype=wp.vec2i),
    contact_worldid: wp.array(dtype=wp.int32),
    geom_friction: wp.array2d(dtype=wp.vec3),
    contact_friction: wp.array(dtype=vec5f),
):
    tid = wp.tid()
    if tid >= nacon[0]:
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

    make_constraint.__wrapped__ = original
    _mw_constraint.make_constraint = make_constraint
    _installed = True
