"""Geometric-mean contact friction for the genesis spoke (quadrants kernel + one re-run call sequence).

The rule is defined in :mod:`sim.metalab.runtime.physics.friction`; this is its genesis implementation.

WHY NOT WARP. genesis' contact buffers are ``quadrants`` tensors (a taichi fork). ``qd.Tensor.to_torch()``
COPIES (measured — and onto the host), so there is no aliased view a warp kernel could write through: the
kernel has to be written in genesis' own DSL. Same formula, second language; both are checked against the
python reference in ``runtime/physics/friction.py``.

WHERE IT HOOKS. genesis mixes friction while writing contact data in ``collider.detection()`` and consumes
it in ``constraint_solver.add_inequality_constraints()`` — both called from
``RigidSolver._func_constraint_force``. So the correction goes between them, which means re-running that
method's sequence from our side with the kernel inserted. ``install()`` binds it onto the live solver
instance (not the vendored class) and asserts every attribute it relies on, so a genesis upgrade that
reshuffles the sequence fails loudly instead of quietly restoring max().

genesis' own floor of 1e-2 on the mixed value is kept: it guards its solver, and it only differs from
newton's 1e-6 floor for mu < 1e-4, far below anything the contracts use.
"""
from __future__ import annotations

import genesis as gs
import genesis.utils.array_class as array_class
import quadrants as qd

_GENESIS_FRICTION_FLOOR = 1.0e-2   # genesis' own guard in collider/contact.py — kept as-is


@qd.kernel(fastcache=True)
def geomean_contact_friction(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
):
    """Rewrite every live contact's friction as the geometric mean of its two geoms' (ratio-scaled) mu.

    Mirrors genesis' own expression from ``collider/contact.py`` — including the per-env ``friction_ratio``
    that friction DR multiplies in — and replaces only the combine step: max(a, b) -> sqrt(a * b)."""
    # flat (contact, env) grid over the fixed buffer, guarded by the live per-env count — genesis' own
    # idiom for walking contacts (see constraint/solver.py add_collision_constraints)
    _B = collider_state.n_contacts.shape[0]
    max_contacts = collider_state.contact_data.geom_a.shape[0]
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        if i_c < collider_state.n_contacts[i_b]:
            i_ga = collider_state.contact_data.geom_a[i_c, i_b]
            i_gb = collider_state.contact_data.geom_b[i_c, i_b]
            mu_a = geoms_info.friction[i_ga] * geoms_state.friction_ratio[i_ga, i_b]
            mu_b = geoms_info.friction[i_gb] * geoms_state.friction_ratio[i_gb, i_b]
            # 1e-2 literal, not the module constant: a global read inside a quadrants kernel is a
            # PURE.VIOLATION warning. Kept equal to _GENESIS_FRICTION_FLOOR (asserted in install()).
            collider_state.contact_data.friction[i_c, i_b] = qd.max(qd.sqrt(mu_a * mu_b), 1.0e-2)


def install(rigid_solver) -> None:
    """Bind a ``_func_constraint_force`` that inserts the kernel between collider and constraint assembly.

    Idempotent per solver instance. Mirrors the vendored sequence (equality → detection → wake → inequality
    → resolve); every piece it calls is asserted to exist first."""
    if getattr(rigid_solver, "_metalab_geomean_friction", False):
        return
    for attr in ("collider", "constraint_solver", "_enable_collision", "_disable_constraint",
                 "_use_hibernation", "entities_info", "_rigid_global_info"):
        assert hasattr(rigid_solver, attr), \
            f"genesis RigidSolver has no '{attr}' — geometric-mean friction hook is stale for this version"
    cs = rigid_solver.constraint_solver
    for attr in ("add_equality_constraints", "add_inequality_constraints", "resolve"):
        assert hasattr(cs, attr), f"genesis constraint solver has no '{attr}' — friction hook is stale"
    assert hasattr(rigid_solver.collider, "detection"), "genesis collider has no 'detection' — hook is stale"
    assert _GENESIS_FRICTION_FLOOR == 1.0e-2, "kernel inlines this floor — keep them in sync"
    assert not rigid_solver._use_hibernation, \
        "hibernation is on: the vendored path also wakes bodies on new contact — port that branch before use"

    def _func_constraint_force():
        if not rigid_solver._disable_constraint:
            cs.add_equality_constraints()
        if rigid_solver._enable_collision:
            rigid_solver.collider.detection()
            geomean_contact_friction(                      # <-- the only change vs the vendored sequence
                rigid_solver.geoms_state, rigid_solver.geoms_info,
                rigid_solver.collider._collider_state,
            )
        if not rigid_solver._disable_constraint:
            cs.add_inequality_constraints()
            cs.resolve(rigid_solver.entities_info, rigid_solver._rigid_global_info)

    rigid_solver._func_constraint_force = _func_constraint_force
    rigid_solver._metalab_geomean_friction = True
    gs.logger.debug("contact friction: geometric mean installed (metalab)")
