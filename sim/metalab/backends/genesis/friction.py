from __future__ import annotations

import genesis as gs
import genesis.utils.array_class as array_class
import quadrants as qd

_GENESIS_FRICTION_FLOOR = 1.0e-2


@qd.kernel(fastcache=True)
def geomean_contact_friction(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    collider_state: array_class.ColliderState,
):
    _B = collider_state.n_contacts.shape[0]
    max_contacts = collider_state.contact_data.geom_a.shape[0]
    for i_c, i_b in qd.ndrange(max_contacts, _B):
        if i_c < collider_state.n_contacts[i_b]:
            i_ga = collider_state.contact_data.geom_a[i_c, i_b]
            i_gb = collider_state.contact_data.geom_b[i_c, i_b]
            mu_a = geoms_info.friction[i_ga] * geoms_state.friction_ratio[i_ga, i_b]
            mu_b = geoms_info.friction[i_gb] * geoms_state.friction_ratio[i_gb, i_b]
            collider_state.contact_data.friction[i_c, i_b] = qd.max(qd.sqrt(mu_a * mu_b), 1.0e-2)


def install(rigid_solver) -> None:
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
            geomean_contact_friction(
                rigid_solver.geoms_state, rigid_solver.geoms_info,
                rigid_solver.collider._collider_state,
            )
        if not rigid_solver._disable_constraint:
            cs.add_inequality_constraints()
            cs.resolve(rigid_solver.entities_info, rigid_solver._rigid_global_info)

    rigid_solver._func_constraint_force = _func_constraint_force
    rigid_solver._metalab_geomean_friction = True
    gs.logger.debug("contact friction: geometric mean installed (metalab)")
