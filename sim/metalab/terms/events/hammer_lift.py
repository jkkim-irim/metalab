"""hammer_lift task-specific event/DR terms (see common.py for the general/reusable ones).

Kept separate from ``common.py`` so the common library stays task-agnostic; the term contract itself
(flat ``fn(env, env_ids, **knobs)``) is documented there. Reference:
sim/isaaclab/envs/hammer_lift/mdp/events.py.
"""
from __future__ import annotations

from sim.metalab.terms.events.common import apply_object_external_force


def _in_z_window(env, env_ids, lift_threshold, z_max):
    """``(K,)`` bool — envs whose object sits in ``[lift_threshold, z_max)`` [m, world z]. The gate these
    wrappers hand to the common term. ``z_max=0`` leaves the window open upward, which is the "lifted" case:
    a floor alone, where raising the object further never takes it back out of the condition."""
    z = env.object_pos()[env_ids, 2]
    return (z >= lift_threshold) if z_max <= 0.0 else ((z >= lift_threshold) & (z < z_max))


def apply_object_external_force_when_lifted(env, env_ids, x_range=(0.0, 0.0), y_range=(0.0, 0.0),
                                            z_range=(0.0, 0.0), lift_threshold=0.9, z_max=0.0,
                                            interval_range_s=(0.0, 0.0)):
    """``apply_object_external_force`` narrowed to a WORLD-Z WINDOW on the object — interval event.

        acts while   lift_threshold <= object_z < z_max        (z_max = 0 → no ceiling, the "lifted" case)

    The window is the only thing this term owns; the absolute per-axis force and the ``interval_range_s``
    cadence are the common term's, called with that condition as its ``eligible`` gate. So the wait between
    fires measures time the object spent INSIDE the window: an env that never gets there neither fires nor
    burns its countdown, which is what makes the interval mean "one tug per N seconds of carrying it".

    The floor alone is the firm-grasp filter this ports from the legacy ``apply_hammer_force_when_lifted``: a
    pull that a solid grip carries and a sloppy one loses, so the usual wiring is a downward ``z_range``. A
    CEILING is the other half — it stops the load once the object is high enough that the pull has made its
    point, so the carry above that height is not fighting a permanent handicap. Knob semantics + why the load
    is absolute force rather than a velocity write:
    :func:`sim.metalab.terms.events.common.apply_object_external_force`.
    """
    if int(env_ids.numel()) == 0:
        return
    apply_object_external_force(
        env, env_ids, x_range=x_range, y_range=y_range, z_range=z_range,
        interval_range_s=interval_range_s, eligible=_in_z_window(env, env_ids, lift_threshold, z_max))
