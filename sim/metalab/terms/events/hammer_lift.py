from __future__ import annotations

from sim.metalab.terms.events.common import apply_object_external_force


def _in_z_window(env, env_ids, lift_threshold, z_max):   # [m]
    z = env.object_pos()[env_ids, 2]
    return (z >= lift_threshold) if z_max <= 0.0 else ((z >= lift_threshold) & (z < z_max))


def apply_object_external_force_when_lifted(env, env_ids,
                                            x_range=(0.0, 0.0), y_range=(0.0, 0.0), z_range=(0.0, 0.0),   # [N]
                                            lift_threshold=0.9, z_max=0.0,   # [m]
                                            interval_range_s=(0.0, 0.0)):   # [s]
    if int(env_ids.numel()) == 0:
        return
    apply_object_external_force(
        env, env_ids, x_range=x_range, y_range=y_range, z_range=z_range,
        interval_range_s=interval_range_s, eligible=_in_z_window(env, env_ids, lift_threshold, z_max))
