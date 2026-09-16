from __future__ import annotations

import torch

#: DR keys this event publishes — read back with ``obs.dr_params(keys=[...])`` / ``env.dr_value(key)``.
COMMAND_KEYS = ("cmd_lin_vel_x", "cmd_lin_vel_y", "cmd_ang_vel_z")


def sample_velocity_command(env, env_ids, lin_vel_x_range, lin_vel_y_range, ang_vel_z_range,   # [m/s] [m/s] [rad/s]
                            interval_range_s=(0.0, 0.0)):   # [s]
    """Draw a new (vx, vy, wz) base-velocity command per env.

    ``interval_range_s=(0, 0)`` (the "reset" form) fires on every call. Otherwise (the "interval" form) each env
    fires every U[lo, hi] seconds counted from its reset — NOT on the first step after the reset, so a contract
    that declares both a reset entry and an interval entry gets one command at reset and the next one an
    interval later, the Genesis ``resampling_time_s`` behaviour."""
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    bounds = torch.tensor([tuple(lin_vel_x_range), tuple(lin_vel_y_range), tuple(ang_vel_z_range)],
                          dtype=torch.float32, device=dev)
    assert (bounds[:, 0] <= bounds[:, 1]).all(), \
        f"each range needs lo <= hi — got x={tuple(lin_vel_x_range)} y={tuple(lin_vel_y_range)} " \
        f"wz={tuple(ang_vel_z_range)}"
    new = torch.rand(k, 3, device=dev) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    lo, hi = float(interval_range_s[0]), float(interval_range_s[1])
    if (lo, hi) == (0.0, 0.0):
        # Reset form: publish outright. (Reading the key back here would fail on the very first reset — the
        # driver asserts on a key no event has published yet.)
        for i, key in enumerate(COMMAND_KEYS):
            env.set_dr_value(key, env_ids, new[:, i])
        return
    assert 0.0 < lo <= hi, f"interval_range_s={interval_range_s} must be (0,0)=every call or 0 < lo <= hi [s]"
    wait = env.buffer("steps_to_fire", dtype=torch.long)   # 0 = fresh after a reset (the driver refills it)
    draw = (torch.empty(k, device=dev).uniform_(lo, hi) / env.step_dt).round().long().clamp(min=1)
    w = wait[env_ids]
    fresh = w == 0
    w = torch.where(fresh, draw - 1, w - 1)               # -1: this call is step 1 of the interval
    fire = (w <= 0) & ~fresh
    wait[env_ids] = torch.where(fire, draw, w)
    for i, key in enumerate(COMMAND_KEYS):
        cur = env.dr_value(key)[env_ids]
        env.set_dr_value(key, env_ids, torch.where(fire, new[:, i], cur))
