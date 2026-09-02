from __future__ import annotations

import torch

from sim.metalab.api.transforms import quat_mul


def reset_object_pose(env, env_ids, active_position, x_range, y_range, yaw_range,
                      base_quat=(1.0, 0.0, 0.0, 0.0)):
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    pos = torch.as_tensor(active_position, dtype=torch.float32, device=dev).unsqueeze(0).expand(k, 3).clone()
    pos[:, 0] += torch.empty(k, device=dev).uniform_(*x_range)
    pos[:, 1] += torch.empty(k, device=dev).uniform_(*y_range)
    yaw = torch.empty(k, device=dev).uniform_(*yaw_range)
    half = yaw * 0.5
    q_yaw = torch.zeros(k, 4, device=dev)
    q_yaw[:, 0] = torch.cos(half)
    q_yaw[:, 3] = torch.sin(half)
    base_rot = torch.as_tensor(base_quat, dtype=torch.float32, device=dev)
    quat = quat_mul(q_yaw, base_rot.unsqueeze(0).expand(k, 4))
    env.set_object_pose(env_ids, pos, quat)


def reset_joints_by_offset(env, env_ids, joints, position_range, velocity_range=(0.0, 0.0)):
    k = int(env_ids.numel())
    if k == 0:
        return
    base = env.joint_pos(joints)[env_ids]
    pos = base + torch.empty_like(base).uniform_(*position_range)
    lo, hi = env.joint_limits(joints)
    pos = torch.clamp(pos, lo, hi)
    vel = torch.empty_like(base).uniform_(*velocity_range)
    env.set_joint_positions(joints, env_ids, pos, vel)


def set_shape_friction(env, env_ids, target, mu=1.0, mu_range=None, mu_scale=1.0, exclude=()):
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    if mu_range is not None:
        vals = torch.empty(k, device=dev).uniform_(*mu_range)
    else:
        vals = torch.full((k,), float(mu), device=dev)
    mu_eff = vals * mu_scale
    env.set_dr_value(f"{target}_friction", env_ids, mu_eff)
    env.set_object_friction(target, env_ids, mu_eff, exclude=tuple(exclude))


def randomize_rigid_body_mass(env, env_ids, scale_range, mass_scale=1.0, operation="scale"):
    assert operation == "scale", \
        f"randomize_rigid_body_mass: only operation='scale' supported (got {operation!r})"
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    scale = torch.empty(k, device=dev).uniform_(*scale_range) * mass_scale
    env.set_dr_value("object_mass_scale", env_ids, scale)
    env.set_object_mass(env_ids, scale)


def randomize_object_scale(env, env_ids, scale_range):
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    assert scale_range[0] > 0.0, f"randomize_object_scale: scale_range must be positive (got {scale_range})"
    scale = torch.empty(k, device=dev).uniform_(*scale_range)
    env.set_dr_value("object_scale", env_ids, scale)
    env.set_object_scale(env_ids, scale)


def _external_wrench_vec(env, env_ids, x_range, y_range, z_range, interval_range_s, eligible):
    k, dev = int(env_ids.numel()), env_ids.device
    ok = torch.ones(k, dtype=torch.bool, device=dev) if eligible is None else eligible
    lo, hi = float(interval_range_s[0]), float(interval_range_s[1])
    if (lo, hi) == (0.0, 0.0):
        fire = ok
    else:
        assert 0.0 < lo <= hi, \
            f"interval_range_s={interval_range_s} must be (0,0)=every step or 0 < lo <= hi [s]"
        wait = env.buffer("next_fire_steps", dtype=torch.long)     # [steps]
        left = wait[env_ids] - ok.long()
        fire = ok & (left <= 0)
        draw = (torch.empty(k, device=dev).uniform_(lo, hi) / env.step_dt).round().long().clamp(min=1)
        wait[env_ids] = torch.where(fire, draw, left)
    bounds = torch.tensor([tuple(x_range), tuple(y_range), tuple(z_range)],
                          dtype=torch.float32, device=dev)
    assert (bounds[:, 0] <= bounds[:, 1]).all(), \
        f"each range needs lo <= hi — got x={tuple(x_range)} y={tuple(y_range)} z={tuple(z_range)}"
    vec = torch.rand(k, 3, device=dev) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    return vec * fire.float().unsqueeze(-1)


def apply_object_external_force(env, env_ids, x_range=(0.0, 0.0), y_range=(0.0, 0.0), z_range=(0.0, 0.0),   # [N]
                                interval_range_s=(0.0, 0.0), eligible=None):   # [s]
    if int(env_ids.numel()) == 0:
        return
    force = _external_wrench_vec(env, env_ids, x_range, y_range, z_range, interval_range_s, eligible)
    env.apply_object_force(env_ids, force)


def randomize_fixed_base_root_height(env, env_ids, z_offset_range):   # [m]
    k, dev = int(env_ids.numel()), env_ids.device
    if k == 0:
        return
    dz = torch.empty(k, device=dev).uniform_(*z_offset_range)
    env.set_dr_value("root_height", env_ids, dz)
    env.set_root_height(env_ids, dz)


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
