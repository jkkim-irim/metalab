from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


class Articulation(Protocol):
    def joint_pos(self, names: list[str]) -> torch.Tensor: ...
    def joint_vel(self, names: list[str]) -> torch.Tensor: ...
    def joint_limits(self, names: list[str]) -> tuple[torch.Tensor, torch.Tensor]: ...
    def joint_torque(self, names: list[str]) -> torch.Tensor: ...   # [Nm]
    def joint_torque_pd(self, names: list[str]) -> torch.Tensor: ...   # [Nm]
    def joint_torque_gravcomp(self, names: list[str]) -> torch.Tensor: ...   # [Nm]
    def body_pos(self, name: str) -> torch.Tensor: ...
    def body_quat(self, name: str) -> torch.Tensor: ...
    def body_lin_vel(self, name: str) -> torch.Tensor: ...
    def body_ang_vel(self, name: str) -> torch.Tensor: ...
    def set_joint_targets(self, names: list[str], targets: torch.Tensor) -> None: ...
    def set_joint_positions(self, names: list[str], env_idx: torch.Tensor,
                            pos: torch.Tensor, vel: torch.Tensor) -> None: ...
    def set_root_height(self, env_idx: torch.Tensor, dz: torch.Tensor) -> None: ...
    def coupled_kq(self) -> list[dict]: ...   # [N·m/rad]
    def coupled_tau_lim(self) -> list[dict]: ...   # [N·m]
    def motor_gain_warnings(self) -> list[str]: ...
    def reload_motor_gains(self) -> list[str]: ...
    def set_coupled_float_damping(self, off: bool) -> None: ...


class RigidObject(Protocol):
    def object_pos(self) -> torch.Tensor: ...
    def object_quat(self) -> torch.Tensor: ...
    def object_lin_vel(self) -> torch.Tensor: ...
    def object_ang_vel(self) -> torch.Tensor: ...
    def set_object_pose(self, env_idx: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> None: ...
    def set_object_mass(self, env_idx: torch.Tensor, scale: torch.Tensor) -> None: ...
    def apply_object_force(self, env_idx: torch.Tensor, force: torch.Tensor) -> None: ...


class Contacts(Protocol):
    def contact_force(self, link_names: list[str]) -> torch.Tensor: ...
    def contact_force_with(self, link_names: list[str], target: str) -> torch.Tensor: ...
    def contact_penetration(self, link_names: list[str], target: str) -> torch.Tensor: ...   # [m]


class Scene(Protocol):
    num_envs: int
    device: torch.device

    def step(self, render: bool = True) -> None: ...
    def reset_idx(self, env_mask: torch.Tensor) -> None: ...
    def nan_world_detected(self) -> torch.Tensor: ...
    def set_gravity(self, gz: float) -> None: ...   # [m/s^2]
    def set_object_friction(self, target: str, env_idx: torch.Tensor, mu: torch.Tensor,
                            exclude: tuple[str, ...] = ()) -> None: ...


class Viewer(Protocol):
    def render_frame(self) -> None: ...
    def viewer_step_allowed(self) -> bool: ...
    def pump_viewer(self) -> None: ...
    def focus_env(self, env_idx: int) -> None: ...


@runtime_checkable
class SimBackend(Articulation, RigidObject, Contacts, Scene, Viewer, Protocol):
    pass


@runtime_checkable
class BatchedStep(Protocol):
    def step_n(self, n: int, render: bool = True) -> None: ...


@runtime_checkable
class ObjectGravity(Protocol):
    def set_object_gravity(self, gz: float) -> None: ...   # [m/s^2]


@runtime_checkable
class ObjectScale(Protocol):
    def set_object_scale(self, env_idx: torch.Tensor, scale: torch.Tensor) -> None: ...


@runtime_checkable
class ObjectVariant(Protocol):
    def object_variant_id(self) -> torch.Tensor: ...
    def object_variant_count(self) -> int: ...


@runtime_checkable
class ContactBudget(Protocol):
    def contact_budget_t(self) -> torch.Tensor: ...


CAPABILITIES: dict[str, type] = {
    "batched_step": BatchedStep,
    "object_gravity": ObjectGravity,
    "object_scale": ObjectScale,
    "object_variant": ObjectVariant,
    "contact_budget": ContactBudget,
}


def _members(proto: type) -> tuple[str, ...]:
    names: set[str] = set()
    for cls in proto.__mro__:
        if cls is object or cls.__module__ == "typing":
            continue
        names.update(getattr(cls, "__annotations__", {}))
        names.update(vars(cls))
    return tuple(sorted(n for n in names if not n.startswith("_")))


def assert_backend(backend) -> frozenset[str]:
    missing = [n for n in _members(SimBackend) if not hasattr(backend, n)]
    assert not missing, (
        f"{type(backend).__name__} does not implement the SimBackend surface — missing {missing}. Every "
        f"member is required; if only one engine can offer it, it belongs in CAPABILITIES "
        f"(sim/metalab/api/backend.py), not in SimBackend.")
    return frozenset(name for name, proto in CAPABILITIES.items() if isinstance(backend, proto))
