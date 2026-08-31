"""SimBackend — the surface :class:`sim.metalab.runtime.env_driver.EnvDriver` drives an engine through.

Extends :class:`sim.metalab.api.state.StateAdapter` (read-only) with action writes, physics step, and reset.
Engine spokes (genesis/newton) implement it; the driver talks to the engine only through this interface, and
obs/reward terms read through the same object as ``state``. All reads follow the canonical convention
(:mod:`sim.metalab.conventions`): quat wxyz, world, SI, ``(N, ...)``.

EVERY member here is REQUIRED of every spoke, and :func:`assert_backend` — called once by the driver's
constructor — says so out loud at load time instead of at the step that first happens to touch a missing
method. A member that only one engine can offer does NOT belong in this Protocol: it gets its own
capability Protocol below and is asked for by name, so "genesis cannot do this" is a statement the code
makes rather than something a ``hasattr`` silently swallows.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SimBackend(Protocol):
    num_envs: int
    device: torch.device

    # --- read (StateAdapter contract) ---
    def joint_pos(self, names: list[str]) -> torch.Tensor: ...
    def joint_vel(self, names: list[str]) -> torch.Tensor: ...
    def joint_torque(self, names: list[str]) -> torch.Tensor: ...
    # Applied-torque COMPONENTS, pre-clamp. Semantics + the per-engine source live with the read contract:
    # api/state.py StateAdapter.
    def joint_torque_pd(self, names: list[str]) -> torch.Tensor: ...
    def joint_torque_gravcomp(self, names: list[str]) -> torch.Tensor: ...
    def joint_limits(self, names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint position (lower, upper) limits, for the position-to-limits clamp."""
        ...
    def body_pos(self, name: str) -> torch.Tensor: ...
    def body_quat(self, name: str) -> torch.Tensor: ...
    def body_lin_vel(self, name: str) -> torch.Tensor: ...
    def body_ang_vel(self, name: str) -> torch.Tensor: ...
    def object_pos(self) -> torch.Tensor: ...
    def object_quat(self) -> torch.Tensor: ...
    def object_lin_vel(self) -> torch.Tensor: ...
    def object_ang_vel(self) -> torch.Tensor: ...

    # Contact reads — semantics (sign, counterpart scoping) live with the read contract: api/contact.py
    # ContactRead, next to the contact_mask predicate that consumes them.
    def contact_force(self, link_names: list[str]) -> torch.Tensor: ...
    def contact_force_with(self, link_names: list[str], target: str) -> torch.Tensor: ...
    def contact_penetration(self, link_names: list[str], target: str) -> torch.Tensor: ...

    # --- control / step / reset ---
    def set_joint_targets(self, names: list[str], targets: torch.Tensor) -> None:
        """Write joint position targets. targets: (N, len(names))."""
        ...

    def step(self) -> None:
        """Advance physics one step."""
        ...

    def reset_idx(self, env_mask: torch.Tensor) -> None:
        """Reset envs where ``env_mask`` (N,) bool to init state (event randomization applied later by env_driver)."""
        ...

    def nan_world_detected(self) -> torch.Tensor:
        """Worlds where physics diverged to NaN/Inf → (N,) bool. The driver always ORs this into ``done``."""
        ...

    # --- event / curriculum write surface (terms call via EnvDriver / EnvDriver) ---
    def set_object_pose(self, env_idx: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """Write object pose (world) for ``env_idx`` (K,) long: pos (K,3), quat (K,4) wxyz; velocity zeroed."""
        ...

    def set_joint_positions(self, names: list[str], env_idx: torch.Tensor,
                            pos: torch.Tensor, vel: torch.Tensor) -> None:
        """Teleport ``names`` joints to pos/vel (K,J) for env_idx + hold PD target. (reset_joints_by_offset)"""
        ...

    def set_object_friction(self, target: str, env_idx: torch.Tensor, mu: torch.Tensor,
                            exclude: tuple[str, ...] = ()) -> None:
        """Set per-env collision-shape friction μ (absolute, (K,)) for ``target`` ({object|robot|table}),
        skipping the shapes of the bodies named in ``exclude``. (set_shape_friction)"""
        ...

    def set_object_mass(self, env_idx: torch.Tensor, scale: torch.Tensor) -> None:
        """Scale object mass (+inertia) per-env by (K,) relative to default. (randomize_rigid_body_mass)"""
        ...

    def set_root_height(self, env_idx: torch.Tensor, dz: torch.Tensor) -> None:
        """Offset fixed-base root z per-env by dz (K,) relative to default. (randomize_fixed_base_root_height)"""
        ...

    def set_gravity(self, gz: float) -> None:
        """World gravity z [m/s^2, signed] — ALL worlds at once, never per-env. (curriculum write surface)"""
        ...

    # Force and torque are written SEPARATELY, each owning its half of the (N,6) external-wrench buffer, so
    # the two event terms that drive them (apply_object_external_force / _torque) can both be wired without
    # the one that runs second zeroing the other's contribution.
    def apply_object_force(self, env_idx: torch.Tensor, force: torch.Tensor) -> None:
        """Apply world-frame force (K,3) at the object COM — held for the coming control step, then auto-cleared
        (step re-applies it each substep). Leaves any torque half untouched. (apply_object_external_force)"""
        ...

    def apply_object_torque(self, env_idx: torch.Tensor, torque: torch.Tensor) -> None:
        """Apply world-frame torque (K,3) about the object COM — same one-control-step lifetime as
        :meth:`apply_object_force`, and leaves the force half untouched. (apply_object_external_torque)"""
        ...

    # --- motor-coupled PD diagnostics (pose/speed dependent → evaluated on demand, not in the step loop) ---
    def coupled_kq(self) -> list[dict]:
        """Per coupled group, joint-space stiffness at the current pose (env 0): ``[{name, joints, K}]``,
        K = Gᵀ·diag(k_phi)·G in N*m/rad. Feeds the Joint Kp readout."""
        ...

    def coupled_tau_lim(self) -> list[dict]:
        """Per coupled group, the joint-space TORQUE LIMIT at the current pose and speed (env 0):
        ``[{name, joints, part, hi, lo}]`` in N*m — the motor torque-speed envelope ∩ rated box mapped
        through Gᵀ. Feeds the Joint Torque Limit readout (see motor_coupling._tau_lim)."""
        ...

    def motor_gain_warnings(self) -> list[str]:
        """Gain-consistency warnings for the coupled-PD gains currently loaded (e.g. a differential
        transmission whose two motors no longer share a gain, which puts a cross term into K_q)."""
        ...

    def reload_motor_gains(self) -> list[str]:
        """Re-read the coupled-PD motor gains from ``robot_model.json`` into the live kernel buffers and
        return the groups that changed. Gains only — the transmission fit / torque envelope still need a
        rebuild. Called on reset by the standalone runner, so a gain edit lands without restarting."""
        ...

    def set_coupled_float_damping(self, off: bool) -> None:
        """Zero (or restore) the coupled D gain on the gravcomp-fold groups — the standalone's Torque/float
        mode, where gravity compensation is feedforward ONLY and dissipation is joint friction."""
        ...

    # --- viewer (--viz); a spoke whose viewer repaints on its own thread implements these as no-ops ---
    def render_frame(self) -> None:
        """Emit ONE viewer frame WITHOUT advancing physics — the standalone Pause (sim frozen, window still
        alive and pumping input)."""
        ...

    def viewer_step_allowed(self) -> bool:
        """May physics advance? False while the viewer's own Pause holds the sim."""
        ...

    def pump_viewer(self) -> None:
        """Keep the window responsive while the sim is held (a viewer that only pumps input inside a draw)."""
        ...

    def focus_env(self, env_idx: int) -> None:
        """Point the viewer camera at env ``env_idx`` — the contract camera pose plus that env's tile offset."""
        ...


# --- engine-partial capabilities ------------------------------------------------------------------------
# One engine can do it and another cannot, so it is asked for BY NAME rather than assumed. Each is declared
# here (not left to a `hasattr` at the call site) so "which engine is missing what" is answerable by reading
# one file, and so the answer arrives through EnvDriver.capabilities — a set computed once, at load time.

@runtime_checkable
class BatchedStep(Protocol):
    def step_n(self, n: int, render: bool = True) -> None:
        """Run n physics steps in ONE launch (newton: the whole control step as a single CUDA graph)."""
        ...


@runtime_checkable
class ObjectGravity(Protocol):
    def set_object_gravity(self, gz: float) -> None:
        """Effective gravity z [m/s^2, signed] on the OBJECT alone, world gravity untouched (newton: per-body
        gravity compensation). Lets a curriculum unload the object without unloading the robot."""
        ...


@runtime_checkable
class ObjectScale(Protocol):
    def set_object_scale(self, env_idx: torch.Tensor, scale: torch.Tensor) -> None:
        """Scale object GEOMETRY per-env by (K,) relative to the asset default (newton: ``shape_scale`` plus
        per-world collision-vertex copies). A spoke that bakes geometry at build time does NOT define this —
        genesis compiles the morph's scale into the vertices and has no per-env vertex array, so its absence
        is the declaration. An ``Event(..., requires="object_scale")`` is dropped on such a backend."""
        ...


@runtime_checkable
class ObjectVariant(Protocol):
    def object_variant_id(self) -> torch.Tensor:
        """Which asset VARIANT each env spawned → (N,) long, ``0 <= id < object_variant_count()``.

        Read back from the spoke that DID the assignment, never recomputed from the round-robin rule: newton
        lays worlds out as ``w % N`` and genesis is patched to match, but genesis keeps its own balanced map
        when ``num_envs < N``, so a second copy of the rule would silently disagree exactly there."""
        ...

    def object_variant_count(self) -> int:
        """How many variants the movable object was built with (1 = a single asset)."""
        ...


@runtime_checkable
class ContactBudget(Protocol):
    def contact_budget(self) -> dict:
        """Peak contact/constraint buffer use vs the caps — ``{nacon, ncollision, naconmax, nefc, njmax}``."""
        ...

    def contact_budget_t(self) -> torch.Tensor:
        """``[nacon, ncollision, nefc, naconmax, njmax]`` → (5,), no host sync (rollout log channel)."""
        ...


CAPABILITIES: dict[str, type] = {
    "batched_step": BatchedStep,
    "object_gravity": ObjectGravity,
    "object_scale": ObjectScale,
    "object_variant": ObjectVariant,
    "contact_budget": ContactBudget,
}


def _members(proto: type) -> tuple[str, ...]:
    return tuple(sorted(n for n in (*getattr(proto, "__annotations__", {}), *vars(proto))
                        if not n.startswith("_")))


def assert_backend(backend) -> frozenset[str]:
    """Fail loud if ``backend`` is missing anything :class:`SimBackend` requires; return the engine-partial
    capabilities it does have (names from :data:`CAPABILITIES`).

    Called once, from the driver's constructor. Before this existed the Protocol was documentation only —
    nothing checked it, so it had already drifted nine methods behind both spokes, and a spoke that dropped
    a method stayed alive until the step that first reached for it."""
    missing = [n for n in _members(SimBackend) if not hasattr(backend, n)]
    assert not missing, (
        f"{type(backend).__name__} does not implement the SimBackend surface — missing {missing}. Every "
        f"member is required; if only one engine can offer it, it belongs in CAPABILITIES instead "
        f"(sim/metalab/runtime/backend.py), not in SimBackend.")
    return frozenset(name for name, proto in CAPABILITIES.items() if isinstance(backend, proto))
