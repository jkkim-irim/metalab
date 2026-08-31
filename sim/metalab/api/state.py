"""StateAdapter — normalized state-read interface implemented by each engine (primitive layer 1).

All returns follow the canonical conventions (:mod:`sim.metalab.conventions`): quaternions **wxyz**,
positions/velocities in the **world** frame, SI units, ``torch.Tensor`` on :attr:`device`,
leading env dim ``(N, ...)``.
Engine-specific details (Genesis/Newton API) hide behind this; computation primitives, obs, and reward
run only on top of it.

**provisional** — the method set is finalized/adjusted by the genesis (Phase 1) and newton (Phase 4)
implementations. For now, only the reads that perceptive_dexdeepmimic obs/reward actually use are derived.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from sim.metalab.api.contact import ContactRead


@runtime_checkable
class StateAdapter(ContactRead, Protocol):
    """Normalized state read. Implementations live in ``sim/<engine>/state_adapter.py``.

    The contact reads are inherited from :class:`sim.metalab.api.contact.ContactRead` — they live next to
    ``contact_mask``, the predicate whose sign convention only makes sense beside them."""

    num_envs: int
    device: torch.device

    # --- joints: in name order (N, len(names)) ---
    def joint_pos(self, names: list[str]) -> torch.Tensor: ...
    def joint_vel(self, names: list[str]) -> torch.Tensor: ...

    def joint_torque(self, names: list[str]) -> torch.Tensor:
        """Torque the ACTUATOR applied [Nm] — what the real robot's joint-torque sensor reads: motor-space
        PD **+ the actuator-routed gravity feedforward**, AFTER the driver's torque clamp, mapped back to
        joint space (``Gᵀ``) on motor-coupled groups. NOT the net generalized force — passive joint
        spring/damping, the gravity/Coriolis bias, and passive-channel gravcomp (an external force, not
        motor torque) are all excluded. An engine with no gravcomp / no motor-level path delivers the PD
        part alone: the same quantity minus a term it never applies."""
        ...

    # Components of joint_torque, both PRE-clamp — OPTIONAL reads: only an engine whose control path is
    # motor-level can separate them, so callers gate on their presence (obs terms joint_pd_torque_obs /
    # joint_gravcomp_torque_obs; the Standalone monitor's PD/Grav tabs). The driver sums PD + gravity in
    # MOTOR space and clamps ONCE, so `pd + gravcomp == joint_torque` holds only while unsaturated — the
    # gap is exactly what the clamp removed, which is why all three are worth watching side by side.
    # Sources per engine: the gravity component is the engine's gravity-compensation force (newton =
    # MuJoCo's `qfrc_gravcomp`; genesis = the actuator gravcomp feedforward g(q)); on motor-coupled groups
    # both components come from the coupled-PD kernel, which already forms them before its clamp.
    def joint_torque_pd(self, names: list[str]) -> torch.Tensor: ...
    def joint_torque_gravcomp(self, names: list[str]) -> torch.Tensor: ...

    # --- body pose/vel: world, quat wxyz. pos/lin/ang (N,3), quat (N,4) ---
    def body_pos(self, name: str) -> torch.Tensor: ...
    def body_quat(self, name: str) -> torch.Tensor: ...
    def body_lin_vel(self, name: str) -> torch.Tensor: ...
    def body_ang_vel(self, name: str) -> torch.Tensor: ...

    # --- object (raw sim read): pose world, quat wxyz ---
    def object_pos(self) -> torch.Tensor: ...
    def object_quat(self) -> torch.Tensor: ...
    def object_lin_vel(self) -> torch.Tensor: ...
    def object_ang_vel(self) -> torch.Tensor: ...

