"""Fingertip contact — the normalized read and the "is it gripping" predicate (engine-agnostic).

Both halves live here on purpose: every consumer needs both, and ``contact_mask`` is the ONE implementation
shared by the reward term, the obs term, the GATE and the curriculum, so the dense signal and the success bar
cannot drift apart.

Which bodies are the fingertips is a ROBOT property (``RobotSpec.fingertips``); the caller passes them in and
this module knows nothing about a task.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ContactRead(Protocol):
    """The contact reads every engine implements — part of the :class:`~sim.metalab.api.state.StateAdapter`
    surface, which inherits this."""

    def contact_force(self, link_names: list[str]) -> torch.Tensor:
        """NET contact force per link, summed over every counterpart → ``(N, len(link_names), 3)`` world.

        The force **ON the link**. Both engines agree on that sign: newton's ``SensorContact`` measures "the
        contact force on each sensing object", genesis sums the queried link's own side of each contact."""
        ...

    def contact_force_with(self, link_names: list[str], target: str) -> torch.Tensor:
        """The same read SCOPED to one counterpart (``target`` = {object|robot|table}) → ``(N, K, 3)`` world.

        Prefer it for anything that means "grasping": a fingertip resting on the TABLE is contact too, and on
        a table-top task the approach keeps the fingers there, so the net read says "pressing something" long
        before the hand is near the object. (optional read — used by the grasp reward / obs / GATE)"""
        ...

    def contact_penetration(self, link_names: list[str], target: str) -> torch.Tensor:
        """How deep each link OVERLAPS one counterpart (``target`` = {object|robot|table}) → ``(N, K)`` [m],
        column k = ``link_names[k]``. POSITIVE is overlap; ``0`` is not overlapping.

        Both engines carry the pair's signed distance in the form their own solver consumes — mjwarp
        ``contact.dist`` (negative = penetrating), genesis ``contact_data.penetration`` (positive =
        penetrating) — so the sign is normalized to "positive = in" here and clamped at 0.

        MAX over every contact between that link and the target, because a link carries several collision
        geoms and one geom pair yields several contact points: one number per link means how far in the
        WORST spot is, which is what a penetration read is asked for.

        Measures overlap, never separation. The engines' contact list IS the detection envelope, and on the
        mjwarp path ``geom_margin`` is forced to 0 with ``rigid_gap`` defaulting to 0 — so a near-touching
        pair generates no contact and reads 0, the same as far away. (optional read)"""
        ...


def contact_mask(force_w: torch.Tensor, force_threshold: float = 0.1) -> torch.Tensor:
    """Is each fingertip pressing hard enough to count as gripping → ``(N, K)`` bool.

        returns  ‖force_w‖ > force_threshold

        force_w         (N,K,3) [N]  contact force ON each tip, world frame — counterpart-scoped
                                     (``contact_force_with(tips, "object")``), not the net read
        force_threshold  0.1 [N]     force that counts as pressing

    Magnitude, with no direction test, because the hand answers "pad or nail?" with GEOMETRY: each fingertip
    PAD is its own collision shell on its own body, so a force reported against a tip body was a pad press.
    The direction test this used to run — projecting onto a body-fixed pad axis — belonged to a hand whose
    tip body carried the nail as well, where the shell could not tell the two apart.
    """
    return force_w.norm(dim=-1) > force_threshold
