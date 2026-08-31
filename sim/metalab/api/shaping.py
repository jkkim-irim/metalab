"""Reward shaping primitives — pure torch (engine-agnostic, stateless).

State needing a latch/buffer (best-so-far, lifted gate) is held by runtime.env_driver;
this module only provides the **pure computation pieces** (e.g. min-dist progress to prevent hover-farming).
"""
from __future__ import annotations

import torch


def exp_kernel(err: torch.Tensor, sigma: float) -> torch.Tensor:
    """Tracking reward kernel ``exp(-err/sigma)`` (N,). err≥0."""
    return torch.exp(-err / sigma)


def min_dist_progress_(best: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """Telescoping min-distance progress. Updates ``best`` IN PLACE (trailing ``_``), returns the payout.

        progress = 0                    where best is inf (first step of the episode)
                 = max(0, best - cur)   otherwise
        best    <- min(best, cur)

    Only NEW ground pays, so approach-retreat-approach cannot farm it and the episode total is bounded by the
    initial distance. ``best`` is a per-env episode buffer the caller owns (EnvDriver.buffer, inf-filled).
    """
    progress = torch.where(torch.isinf(best), torch.zeros_like(cur), (best - cur).clamp_min(0.0))
    torch.minimum(best, cur, out=best)
    return progress


def risen_above(z: torch.Tensor, spawn_z: torch.Tensor, height: float) -> torch.Tensor:
    """Has the object risen ``height`` above where THIS episode spawned it? → (N,) bool.

    Spawn-relative, so table height and the spawn DR do not move the bar. The absolute counterpart is a
    plain ``z >= bar`` and belongs in whichever term wants it (``obs.object_lifed``)."""
    return (z - spawn_z) > height


def hold_count(count: torch.Tensor, ok: torch.Tensor, mode: str) -> torch.Tensor:
    """Advance a success-hold counter by one step → same shape/dtype as ``count``.

        consecutive:  count + 1 while ok, 0 the step it breaks     — "held for N in a row"
        cumulative:   count + 1 while ok, unchanged otherwise      — "N qualifying steps in total"

    ONE implementation, shared by the two bars the driver counts — the GATE (``_advance_gate``) and the
    curriculum level (``_advance_curriculum``): the same law at two different bars, so the counting rule must
    not be written twice. ``GATE.hold_mode`` picks the mode for both, so a contract states it once.

    Cumulative is STRICTLY easier at equal ``hold_steps`` — a consecutive run of N is also N cumulative
    steps — and it stops distinguishing a firm hold from contact flicker, which is the thing the
    consecutive form exists to catch.
    """
    if mode == "cumulative":
        return count + ok.to(count.dtype)
    assert mode == "consecutive", f"hold_count: unknown mode {mode!r} (consecutive | cumulative)"
    return torch.where(ok, count + 1, torch.zeros_like(count))
