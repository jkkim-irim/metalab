from __future__ import annotations

import math
from typing import Any

import torch

from sim.metalab.contract.spec import ActionCfg


class JointPositionToLimits(ActionCfg):
    scale: float = 1.0
    range_deg: dict[str, tuple[float, float]] = {}

    def _range(self, limits) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = limits
        if not self.range_deg:
            return lo, hi
        lo, hi = lo.clone(), hi.clone()
        for name, (a, b) in self.range_deg.items():
            i = self.joints.index(name)
            lo[i], hi[i] = math.radians(a), math.radians(b)
        return lo, hi

    def check(self, *, default, limits) -> None:
        assert limits is not None, f"{type(self).__name__} needs joint limits, the backend reported none"
        unknown = set(self.range_deg) - set(self.joints)
        assert not unknown, f"range_deg names joints outside the group: {sorted(unknown)}"
        lo, hi = self._range(limits)
        assert bool((hi > lo).all()), f"range_deg has hi <= lo: {self.range_deg}"
        assert bool(((lo >= limits[0] - 1e-6) & (hi <= limits[1] + 1e-6)).all()), \
            f"range_deg leaves the joint limits: {self.range_deg}"

    def decode(self, a: torch.Tensor, *, default, limits) -> torch.Tensor:
        if self.scale == 0.0:
            return default.expand_as(a)
        lo, hi = self._range(limits)
        u = torch.clamp(a * self.scale, -1.0, 1.0)
        return lo + (u + 1.0) * 0.5 * (hi - lo)

    def spawn_action(self, *, default, limits) -> torch.Tensor:
        lo, hi = self._range(limits)
        u = 2.0 * (default - lo) / (hi - lo) - 1.0
        assert bool(((u >= -1.0 - 1e-6) & (u <= 1.0 + 1e-6)).all()), "spawn pose outside the joint limits"
        return torch.zeros_like(default) if self.scale == 0.0 else u / self.scale

    def deploy(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": "position_to_limits", "scale": float(self.scale)}
        if self.range_deg:
            out["range_deg"] = {k: [float(a), float(b)] for k, (a, b) in self.range_deg.items()}
        return out


class JointDeltaPosition(ActionCfg):
    scale: float = 1.0

    def check(self, *, default, limits) -> None:
        pass

    def decode(self, a: torch.Tensor, *, default, limits) -> torch.Tensor:
        tgt = a * self.scale if default is None else default + a * self.scale
        return tgt if limits is None else torch.clamp(tgt, limits[0], limits[1])

    def spawn_action(self, *, default, limits) -> torch.Tensor:
        return torch.zeros_like(default)

    def deploy(self) -> dict[str, Any]:
        return {"mode": "position", "scale": float(self.scale)}
