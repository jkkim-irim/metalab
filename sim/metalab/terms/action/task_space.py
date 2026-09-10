from __future__ import annotations

import math
from typing import Any

from pydantic import Field, PrivateAttr, model_validator
import torch

from sim.metalab.contract.spec import ActionCfg

Range = tuple[float, float]


class TaskDeltaPose(ActionCfg):
    pos_m: dict[str, Range] = Field(default_factory=dict)
    rot_deg: dict[str, Range] = Field(default_factory=dict)
    scale: float = 1.0
    _box: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_box(self) -> "TaskDeltaPose":
        assert self.joints, "TaskDeltaPose: `joints` is required — it fixes the action order the box is read in"
        both = set(self.pos_m) & set(self.rot_deg)
        assert not both, f"TaskDeltaPose: joints in both pos_m and rot_deg: {sorted(both)}"
        named = set(self.pos_m) | set(self.rot_deg)
        assert named == set(self.joints), (
            f"TaskDeltaPose: pos_m/rot_deg name {sorted(named)} but joints={self.joints} — every joint gets one box")
        for j, (lo, hi) in {**self.pos_m, **self.rot_deg}.items():
            assert lo < hi and lo <= 0.0 <= hi, f"TaskDeltaPose: {j} box {(lo, hi)} must satisfy lo <= 0 <= hi, lo < hi"
        return self

    def box_si(self) -> list[Range]:
        return [tuple(self.pos_m[j]) if j in self.pos_m
                else (math.radians(self.rot_deg[j][0]), math.radians(self.rot_deg[j][1])) for j in self.joints]

    def _tensors(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(like.device), like.dtype)
        if key not in self._box:
            box = self.box_si()
            self._box[key] = tuple(torch.tensor([b[i] for b in box], device=like.device, dtype=like.dtype)
                                   for i in (0, 1))
        return self._box[key]

    def check(self, *, default, limits) -> None:
        assert default is not None, "TaskDeltaPose needs the group's spawn pose"
        if limits is None:
            return
        lo, hi = self._tensors(default)
        lo_ok = bool((default + lo >= limits[0] - 1e-6).all())
        hi_ok = bool((default + hi <= limits[1] + 1e-6).all())
        assert lo_ok and hi_ok, (
            f"TaskDeltaPose: the offset box about the spawn pose leaves the joint limits — "
            f"spawn+lo={(default[0] + lo).tolist()} spawn+hi={(default[0] + hi).tolist()} "
            f"limits lo={limits[0].tolist()} hi={limits[1].tolist()}")

    def decode(self, a: torch.Tensor, *, default, limits) -> torch.Tensor:
        if self.scale == 0.0:
            return default.expand_as(a)
        lo, hi = self._tensors(default)
        u = torch.clamp(a * self.scale, -1.0, 1.0)
        return default + lo + (u + 1.0) * 0.5 * (hi - lo)

    def spawn_action(self, *, default, limits) -> torch.Tensor:
        if self.scale == 0.0:
            return torch.zeros_like(default)
        lo, hi = self._tensors(default)
        return (-(lo + hi) / (hi - lo) / self.scale).expand_as(default)

    def deploy(self) -> dict[str, Any]:
        return {"mode": "position_delta", "scale": float(self.scale),
                "delta": [[float(lo), float(hi)] for lo, hi in self.box_si()]}
