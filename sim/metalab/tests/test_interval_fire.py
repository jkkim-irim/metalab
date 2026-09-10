"""``events._interval_fire`` — the one cadence book every interval event shares, and ``seed`` redrawing lags.

Runs under pytest, or directly: ``python3 sim/metalab/tests/test_interval_fire.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.metalab.terms import events  # noqa: E402
from sim.metalab.terms.events.common import _interval_fire  # noqa: E402

N = 4


class _Env:
    def __init__(self, step_dt=0.1):
        self.step_dt = step_dt
        self.goal_pos = torch.zeros(N, 3)
        self._bufs: dict = {}

    def buffer(self, key, shape=(), fill=0.0, dtype=torch.float32):
        buf = self._bufs.get(key)
        if buf is None:
            buf = torch.full((N, *shape), fill, dtype=dtype)
            self._bufs[key] = buf
        return buf


def _ids():
    return torch.arange(N)


def test_zero_range_means_every_step_and_passes_ok_through():
    e = _Env()
    ok = torch.tensor([True, False, True, False])
    assert torch.equal(_interval_fire(e, _ids(), (0.0, 0.0), ok), ok)
    assert "next_fire_steps" not in e._bufs


def test_first_eligible_call_fires_then_waits_the_drawn_interval():
    e = _Env(step_dt=0.5)
    ok = torch.ones(N, dtype=torch.bool)
    fire = _interval_fire(e, _ids(), (1.0, 1.0), ok)
    assert fire.all()
    assert e._bufs["next_fire_steps"].tolist() == [2] * N
    assert not _interval_fire(e, _ids(), (1.0, 1.0), ok).any()
    assert _interval_fire(e, _ids(), (1.0, 1.0), ok).all()


def test_ineligible_envs_neither_fire_nor_count_down():
    e = _Env(step_dt=0.5)
    ok = torch.tensor([True, True, False, False])
    fire = _interval_fire(e, _ids(), (1.0, 1.0), ok)
    assert fire.tolist() == [True, True, False, False]
    assert e._bufs["next_fire_steps"].tolist() == [2, 2, 0, 0]
    _interval_fire(e, _ids(), (1.0, 1.0), ok)
    assert e._bufs["next_fire_steps"].tolist() == [1, 1, 0, 0]


def test_bad_range_fails_loud():
    with pytest.raises(AssertionError):
        _interval_fire(_Env(), _ids(), (2.0, 1.0), torch.ones(N, dtype=torch.bool))


def test_sample_goal_position_uses_the_shared_cadence():
    torch.manual_seed(0)
    e = _Env(step_dt=0.5)
    events.sample_goal_position(e, _ids(), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), interval_range_s=(1.0, 1.0))
    assert torch.equal(e.goal_pos, torch.tensor([[1.0, 2.0, 3.0]]).expand(N, 3))
    events.sample_goal_position(e, _ids(), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0), interval_range_s=(1.0, 1.0))
    assert torch.equal(e.goal_pos, torch.tensor([[1.0, 2.0, 3.0]]).expand(N, 3))
    events.sample_goal_position(e, _ids(), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0), interval_range_s=(1.0, 1.0))
    assert torch.equal(e.goal_pos, torch.full((N, 3), 5.0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
