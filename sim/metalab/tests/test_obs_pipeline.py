"""``ObsPipeline`` (noise gating, scale, history stacking) and the ``time_out`` termination term.

The double provides ``run_term`` the way ``EnvDriver`` does — it just calls the term. Runs under pytest, or
directly: ``python3 sim/metalab/tests/test_obs_pipeline.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.metalab.contract.loader import load_task  # noqa: E402
from sim.metalab.contract.spec import ObsNoise, ObsTerm  # noqa: E402
from sim.metalab.runtime.obs_pipeline import ObsPipeline  # noqa: E402
from sim.metalab.terms import terminate  # noqa: E402

N = 2


class _Env:
    def __init__(self):
        self.x = torch.zeros(N, 3)
        self.calls = 0

    def run_term(self, kind, t):
        assert kind == "obs"
        self.calls += 1
        return t.fn(self, **t.params)


def _x(env):
    return env.x


def _spec(hist=None, noise_groups=(), noise=None, scale=1.0):
    t = ObsTerm(name="x", fn=_x, scale=scale, noise=noise)
    return SimpleNamespace(obs={"actor": [t], "critic": [t]}, obs_history_length=hist or {},
                           obs_noise_groups=list(noise_groups))


def test_noise_applies_only_in_the_listed_groups_and_scale_after_it():
    torch.manual_seed(0)
    p = ObsPipeline(_spec(noise_groups=["actor"], noise=ObsNoise(std=1.0), scale=10.0), N)
    e = _Env()
    e.x[:] = 1.0
    frames = p.capture(e)
    assert torch.equal(frames["critic"], torch.full((N, 3), 10.0))
    assert not torch.equal(frames["actor"], torch.full((N, 3), 10.0))
    assert e.calls == 2


def test_history_stacks_oldest_first_and_reset_envs_repeat_the_current_frame():
    p = ObsPipeline(_spec(hist={"critic": 3}), N)
    e = _Env()
    e.x[:] = 1.0
    obs = p.observe(e)
    assert obs["critic"].shape == (N, 9) and torch.equal(obs["critic"], torch.ones(N, 9))
    e.x[:] = 2.0
    f = p.capture(e)
    p.advance_history(f, None)
    obs = p.stack(f)
    assert obs["critic"][0].tolist() == [1.0] * 3 + [1.0] * 3 + [2.0] * 3
    assert torch.equal(obs["actor"], torch.full((N, 3), 2.0))
    e.x[:] = 3.0
    f = p.capture(e)
    p.advance_history(f, torch.tensor([1]))
    obs = p.stack(f)
    assert obs["critic"][0].tolist() == [1.0] * 3 + [2.0] * 3 + [3.0] * 3
    assert obs["critic"][1].tolist() == [3.0] * 9


def test_observe_does_not_advance_the_history():
    p = ObsPipeline(_spec(hist={"critic": 2}), N)
    e = _Env()
    p.observe(e)
    e.x[:] = 5.0
    p.observe(e)
    assert torch.equal(p.hist["critic"], torch.zeros(N, 2, 3))
    p.reset()
    assert p.hist == {}


def test_time_out_fires_at_the_horizon():
    e = SimpleNamespace(episode_length_buf=torch.tensor([3, 4, 5]), max_episode_length=4)
    assert terminate.time_out(e).tolist() == [False, True, True]


def test_every_mdp_contract_declares_a_time_out_term():
    for task, recipe in (("franka_reach", "position"), ("hammer_lift_teacher", "privileged"), ("parity_mdp", None)):
        spec = load_task(task, recipe) if recipe else load_task(task)
        flagged = [t.name for t in spec.terminate if t.time_out]
        assert "time_out" in flagged, (task, flagged)


def test_loader_rejects_an_mdp_contract_without_a_time_out(monkeypatch):
    from sim.metalab.contract.tasks.rl.manipulation.franka_reach import _base, position

    monkeypatch.setattr(_base, "TERMINATE", type("TERMINATE", (), {}))
    monkeypatch.setattr(position, "TASK", _base.build_task("no_time_out", reward=position.REWARD, gate=position.GATE))
    with pytest.raises(AssertionError, match="horizon termination"):
        load_task("franka_reach", "position")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
