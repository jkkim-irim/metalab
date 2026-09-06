"""The body-reach terms (gate / reward / obs / events) against a minimal env double.

The double carries only what these terms read: ``goal_pos`` (N,3), ``palm_body`` and ``body_pos(name)``.
Runs under pytest, or directly: ``python3 sim/metalab/tests/test_reach_terms.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.metalab.terms import events, gate, obs, reward  # noqa: E402


class _Env:
    def __init__(self, goal, palm):
        self.goal_pos = torch.tensor(goal, dtype=torch.float32)
        self.palm_body = "palm"
        self._palm = torch.tensor(palm, dtype=torch.float32)

    def body_pos(self, name):
        assert name == "palm", name
        return self._palm


def _env():
    return _Env(goal=[[0.5, 0.0, 0.4], [0.5, 0.0, 0.4]],
                palm=[[0.5, 0.0, 0.4], [0.5, 0.0, 0.5]])


def test_body_goal_dist_is_the_euclidean_norm():
    d = gate.body_goal_dist(_env(), "palm")
    assert torch.allclose(d, torch.tensor([0.0, 0.1]))


def test_body_at_goal_thresholds_the_distance():
    at = gate.body_at_goal(_env(), goal_dist_tol=0.05)
    assert at.tolist() == [True, False]
    assert gate.body_at_goal(_env(), goal_dist_tol=0.1).tolist() == [True, True]


def test_body_at_goal_rejects_grasp_conditions():
    with pytest.raises(AssertionError):
        gate.body_at_goal(_env(), goal_dist_tol=0.05, contact_count=1)


def test_body_at_goal_needs_a_palm_frame():
    e = _env()
    e.palm_body = None
    with pytest.raises(AssertionError):
        gate.body_at_goal(e, goal_dist_tol=0.05)


def test_body_goal_proximity_is_one_at_the_goal_and_decays():
    r = reward.body_goal_proximity(_env(), body="palm", std=0.1)
    assert torch.allclose(r, torch.tensor([1.0, torch.e ** -1.0]))


def test_obs_terms_expose_goal_and_error():
    e = _env()
    assert torch.equal(obs.goal_position(e), e.goal_pos)
    err = obs.body_goal_error(e, body="palm")
    assert torch.allclose(err, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -0.1]]))


def test_reset_goal_position_writes_only_the_reset_envs_inside_the_box():
    e = _env()
    before = e.goal_pos.clone()
    torch.manual_seed(0)
    events.reset_goal_position(e, torch.tensor([1]), x_range=[0.3, 0.6], y_range=[-0.2, 0.2], z_range=[0.2, 0.5])
    assert torch.equal(e.goal_pos[0], before[0])
    g = e.goal_pos[1]
    assert 0.3 <= g[0] <= 0.6 and -0.2 <= g[1] <= 0.2 and 0.2 <= g[2] <= 0.5


def test_reset_goal_position_rejects_an_inverted_range():
    with pytest.raises(AssertionError):
        events.reset_goal_position(_env(), torch.tensor([0]), x_range=[0.6, 0.3], y_range=[0, 0], z_range=[0, 0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
