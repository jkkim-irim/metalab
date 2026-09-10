"""The gate's hold/pass books (``gate.hold_and_pass``) and ``GateSpec.bars()`` against real predicates.

Runs under pytest, or directly: ``python3 sim/metalab/tests/test_gate.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.metalab.contract.loader import load_task  # noqa: E402
from sim.metalab.contract.spec import GateSpec  # noqa: E402
from sim.metalab.terms import gate  # noqa: E402

N = 3


class _Env:
    def __init__(self):
        self.near = torch.zeros(N, dtype=torch.bool)
        self._bufs: dict = {}
        self._fill: dict = {}

    def buffer(self, key, shape=(), fill=0.0, dtype=torch.float32):
        buf = self._bufs.get(key)
        if buf is None:
            buf = torch.full((N, *shape), fill, dtype=dtype)
            self._bufs[key] = buf
            self._fill[key] = fill
        return buf

    def reset_ids(self, ids):
        for k, buf in self._bufs.items():
            buf[torch.as_tensor(ids)] = self._fill[k]


def _pred(env, tol: float):
    return env.near


def test_consecutive_hold_latches_after_hold_steps_and_restarts_on_a_miss():
    e = _Env()
    e.near[:] = torch.tensor([True, True, False])
    _, passed, hold = gate.hold_and_pass(e, _pred, {"tol": 0.1}, 2, "consecutive")
    assert hold.tolist() == [1, 1, 0] and not passed.any()
    e.near[1] = False
    _, passed, hold = gate.hold_and_pass(e, _pred, {"tol": 0.1}, 2, "consecutive")
    assert hold.tolist() == [2, 0, 0] and passed.tolist() == [True, False, False]
    e.near[0] = False
    _, passed, hold = gate.hold_and_pass(e, _pred, {"tol": 0.1}, 2, "consecutive")
    assert hold.tolist() == [0, 0, 0] and passed.tolist() == [True, False, False]


def test_cumulative_hold_keeps_counting_across_misses():
    e = _Env()
    e.near[:] = True
    gate.hold_and_pass(e, _pred, {"tol": 0.1}, 3, "cumulative")
    e.near[:] = False
    gate.hold_and_pass(e, _pred, {"tol": 0.1}, 3, "cumulative")
    e.near[:] = True
    _, passed, hold = gate.hold_and_pass(e, _pred, {"tol": 0.1}, 3, "cumulative")
    assert hold.tolist() == [2, 2, 2] and not passed.any()
    _, passed, hold = gate.hold_and_pass(e, _pred, {"tol": 0.1}, 3, "cumulative")
    assert passed.all()


def test_reset_fill_clears_the_books_for_the_reset_envs_only():
    e = _Env()
    e.near[:] = True
    _, passed, hold = gate.hold_and_pass(e, _pred, {"tol": 0.1}, 1, "consecutive")
    assert passed.all()
    e.reset_ids([1])
    assert passed.tolist() == [True, False, True] and hold.tolist() == [1, 0, 1]


def test_hold_count_rejects_an_unknown_mode():
    with pytest.raises(AssertionError):
        gate.hold_count(torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.bool), "sometimes")


def test_bars_hands_the_predicate_only_the_knobs_it_takes():
    g = GateSpec(predicate=gate.body_at_goal, goal_dist_tol=0.02, hold_steps=10)
    assert g.bars() == {"goal_dist_tol": 0.02, "palm_distance": 0.0, "contact_count": 0, "contact_fingers": (),
                        "force_threshold": 0.1, "joint_pose": {}, "joint_pose_tolerance": 0.0}
    g = GateSpec(predicate=lambda env, goal_dist_tol: None, goal_dist_tol=0.02)
    assert g.bars() == {"goal_dist_tol": 0.02}


def test_bars_fails_loud_when_the_contract_states_a_knob_the_predicate_ignores():
    def only_dist(env, goal_dist_tol):
        return None

    g = GateSpec(predicate=only_dist, goal_dist_tol=0.02, contact_count=2)
    with pytest.raises(AssertionError, match="contact_count"):
        g.bars()


def test_loaded_contracts_bind_their_gate_bars():
    for task, recipe in (("franka_reach", "position"), ("hammer_lift_student", "4hammers")):
        spec = load_task(task, recipe)
        bars = spec.gate.bars()
        assert bars["goal_dist_tol"] == spec.gate.goal_dist_tol
        assert bars["contact_fingers"] == tuple(spec.gate.contact_fingers)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
