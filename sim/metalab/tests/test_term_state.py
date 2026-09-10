"""Terms that keep their own per-env episode state through ``env.buffer()`` and ``step_edge``.

The double provides only the driver surface these terms read: ``buffer`` (a keyed store refilled by the
test the way ``EnvDriver._fill_term_buffers`` does), ``common_step_counter``, ``episode_length_buf``,
``step_dt`` and the backend reads. Runs under pytest, or directly: ``python3 sim/metalab/tests/test_term_state.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.metalab.runtime.episode import step_edge  # noqa: E402
from sim.metalab.terms import obs  # noqa: E402

N = 3
TIPS = ["tip_a", "tip_b"]


class _Env:
    def __init__(self):
        self.num_envs = N
        self.common_step_counter = 0
        self.step_dt = 0.1
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        self.spec = SimpleNamespace(robot=SimpleNamespace(fingertips=TIPS))
        self.curriculum_values: dict = {}
        self._bufs: dict = {}
        self._fill: dict = {}
        self.vel = torch.zeros(N, 2)
        self.force = torch.zeros(N, len(TIPS), 3)
        self.opos = torch.zeros(N, 3)
        self.oquat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(N, 4).clone()

    def buffer(self, key, shape=(), fill=0.0, dtype=torch.float32):
        buf = self._bufs.get(key)
        if buf is None:
            buf = torch.full((N, *shape), fill, dtype=dtype)
            self._bufs[key] = buf
            self._fill[key] = fill
        return buf

    def reset_ids(self, ids):
        ids = torch.as_tensor(ids)
        for k, buf in self._bufs.items():
            buf[ids] = self._fill[k]
        self.episode_length_buf[ids] = 0

    def step(self):
        self.common_step_counter += 1
        self.episode_length_buf += 1

    def joint_vel(self, names):
        return self.vel

    def contact_force(self, names):
        return self.force

    def contact_force_with(self, names, target):
        return self.force

    def object_pos(self):
        return self.opos

    def object_quat(self):
        return self.oquat


def test_step_edge_reports_first_read_then_only_step_advances():
    e = _Env()
    adv, first = step_edge(e)
    assert adv.all() and first.all()
    adv, first = step_edge(e)
    assert not adv.any() and not first.any()
    e.step()
    adv, first = step_edge(e)
    assert adv.all() and not first.any()
    e.reset_ids([1])
    adv, first = step_edge(e)
    assert adv.tolist() == [False, True, False] and first.tolist() == [False, True, False]


def test_joint_accelerations_is_zero_on_the_first_frame_then_a_finite_difference():
    e = _Env()
    e.vel[:] = 1.0
    assert torch.equal(obs.joint_accelerations(e, ["a", "b"]), torch.zeros(N, 2))
    e.step()
    e.vel[:] = 1.5
    acc = obs.joint_accelerations(e, ["a", "b"])
    assert torch.allclose(acc, torch.full((N, 2), 5.0))
    assert torch.allclose(obs.joint_accelerations(e, ["a", "b"]), acc)


def test_joint_accelerations_restarts_on_a_reset_env():
    e = _Env()
    obs.joint_accelerations(e, ["a", "b"])
    e.step()
    e.vel[:] = 2.0
    obs.joint_accelerations(e, ["a", "b"])
    e.reset_ids([0])
    e.vel[0] = 7.0
    e.step()
    e.vel[1:] = 3.0
    acc = obs.joint_accelerations(e, ["a", "b"])
    assert torch.allclose(acc[0], torch.zeros(2))
    assert torch.allclose(acc[1:], torch.full((2, 2), 10.0))


def test_fingertip_contact_steps_counts_consecutive_presses_once_per_step():
    e = _Env()
    e.force[:, 0, 2] = 1.0
    s = obs.fingertip_contact_steps(e, force_threshold=0.5)
    assert torch.equal(s, torch.tensor([[1.0, 0.0]]).expand(N, 2))
    assert torch.equal(obs.fingertip_contact_steps(e, force_threshold=0.5), s)
    e.step()
    s = obs.fingertip_contact_steps(e, force_threshold=0.5)
    assert torch.equal(s, torch.tensor([[2.0, 0.0]]).expand(N, 2))
    e.force[:, 0, 2] = 0.0
    e.step()
    s = obs.fingertip_contact_steps(e, force_threshold=0.5)
    assert torch.equal(s, torch.zeros(N, 2))


def test_object_seen_pose_freezes_after_the_window():
    e = _Env()
    e.opos[:, 2] = 1.0
    w = obs.object_seen_pose_world(e)
    assert torch.allclose(w[:, 2], torch.ones(N))
    e.step()
    e.opos[:, 2] = 2.0
    w = obs.object_seen_pose_world(e)
    assert torch.allclose(w[:, 2], torch.ones(N))
    e.reset_ids([2])
    w = obs.object_seen_pose_world(e)
    assert w[:, 2].tolist() == [1.0, 1.0, 2.0]


def test_object_seen_pose_window_from_the_curriculum():
    e = _Env()
    e.curriculum_values["seen_steps"] = torch.tensor(2.0)
    e.opos[:, 2] = 1.0
    obs.object_seen_pose_world(e, seen_steps_key="seen_steps")
    e.step()
    e.opos[:, 2] = 2.0
    assert torch.allclose(obs.object_seen_pose_world(e, seen_steps_key="seen_steps")[:, 2], torch.full((N,), 2.0))
    e.step()
    e.opos[:, 2] = 3.0
    assert torch.allclose(obs.object_seen_pose_world(e, seen_steps_key="seen_steps")[:, 2], torch.full((N,), 2.0))
    with pytest.raises(AssertionError):
        obs.object_seen_pose_world(e, seen_steps_key="missing")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
