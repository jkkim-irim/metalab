"""Action mapping terms — the classes a contract's ACTION block is written with, and the loader's checks.

Runs under pytest, or directly: ``python3 sim/metalab/tests/test_action_terms.py``.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.metalab.contract.loader import load_task  # noqa: E402
from sim.metalab.contract.spec import ActionCfg, values  # noqa: E402
from sim.metalab.terms import action  # noqa: E402

JOINTS = ["j0", "j1"]
DEFAULT = torch.tensor([[0.5, -0.5], [0.0, 0.0]])
LIMITS = (torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0]))
A = torch.tensor([[0.2, 0.0], [10.0, -10.0]])


def test_delta_position_is_the_old_driver_formula_with_the_clamp():
    t = action.JointDeltaPosition(joints=JOINTS, scale=0.5)
    t.check(default=DEFAULT, limits=LIMITS)
    want = torch.clamp(DEFAULT + A * 0.5, LIMITS[0], LIMITS[1])
    assert torch.allclose(t.decode(A, default=DEFAULT, limits=LIMITS), want)


def test_delta_position_without_limits_does_not_clamp():
    t = action.JointDeltaPosition(joints=JOINTS, scale=1.0)
    assert torch.allclose(t.decode(A, default=DEFAULT, limits=None), DEFAULT + A)


def test_position_to_limits_maps_minus_one_zero_plus_one_onto_lo_mid_hi():
    t = action.JointPositionToLimits(joints=JOINTS, scale=1.0)
    t.check(default=DEFAULT, limits=LIMITS)
    a = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [5.0, -5.0]])
    lo, hi = LIMITS
    want = torch.stack([lo, (lo + hi) * 0.5, hi, torch.stack([hi[0], lo[1]])])
    assert torch.allclose(t.decode(a, default=DEFAULT[:1].expand(4, 2), limits=LIMITS), want)


def test_position_to_limits_scale_zero_pins_the_spawn_pose():
    t = action.JointPositionToLimits(joints=JOINTS, scale=0.0)
    assert torch.equal(t.decode(A, default=DEFAULT, limits=LIMITS), DEFAULT)


def test_position_to_limits_needs_limits():
    t = action.JointPositionToLimits(joints=JOINTS)
    with pytest.raises(AssertionError):
        t.check(default=DEFAULT, limits=None)


@pytest.mark.parametrize("cls", [action.JointDeltaPosition, action.JointPositionToLimits])
def test_mode_is_the_class_and_cannot_be_repointed(cls):
    with pytest.raises(Exception):
        cls(joints=JOINTS, mode="velocity")


def test_the_base_class_is_not_a_mapping():
    with pytest.raises(NotImplementedError):
        ActionCfg().decode(A, default=DEFAULT, limits=LIMITS)


def test_values_passes_a_mapping_instance_through():
    class ACTION:
        min_delay = 0
        arm = action.JointDeltaPosition(scale=0.5)

    out = values(ACTION)
    assert out["min_delay"] == 0
    assert isinstance(out["arm"], action.JointDeltaPosition)


def test_loader_resolves_joints_into_the_term():
    spec = load_task("franka_reach", "position")
    g = spec.action["arm"]
    assert isinstance(g.term, action.JointDeltaPosition)
    assert g.term.joints == g.joints and len(g.joints) == 7
    assert g.term.scale == 0.5 and g.ema_tau == 0.1


def test_loader_defaults_an_omitted_action_block_to_delta_position():
    spec = load_task("franka")
    assert {g: type(a.term) for g, a in spec.action.items()} == {
        "arm": action.JointDeltaPosition, "gripper": action.JointDeltaPosition}


def test_position_to_limits_spawn_action_decodes_back_to_the_spawn_pose():
    t = action.JointPositionToLimits(joints=JOINTS, scale=0.5)
    sa = t.spawn_action(default=DEFAULT, limits=LIMITS)
    assert torch.allclose(t.decode(sa, default=DEFAULT, limits=LIMITS), DEFAULT)
    assert torch.equal(action.JointDeltaPosition(joints=JOINTS).spawn_action(default=DEFAULT, limits=LIMITS),
                       torch.zeros_like(DEFAULT))


def test_position_to_limits_range_deg_narrows_one_joint_inside_its_limits():
    t = action.JointPositionToLimits(joints=JOINTS, range_deg={"j1": (0.0, 90.0)})
    t.check(default=DEFAULT, limits=LIMITS)
    out = t.decode(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]), default=DEFAULT, limits=LIMITS)
    assert torch.allclose(out[:, 1], torch.tensor([math.pi / 2, 0.0]))
    assert torch.allclose(out[:, 0], torch.tensor([1.0, -1.0]))
    with pytest.raises(AssertionError):
        action.JointPositionToLimits(joints=JOINTS, range_deg={"j1": (0.0, 200.0)}).check(default=DEFAULT, limits=LIMITS)
    with pytest.raises(AssertionError):
        action.JointPositionToLimits(joints=JOINTS, range_deg={"nope": (0.0, 1.0)}).check(default=DEFAULT, limits=LIMITS)


def test_task_delta_pose_is_an_offset_box_about_the_spawn_pose():
    t = action.TaskDeltaPose(joints=JOINTS, pos_m={"j0": (-0.1, 0.1)}, rot_deg={"j1": (-45.0, 0.0)})
    t.check(default=DEFAULT, limits=LIMITS)
    out = t.decode(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]), default=DEFAULT, limits=LIMITS)
    assert torch.allclose(out[:, 0], DEFAULT[:, 0] + torch.tensor([0.1, -0.1]))
    assert torch.allclose(out[:, 1], DEFAULT[:, 1] + torch.tensor([0.0, -math.pi / 4]))
    sa = t.spawn_action(default=DEFAULT, limits=LIMITS)
    assert torch.allclose(t.decode(sa, default=DEFAULT, limits=LIMITS), DEFAULT)
    assert t.deploy()["mode"] == "position_delta"


def test_task_delta_pose_rejects_a_box_that_misses_a_joint_or_the_zero():
    with pytest.raises(Exception):
        action.TaskDeltaPose(joints=JOINTS, pos_m={"j0": (-0.1, 0.1)})
    with pytest.raises(Exception):
        action.TaskDeltaPose(joints=JOINTS, pos_m={"j0": (0.1, 0.2)}, rot_deg={"j1": (-1.0, 1.0)})


def test_task_delta_pose_check_fails_when_the_box_leaves_the_limits():
    t = action.TaskDeltaPose(joints=JOINTS, pos_m={"j0": (-1.0, 1.0)}, rot_deg={"j1": (-1.0, 1.0)})
    with pytest.raises(AssertionError):
        t.check(default=DEFAULT, limits=LIMITS)


def test_deploy_names_the_mapping():
    assert action.JointDeltaPosition(joints=JOINTS, scale=0.5).deploy() == {"mode": "position", "scale": 0.5}
    assert action.JointPositionToLimits(joints=JOINTS).deploy() == {"mode": "position_to_limits", "scale": 1.0}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
