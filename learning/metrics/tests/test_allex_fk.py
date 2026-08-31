"""Unit tests for the ALLEX forward-kinematics layer (learning/metrics/allex_fk.py).

Drives the real AllexFK against the committed URDF and asserts on independent geometric invariants
(joint coverage, the documented mimic couplings, the 44->DOF mapping, left/right mirror symmetry) —
not against a re-implemented FK.

Needs pytorch_kinematics (the `fk` extra) + the URDF; skipped cleanly if either is absent.
Run in the node env from the repo root:  python -m pytest learning -q
"""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_kinematics")

from learning.metrics.allex_fk import DEFAULT_URDF, FINGERS, AllexFK  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_URDF), reason=f"URDF not found: {DEFAULT_URDF}")


@pytest.fixture(scope="module")
def fk():
    return AllexFK(DEFAULT_URDF, device="cpu")


def test_all_44_logical_joints_resolve_to_dofs(fk):
    assert len(fk.vec_joints) == 44
    assert len(set(fk.vec_joints)) == 44                      # no dup resolution
    assert [j for j in fk.vec_joints if j not in fk.pk_set] == []


def test_fingertips_named_thumb_to_little(fk):
    assert set(fk.tips) == {f"{s}_finger_{n}" for s in ("R", "L") for n in range(1, 6)}
    for s in ("R", "L"):
        assert "Thumb" in fk.tips[f"{s}_finger_1"]
        for n, name in enumerate(FINGERS, start=2):           # finger_2..5 = Index..Little
            assert name in fk.tips[f"{s}_finger_{n}"]
    assert set(fk.wrists) >= {"R", "L"}


def test_mimic_couplings_use_documented_multipliers(fk):
    for child, (parent, mult) in fk.mimic.items():
        assert parent in fk.vec_idx                            # parent is one of the 44
        if "Thumb_IP" in child:
            assert mult == pytest.approx(0.7319)
            assert "Thumb_MCP" in parent
        else:
            assert "DIP" in child and "PIP" in parent
            assert mult == pytest.approx(0.6361)


def test_full_th_places_44_inputs_and_couples_mimics(fk):
    vec = (torch.arange(44, dtype=torch.float32) * 0.013 + 0.05)[None]   # distinct values
    th = fk._full_th(vec)
    assert th.shape == (1, len(fk.pk_joints))
    for k, j in enumerate(fk.pk_joints):
        if j in fk.vec_idx:
            assert th[0, k] == pytest.approx(float(vec[0, fk.vec_idx[j]]))   # identity map
        elif j in fk.mimic:
            parent, mult = fk.mimic[j]
            assert th[0, k] == pytest.approx(float(vec[0, fk.vec_idx[parent]]) * mult)
        else:
            assert th[0, k] == 0.0                              # waist/neck/base stay at init


def test_zero_pose_is_left_right_mirror_symmetric(fk):
    # At all-zero joint angles each joint contributes only its (mirrored) origin transform, so the
    # right/left chains must mirror across the sagittal plane (y = left/right): same x & z, opposite
    # y. Catches a bad URDF load, wrong link resolution, or a side swap.
    tips, wrists = fk.tip_wrist_positions(torch.zeros(1, 44))
    wr, wl = wrists["R"][0], wrists["L"][0]
    assert wr[0] == pytest.approx(float(wl[0]), abs=2e-3)
    assert wr[2] == pytest.approx(float(wl[2]), abs=2e-3)
    assert wr[1] == pytest.approx(float(-wl[1]), abs=2e-3)
    assert abs(float(wr[1])) > 5e-3                             # and actually off-center
    for n in range(1, 6):
        r, lt = tips[f"R_finger_{n}"][0], tips[f"L_finger_{n}"][0]
        assert r[0] == pytest.approx(float(lt[0]), abs=2e-3)
        assert r[2] == pytest.approx(float(lt[2]), abs=2e-3)
        assert r[1] == pytest.approx(float(-lt[1]), abs=2e-3)
