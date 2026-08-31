"""Unit tests for the task-space validation metrics (learning/metrics/validation.py).

compute_group_metrics is robot-agnostic plumbing (tested with a fake policy, no deps).
compute_fk_metrics needs pytorch_kinematics + the URDF (skipped otherwise); tested for the
plumbing invariant that pred == GT ⇒ 0 mm error, and that a perturbed prediction is > 0.

Run in the node env from the repo root:  python -m pytest learning -q
"""
import contextlib
import os
from types import SimpleNamespace

import pytest
import torch

from learning.metrics.validation import compute_group_metrics
from learning.model.act.constants import ACTION

ACTION_DIM, CHUNK, BATCH = 44, 4, 2


class _FakeAccel:
    def autocast(self):
        return contextlib.nullcontext()


class _FakePolicy:
    """Returns a fixed (or GT-echoing) action chunk — exercises the metric, not a real model."""

    def __init__(self, pred=None, echo_gt=False):
        self._pred = pred
        self._echo = echo_gt
        self.config = SimpleNamespace(chunk_size=CHUNK)

    def eval(self):
        pass

    def predict_action_chunk(self, batch):
        return batch[ACTION] if self._echo else self._pred


def test_compute_group_metrics_slices_by_body_part():
    gt = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    pred = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    pred[:, :, 14:44] = 1.0  # arm (0:14) exact; fingers (14:44) off by 1.0
    batch = {ACTION: gt, "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool)}
    out = compute_group_metrics(_FakePolicy(pred), [batch], lambda b: b, _FakeAccel())
    assert out["l1_arm"] == pytest.approx(0.0)
    assert out["l1_r_arm"] == pytest.approx(0.0)
    assert out["l1_finger"] == pytest.approx(1.0)
    assert out["l1_r_hand"] == pytest.approx(1.0)
    assert out["l1_all"] == pytest.approx(30.0 / 44.0)  # 14 exact + 30 off-by-1, every chunk step


def test_compute_group_metrics_respects_pad_mask():
    gt = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    pred = torch.ones(BATCH, CHUNK, ACTION_DIM)
    pad = torch.ones(BATCH, CHUNK, dtype=torch.bool)  # every chunk step padded -> no valid pairs
    batch = {ACTION: gt, "action_is_pad": pad}
    assert compute_group_metrics(_FakePolicy(pred), [batch], lambda b: b, _FakeAccel()) == {}


def test_compute_group_metrics_accepts_custom_groups():
    gt = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    pred = torch.full((BATCH, CHUNK, ACTION_DIM), 2.0)
    batch = {ACTION: gt, "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool)}
    out = compute_group_metrics(_FakePolicy(pred), [batch], lambda b: b, _FakeAccel(),
                                groups={"first_half": (0, 22), "second_half": (22, 44)})
    assert set(out) == {"l1_first_half", "l1_second_half"}
    assert out["l1_first_half"] == pytest.approx(2.0)


_LIBERO_CART = [
    {"key": "mm_pos", "slice": (0, 3), "scale": 50.0, "unit": "mm"},
    {"key": "deg_rot", "slice": (3, 6), "scale": 28.6479, "unit": "deg"},
]


def test_cartesian_gripper_metrics_scales_masks_and_gripper_sign():
    # LibERO-style 7-D EE-delta action [dpos(3), drot(3), gripper(1)]. Hand-built pred/gt with a KNOWN
    # per-axis error verify: (1) mm = L2(pos err) * 50; (2) deg = L2(rot err) * 28.6479; (3) gripper
    # L1 + sign-accuracy; (4) a padded chunk step with garbage is masked out (doesn't move any metric).
    from learning.metrics.validation import compute_cartesian_gripper_metrics
    B, K = 2, 2
    gt = torch.zeros(B, K, 7)
    pred = torch.zeros(B, K, 7)
    pred[..., 0] = 0.1                 # dpos err = (0.1, 0, 0) -> L2 = 0.1 -> mm_pos = 0.1*50 = 5.0
    pred[..., 4] = 0.2                 # drot err = (0, 0.2, 0) -> L2 = 0.2 -> deg_rot = 0.2*28.6479
    gt[..., 6] = 1.0                   # gripper GT = +1 (close) on every step
    pred[0, 0, 6] = 0.5                # sign(+)==sign(+) match ; |0.5-1| = 0.5
    pred[0, 1, 6] = 0.8                # match                 ; |0.8-1| = 0.2
    pred[1, 0, 6] = -0.3               # sign(-)!=sign(+) MISMATCH ; |-0.3-1| = 1.3
    pred[1, 1, :] = 999.0              # padded step: garbage that MUST be masked away
    gt[1, 1, :] = -999.0
    pad = torch.zeros(B, K, dtype=torch.bool)
    pad[1, 1] = True                   # -> 3 valid (sample, step) pairs: (0,0) (0,1) (1,0)
    batch = {ACTION: gt, "action_is_pad": pad}
    out = compute_cartesian_gripper_metrics(
        _FakePolicy(pred), [batch], lambda b: b, _FakeAccel(),
        cartesian_metrics=_LIBERO_CART, gripper_dim=6)
    assert out["mm_pos"] == pytest.approx(5.0)                 # 0.1 * 50, uniform over valid pairs
    assert out["deg_rot"] == pytest.approx(0.2 * 28.6479)
    assert out["gripper_l1"] == pytest.approx((0.5 + 0.2 + 1.3) / 3)   # padded 1.3k+ excluded
    assert out["gripper_acc"] == pytest.approx(2.0 / 3)               # 2 of 3 valid steps match sign


def test_cartesian_gripper_metrics_gripper_dim_none_skips_gripper():
    from learning.metrics.validation import compute_cartesian_gripper_metrics
    B, K = 1, 2
    gt = torch.zeros(B, K, 7)
    pred = torch.zeros(B, K, 7)
    pred[..., 1] = 0.04               # dpos err = (0, 0.04, 0) -> L2 = 0.04 -> mm_pos = 2.0
    batch = {ACTION: gt, "action_is_pad": torch.zeros(B, K, dtype=torch.bool)}
    out = compute_cartesian_gripper_metrics(
        _FakePolicy(pred), [batch], lambda b: b, _FakeAccel(),
        cartesian_metrics=_LIBERO_CART, gripper_dim=None)
    assert set(out) == {"mm_pos", "deg_rot"}                   # no gripper_* keys when dim is None
    assert out["mm_pos"] == pytest.approx(2.0)
    assert out["deg_rot"] == pytest.approx(0.0)


def test_cartesian_gripper_metrics_empty_when_all_padded():
    from learning.metrics.validation import compute_cartesian_gripper_metrics
    B, K = 2, 2
    gt = torch.zeros(B, K, 7)
    pred = torch.ones(B, K, 7)
    pad = torch.ones(B, K, dtype=torch.bool)   # every step padded -> no valid pairs -> {}
    batch = {ACTION: gt, "action_is_pad": pad}
    out = compute_cartesian_gripper_metrics(
        _FakePolicy(pred), [batch], lambda b: b, _FakeAccel(),
        cartesian_metrics=_LIBERO_CART, gripper_dim=6)
    assert out == {}


def test_compute_fk_metrics_zero_when_pred_equals_gt():
    pytest.importorskip("pytorch_kinematics")
    from learning.metrics.allex_fk import DEFAULT_URDF, AllexFK
    from learning.metrics.validation import compute_fk_metrics
    if not os.path.exists(DEFAULT_URDF):
        pytest.skip(f"URDF not found: {DEFAULT_URDF}")
    fk = AllexFK(DEFAULT_URDF, device="cpu")
    gt = torch.randn(BATCH, CHUNK, ACTION_DIM) * 0.1  # small radians
    batch = {ACTION: gt, "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool)}
    # identity pre/post processors; policy echoes GT -> raw_pred == raw_gt -> 0 mm error.
    out = compute_fk_metrics(
        _FakePolicy(echo_gt=True), [batch], lambda b: b, lambda x: x, _FakeAccel(), fk)
    assert out, "expected FK metrics"
    for k, v in out.items():
        # position is exactly 0; orientation (deg) has a tiny arccos float floor near 0 (~0.03 deg).
        tol = 0.1 if k.startswith("fk_deg") else 1e-3
        assert v == pytest.approx(0.0, abs=tol), f"{k}={v} should be ~0 when pred==GT"


def test_prediction_metrics_unified_equals_separate_no_fk():
    # The unified one-pass compute_prediction_metrics must return EXACTLY what the separate
    # per-horizon + group passes did (eval-mode predict is deterministic, so folding them is exact).
    # It now emits each L1/horizon key in two spaces (_norm/_unnorm); with identity pre/post-processors
    # both spaces collapse to the single reference value.
    from learning.metrics.validation import (
        ALLEX_ACTION_GROUPS,
        compute_per_horizon_metrics,
        compute_prediction_metrics,
    )
    torch.manual_seed(0)
    gt = torch.randn(BATCH, CHUNK, ACTION_DIM)
    pred = torch.randn(BATCH, CHUNK, ACTION_DIM)
    pad = torch.zeros(BATCH, CHUNK, dtype=torch.bool)
    pad[0, -1] = True                                   # mask one chunk step
    batch = {ACTION: gt, "action_is_pad": pad}
    sep = {}
    sep.update(compute_per_horizon_metrics(_FakePolicy(pred), [batch], lambda b: b, _FakeAccel(), [0, 2]))
    sep.update(compute_group_metrics(_FakePolicy(pred), [batch], lambda b: b, _FakeAccel()))
    uni = compute_prediction_metrics(_FakePolicy(pred), [batch], lambda b: b, lambda x: x, _FakeAccel(),
                                     horizons=[0, 2], groups=ALLEX_ACTION_GROUPS, fk=None)
    assert set(uni) == {f"{k}_{sp}" for k in sep for sp in ("norm", "unnorm")}
    for k in sep:
        for sp in ("norm", "unnorm"):
            assert uni[f"{k}_{sp}"] == pytest.approx(sep[k]), f"{k}_{sp}: unified != separate {sep[k]}"


def test_prediction_metrics_unified_equals_separate_with_fk():
    pytest.importorskip("pytorch_kinematics")
    from learning.metrics.allex_fk import DEFAULT_URDF, AllexFK
    from learning.metrics.validation import (
        ALLEX_ACTION_GROUPS,
        compute_fk_metrics,
        compute_per_horizon_metrics,
        compute_prediction_metrics,
    )
    if not os.path.exists(DEFAULT_URDF):
        pytest.skip(f"URDF not found: {DEFAULT_URDF}")
    fk = AllexFK(DEFAULT_URDF, device="cpu")
    torch.manual_seed(0)
    gt = torch.randn(BATCH, CHUNK, ACTION_DIM) * 0.2
    pred = torch.randn(BATCH, CHUNK, ACTION_DIM) * 0.2
    batch = {ACTION: gt, "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool)}
    sep = {}
    sep.update(compute_per_horizon_metrics(_FakePolicy(pred), [batch], lambda b: b, _FakeAccel(), [0, 2]))
    sep.update(compute_group_metrics(_FakePolicy(pred), [batch], lambda b: b, _FakeAccel()))
    sep.update(compute_fk_metrics(_FakePolicy(pred), [batch], lambda b: b, lambda x: x, _FakeAccel(), fk))
    uni = compute_prediction_metrics(_FakePolicy(pred), [batch], lambda b: b, lambda x: x, _FakeAccel(),
                                     horizons=[0, 2], groups=ALLEX_ACTION_GROUPS, fk=fk)
    # fk_* keys are unsuffixed (always raw-space); L1/horizon keys are doubled into _norm/_unnorm.
    fk_keys = {k for k in sep if k.startswith("fk_")}
    l1h_keys = set(sep) - fk_keys
    assert set(uni) == fk_keys | {f"{k}_{sp}" for k in l1h_keys for sp in ("norm", "unnorm")}
    for k in fk_keys:
        assert uni[k] == pytest.approx(sep[k], rel=1e-5, abs=1e-5), f"{k}: {uni[k]} != {sep[k]}"
    for k in l1h_keys:
        for sp in ("norm", "unnorm"):
            assert uni[f"{k}_{sp}"] == pytest.approx(sep[k], rel=1e-5, abs=1e-5), f"{k}_{sp} != {sep[k]}"


def test_prediction_metrics_norm_and_unnorm_are_distinct_spaces():
    # With a non-identity preprocessor (normalize) + its inverse postprocessor (unnormalize) — the
    # ACT case — _norm must be the normalized-space L1 and _unnorm the raw-space L1, and _unnorm must
    # equal the L1 on the ORIGINAL (pre-normalization) actions. This is the guarantee that makes
    # _unnorm comparable across ACT and GR00T (whose identity processors leave it in radians).
    from learning.metrics.validation import ALLEX_ACTION_GROUPS, compute_prediction_metrics
    torch.manual_seed(0)
    raw_gt = torch.randn(BATCH, CHUNK, ACTION_DIM)
    raw_pred = torch.randn(BATCH, CHUNK, ACTION_DIM)
    scale = 4.0

    def pre(b):                                   # trainer normalizes the GT action it feeds the model
        b = dict(b)
        b[ACTION] = b[ACTION] / scale
        return b

    def post(x):                                  # postprocessor un-normalizes the model's prediction
        return x * scale

    # Model predicts in normalized space; postprocessor recovers raw_pred. _FakePolicy ignores the
    # (normalized) batch and returns this fixed normalized chunk.
    batch = {ACTION: raw_gt, "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool)}
    out = compute_prediction_metrics(_FakePolicy(raw_pred / scale), [batch], pre, post, _FakeAccel(),
                                     horizons=[0], groups=ALLEX_ACTION_GROUPS, fk=None)
    raw_l1 = (raw_pred - raw_gt).abs().mean().item()
    assert out["l1_all_unnorm"] == pytest.approx(raw_l1, rel=1e-5)          # raw (radian) space
    assert out["l1_all_norm"] == pytest.approx(raw_l1 / scale, rel=1e-5)    # normalized space = raw/scale
    assert out["l1_all_norm"] != pytest.approx(out["l1_all_unnorm"])        # genuinely different scales


def test_compute_fk_metrics_positive_for_perturbed_pred():
    pytest.importorskip("pytorch_kinematics")
    from learning.metrics.allex_fk import DEFAULT_URDF, AllexFK
    from learning.metrics.validation import compute_fk_metrics
    if not os.path.exists(DEFAULT_URDF):
        pytest.skip(f"URDF not found: {DEFAULT_URDF}")
    fk = AllexFK(DEFAULT_URDF, device="cpu")
    gt = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    pred = torch.zeros(BATCH, CHUNK, ACTION_DIM)
    pred[:, :, :] = 0.3  # perturb every joint by 0.3 rad
    batch = {ACTION: gt, "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool)}
    out = compute_fk_metrics(
        _FakePolicy(pred), [batch], lambda b: b, lambda x: x, _FakeAccel(), fk)
    assert out["fk_mm_finger_mean"] > 1.0       # position clearly nonzero (mm)
    assert out["fk_deg_wrist_mean"] > 0.0       # wrist orientation also moved (deg)
