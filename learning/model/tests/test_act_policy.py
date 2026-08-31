"""Unit tests for the ACT outer builder (``learning/model/act_policy.py``): the optional
warmup->cosine LR scheduler (LeRobot's ACT has none; ``build_act`` still returns None — pinned in
test_act). The scheduler is built here because the run horizon (``cfg.steps``) is only in scope here.
"""
from types import SimpleNamespace

import pytest
from torch.optim.lr_scheduler import LambdaLR

from learning.model import act_policy
from learning.model.act.configuration import ACTConfig
from learning.model.act.constants import ACTION, OBS_STATE

CAM = "observation.images.cam0"


class _Meta:
    features = {
        OBS_STATE: {"shape": (44,), "dtype": "float32", "names": None},
        CAM: {"shape": (3, 64, 64), "dtype": "video", "names": ["channel", "height", "width"]},
        ACTION: {"shape": (44,), "dtype": "float32", "names": None},
    }
    stats = None


class _Dataset:
    meta = _Meta()


def _cfg(steps=1000, **policy_over):
    kw = dict(chunk_size=4, n_action_steps=4, dim_model=64, n_heads=2, n_encoder_layers=1,
              n_decoder_layers=1, dim_feedforward=128, latent_dim=8, n_vae_encoder_layers=1,
              vision_backbone="resnet18", pretrained_backbone_weights=None)
    kw.update(policy_over)
    return SimpleNamespace(policy=ACTConfig(**kw), steps=steps)


def test_no_scheduler_by_default():
    *_, sched = act_policy.build(_cfg(), _Dataset())
    assert sched is None


def test_cosine_scheduler_warms_up_then_peaks():
    *_, opt, sched = act_policy.build(
        _cfg(steps=100, lr_scheduler="cosine", warmup_steps=10), _Dataset()
    )
    assert isinstance(sched, LambdaLR)
    base = sched.base_lrs[0]
    # Step 0 of warmup -> factor 0.
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)
    sched.step()  # step 1/10 of warmup
    assert opt.param_groups[0]["lr"] == pytest.approx(base * 0.1, rel=1e-6)
    for _ in range(9):  # advance to step 10 = end of warmup, start of cosine (progress 0 -> factor 1)
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(base, rel=1e-6)


def test_wsd_scheduler_holds_peak_then_anneals():
    # steps=100, warmup=10, decay_ratio=0.2 -> decay_steps=20, stable body [10, 80), anneal [80, 100].
    *_, opt, sched = act_policy.build(
        _cfg(steps=100, lr_scheduler="wsd", warmup_steps=10, lr_decay_ratio=0.2), _Dataset()
    )
    assert isinstance(sched, LambdaLR)
    base = sched.base_lrs[0]
    for _ in range(10):  # -> step 10: end of warmup, start of stable
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(base, rel=1e-6)  # at peak
    for _ in range(40):  # -> step 50: still in the constant body (this is what cosine-over-total lacks)
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(base, rel=1e-6)  # STILL peak — decoupled from length
    for _ in range(50):  # -> step 100: end of the final cosine anneal
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_unknown_scheduler_fails_loud():
    with pytest.raises(ValueError):
        act_policy.build(_cfg(lr_scheduler="linear"), _Dataset())
