"""Unit tests for the self-contained (un)normalizers (learning/model/act/normalize.py).

Checks the MEAN_STD / MIN_MAX transform math, broadcast of image stats stored as (C,1,1), the
IDENTITY fall-throughs (no stats / IDENTITY mode / a missing required stat), the bare-tensor
Unnormalize path the postprocessor uses, and Normalize∘Unnormalize round-trip. No lerobot needed.

Run in the node env from the repo root:  python -m pytest learning -q
"""
import torch

from learning.model.act.configuration import FeatureType, NormalizationMode, PolicyFeature
from learning.model.act.constants import ACTION, OBS_STATE
from learning.model.act.normalize import Normalize, Unnormalize


def test_mean_std_forward_values():
    feats = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD}
    stats = {
        OBS_STATE: {"mean": torch.tensor([1.0, 2.0, 3.0]), "std": torch.tensor([1.0, 1.0, 2.0])},
        ACTION: {"mean": torch.tensor([0.0, 10.0]), "std": torch.tensor([2.0, 5.0])},
    }
    n = Normalize(feats, norm_map, stats)
    out = n({OBS_STATE: torch.tensor([[2.0, 4.0, 7.0]]), ACTION: torch.tensor([[4.0, 20.0]])})
    assert torch.allclose(out[OBS_STATE], torch.tensor([[1.0, 2.0, 2.0]]), atol=1e-5)
    assert torch.allclose(out[ACTION], torch.tensor([[2.0, 2.0]]), atol=1e-5)


def test_mean_std_roundtrip():
    feats = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    norm_map = {"ACTION": NormalizationMode.MEAN_STD}
    stats = {ACTION: {"mean": torch.tensor([0.0, 10.0]), "std": torch.tensor([2.0, 5.0])}}
    n = Normalize(feats, norm_map, stats)
    un = Unnormalize(feats, norm_map, stats)
    a = torch.randn(4, 2)
    back = un(n({ACTION: a}))[ACTION]
    assert torch.allclose(a, back, atol=1e-5)


def test_min_max_forward_and_roundtrip():
    feats = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    norm_map = {"STATE": NormalizationMode.MIN_MAX}
    stats = {OBS_STATE: {"min": torch.tensor([0.0, 0.0]), "max": torch.tensor([10.0, 4.0])}}
    n = Normalize(feats, norm_map, stats)
    un = Unnormalize(feats, norm_map, stats)
    out = n({OBS_STATE: torch.tensor([[5.0, 1.0]])})[OBS_STATE]
    assert torch.allclose(out, torch.tensor([[0.0, -0.5]]), atol=1e-5)  # 2*(x-min)/(max-min)-1
    x = torch.rand(3, 2) * torch.tensor([10.0, 4.0])
    assert torch.allclose(un(n({OBS_STATE: x}))[OBS_STATE], x, atol=1e-5)


def test_image_stats_broadcast_over_bchw():
    feats = {"observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2))}
    norm_map = {"VISUAL": NormalizationMode.MEAN_STD}
    half = torch.full((3, 1, 1), 0.5)  # stored (C,1,1) so it broadcasts against (B,C,H,W)
    stats = {"observation.images.cam": {"mean": half, "std": half}}
    n = Normalize(feats, norm_map, stats)
    out = n({"observation.images.cam": torch.ones(2, 3, 2, 2)})["observation.images.cam"]
    assert torch.allclose(out, torch.ones(2, 3, 2, 2), atol=1e-5)  # (1-0.5)/0.5 == 1


def test_identity_when_stats_none():
    feats = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    n = Normalize(feats, {"STATE": NormalizationMode.MEAN_STD}, stats=None)
    x = torch.randn(2, 2)
    assert torch.equal(n({OBS_STATE: x})[OBS_STATE], x)


def test_identity_mode_passes_through_even_with_stats():
    feats = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    stats = {OBS_STATE: {"mean": torch.zeros(2), "std": torch.ones(2)}}
    n = Normalize(feats, {"STATE": NormalizationMode.IDENTITY}, stats)
    x = torch.randn(2, 2)
    assert torch.equal(n({OBS_STATE: x})[OBS_STATE], x)


def test_missing_required_stat_falls_through_to_identity():
    # mean present but std absent -> the key must not be half-normalized (LeRobot behaviour).
    feats = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    stats = {OBS_STATE: {"mean": torch.zeros(2)}}
    n = Normalize(feats, {"STATE": NormalizationMode.MEAN_STD}, stats)
    x = torch.randn(2, 2)
    assert torch.equal(n({OBS_STATE: x})[OBS_STATE], x)


def test_unnormalize_bare_action_tensor():
    feats = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    stats = {ACTION: {"mean": torch.tensor([1.0, 1.0]), "std": torch.tensor([2.0, 2.0])}}
    un = Unnormalize(feats, {"ACTION": NormalizationMode.MEAN_STD}, stats)
    out = un(torch.zeros(3, 2))  # bare tensor -> x*std + mean
    assert torch.allclose(out, torch.ones(3, 2), atol=1e-5)


def test_save_from_pretrained_round_trips_stats(tmp_path):
    # Regression for the reviewed bug: stat buffers are persistent=False, so without an explicit
    # save_pretrained a checkpoint carries no stats and reloads as IDENTITY. The reloaded normalizer
    # must reproduce the original's output exactly.
    feats = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    norm_map = {"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD}
    stats = {
        OBS_STATE: {"mean": torch.tensor([1.0, 2.0, 3.0]), "std": torch.tensor([0.5, 1.0, 2.0])},
        ACTION: {"mean": torch.tensor([0.0, 10.0]), "std": torch.tensor([2.0, 5.0])},
    }
    n = Normalize(feats, norm_map, stats)
    n.save_pretrained(tmp_path)
    assert (tmp_path / "preprocessor_stats.pt").is_file()

    n2 = Normalize.from_pretrained(tmp_path)
    x = {OBS_STATE: torch.randn(4, 3), ACTION: torch.randn(4, 2)}
    out1, out2 = n(x), n2(x)
    assert torch.allclose(out1[OBS_STATE], out2[OBS_STATE], atol=1e-6)
    assert torch.allclose(out1[ACTION], out2[ACTION], atol=1e-6)
