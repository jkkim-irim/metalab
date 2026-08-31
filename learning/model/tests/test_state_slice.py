"""Unit tests for the ACT observation.state slice (``state_keys``): ``build_act`` resolves the named
state groups to a leading-N prefix, rewrites the state feature shape + truncates its normalization
stats WITHOUT mutating the shared ds_meta.stats, and ``Normalize`` slices the raw incoming state
tensor to match (persisted across save/reload). Guards the S1 pos-only (``["q"]`` = 44-D) state path.
"""
import numpy as np
import pytest
import torch

from learning.model.act.build import build_act
from learning.model.act.configuration import ACTConfig, FeatureType, PolicyFeature
from learning.model.act.constants import ACTION, OBS_STATE
from learning.model.act.normalize import Normalize

FULL_STATE, SLICE, ACTION_DIM = 132, 44, 44
CAM = "observation.images.cam0"
IMG = (3, 64, 64)


def _features(state_dim=FULL_STATE):
    return {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        CAM: PolicyFeature(type=FeatureType.VISUAL, shape=IMG),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }


def _cfg(**over):
    kw = dict(chunk_size=4, n_action_steps=4, dim_model=64, n_heads=2, n_encoder_layers=1,
              n_decoder_layers=1, dim_feedforward=128, latent_dim=8, n_vae_encoder_layers=1,
              vision_backbone="resnet18", pretrained_backbone_weights=None)
    kw.update(over)
    return ACTConfig(**kw)


class _Meta:
    """Minimal ds_meta stand-in: lerobot-style features dict + numpy stats."""

    def __init__(self, stats):
        self.features = {
            OBS_STATE: {"shape": (FULL_STATE,), "dtype": "float32", "names": None},
            CAM: {"shape": IMG, "dtype": "video", "names": ["channel", "height", "width"]},
            ACTION: {"shape": (ACTION_DIM,), "dtype": "float32", "names": None},
        }
        self.stats = stats


def _stats():
    return {
        OBS_STATE: {"mean": np.arange(FULL_STATE, dtype=np.float32),
                    "std": np.ones(FULL_STATE, dtype=np.float32)},
        ACTION: {"mean": np.zeros(ACTION_DIM, dtype=np.float32),
                 "std": np.ones(ACTION_DIM, dtype=np.float32)},
        CAM: {"mean": np.zeros((3, 1, 1), dtype=np.float32),
              "std": np.ones((3, 1, 1), dtype=np.float32)},
    }


def _state_batch(dim):
    return {OBS_STATE: torch.randn(2, dim)}


def test_feature_shape_rewritten_and_tensor_sliced():
    cfg = _cfg(state_keys=["q"])           # q -> leading 44 dims
    _, pre, _, _, _ = build_act(cfg, _features())
    assert cfg.input_features[OBS_STATE].shape == (SLICE,)
    out = pre(_state_batch(FULL_STATE))
    assert out[OBS_STATE].shape == (2, SLICE)


def test_default_none_keeps_full_state():
    cfg = _cfg()  # state_keys default None -> full state, no slice
    _, pre, _, _, _ = build_act(cfg, _features())
    assert cfg.input_features[OBS_STATE].shape == (FULL_STATE,)
    out = pre(_state_batch(FULL_STATE))
    assert out[OBS_STATE].shape == (2, FULL_STATE)


def test_full_state_keys_keep_full_state():
    cfg = _cfg(state_keys=["q", "dq", "tau"])   # 132 == full -> no slice
    _, pre, _, _, _ = build_act(cfg, _features())
    assert cfg.input_features[OBS_STATE].shape == (FULL_STATE,)
    out = pre(_state_batch(FULL_STATE))
    assert out[OBS_STATE].shape == (2, FULL_STATE)


def test_fail_loud_when_slice_exceeds_dim():
    # state_keys resolve to more dims (q+dq=88) than the dataset's observation.state has (44).
    with pytest.raises(ValueError):
        build_act(_cfg(state_keys=["q", "dq"]), _features(state_dim=SLICE))


def test_fail_loud_on_non_contiguous_or_unknown():
    with pytest.raises(ValueError):
        build_act(_cfg(state_keys=["q", "tau"]), _features())   # skips dq -> non-contiguous
    with pytest.raises(KeyError):
        build_act(_cfg(state_keys=["bogus"]), _features())      # unknown group name


def test_stats_truncated_and_meta_not_mutated():
    meta = _Meta(_stats())
    _, pre, _, _, _ = build_act(_cfg(state_keys=["q"]), meta)
    # The preprocessor's registered OBS_STATE stats are truncated to the slice...
    mean_buf = pre._stats_for(OBS_STATE)["mean"]
    assert mean_buf.shape == (SLICE,)
    assert torch.equal(mean_buf, torch.arange(SLICE, dtype=torch.float32))
    # ...and the shared ds_meta.stats is untouched (still full length).
    assert len(meta.stats[OBS_STATE]["mean"]) == FULL_STATE


def test_state_slice_persisted_across_save_reload(tmp_path):
    _, pre, _, _, _ = build_act(_cfg(state_keys=["q"]), _Meta(_stats()))
    pre.save_pretrained(tmp_path)
    reloaded = Normalize.from_pretrained(tmp_path)
    out = reloaded(_state_batch(FULL_STATE))
    assert out[OBS_STATE].shape == (2, SLICE)
