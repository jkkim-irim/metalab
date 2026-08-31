"""Unit tests for ACT observation history (n_obs_steps > 1): the delta-index window, the guard, the
temporal positional embedding (created only when history is on, so n_obs_steps=1 stays byte-identical
to the pre-history model — see test_act.py for the single-frame forward/predict pins), and a k>1
forward/predict on a stacked (B, T, ...) batch.
"""
import pytest
import torch

from learning.model.act.build import build_act
from learning.model.act.configuration import ACTConfig, FeatureType, PolicyFeature
from learning.model.act.constants import ACTION, OBS_STATE

STATE_DIM, ACTION_DIM, CHUNK, BATCH = 8, 4, 4, 2
IMG = (3, 64, 64)
CAMS = ["observation.images.cam0", "observation.images.cam1"]


def _features():
    f = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
    for c in CAMS:
        f[c] = PolicyFeature(type=FeatureType.VISUAL, shape=IMG)
    f[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))
    return f


def _cfg(**over):
    kw = dict(chunk_size=CHUNK, n_action_steps=CHUNK, dim_model=64, n_heads=2, n_encoder_layers=1,
              n_decoder_layers=1, dim_feedforward=128, latent_dim=8, n_vae_encoder_layers=1,
              vision_backbone="resnet18", pretrained_backbone_weights=None)
    kw.update(over)
    return ACTConfig(**kw)


def _hist_batch(n_obs, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {
        OBS_STATE: torch.randn(BATCH, n_obs, STATE_DIM, generator=g),
        ACTION: torch.randn(BATCH, CHUNK, ACTION_DIM, generator=g),
        "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool),
    }
    for c in CAMS:
        b[c] = torch.rand(BATCH, n_obs, *IMG, generator=g)
    return b


def test_observation_delta_indices():
    assert _cfg(n_obs_steps=1).observation_delta_indices is None
    assert _cfg(n_obs_steps=2).observation_delta_indices == [-1, 0]
    assert _cfg(n_obs_steps=3).observation_delta_indices == [-2, -1, 0]


def test_n_obs_steps_guard_fails_loud():
    with pytest.raises(ValueError):
        _cfg(n_obs_steps=0)


def test_temporal_embed_created_only_with_history():
    p1, *_ = build_act(_cfg(n_obs_steps=1), _features())
    assert not hasattr(p1.model, "encoder_temporal_pos_embed")  # n_obs=1 -> byte-identical module set
    p3, *_ = build_act(_cfg(n_obs_steps=3), _features())
    assert hasattr(p3.model, "encoder_temporal_pos_embed")
    assert p3.model.encoder_temporal_pos_embed.weight.shape[0] == 3


def test_history_forward_and_predict_shapes():
    policy, *_ = build_act(_cfg(n_obs_steps=3), _features())
    policy.train()
    loss, loss_dict = policy.forward(_hist_batch(3))
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict
    loss.backward()
    assert any(p.grad is not None for p in policy.parameters())
    actions = policy.predict_action_chunk(_hist_batch(3, seed=1))
    assert actions.shape == (BATCH, CHUNK, ACTION_DIM)


def test_history_composes_with_no_vae():
    # history + plain-DETR (use_vae False) path
    policy, *_ = build_act(_cfg(n_obs_steps=2, use_vae=False), _features())
    policy.train()
    loss, loss_dict = policy.forward(_hist_batch(2))
    assert torch.isfinite(loss)
    assert "kld_loss" not in loss_dict
