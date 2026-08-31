"""Unit tests for checkpoint save/reload (learning/utils/train_utils.py).

Recreates the (lerobot-free) save_checkpoint round-trip coverage and adds a regression test for the
reviewed bug: the normalization stats must be persisted into the checkpoint (the Normalize buffers
are persistent=False, so a missing save_pretrained would silently drop them and reload as IDENTITY).

Run in the node env from the repo root:  python -m pytest learning -q
"""
import json

import torch

from learning.configs.config import DatasetConfig, TrainPipelineConfig
from learning.model.act.build import build_act
from learning.model.act.configuration import (
    ACTConfig,
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)
from learning.model.act.constants import ACTION, OBS_STATE
from learning.model.act.modeling import ACTPolicy
from learning.model.act.normalize import Normalize, Unnormalize
from learning.utils.train_utils import save_checkpoint, update_last_checkpoint

STATE_DIM, ACTION_DIM = 8, 4


def _tiny_policy():
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    cfg_pol = ACTConfig(
        chunk_size=4, n_action_steps=4, dim_model=64, n_heads=2, n_encoder_layers=1,
        n_decoder_layers=1, dim_feedforward=128, latent_dim=8, n_vae_encoder_layers=1,
        vision_backbone="resnet18", pretrained_backbone_weights=None,
    )
    policy, _, _, opt, sched = build_act(cfg_pol, features)
    return cfg_pol, policy, opt, sched


def _processors_with_stats():
    feats = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    norm_map = {"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MEAN_STD}
    stats = {
        OBS_STATE: {"mean": torch.randn(STATE_DIM), "std": torch.rand(STATE_DIM) + 0.1},
        ACTION: {"mean": torch.randn(ACTION_DIM), "std": torch.rand(ACTION_DIM) + 0.1},
    }
    pre = Normalize(feats, norm_map, stats)
    post = Unnormalize({ACTION: feats[ACTION]}, norm_map, stats)
    return pre, post


def test_save_checkpoint_round_trips(tmp_path):
    torch.manual_seed(0)
    cfg_pol, policy, opt, sched = _tiny_policy()
    pre, post = _processors_with_stats()
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="r/x", s3_uri="s3://b/x"), policy=cfg_pol)

    ckpt = tmp_path / "checkpoints" / "epoch_00001"
    ckpt.mkdir(parents=True)
    save_checkpoint(checkpoint_dir=ckpt, step=42, cfg=cfg, policy=policy, optimizer=opt,
                    scheduler=sched, preprocessor=pre, postprocessor=post)
    update_last_checkpoint(ckpt)

    pretrained = ckpt / "pretrained_model"
    training_state = ckpt / "training_state"
    for f in ["config.json", "model.pt", "train_config.json"]:
        assert (pretrained / f).is_file()
    for f in ["training_step.json", "optimizer_state.pt", "rng_state.pt"]:
        assert (training_state / f).is_file()
    assert (tmp_path / "checkpoints" / "last").resolve() == ckpt.resolve()

    # Policy weights round-trip bit-for-bit.
    reloaded = ACTPolicy.from_pretrained(pretrained)
    sd_a, sd_b = policy.state_dict(), reloaded.state_dict()
    assert set(sd_a) == set(sd_b)
    for k in sd_a:
        assert torch.equal(sd_a[k].cpu(), sd_b[k].cpu()), f"weight mismatch at {k}"

    assert json.loads((training_state / "training_step.json").read_text())["step"] == 42
    assert json.loads((pretrained / "train_config.json").read_text())["policy"]["type"] == "act"


def test_save_checkpoint_persists_normalization_stats(tmp_path):
    # The regression: stats must be written (and reload to identical normalization), not silently
    # dropped. Without save_pretrained the reloaded normalizer would act as IDENTITY.
    cfg_pol, policy, opt, sched = _tiny_policy()
    pre, post = _processors_with_stats()
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="r/x", s3_uri="s3://b/x"), policy=cfg_pol)

    ckpt = tmp_path / "checkpoints" / "epoch_00001"
    ckpt.mkdir(parents=True)
    save_checkpoint(checkpoint_dir=ckpt, step=1, cfg=cfg, policy=policy, optimizer=opt,
                    scheduler=sched, preprocessor=pre, postprocessor=post)

    pretrained = ckpt / "pretrained_model"
    assert (pretrained / "preprocessor_stats.pt").is_file()
    assert (pretrained / "postprocessor_stats.pt").is_file()

    pre2 = Normalize.from_pretrained(pretrained)
    x = {OBS_STATE: torch.randn(2, STATE_DIM), ACTION: torch.randn(2, ACTION_DIM)}
    out1, out2 = pre(x), pre2(x)
    assert torch.allclose(out1[OBS_STATE], out2[OBS_STATE], atol=1e-6)
    assert torch.allclose(out1[ACTION], out2[ACTION], atol=1e-6)
    # And not IDENTITY (the bug would make the reloaded output == the raw input).
    assert not torch.allclose(out2[OBS_STATE], x[OBS_STATE], atol=1e-6)
