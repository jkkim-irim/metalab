"""Unit tests for the self-contained ACT model + builder (learning/model/act/).

Builds a tiny ACT policy via the real ``build_act`` and checks the production paths the trainer
uses: forward() returns a finite L1+KL loss with grads, predict_action_chunk() has the right shape,
the optimizer has ACT's two param groups (backbone at its own lr), and save/from_pretrained
round-trips weights bit-for-bit. No lerobot needed (these are intrinsic-correctness checks; the
lerobot bit-equivalence proof lives in the PR history).

Run in the node env from the repo root:  python -m pytest learning -q
"""
import torch

from learning.model.act.build import build_act
from learning.model.act.configuration import ACTConfig, FeatureType, PolicyFeature
from learning.model.act.constants import ACTION, OBS_STATE
from learning.model.act.modeling import ACTPolicy

STATE_DIM, ACTION_DIM, CHUNK, BATCH = 8, 4, 4, 2
IMG = (3, 64, 64)
CAMS = ["observation.images.cam0", "observation.images.cam1"]


def _features():
    f = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
    for c in CAMS:
        f[c] = PolicyFeature(type=FeatureType.VISUAL, shape=IMG)
    f[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))
    return f


def _tiny_cfg(**over):
    kwargs = dict(
        chunk_size=CHUNK, n_action_steps=CHUNK, dim_model=64, n_heads=2,
        n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=128,
        latent_dim=8, n_vae_encoder_layers=1, vision_backbone="resnet18",
        pretrained_backbone_weights=None,  # no network download
    )
    kwargs.update(over)
    return ACTConfig(**kwargs)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {
        OBS_STATE: torch.randn(BATCH, STATE_DIM, generator=g),
        ACTION: torch.randn(BATCH, CHUNK, ACTION_DIM, generator=g),
        "action_is_pad": torch.zeros(BATCH, CHUNK, dtype=torch.bool),
    }
    for c in CAMS:
        b[c] = torch.rand(BATCH, *IMG, generator=g)
    return b


def test_build_returns_components_and_splits_features():
    cfg = _tiny_cfg()
    policy, pre, post, opt, sched = build_act(cfg, _features())
    assert isinstance(policy, ACTPolicy)
    assert isinstance(opt, torch.optim.AdamW)
    assert sched is None
    # outputs = ACTION feature only; inputs = state + cameras.
    assert set(cfg.output_features) == {ACTION}
    assert set(cfg.input_features) == {OBS_STATE, *CAMS}


def test_forward_train_returns_finite_loss_with_grad():
    policy, *_ = build_act(_tiny_cfg(), _features())
    policy.train()
    loss, loss_dict = policy.forward(_batch())
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict and "kld_loss" in loss_dict  # use_vae default True
    loss.backward()
    assert any(p.grad is not None for p in policy.parameters())


def test_predict_action_chunk_shape():
    policy, *_ = build_act(_tiny_cfg(), _features())
    actions = policy.predict_action_chunk(_batch(seed=1))
    assert actions.shape == (BATCH, CHUNK, ACTION_DIM)


def test_optimizer_has_backbone_param_group_at_its_own_lr():
    cfg = _tiny_cfg(optimizer_lr=1e-3, optimizer_lr_backbone=1e-4)
    _, _, _, opt, _ = build_act(cfg, _features())
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-3        # everything-but-backbone at base lr
    assert opt.param_groups[1]["lr"] == 1e-4        # backbone at lr_backbone


def test_save_pretrained_round_trips_weights(tmp_path):
    torch.manual_seed(0)
    policy, *_ = build_act(_tiny_cfg(), _features())
    policy.save_pretrained(tmp_path)
    assert (tmp_path / "model.pt").is_file()
    assert (tmp_path / "config.json").is_file()

    reloaded = ACTPolicy.from_pretrained(tmp_path)
    sd_a, sd_b = policy.state_dict(), reloaded.state_dict()
    assert set(sd_a) == set(sd_b)
    for k in sd_a:
        assert torch.equal(sd_a[k].cpu(), sd_b[k].cpu()), f"weight mismatch at {k}"
