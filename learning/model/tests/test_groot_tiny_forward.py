# SPDX-FileCopyrightText: Copyright (c) 2026 WIRobotics. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-free integration test for the internalized GR00T-N1.7 model.

Builds a TINY Gr00tN1d7 (2 LLM layers, 2 vision blocks, 128-d, 16-d state/action) purely from config
— no base checkpoint, no tokenizer, no GPU — and runs the real compute path end to end on CPU (sdpa):
the Qwen3-VL backbone (vision + LLM, constructed via ``_from_config``) feeds the flow-matching action
head, producing a finite training loss (+ backward) and an action chunk (denoising inference). This is
a fast wiring/shape guard for the vendored model — it catches dim mismatches and forward breakage in
seconds without the 3B weights.

Beyond wiring/shape checks, ``test_golden_*`` pins the actual output *values* against a committed
reference tensor (``fixtures/groot_tiny_golden.pt``): a slip in the vendored backbone (a swapped
activation, a dropped norm, a broken RoPE) still yields finite, correctly-shaped output — only a
value check catches it. The reference is generated on the GR00T venv itself (``GROOT_DUMP_GOLDEN=1
pytest -k golden``); regenerate + re-commit it whenever the shared tiny config (in
``learning.model.qwen3vl.testing``) legitimately changes. The vendored Qwen3-VL model + processor have
their own package-boundary tests under ``learning/model/qwen3vl/tests``.

Runs in the GR00T env (torch + diffusers; the Qwen3-VL config/backbone/BatchFeature are the vendored
`learning.model.qwen3vl`, no `transformers` package). Imports are unguarded on purpose: if the deps
are missing this fails loudly rather than silently skipping — GR00T tests belong on the GR00T venv.
"""
import os
import tempfile

import pytest
import torch

from learning.model.groot.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from learning.model.groot.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from learning.model.groot.model.modules.qwen3_backbone import Qwen3Backbone
from learning.model.qwen3vl.pretrained_base import BatchFeature
from learning.model.qwen3vl.testing import HIDDEN, SEQ_LEN, backbone_inputs, tiny_config

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "groot_tiny_golden.pt")

DIM = HIDDEN       # backbone hidden == DiT cross-attn == action-head embedding dims (kept equal)
BATCH = 2
STATE_DIM = 16
ACTION_DIM = 16
ACTION_HORIZON = 4


def _tiny_backbone_dir():
    """Save the shared tiny Qwen3-VL config (``qwen3vl.testing.tiny_config``) to a local dir whose path
    contains ``nvidia/Cosmos-Reason2-2B`` so ``get_backbone_cls`` dispatches to Qwen3Backbone and the
    resolver treats it as a local dir."""
    d = os.path.join(tempfile.mkdtemp(), "nvidia", "Cosmos-Reason2-2B")
    os.makedirs(d)
    tiny_config().save_pretrained(d)
    return d


def _tiny_config(model_dir):
    return Gr00tN1d7Config(
        model_name=model_dir, backbone_embedding_dim=DIM, select_layer=2, use_flash_attention=False,
        load_bf16=False, max_state_dim=STATE_DIM, max_action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON, hidden_size=DIM, input_embedding_dim=DIM, state_history_length=1,
        max_num_embodiments=2, max_seq_len=64, use_vlln=True, add_pos_embed=True,
        use_alternate_vl_dit=True, attend_text_every_n_blocks=1, state_dropout_prob=0.0,
        num_inference_timesteps=2,
        diffusion_model_cfg=dict(
            positional_embeddings=None, num_layers=2, num_attention_heads=4, attention_head_dim=32,
            norm_type="ada_norm", dropout=0.0, final_dropout=False, output_dim=DIM,
            interleave_self_attention=True,
        ),
    )


def _backbone_inputs():
    return backbone_inputs(batch=BATCH)


def _action_inputs():
    return BatchFeature(dict(
        state=torch.randn(BATCH, 1, STATE_DIM), action=torch.randn(BATCH, ACTION_HORIZON, ACTION_DIM),
        action_mask=torch.ones(BATCH, ACTION_HORIZON, ACTION_DIM),
        embodiment_id=torch.tensor([0, 1], dtype=torch.long)))


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    model_dir = _tiny_backbone_dir()
    cfg = _tiny_config(model_dir)
    backbone = Qwen3Backbone(model_name=model_dir, tune_llm=True, tune_visual=True, select_layer=2,
                             use_flash_attention=False, load_bf16=False, transformers_loading_kwargs={})
    head = Gr00tN1d7ActionHead(cfg)
    return backbone, head


def test_builds_from_config_without_checkpoint(tiny_model):
    backbone, head = tiny_model
    n = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters())
    assert 0 < n < 5_000_000  # tiny — construction never touches the 3B base checkpoint


def test_backbone_then_head_forward_backward(tiny_model):
    backbone, head = tiny_model
    backbone.train()
    head.train()
    for p in head.parameters():
        p.grad = None
    features = backbone(_backbone_inputs())
    assert tuple(features["backbone_features"].shape) == (BATCH, SEQ_LEN, DIM)
    out = head(features, _action_inputs())
    loss = out["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_get_action_inference(tiny_model):
    backbone, head = tiny_model
    backbone.eval()
    head.eval()
    with torch.no_grad():
        features = backbone(_backbone_inputs())
        obs = BatchFeature(dict(state=torch.randn(BATCH, 1, STATE_DIM),
                                embodiment_id=torch.tensor([0, 1], dtype=torch.long)))
        out = head.get_action(features, obs, None)
    action_pred = out["action_pred"]
    assert tuple(action_pred.shape) == (BATCH, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(action_pred).all()


def _build_deterministic():
    """A fresh, fully-seeded tiny model — independent of the module fixture so the golden values are
    reproducible regardless of test order. Seed(0) fixes the (random) weight init; construction order
    is fixed, so on the pinned GR00T venv the init is deterministic."""
    torch.manual_seed(0)
    model_dir = _tiny_backbone_dir()
    backbone = Qwen3Backbone(model_name=model_dir, tune_llm=True, tune_visual=True, select_layer=2,
                             use_flash_attention=False, load_bf16=False, transformers_loading_kwargs={})
    head = Gr00tN1d7ActionHead(_tiny_config(model_dir))
    backbone.eval()
    head.eval()
    return backbone, head


def test_golden_backbone_and_action_values():
    """Value regression: pin backbone_features + action_pred to a committed reference. A pruning slip in
    the vendored qwen3vl that keeps shapes/finiteness (swapped activation, dropped norm, broken RoPE)
    diverges here by orders of magnitude, past the tolerance. Regenerate: GROOT_DUMP_GOLDEN=1 -k golden."""
    backbone, head = _build_deterministic()
    torch.manual_seed(1)                                   # inputs seeded independently of init RNG
    bb_in = _backbone_inputs()
    state = torch.randn(BATCH, 1, STATE_DIM)
    with torch.no_grad():
        features = backbone(bb_in)
        obs = BatchFeature(dict(state=state, embodiment_id=torch.tensor([0, 1], dtype=torch.long)))
        torch.manual_seed(2)                               # denoising noise seeded for reproducibility
        out = head.get_action(features, obs, None)
    got = {"backbone_features": features["backbone_features"].float(),
           "action_pred": out["action_pred"].float()}

    if os.environ.get("GROOT_DUMP_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        torch.save(got, GOLDEN_PATH)
        pytest.skip(f"dumped golden reference to {GOLDEN_PATH}")

    assert os.path.exists(GOLDEN_PATH), f"missing golden fixture {GOLDEN_PATH} (GROOT_DUMP_GOLDEN=1 to make it)"
    ref = torch.load(GOLDEN_PATH)
    for key in got:
        torch.testing.assert_close(got[key], ref[key], atol=1e-4, rtol=1e-3,
                                   msg=lambda m, k=key: f"{k} diverged from golden — {m}")
