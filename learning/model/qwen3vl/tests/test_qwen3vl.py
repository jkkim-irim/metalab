# SPDX-FileCopyrightText: Copyright (c) 2026 WIRobotics. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package-boundary tests for the vendored Qwen3-VL: exercise it DIRECTLY (no GR00T wrapper), the way a
consumer uses it — build from config, ``forward(output_hidden_states=True)``, round-trip the config, and
run the image processor. Checkpoint-free (tiny model from config via `learning.model.qwen3vl.testing`,
CPU/sdpa); runs in the GR00T venv (torch + this package, no `transformers`).

``test_forward_golden`` pins the backbone's last hidden state bit-exactly to a committed reference
(``fixtures/tiny_hidden_states.pt``): a numeric regression in the vendored modeling (swapped activation,
dropped norm, broken RoPE) diverges past the tolerance. Regenerate on the GR00T venv with
``QWEN3VL_DUMP_GOLDEN=1 pytest learning/model/qwen3vl`` whenever the tiny arch in `testing.py` changes.
"""
import os

import numpy as np
import pytest
import torch

from learning.model.qwen3vl.modeling import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from learning.model.qwen3vl.processing import Qwen3VLImageProcessor, smart_resize
from learning.model.qwen3vl.testing import HIDDEN, SEQ_LEN, backbone_inputs, tiny_config, tiny_model

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "tiny_hidden_states.pt")


def _forward_last_hidden():
    """Deterministic tiny forward -> last hidden state. seed(0) fixes init, seed(1) fixes the inputs."""
    torch.manual_seed(0)
    model = tiny_model()
    torch.manual_seed(1)
    inputs = backbone_inputs()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1].float()


def test_build_from_config_without_checkpoint():
    model = tiny_model()
    n = sum(p.numel() for p in model.parameters())
    assert 0 < n < 5_000_000  # tiny — construction never touches the 3B checkpoint
    assert isinstance(model, Qwen3VLForConditionalGeneration)
    assert model.config.image_token_id == 150


def test_forward_output_hidden_states():
    hs = _forward_last_hidden()
    assert tuple(hs.shape) == (2, SEQ_LEN, HIDDEN)
    assert torch.isfinite(hs).all()


def test_forward_golden():
    """Numeric regression guard for the vendored modeling (see module docstring)."""
    hs = _forward_last_hidden()
    if os.environ.get("QWEN3VL_DUMP_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        torch.save(hs, GOLDEN_PATH)
        pytest.skip(f"dumped golden reference to {GOLDEN_PATH}")
    assert os.path.exists(GOLDEN_PATH), f"missing golden {GOLDEN_PATH} (QWEN3VL_DUMP_GOLDEN=1 to make it)"
    torch.testing.assert_close(hs, torch.load(GOLDEN_PATH), atol=1e-4, rtol=1e-3)


def test_config_round_trip(tmp_path):
    cfg = tiny_config()
    cfg.save_pretrained(tmp_path)
    reloaded = Qwen3VLConfig.from_pretrained(tmp_path)
    assert reloaded.image_token_id == cfg.image_token_id
    assert reloaded.vision_start_token_id == cfg.vision_start_token_id
    assert reloaded.text_config.hidden_size == cfg.text_config.hidden_size
    assert reloaded.vision_config.out_hidden_size == cfg.vision_config.out_hidden_size


def test_smart_resize_divisible_and_bounded():
    """smart_resize keeps H,W divisible by factor, total pixels within [min,max], aspect ~preserved."""
    factor, min_px, max_px = 32, 65536, 16777216           # patch16 * merge2
    for h, w in [(120, 200), (33, 4000), (2000, 2000), (7, 9)]:
        rh, rw = smart_resize(h, w, factor, min_px, max_px)
        assert rh % factor == 0 and rw % factor == 0
        assert min_px <= rh * rw <= max_px
        assert abs((rh / rw) - (h / w)) < 0.15 * (h / w)   # aspect roughly kept


def test_image_processor_preprocess_shapes():
    """Qwen3VLImageProcessor.preprocess: patch extraction -> (num_patches, C*T*P*P) + [t,h,w] grid."""
    proc = Qwen3VLImageProcessor()                         # defaults: patch16, temporal2, merge2
    img = np.random.default_rng(0).integers(0, 256, (120, 200, 3), dtype=np.uint8)
    pixel_values, grid = proc.preprocess([img])
    assert tuple(grid.shape) == (1, 3)
    gt, gh, gw = grid[0].tolist()
    assert gt == 1                                         # single image -> one temporal group
    assert gh % proc.merge_size == 0 and gw % proc.merge_size == 0
    assert pixel_values.shape[0] == gt * gh * gw
    assert pixel_values.shape[1] == 3 * proc.temporal_patch_size * proc.patch_size ** 2  # 1536
    assert torch.isfinite(pixel_values).all()
