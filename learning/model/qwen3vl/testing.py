# SPDX-FileCopyrightText: Copyright (c) 2026 WIRobotics. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable builders for a tiny-but-real Qwen3-VL — for any test that needs the model without the 3B
checkpoint. Builds a 2-layer / 2-vision-block Qwen3-VL from config on CPU: no weights, no tokenizer,
no GPU.

Importable as ``learning.model.qwen3vl.testing`` (kept OUT of ``tests/`` so pytest does not collect it),
so this package's tests, GR00T's, and any future consumer share ONE definition of "a tiny Qwen3-VL" and
its sample inputs — change the tiny arch in one place. Imports only torch + this package (no pytest), so
it is safe to import anywhere.
"""
import torch

from learning.model.qwen3vl.modeling import Qwen3VLConfig, Qwen3VLForConditionalGeneration

HIDDEN = 128                       # backbone hidden == vision out_hidden (kept equal for simple wiring)
IMAGE_TOKEN_ID = 150
VISION_START_ID = 152
VISION_END_ID = 153
VOCAB = 200
PATCH_DIM = 3 * 2 * 16 * 16        # in_channels * temporal_patch_size * patch_size**2
N_TEXT = 5                         # text tokens after the vision block
SEQ_LEN = 3 + N_TEXT               # vision_start + <image> (one merged token) + vision_end + text


def tiny_config(hidden_size: int = HIDDEN, **overrides) -> Qwen3VLConfig:
    """A minimal but structurally-real Qwen3-VL config (2 LLM layers, 2 vision blocks). ``overrides``
    replace top-level config keys (e.g. ``image_token_id=...``)."""
    kwargs = dict(
        text_config=dict(
            hidden_size=hidden_size, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, intermediate_size=256, vocab_size=VOCAB, hidden_act="silu", rms_norm_eps=1e-6,
            rope_theta=5000000.0, max_position_embeddings=512, attention_bias=False,
            tie_word_embeddings=True,
            rope_scaling=dict(mrope_interleaved=True, mrope_section=[6, 5, 5], rope_type="default"),
        ),
        vision_config=dict(
            depth=2, hidden_size=64, out_hidden_size=hidden_size, num_heads=2, in_channels=3,
            patch_size=16, spatial_merge_size=2, temporal_patch_size=2, intermediate_size=128,
            hidden_act="gelu_pytorch_tanh", num_position_embeddings=64, deepstack_visual_indexes=[0],
        ),
        image_token_id=IMAGE_TOKEN_ID, video_token_id=151, vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID, tie_word_embeddings=True,
    )
    kwargs.update(overrides)
    return Qwen3VLConfig(**kwargs)


def tiny_model(config: Qwen3VLConfig | None = None) -> Qwen3VLForConditionalGeneration:
    """Build a tiny Qwen3-VL from config (no checkpoint), eval mode, sdpa attention, CPU. Seed before
    calling if you need reproducible random init."""
    return Qwen3VLForConditionalGeneration._from_config(config or tiny_config()).eval()


def backbone_inputs(batch: int = 2, image_grid=(1, 2, 2)) -> dict:
    """One image per sample (grid t,h,w) + a short text tail — the tensors the backbone forward wants.
    ``pixel_values`` is ``torch.randn``, so seed before calling for reproducible inputs."""
    t, h, w = image_grid
    grid = torch.tensor([[t, h, w]] * batch, dtype=torch.long)
    pixel_values = torch.randn(int(grid.prod(dim=1).sum()), PATCH_DIM)
    text = list(range(10, 10 + N_TEXT))
    ids = torch.tensor([[VISION_START_ID, IMAGE_TOKEN_ID, VISION_END_ID, *text]] * batch, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": pixel_values, "image_grid_thw": grid}
