"""Unit tests for the one device-agnostic image augmentation (learning/data/image_aug.py).

Runs on CPU (pure torch). Covers: config-driven ranges, shape handling, per-sample independence,
temporal cohesion, no-black-border geometry, determinism, and — the point of the refactor — that the
cpu and gpu paths produce the SAME augmented output. Tests assert on OUTPUT only, not on how either
path is wired internally.
"""
import torch

from learning.data.image_aug import (
    AUG_ORDER,
    ImageAugConfig,
    _hsv_to_rgb,
    _rgb_to_hsv,
    apply_image_aug,
    make_cpu_transform,
)

CAM = "observation.images.camera_1"


def _color_batch(b=8, H=32, W=32):
    # a color image (R != G != B) so saturation/hue actually do something
    r = torch.linspace(0.2, 0.8, W).view(1, 1, W).expand(1, H, W)
    g = torch.linspace(0.8, 0.2, H).view(1, H, 1).expand(1, H, W)
    bl = torch.full((1, H, W), 0.5)
    return torch.cat([r, g, bl], 0).unsqueeze(0).repeat(b, 1, 1, 1).clone()


def _identity_cfg():
    """A config whose ranges are all the identity -> a full no-op (both photometric AND geometric)."""
    return ImageAugConfig(
        brightness=(1.0, 1.0), contrast=(1.0, 1.0), saturation=(1.0, 1.0),
        hue=(0.0, 0.0), sharpness=(1.0, 1.0), rotate_deg=(0.0, 0.0), zoom=(1.0, 1.0),
    )


def test_hsv_roundtrip_is_identity():
    x = torch.rand(4, 3, 16, 16)
    assert torch.allclose(x, _hsv_to_rgb(_rgb_to_hsv(x)), atol=1e-4)


def test_aug_order_is_the_documented_fixed_order():
    assert AUG_ORDER == ("brightness", "contrast", "saturation", "hue", "sharpness",
                         "rotate_deg", "zoom")
    # every AUG_ORDER name is a real config field (so viz + executor can't drift from the config)
    fields = set(ImageAugConfig().__dataclass_fields__)
    assert set(AUG_ORDER) == fields


def test_identity_ranges_are_a_full_noop():
    x = _color_batch(4)
    out = apply_image_aug(x.clone(), _identity_cfg())
    assert torch.allclose(out, x, atol=1e-5), "identity ranges must not change the image"


def test_output_in_unit_range_and_shape_preserved():
    for shape in [(3, 32, 32), (8, 3, 32, 32), (4, 2, 3, 32, 32)]:
        x = torch.rand(*shape)
        out = apply_image_aug(x, ImageAugConfig())
        assert out.shape == x.shape
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_chw_single_frame_matches_batch_of_one():
    torch.manual_seed(0)
    one = apply_image_aug(_color_batch(1)[0], ImageAugConfig())     # (C,H,W)
    torch.manual_seed(0)
    batched = apply_image_aug(_color_batch(1), ImageAugConfig())    # (1,C,H,W)
    assert one.shape == (3, 32, 32)
    assert torch.allclose(one, batched[0], atol=1e-6)


def test_per_sample_independent_no_black_border():
    torch.manual_seed(0)
    out = apply_image_aug(_color_batch(8), ImageAugConfig())
    diffs = [(out[i] - out[0]).abs().mean().item() for i in range(1, 8)]
    assert all(d > 1e-4 for d in diffs), "each sample must get its own draw"
    # zoom-in covers the <=4deg rotation -> no zero-filled border
    assert (out == 0).float().mean().item() < 1e-3


def test_history_is_temporally_cohesive():
    # (B,T,C,H,W): the T frames of one sample share its draw (a coherent clip); samples differ.
    B, T = 6, 4
    x = _color_batch(B).unsqueeze(1).repeat(1, T, 1, 1, 1).clone()   # T IDENTICAL frames/sample
    torch.manual_seed(0)
    out = apply_image_aug(x, ImageAugConfig())
    assert out.shape == (B, T, 3, 32, 32)
    for i in range(B):
        assert torch.allclose(out[i], out[i, 0:1].expand_as(out[i]), atol=1e-6), \
            "history frames of a sample were augmented inconsistently"
    diffs = [(out[i, 0] - out[0, 0]).abs().mean().item() for i in range(1, B)]
    assert all(d > 1e-4 for d in diffs)


def test_config_ranges_drive_the_augmentation():
    # widen brightness far past default -> mean brightness variance across samples grows.
    x = _color_batch(64)
    torch.manual_seed(0)
    narrow = apply_image_aug(x.clone(), _identity_cfg())            # brightness (1,1)
    wide_cfg = _identity_cfg()
    wide_cfg.brightness = (0.3, 1.7)
    torch.manual_seed(0)
    wide = apply_image_aug(x.clone(), wide_cfg)
    v_narrow = narrow.mean((1, 2, 3)).var().item()
    v_wide = wide.mean((1, 2, 3)).var().item()
    assert v_wide > v_narrow + 1e-4, "widening a config range must widen the applied augmentation"


def test_deterministic_with_seeded_generator():
    x = _color_batch(4)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = apply_image_aug(x.clone(), ImageAugConfig(), generator=g1)
    b = apply_image_aug(x.clone(), ImageAugConfig(), generator=g2)
    assert torch.allclose(a, b), "same seed must give the same augmentation"


def test_runs_on_input_device_agnostically():
    x = torch.rand(2, 3, 16, 16)                                    # cpu tensor
    out = apply_image_aug(x, ImageAugConfig())
    assert out.device == x.device                                  # follows the tensor, no hardcoded device


def test_cpu_transform_preserves_per_item_shape():
    """Fundamental invariant of the cpu per-item path: it returns the SAME shape it was handed, so
    the default collate can stack items into a batch. Covers n_obs_steps=1 ``(C,H,W)`` and the
    n_obs_steps>1 history ``(T,C,H,W)`` — the batch dim the path adds internally must be removed
    again (a wrapper that leaked a leading 1 would corrupt the collated batch)."""
    cpu_aug = make_cpu_transform(ImageAugConfig())
    for shape in [(3, 32, 32), (4, 3, 32, 32)]:            # (C,H,W) and (T,C,H,W) history
        out = cpu_aug(torch.rand(*shape))
        assert out.shape == shape
        assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_cpu_history_preserves_temporal_cohesion_like_gpu():
    """REGRESSION (n_obs_steps>1): a single sample's history is (T,C,H,W) on the cpu per-item path.
    It must get ONE shared draw across the T frames (temporal cohesion) — matching the gpu
    (B,T,C,H,W) fold — not T independent draws. Before the batch-dim fix, the bare (T,C,H,W) was read
    as a T-batch and each history frame got its own draw (cohesion broken)."""
    T = 4
    frame = _color_batch(1)[0]                              # (C,H,W)
    hist = frame.unsqueeze(0).repeat(T, 1, 1, 1).clone()   # (T,C,H,W): T IDENTICAL history frames
    torch.manual_seed(0)
    out = make_cpu_transform(ImageAugConfig())(hist)
    assert out.shape == (T, *frame.shape)
    for t in range(1, T):
        assert torch.allclose(out[t], out[0], atol=1e-6), \
            "cpu history frames were augmented inconsistently (temporal cohesion broken)"


def test_rank4_batch_is_still_independent_per_item():
    """The other side of the rank-4 contract: a bare (B,C,H,W) (the gpu no-history / batch case) must
    still get INDEPENDENT per-item draws — so the cpu history fix didn't turn batches cohesive."""
    B = 4
    frame = _color_batch(1)[0]
    batch = frame.unsqueeze(0).repeat(B, 1, 1, 1).clone()  # (B,C,H,W): B IDENTICAL frames
    torch.manual_seed(0)
    out = apply_image_aug(batch, ImageAugConfig())
    diffs = [(out[i] - out[0]).abs().mean().item() for i in range(1, B)]
    assert all(d > 1e-4 for d in diffs), "a (B,C,H,W) batch must get independent per-item draws"


def test_cpu_transform_is_picklable():
    # DataLoader workers pickle the dataset (incl. its image_transforms callable) — must not be a lambda.
    import pickle
    fn = make_cpu_transform(ImageAugConfig())
    restored = pickle.loads(pickle.dumps(fn))
    x = _color_batch(1)[0]
    torch.manual_seed(1)
    a = fn(x.clone())
    torch.manual_seed(1)
    b = restored(x.clone())
    assert torch.allclose(a, b, atol=1e-6)
