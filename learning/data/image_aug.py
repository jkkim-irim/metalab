"""Image augmentation — one device-agnostic implementation for both the cpu and gpu paths.

There is a single augmentation here, ``apply_image_aug``. It is plain torch, so it runs on whatever
device its input tensor is on:

  * ``--dataset.image_aug gpu`` → the trainer calls it on the assembled batch, on-device, in the
    train loop (workers just read raw frames).
  * ``--dataset.image_aug cpu`` → the dataset calls the SAME function per-item in ``__getitem__``,
    on cpu, inside the dataloader workers (offloads aug from the GPU).

So ``cpu`` vs ``gpu`` is only *where* it runs — the transforms, ranges, order, and per-sample
semantics are identical. (The two paths draw their random magnitudes from different RNG streams —
per-item in workers vs per-batch in the main process — so a given frame gets a different *draw* from
the identical distribution; they are equivalent in every sense except the literal per-sample seed.)

Every configured transform is applied to every sample, per-sample-independent, in a FIXED order
(the order matters — the ops don't commute):

    brightness → contrast → saturation → hue → sharpness   (photometric, in [0,1])
    → rotate + zoom-in                                      (geometric, one affine, no black border)

Set a transform's range to its identity to disable it (brightness/contrast/saturation/sharpness/zoom
→ ``(1, 1)``; hue/rotate_deg → ``(0, 0)``). The zoom-in is calibrated so the rotation never exposes a
black border: a corner stays inside the normalized [-1,1] frame iff ``zoom >= |cos θ| + |sin θ|``
(= 1.067 at 4°), and ``zoom >= 1.1`` clears it. With observation history ((B,T,C,H,W)) the per-sample
draw is shared across a sample's T frames (temporal cohesion — a coherent clip). Applied to the [0,1]
camera images BEFORE the policy preprocessor's normalization, and only during TRAINING.

(Ported from the previous ``learning/utils/gpu_aug.py``; the lerobot ``ImageTransforms`` /
``RandomSubsetApply`` subset machinery — weighted subset, random order, ``enable`` — was retired in
favour of this single apply-all, fixed-order, device-agnostic executor.)
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math

import torch
import torch.nn.functional as F

_GRAY = (0.299, 0.587, 0.114)
_EPS = 1e-8

# Fixed application order (also the order the ranges are documented + annotated in).
PHOTOMETRIC = ("brightness", "contrast", "saturation", "hue", "sharpness")
GEOMETRIC = ("rotate_deg", "zoom")
AUG_ORDER = PHOTOMETRIC + GEOMETRIC


@dataclass
class ImageAugConfig:
    """Per-sample image augmentation ranges, applied in the fixed ``AUG_ORDER``.

    Each ``(lo, hi)`` is sampled uniformly per sample. Disable a transform by setting its range to
    the identity (see module docstring). Photometric ranges match the historical ALLEX/lerobot
    defaults; the geometric ranges are calibrated for the no-black-border zoom.
    """

    brightness: tuple[float, float] = (0.8, 1.2)
    contrast: tuple[float, float] = (0.8, 1.2)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.05, 0.05)
    sharpness: tuple[float, float] = (0.5, 1.5)
    rotate_deg: tuple[float, float] = (-4.0, 4.0)
    zoom: tuple[float, float] = (1.1, 1.2)


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) RGB in [0,1] -> (B,3,H,W) HSV with H,S,V in [0,1]."""
    maxc, argmax = img.max(dim=1)
    minc = img.min(dim=1).values
    v = maxc
    delta = maxc - minc
    s = delta / maxc.clamp(min=_EPS)
    deltac = delta.clamp(min=_EPS)
    r, g, b = img.unbind(1)
    rc, gc, bc = (maxc - r) / deltac, (maxc - g) / deltac, (maxc - b) / deltac
    h = torch.zeros_like(maxc)
    h = torch.where(argmax == 0, bc - gc, h)
    h = torch.where(argmax == 1, 2.0 + rc - bc, h)
    h = torch.where(argmax == 2, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = torch.where(delta <= _EPS, torch.zeros_like(h), h)
    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) HSV -> (B,3,H,W) RGB in [0,1]."""
    h, s, v = hsv.unbind(1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    idx = (i.long() % 6).unsqueeze(0)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = torch.stack([v, q, p, p, t, v], dim=0).gather(0, idx).squeeze(0)
    g = torch.stack([t, v, v, q, p, p], dim=0).gather(0, idx).squeeze(0)
    b = torch.stack([p, p, t, v, v, q], dim=0).gather(0, idx).squeeze(0)
    return torch.stack([r, g, b], dim=1)


@torch.no_grad()
def apply_image_aug(
    imgs: torch.Tensor, cfg: ImageAugConfig, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Augment camera images in [0,1], per-sample independent, on ``imgs.device`` (cpu OR cuda).

    Accepts ``(C,H,W)`` (single frame), ``(B,C,H,W)`` (a batch — one independent draw per item), or
    ``(B,T,C,H,W)`` (a batch WITH obs history — the T frames of one sample share its draw). Returns
    the same shape. Pure torch (no torchvision/kornia), so it is identical on cpu and gpu.

    NOTE ON RANK-4: a 4-D input is treated as ``(B,C,H,W)`` — each leading item is an INDEPENDENT
    sample. So a single sample's history ``(T,C,H,W)`` must NOT be passed bare here (it would get T
    independent draws, breaking temporal cohesion); add a batch dim -> ``(1,T,C,H,W)``. The cpu
    per-item path does exactly that via ``make_cpu_transform`` (the gpu path is already batched).

    ``generator`` (optional) makes the draws reproducible; it must live on ``imgs.device``.
    """
    device = imgs.device
    x = imgs
    orig_dim = x.dim()
    if orig_dim == 3:                    # (C,H,W) -> (1,C,H,W)
        x = x.unsqueeze(0)
    fold = x.dim() == 5                  # (B,T,C,H,W): fold time into batch, one draw per sample
    reps = 1
    if fold:
        bsz, reps = x.shape[0], x.shape[1]
        x = x.reshape(bsz * reps, *x.shape[2:])
    b = x.shape[0]                       # folded batch = B*T (or B)
    nb = b // reps                       # independent samples; one draw each, shared over T

    gray = torch.tensor(_GRAY, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    blur_w = torch.full((3, 1, 3, 3), 1.0 / 9.0, device=device, dtype=x.dtype)  # depthwise 3x3 box

    def _s(rng: tuple[float, float]) -> torch.Tensor:
        """Per-sample factor in ``rng``: ``nb`` draws, each repeated across the sample's ``reps``
        frames -> ``(nb*reps, 1, 1, 1)`` (temporal cohesion; reps=1 is unchanged)."""
        lo, hi = rng
        t = torch.empty(nb, 1, 1, 1, device=device, dtype=x.dtype).uniform_(lo, hi, generator=generator)
        return t.repeat_interleave(reps, dim=0) if reps > 1 else t

    # --- photometric (fixed order; per-sample, shared across a sample's T frames) ---
    x = x * _s(cfg.brightness)                                      # brightness
    m = (x * gray).sum(1, keepdim=True).mean((1, 2, 3), keepdim=True)
    x = (x - m) * _s(cfg.contrast) + m                             # contrast (around mean luminance)
    lum = (x * gray).sum(1, keepdim=True)
    x = (x - lum) * _s(cfg.saturation) + lum                       # saturation (toward gray)
    x = x.clamp(0, 1)
    hsv = _rgb_to_hsv(x)                                           # hue (HSV H-shift)
    hsv[:, 0] = (hsv[:, 0] + _s(cfg.hue).squeeze(1)) % 1.0
    x = _hsv_to_rgb(hsv)
    deg = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), blur_w, groups=3)
    fac = _s(cfg.sharpness)
    x = (fac * x + (1 - fac) * deg).clamp(0, 1)                    # sharpness (blur<->orig blend)

    # --- geometric: rotate + zoom-in in ONE affine, no black border, per-sample ---
    ang = _s(cfg.rotate_deg).view(-1) * (math.pi / 180)
    sc = _s(cfg.zoom).view(-1)
    cs, sn = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(b, 2, 3, device=device, dtype=x.dtype)
    theta[:, 0, 0] = cs / sc                                       # sample a (1/scale)-sized rotated
    theta[:, 0, 1] = -sn / sc                                      # region -> zoom in, no black border
    theta[:, 1, 0] = sn / sc
    theta[:, 1, 1] = cs / sc
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    if fold:
        x = x.reshape(bsz, reps, *x.shape[1:])
    if orig_dim == 3:
        x = x.squeeze(0)
    return x


def _augment_one_sample(img: torch.Tensor, cfg: ImageAugConfig) -> torch.Tensor:
    """Augment ONE sample's camera image(s) for the cpu (per-item) path.

    ``img`` is a single sample: ``(C,H,W)`` (no history) or ``(T,C,H,W)`` (obs history). We add a
    leading batch dim so ``apply_image_aug`` reads a leading T as the HISTORY dim — ONE shared draw
    across the T frames (temporal cohesion) — exactly like the gpu ``(B,T,C,H,W)`` path, instead of
    mistaking the T frames for T independent batch items. ``(C,H,W)`` -> one frame, one draw."""
    return apply_image_aug(img.unsqueeze(0), cfg).squeeze(0)


def make_cpu_transform(cfg: ImageAugConfig):
    """A picklable per-item callable for the cpu path (``LeRobotDataset.image_transforms``).

    Returns a ``functools.partial`` (picklable, so DataLoader workers can receive it) over
    ``_augment_one_sample`` — the SAME ``apply_image_aug`` as the gpu path, wrapped only to add/remove
    the single-sample batch dim so obs-history cohesion matches the gpu path exactly."""
    return functools.partial(_augment_one_sample, cfg=cfg)
