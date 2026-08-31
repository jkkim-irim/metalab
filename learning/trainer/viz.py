"""Trainer-side visualization — the first-batch augmentation grid logged to wandb.

Factored out of learning/trainer/bc_trainer.py: PIL image annotation + grid assembly are self-contained
and don't belong in the training loop. ``log_first_batch_aug`` is the only entry point the trainer
calls (once, at step 0); it samples nothing and never touches the training RNG. Pure presentation —
no training/validation logic here.
"""
import logging
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.utils import make_grid

from learning.data import image_aug

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSansMono.ttf",
)


def _load_font(size):
    """A monospace TrueType font at ``size`` px, falling back to PIL's (scalable on Pillow>=10.1)."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _aug_summary_lines(aug_cfg, mode):
    """Human-readable description of the augmentation actually applied to the frames.

    One augmentation (``learning/data/image_aug.apply_image_aug``), per-sample, every transform in
    the fixed ``AUG_ORDER``. ``mode`` is ``off | cpu | gpu`` — cpu and gpu run the identical function,
    so this describes the same ranges either way; only *where* it runs differs.
    """
    if mode == "off":
        return ["aug: OFF  (frames are the raw resized inputs — no augmentation)"]
    where = "gpu — train loop, on-device" if mode == "gpu" else "cpu — dataloader workers, per-item"
    lines = [f"aug: ON   [{where}]   per-sample, fixed order, before normalization"]
    for name in image_aug.AUG_ORDER:
        lines.append(f"  {name}: {tuple(getattr(aug_cfg, name))}")
    return lines


def _annotate(arr, lines):
    """Overlay ``lines`` of text in a translucent dark box at the top-left of an HWC-uint8 RGB image.
    Returns a new HWC-uint8 numpy array (so wandb.Image gets the annotated frame)."""
    img = Image.fromarray(arr).convert("RGBA")
    fsize = max(12, img.width // 90)               # scale text to the grid width so it stays legible
    font = _load_font(fsize)
    pad = max(4, fsize // 2)
    line_h = fsize + max(2, fsize // 4)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    box_w = int(min(img.width, pad * 2 + max(draw.textlength(s, font=font) for s in lines)))
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 175))
    y = pad
    for s in lines:
        draw.text((pad, y), s, fill=(255, 255, 255, 255), font=font)
        y += line_h
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def _prompt_for(batch, i):
    """The task/prompt string for sample ``i`` (``batch["task"][i]``), or ``"(none)"`` if absent —
    keeps the per-sample info table honest about which task each frame was drawn for."""
    t = batch.get("task")
    if isinstance(t, (list, tuple)) and i < len(t) and isinstance(t[i], str) and t[i]:
        return t[i]
    if isinstance(t, str) and t:
        return t
    return "(none)"


def _round_vec(row, n, ndigits=3):
    """A short, table-friendly string of the first ``n`` entries of a 1-D tensor, rounded."""
    return str([round(float(x), ndigits) for x in row[:n]])


def _first_step(v):
    """Collapse a possibly time-stacked tensor to its first step: (B,T,...) -> (B,...); else as-is."""
    return v[:, 0] if torch.is_tensor(v) and v.ndim >= 3 else v


@torch.no_grad()
def log_first_batch_aug(wandb_logger, batch, aug_cfg, mode, policy=None):
    """Log the first training batch to wandb: per-camera image grids PLUS non-truncating info tables.

    Grids (``media/first_batch_aug/<cam>``, always logged) carry only a short pixel overlay
    (``cam / B / C×H×W / range / dtype``) so nothing clips. The prose that used to be baked into the
    pixels now lives in two ``wandb.Table``s (only logged when the wandb client exposes ``.Table`` —
    a test stub may not):
      * ``media/first_batch/info`` — ONE ROW PER SAMPLE (``sample_idx``, ``prompt``, a per-camera
        ``C×H×W [min,max]`` column, ``state[:8]``, ``action[:7]``); untruncated + a per-task sanity check.
      * ``media/first_batch/model`` — the model-specific input summary. If ``policy`` exposes
        ``input_summary(batch)`` (e.g. GR00T) that is rendered verbatim (model-owned, so it stays
        correct as the model changes); otherwise a generic pre-normalization line — ACT is unchanged.

    Receives the real first training batch (frames in [0,1], pre-normalization), so it samples nothing
    itself and never touches the training RNG.
    """
    w = wandb_logger._wandb
    aug_lines = _aug_summary_lines(aug_cfg, mode)
    cam_frames = {}                                            # cam -> (B,C,H,W) float[0,1] cpu
    imgs = {}
    for k, v in batch.items():
        if "observation.images." in k and torch.is_tensor(v) and v.ndim >= 4:
            x = v[:, 0] if v.ndim == 5 else v                 # (B,T,C,H,W) -> first step; else (B,C,H,W)
            vmin, vmax = float(x.min()), float(x.max())       # report range on the real (unclamped) frames
            x = x.float().clamp(0, 1).cpu()
            b, c, h, w_ = x.shape
            cam = k.replace("observation.images.", "")
            cam_frames[cam] = x
            nrow = max(1, int(math.ceil(b ** 0.5)))
            grid = make_grid(x, nrow=nrow, padding=2)
            arr = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")   # HWC uint8
            dtype = str(v.dtype).replace("torch.", "")
            # Trimmed overlay: essentials only, so it never clips (prose moved to the info tables).
            header = [f"{cam}  B={b}  {c}x{h}x{w_}  [{vmin:.3f},{vmax:.3f}]  {dtype}"]
            arr = _annotate(arr, header)
            imgs[f"media/first_batch_aug/{cam}"] = w.Image(
                arr, caption=f"{cam}: first train batch ({b} samples) — pre-normalization model input")
    if imgs:
        w.log(imgs)
        logging.info(f"logged first-batch aug grids: {sorted(imgs)}")

    # Tables live off the wandb client; a test stub may not implement .Table -> skip (grids still logged).
    if not hasattr(w, "Table"):
        return

    # Model-specific summary if the policy owns one (GR00T); else the generic pre-normalization line.
    # Append the aug description either way, so the prose that left the pixels still has a home.
    if policy is not None and hasattr(policy, "input_summary"):
        model_lines = list(policy.input_summary(batch))
    else:
        model_lines = ["model input BEFORE normalization; each tile = 1 sample, independently augmented"]
    model_lines = model_lines + aug_lines
    logged = {"media/first_batch/model": w.Table(
        columns=["model input summary"], data=[[s] for s in model_lines])}

    # Per-sample info table (untruncated prompts + full per-sample tensor summary).
    if cam_frames:
        cams = sorted(cam_frames)
        B = next(iter(cam_frames.values())).shape[0]
        state = _first_step(batch.get("observation.state"))
        action = batch.get("action")                          # (B, K, adim) -> first chunk step below
        columns = ["sample_idx", "prompt"] + cams + ["state[:8]", "action[:7]"]
        data = []
        for i in range(B):
            row = [i, _prompt_for(batch, i)]
            for cam in cams:
                fr = cam_frames[cam][i]                        # (C,H,W)
                c, h, w_ = fr.shape
                row.append(f"{c}x{h}x{w_} [{float(fr.min()):.3f},{float(fr.max()):.3f}]")
            row.append(_round_vec(state[i], 8) if torch.is_tensor(state) else "(n/a)")
            if torch.is_tensor(action):
                a = action[i, 0] if action.ndim == 3 else action[i]   # first chunk step
                row.append(_round_vec(a, 7))
            else:
                row.append("(n/a)")
            data.append(row)
        logged["media/first_batch/info"] = w.Table(columns=columns, data=data)

    w.log(logged)
    logging.info(f"logged first-batch info tables: {sorted(logged)}")
