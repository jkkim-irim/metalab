"""Unit tests for the BC trainer helpers (learning/trainer/bc_trainer.py).

Covers the validation-driven bookkeeping (_update_best: true-minimum best_val + early-stop by
consecutive non-improving validations) and the RNG-safety of compute_val_loss. The full step-driven
loop (BCTrainer.run) is exercised end-to-end by the training runs; the first-batch-aug visualization
is tested in test_viz.py. Deps are imported directly — a missing one fails loudly.
Run in the node env from the repo root:  python -m pytest learning -q
"""
import contextlib
from pathlib import Path

import torch
import torch.nn as nn

from learning.metrics.validation import compute_val_loss
from learning.trainer.bc_trainer import (
    _prune_step_dirs,
    _raw_frame_view,
    _step_of,
    _update_best,
)


class _FakeAccel:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()

    def backward(self, loss):
        loss.backward()


class _DropoutPolicy(nn.Module):
    """Tiny policy whose forward draws from the RNG (dropout) — like ACT's train-mode forward."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, batch):
        return (self.drop(self.lin(batch["x"])) ** 2).mean(), None


def test_compute_val_loss_does_not_perturb_global_rng():
    # compute_val_loss runs the policy in train mode (ACT's VAE/KL path needs it), so dropout +
    # VAE reparameterization draw from the RNG. It must NOT advance the *global* RNG, otherwise
    # step-cadence validation (--val_every_n_steps) would shift later training dropout masks and
    # change the trajectory. validation.py forks the RNG around the val forward to prevent that.
    policy = _DropoutPolicy()
    val_loader = [{"x": torch.randn(2, 4)} for _ in range(5)]
    torch.manual_seed(0)
    before = torch.get_rng_state()
    vloss = compute_val_loss(policy, val_loader, lambda b: b, _FakeAccel(), max_batches=3)
    after = torch.get_rng_state()
    assert isinstance(vloss, float)
    assert torch.equal(before, after), \
        "compute_val_loss advanced the global RNG — val leaked into the training trajectory"


def test_best_val_tracks_true_minimum_not_last_or_save():
    # best_val must track the true minimum across validations, independent of any save cadence
    # (regression: it once updated only inside the save block, missing earlier-better points).
    # The 3rd validation is the true best; the 4th-5th are worse.
    losses = [0.21, 0.20, 0.18, 0.19, 0.22]
    best_val, no_improve = float("inf"), 0
    improved_at = []
    for epoch, vl in enumerate(losses, 1):
        best_val, no_improve, improved, stop = _update_best(vl, best_val, no_improve, patience=0)
        if improved:
            improved_at.append(epoch)
        assert not stop                      # patience=0 never stops
    assert best_val == 0.18                  # true minimum (epoch 3), not the last (0.22)
    assert improved_at == [1, 2, 3]


def test_early_stop_fires_after_patience_consecutive_non_improvements():
    losses = [0.21, 0.20, 0.205, 0.206, 0.207]   # best at epoch 2, then 3 non-improving
    best_val, no_improve = float("inf"), 0
    stop_epoch = None
    for epoch, vl in enumerate(losses, 1):
        best_val, no_improve, improved, stop = _update_best(vl, best_val, no_improve, patience=3)
        if stop:
            stop_epoch = epoch
            break
    assert stop_epoch == 5                   # epochs 3,4,5 didn't beat 0.20 -> stop at epoch 5
    assert best_val == 0.20


def test_update_best_no_validation_is_noop():
    # val_loss=None (no validation this call) must not touch state or stop.
    assert _update_best(None, 0.2, 1, patience=3) == (0.2, 1, False, False)


def _steps(nums):
    return [Path(f"/run/checkpoints/step_{n:08d}") for n in nums]


def _nums(dirs):
    return sorted(_step_of(d) for d in dirs)


def test_prune_keeps_newest_n():
    # Rolling window: keep the newest N step dirs, delete the rest.
    to_delete = _prune_step_dirs(_steps([10, 20, 30, 40, 50, 60, 70, 80]), keep_n=5, protect_step=None)
    assert _nums(to_delete) == [10, 20, 30]           # newest 5 (40..80) kept


def test_prune_protects_best_step_even_if_old():
    # The best-select_metric step must survive pruning even when it's older than the rolling window —
    # this is what preserves the good model a rising surrogate loss would otherwise discard.
    to_delete = _prune_step_dirs(_steps([10, 20, 30, 40, 50, 60, 70, 80]), keep_n=3, protect_step=20)
    assert _nums(to_delete) == [10, 30, 40, 50]       # newest 3 (60,70,80) + protected 20 kept


def test_prune_nothing_when_window_covers_all():
    # keep_n >= number of dirs -> delete nothing; protecting an in-window step deletes nothing extra.
    assert _prune_step_dirs(_steps([10, 20, 30]), keep_n=5, protect_step=None) == []
    assert _prune_step_dirs(_steps([10, 20, 30]), keep_n=3, protect_step=30) == []


def test_raw_frame_view_is_train_only():
    # Image augmentation is TRAIN-ONLY: the val loader gets a transform-free shallow view. The view
    # must null image_transforms WITHOUT disturbing the train dataset, and share heavy state (meta)
    # by reference (not deep-copy it). This guards the "validation sees raw frames" guarantee for
    # image_aug=cpu (where transforms are baked into the shared dataset).
    class _DS:
        pass

    ds = _DS()
    ds.image_transforms = object()    # train carries CPU transforms (image_aug=cpu)
    ds.meta = object()                # heavy shared state (metadata, caches)
    view = _raw_frame_view(ds)
    assert view is not ds
    assert view.image_transforms is None         # val sees RAW frames
    assert ds.image_transforms is not None       # train dataset untouched
    assert view.meta is ds.meta                  # heavy state shared, not deep-copied
