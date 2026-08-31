"""ALLEX learning — data loading for the ACT train/eval pipeline.

Episode-wise train/val split (seeded) + EpisodeAwareSampler DataLoader over a subset of
episodes, plus a thin re-export of LeRobot's make_dataset.
"""
import numpy as np
import torch

from learning.data.factory import make_dataset  # noqa: F401  (re-exported)
from learning.data.sampler import EpisodeAwareSampler


def split_single(num_episodes: int, val_ratio: float, seed: int):
    """Single train/val episode split (seeded, reproducible)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(num_episodes)
    rng.shuffle(idx)
    n_val = max(1, int(round(num_episodes * val_ratio))) if val_ratio > 0 else 0
    val_ep = sorted(idx[:n_val].tolist()) if n_val > 0 else []
    train_ep = sorted(idx[n_val:].tolist())
    return train_ep, val_ep


def _deterministic_subset(indices, n, seed):
    """A fixed, seeded-random subset of ``n`` of ``indices`` (sorted for a stable iteration order).

    Caps validation to a small but REPRESENTATIVE sample of the val set: a seeded random draw
    spread across all val episodes — NOT the first-N prefix — that is identical on every validation
    pass (same seed), so the val curve is a consistent estimator across training. ``n <= 0`` or
    ``n >= len(indices)`` returns the full set unchanged.
    """
    indices = list(indices)
    if n <= 0 or n >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(indices), size=n, replace=False)
    return sorted(int(indices[i]) for i in sel)


def make_dataloader(dataset, cfg, episode_indices, shuffle, device, max_samples=0, sample_seed=0):
    """DataLoader over a subset of episodes via EpisodeAwareSampler.

    ``max_samples`` > 0 restricts the loader to a fixed, seeded-random sample of that many frames
    drawn across the given episodes — a cheap, REPRESENTATIVE, reproducible validation subset (see
    _deterministic_subset), reused identically on every pass; 0 = every frame. ``sample_seed`` seeds
    the draw. (Used for validation; training always passes the full shuffled sampler.)
    """
    kwargs = dict(
        dataset_from_indices=dataset.meta.episodes["dataset_from_index"],
        dataset_to_indices=dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=episode_indices,
        shuffle=shuffle,
    )
    if hasattr(cfg.policy, "drop_n_last_frames"):
        kwargs["drop_n_last_frames"] = cfg.policy.drop_n_last_frames
    sampler = EpisodeAwareSampler(**kwargs)
    if max_samples > 0:
        # Fix the val set to a seeded-random, representative subset of frame indices (NOT a prefix),
        # so step-cadence validation is cheap yet samples the whole val set the same way every pass.
        sampler = _deterministic_subset(list(sampler), max_samples, sample_seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        # Keep workers alive across iterations. The val loader is re-iterated every validation
        # (step-cadence --val_every_n_steps), so without this each validation would respawn all
        # num_workers + reopen the video readers — the dominant validation cost.
        persistent_workers=cfg.num_workers > 0,
    )
