"""Tests for the episode-wise split + val subsampling (learning/data/dataset.py): no train/val
leakage, deterministic split, and a representative (non-prefix) seeded-random validation subset.

Run in the LeRobot env from the repo root:  python -m pytest learning -q
"""
from learning.data.dataset import _deterministic_subset, split_single


def test_split_single_no_leakage_and_deterministic():
    train, val = split_single(100, val_ratio=0.2, seed=42)
    assert len(val) == 20 and len(train) == 80
    assert set(train).isdisjoint(val), "train/val episodes must not overlap"
    assert sorted(train + val) == list(range(100)), "split must cover each episode exactly once"

    # deterministic for a fixed seed
    assert (train, val) == split_single(100, val_ratio=0.2, seed=42)
    # a different seed gives a different val partition
    assert val != split_single(100, val_ratio=0.2, seed=7)[1]


def test_deterministic_subset_is_seeded_representative_and_bounded():
    # The validation subsampler: a fixed seeded-random sample of frame indices spread across the
    # whole val set (NOT the first-N prefix the old --val_max_batches cap took).
    pool = list(range(1000))
    s = _deterministic_subset(pool, 40, seed=42)
    assert len(s) == 40 and len(set(s)) == 40        # 40 distinct samples (without replacement)
    assert s == sorted(s)                            # stable iteration order
    assert set(s).issubset(pool)                     # all drawn from the pool

    # deterministic: same seed -> identical subset every call (so every val pass is comparable)
    assert s == _deterministic_subset(pool, 40, seed=42)
    # a different seed -> a different subset
    assert s != _deterministic_subset(pool, 40, seed=7)

    # representative, not a prefix: spans the whole pool rather than just the first 40 indices
    assert s != pool[:40]
    assert min(s) < 100 and max(s) > 900

    # degenerate caps fall back to the full set unchanged
    assert _deterministic_subset(pool, 0, seed=1) == pool
    assert _deterministic_subset(pool, 5000, seed=1) == pool
