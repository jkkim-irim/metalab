"""Unit tests for EpisodeAwareSampler (learning/data/sampler.py).

Verifies the per-episode index construction (drop_n_first/last_frames boundary math, episode
subsetting) and that shuffling is a torch-RNG-seeded permutation. Synthetic episode boundaries, so
no dataset and no lerobot needed.

Run in the node env from the repo root:  python -m pytest learning -q
"""
import torch

from learning.data.sampler import EpisodeAwareSampler

# 3 episodes of 5 frames each -> global frames 0..14.
FROM = [0, 5, 10]
TO = [5, 10, 15]


def test_all_frames_unshuffled():
    s = EpisodeAwareSampler(FROM, TO, shuffle=False)
    assert s.indices == list(range(15))
    assert list(iter(s)) == list(range(15))
    assert len(s) == 15


def test_drop_n_last_frames():
    s = EpisodeAwareSampler(FROM, TO, drop_n_last_frames=2)
    assert s.indices == [0, 1, 2, 5, 6, 7, 10, 11, 12]


def test_drop_n_first_frames():
    s = EpisodeAwareSampler(FROM, TO, drop_n_first_frames=1)
    assert s.indices == [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]


def test_episode_subset():
    s = EpisodeAwareSampler(FROM, TO, episode_indices_to_use=[0, 2])
    assert s.indices == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]


def test_shuffle_is_seeded_permutation():
    s = EpisodeAwareSampler(FROM, TO, shuffle=True)
    torch.manual_seed(0)
    order1 = list(iter(s))
    torch.manual_seed(0)
    order2 = list(iter(s))
    assert order1 == order2                       # same seed -> same order
    assert sorted(order1) == list(range(15))      # a permutation of all frames
    torch.manual_seed(0)
    assert list(iter(s)) != list(range(15))       # actually shuffled
