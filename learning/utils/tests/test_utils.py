"""Unit tests for the de-lerobot'd utils (learning/utils/).

``format_big_number`` suffix/precision formatting and ``set_seed`` determinism (same seed -> same
random/numpy/torch draws; different seeds diverge). No lerobot needed.

Run in the node env from the repo root:  python -m pytest learning -q
"""
import random

import numpy as np
import pytest
import torch

from learning.utils.logging_utils import format_big_number
from learning.utils.seed import set_seed


@pytest.mark.parametrize("value,precision,expected", [
    (0, 0, "0"),
    (12, 0, "12"),
    (999, 0, "999"),
    (1000, 0, "1K"),
    (1_500_000, 2, "1.50M"),
    (2_000_000_000, 0, "2B"),
    (1_000_000_000_000, 0, "1T"),
])
def test_format_big_number(value, precision, expected):
    assert format_big_number(value, precision) == expected


def test_set_seed_is_deterministic():
    set_seed(123)
    a = (random.random(), float(np.random.rand()), torch.randn(3).tolist())
    set_seed(123)
    b = (random.random(), float(np.random.rand()), torch.randn(3).tolist())
    assert a == b


def test_set_seed_differs_across_seeds():
    set_seed(1)
    a = torch.randn(5)
    set_seed(2)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_set_seed_without_accelerator_is_a_noop_import():
    # accelerator=None must not require importing accelerate.
    set_seed(0)  # no exception
