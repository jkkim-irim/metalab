"""Unit tests for the contact-friction mixing rule — the plain-python definition both engines mirror.

``geomean_friction`` takes two friction triples and returns one: no engine, no GPU, no contacts, so it runs
under plain pytest. What DOES need a live solver is whether each backend's kernel actually produces this
value in its contact buffer — that is ``check_contact_friction.py``, and these tests do not replace it.

They pin the DEFINITION, which the live check cannot: both kernels are validated *against* this function, so
if it silently changed the two engines would still agree with each other while both departing from the rule
every policy so far was trained under. Concretely, this file is what catches

* the mean turning arithmetic (or back into MuJoCo/genesis' ``max``), which no cross-check would flag,
* the ``FRICTION_EPS`` floor going away — only a scene holding a zero-mu geom would show it,
* a transposed component (slide ↔ rolling),
* a numpy scalar leaking out, which breaks the warp / quadrants call site rather than this function,
* the two geoms' order mattering, which the broadphase does not promise.

Today robot, table and hammer all carry mu = 1.0, where ``max(1, 1) == sqrt(1 * 1)`` — the shipped assets
cannot tell the two mixing rules apart at all, which is why the live check has to write a low mu by hand.
Until per-body mu values actually differ, this file is the only cheap guard on the convention.

    python -m pytest sim/metalab/tests/test_contact_friction.py -q       # from the repo root
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.metalab.runtime.physics.friction import FRICTION_EPS, geomean_friction

# (slide, torsional, rolling) pair → the mixed triple. Hand-computed, NOT recomputed with the formula under
# test, and every component distinct so a transposed one cannot pass.
PAIRS = [
    ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), [1.0, 1.0, 1.0]),                # today's assets: mixing is a no-op
    ((0.25, 0.25, 0.25), (1.0, 1.0, 1.0), [0.5, 0.5, 0.5]),             # check_contact_friction's scenario
    ((0.25, 0.1, 2.0), (4.0, 0.4, 8.0), [1.0, 0.2, 4.0]),
    ((9.0, 0.04, 0.0625), (4.0, 0.09, 0.25), [6.0, 0.06, 0.125]),
]


@pytest.mark.parametrize("mu_a, mu_b, want", PAIRS)
def test_hand_computed_pairs(mu_a, mu_b, want):
    assert geomean_friction(mu_a, mu_b) == pytest.approx(want)


@pytest.mark.parametrize("mu_a, mu_b, want", PAIRS)
def test_geom_order_does_not_matter(mu_a, mu_b, want):
    """The solver hands over the two geoms in whatever order its broadphase found them."""
    assert geomean_friction(mu_b, mu_a) == pytest.approx(want)


def test_is_the_geometric_mean_not_max_or_arithmetic():
    """The WHY of the module. Under ``max`` (the MuJoCo/genesis default) the stickier surface wins outright
    and the other material is hidden; the geometric mean must land strictly between the two, and below the
    arithmetic mean — which is the drift a reader of "mean" could plausibly introduce."""
    low, high = (0.25, 0.25, 0.25), (1.0, 1.0, 1.0)
    got = geomean_friction(low, high)
    assert got == pytest.approx([0.5, 0.5, 0.5])
    for g, lo, hi in zip(got, low, high):
        assert lo < g < hi                                  # neither material can hide the other (not max)
        assert g < (lo + hi) / 2.0                          # and not the arithmetic mean (0.625) either


def test_zero_mu_floors_at_eps():
    """A frictionless geom must not zero the contact — the floor is what keeps the solver off a 0/NaN path."""
    got = geomean_friction((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert got == [FRICTION_EPS] * 3
    assert all(g > 0.0 for g in got)


def test_returns_plain_floats():
    """numpy in, plain float out: the result is handed to a warp kernel / a quadrants kernel, where a leaked
    numpy scalar fails at the call site instead of here."""
    got = geomean_friction(np.array([0.25, 0.1, 2.0]), np.array([4.0, 0.4, 8.0]))
    assert got == pytest.approx([1.0, 0.2, 4.0])
    assert all(type(g) is float for g in got)


def test_sequence_and_array_inputs_agree():
    a, b = (0.25, 0.1, 2.0), (4.0, 0.4, 8.0)
    assert (geomean_friction(a, b)
            == geomean_friction(list(a), list(b))
            == geomean_friction(np.array(a), np.array(b)))
