"""Contact friction mixing — GEOMETRIC mean, the single definition both engines implement.

A contact needs one friction coefficient from the two geoms that touch. The default in MuJoCo (and in
genesis) is ``max(mu_a, mu_b)``: the stickier surface wins outright, so a low-friction object dragged over a
high-friction table behaves as if it were the table's material. ALLEX's legacy isaaclab stack replaced that
with the geometric mean ``sqrt(mu_a * mu_b)`` — a mix in which BOTH materials matter and neither can be
hidden by the other — and every trained policy so far saw that convention. This module is that rule, made
engine-agnostic:

    mu_contact = max(sqrt(mu_a * mu_b), FRICTION_EPS)      per friction component

applied component-wise to (slide, torsional, rolling); the solver's 5-slot friction vector repeats the
slide and rolling entries (MuJoCo's ``vec5`` layout: slide, slide, torsional, roll, roll).

Where the kernels live, and why not here:

* **newton** — contacts are warp arrays, so ``backends/newton/friction.py`` holds a warp kernel and wraps
  ``mujoco_warp``'s ``make_constraint`` (the last place the mixed value can still be corrected).
* **genesis** — contacts are ``quadrants`` (taichi-fork) fields that warp cannot address, so
  ``backends/genesis/friction.py`` holds a ``quadrants`` kernel and re-runs the collider→constraint sequence
  with it inserted.

Both are checked against :func:`geomean_friction` here, which is the only place the formula is written in
plain python — if the two kernels ever disagree with it, the test says so.

NOTE the numbers today: robot, table and hammer all carry mu = 1.0, and ``max(1, 1) == sqrt(1 * 1)``, so
this changes nothing measurable until per-body mu values actually differ. It exists so that when they do
differ, both engines answer the same way and match how the legacy policies were trained.
"""
from __future__ import annotations

FRICTION_EPS = 1.0e-6   # floor: a zero-mu geom must not make the contact frictionless-with-NaN-risk


def geomean_friction(mu_a, mu_b):
    """Reference mixing of two geoms' friction triples ``(slide, torsional, rolling)`` → the contact's.

    Pure python/numpy (no engine import) so tests and analysis can call it; the engine kernels mirror it.
    Accepts sequences or numpy arrays and returns a list of 3 floats per the input's component order."""
    return [max((float(a) * float(b)) ** 0.5, FRICTION_EPS) for a, b in zip(mu_a, mu_b)]
