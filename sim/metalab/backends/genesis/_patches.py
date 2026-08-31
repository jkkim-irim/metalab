"""genesis monkeypatches — build-time fixes applied from OUR code, never by editing the vendored source.

Two, both installed by :func:`apply`:

1. :func:`apply` (below) — removes the O(n_envs) GPU-scalar indexing bottleneck in scene build init.
2. :func:`apply_round_robin_variants` — makes the per-env variant map ROUND-ROBIN, matching newton.


genesis 1.2.1 ``RigidSolver._init_invweight_and_meaninertia`` iterates ``envs_idx`` (a torch **CUDA**
tensor) directly, using the GPU scalar ``i_b`` as a numpy-array index **millions of times**
(``mass_mat_L[i_d, j_d, i_b]`` etc.). Each access is a GPU→CPU sync plus ``__torch_function__``
overhead, so ``scene.build`` blows up to O(n_envs) for heterogeneous scenes (per-env shapes →
``batch_links_info=True`` recomputes invweight per env): ~2-3 min at 8192 envs (profile: 11 of 14 s here).

The math is already all CPU numpy (fetched via ``qd_to_numpy``); iterating the **loop counter over CPU
numpy** removes the GPU round-trips and makes build ~3-4× faster (measured: 3 variants @512 17.7s→6.4s).
Results unchanged (same index values); only the one-time startup init is affected — the training rollout /
``scene.step`` GPU path is untouched.

This is a naive loop genesis itself flagged ``# TODO: ... not performance critical`` at line 744.

Per CLAUDE.md "no vendor edits — wrap/override in our own code", we don't touch the genesis source. Instead
of forking/copying genesis, we **read the current method source, replace that one line, recompile, rebind**:
- if upstream changes an *other* line of the method, it's picked up (we work from current source).
- if upstream changes *that line*, we ``assert`` loud-fail — no silent staleness (fail-loud rule).
"""
from __future__ import annotations

import inspect
import sys
import textwrap

import numpy as np

# Replace: iterating the CUDA tensor envs_idx directly → iterate plain numpy ints.
_TARGET = "for i_b_, i_b in enumerate(envs_idx):"
_FIXED = "for i_b_, i_b in enumerate(envs_idx.cpu().numpy()):"
_METHOD = "_init_invweight_and_meaninertia"


def apply() -> None:
    """Swap RigidSolver's env loop to CPU-int iteration (idempotent). Call **after gs.init()**.

    The ``rigid_solver`` module references ``genesis.qd_float`` at load, which is only set by
    ``gs.init()``, so the import lives here (called post-gs.init) rather than at module top — a
    justified exception.
    """
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    fn = getattr(RigidSolver, _METHOD)
    if getattr(fn, "_allex_invweight_patched", False):
        return  # already applied (process-global class change) — idempotent

    src = textwrap.dedent(inspect.getsource(fn))
    assert _TARGET in src, (
        f"genesis {RigidSolver.__module__}.{_METHOD} env-loop line changed upstream "
        f"— invweight patch stale. Check genesis-world version and update sim/metalab/backends/genesis/_patches.py."
    )
    src = src.replace(_TARGET, _FIXED)

    # Give the original module's globals (np, gs, qd_to_numpy, kernel_*, ...) for name resolution.
    ns: dict = {}
    exec(compile(src, f"<allex_patch:{_METHOD}>", "exec"),
         sys.modules[RigidSolver.__module__].__dict__, ns)
    patched = ns[_METHOD]
    patched._allex_invweight_patched = True
    setattr(RigidSolver, _METHOD, patched)


def apply_round_robin_variants() -> None:
    """Make genesis map env -> object variant ROUND-ROBIN (env i -> variant i % N), like newton (idempotent).

    genesis assigns variants in contiguous BLOCKS (``kinematic_solver._balanced_variant_mapping``: 9 envs / 3
    variants -> 0,0,0,1,1,1,2,2,2), newton's parser does ``w % N``. Both give each variant an equal share, so
    a plain PPO population is unaffected — but SAPG splits envs into contiguous blocks too
    (``arange(n) // block_size``), and the two block structures then ALIGN: a genesis SAPG block can end up
    training on one hammer shape while every newton block sees all three. The block embedding then encodes
    "which object" instead of "how exploratory", which is not what it is for.

    ``add_entity`` takes no mapping argument and the solver computes it from ``(n_variants, B)`` alone, so the
    only seam is the function itself. ``_dispatch_heterogeneous_vgeoms`` imports it INSIDE the method, so
    rebinding the module attribute is picked up at call time — no source rewriting needed.

    Same shape as the balanced map it replaces: a length-B array of variant indices, each variant used
    ``B // N`` times (+1 for the first ``B % N``), so nothing downstream changes except WHICH env gets which.
    """
    from genesis.engine.solvers import kinematic_solver as _ks

    if getattr(_ks._balanced_variant_mapping, "_allex_round_robin", False):
        return                                    # already applied (module attribute) — idempotent
    orig = _ks._balanced_variant_mapping
    assert callable(orig), "genesis kinematic_solver._balanced_variant_mapping missing — variant patch stale"

    def _round_robin(n_variants, B):
        if B < n_variants:                        # genesis' own fallback: one variant per env, in order
            return orig(n_variants, B)
        return np.arange(B) % n_variants

    _round_robin._allex_round_robin = True
    _ks._balanced_variant_mapping = _round_robin
