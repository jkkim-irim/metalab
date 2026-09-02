from __future__ import annotations

import inspect
import sys
import textwrap

import numpy as np

_TARGET = "for i_b_, i_b in enumerate(envs_idx):"
_FIXED = "for i_b_, i_b in enumerate(envs_idx.cpu().numpy()):"
_METHOD = "_init_invweight_and_meaninertia"


def apply() -> None:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    fn = getattr(RigidSolver, _METHOD)
    if getattr(fn, "_allex_invweight_patched", False):
        return

    src = textwrap.dedent(inspect.getsource(fn))
    assert _TARGET in src, (
        f"genesis {RigidSolver.__module__}.{_METHOD} env-loop line changed upstream "
        f"— invweight patch stale. Check genesis-world version and update sim/metalab/backends/genesis/_patches.py."
    )
    src = src.replace(_TARGET, _FIXED)

    ns: dict = {}
    exec(compile(src, f"<allex_patch:{_METHOD}>", "exec"),
         sys.modules[RigidSolver.__module__].__dict__, ns)
    patched = ns[_METHOD]
    patched._allex_invweight_patched = True
    setattr(RigidSolver, _METHOD, patched)


def apply_round_robin_variants() -> None:
    from genesis.engine.solvers import kinematic_solver as _ks

    if getattr(_ks._balanced_variant_mapping, "_allex_round_robin", False):
        return
    orig = _ks._balanced_variant_mapping
    assert callable(orig), "genesis kinematic_solver._balanced_variant_mapping missing — variant patch stale"

    def _round_robin(n_variants, B):
        if B < n_variants:
            return orig(n_variants, B)
        return np.arange(B) % n_variants

    _round_robin._allex_round_robin = True
    _ks._balanced_variant_mapping = _round_robin
