"""Motor-to-joint coupling — group loaders + buffer packing (numpy-only, engine-agnostic).

Shared by both engines' coupled-PD path (they run the same ``motor_coupling`` warp kernels — newton
natively, genesis through its backend adapter). Deliberately numpy-only (no warp/torch import) so
loading a group never depends on a GPU runtime. A "group" dict is the packed contract of one coupled transmission
(models, motor gains, fit envelope, torque limits, gravcomp-fold flag); ``_pack`` flattens a list of
same-DOF groups into the fixed flat buffers the kernels evaluate.

See ``motor_coupling.py`` for the control law and ``README.md`` for the provenance of the
``mj_mapping/*.json`` firmware maps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEFAULT_PARAMS_FILE = _HERE / "robot_model.json"                # per-group motor_control_param
DEFAULT_MODEL_DIR = _HERE / "mj_mapping"                         # firmware-derived J2M maps

# HAND (finger/thumb) layout inside the packed buffers: 3-DOF lower-triangular — the 3 J2M fits
# m1(q1), m2(q1,q2), m3(q1,q2,q3).
MODELS_PER_GROUP_HAND = 3
_MODEL_ORDER_HAND = (("m1", 1), ("m2", 2), ("m3", 3))
# ARM layout (wrist Roll/Pitch ballscrew; elbow+wristYaw differential pulley) = 2-DOF 2×2 — 2 models,
# each a 2-var fit.
MODELS_PER_GROUP_ARM = 2
_MODEL_ORDER_ARM = (("m1", 2), ("m2", 2))


def read_gains(src: dict) -> tuple[np.ndarray, np.ndarray]:
    """Re-read ONE group's motor gains (k_phi, k_d) from ``robot_model.json``.

    ``src`` is the ``gain_src`` provenance every loaded group carries (source file + ``params_key`` + the
    motor slice it owns) — the same two lines the loaders run at build time, isolated so a gain edit can be picked up
    WITHOUT rebuilding the env: the caller assigns the result into the live kernel buffers (see
    ``MotorCoupledPD*.reload_gains``). Only gains: the transmission fit, envelope and torque limits are
    model identity, not a tuning knob, and changing those still means a restart."""
    params_file = Path(src["file"])
    with open(params_file) as f:
        params = json.load(f)
    key, n = src["params_key"], src["n"]
    assert key in params, f"{params_file}: no group '{key}'"
    sl = slice(*src["slice"]) if src["slice"] else slice(None)
    mcp = params[key]["motor_control_param"][0]
    k_phi = np.asarray(mcp["k_pos"][sl], dtype=np.float64)
    k_d = np.asarray(mcp["k_vel"][sl], dtype=np.float64)
    assert k_phi.shape == k_d.shape == (n,), \
        f"{key}: k_pos/k_vel slice must stay {n} values (got {k_phi.shape}/{k_d.shape})"
    return k_phi, k_d


def jacobian(models: list, dim: int, q: np.ndarray) -> np.ndarray:
    """G = ∂m/∂q at one joint config — the numpy twin of the kernel's ``_jac_j2m_*``, for HOST-side
    diagnostics (the Joint Kp readout). Same packed ``models`` the kernel evaluates, so the two cannot
    drift; kept here because this module is the numpy-only side of the coupling.

    Callers clamp ``q`` to the group's fit envelope first, exactly as the kernel does before evaluating."""
    G = np.zeros((len(models), dim))
    x = np.asarray(q, dtype=np.float64)
    for k, (terms, coeffs) in enumerate(models):
        e = terms[:, :dim]                       # (nt, dim) exponents
        p = x ** e                               # (nt, dim) per-variable powers, monomial = row product
        for j in range(dim):
            pj = p.copy()                        # ∂/∂x_j: swap column j for its derivative factor
            pj[:, j] = np.where(e[:, j] > 0, e[:, j] * x[j] ** np.maximum(e[:, j] - 1, 0), 0.0)
            G[k, j] = (coeffs * pj.prod(axis=1)).sum()
    return G


def const_jacobian(models: list, dim: int) -> np.ndarray | None:
    """G = ∂m/∂q when the transmission is LINEAR (every term degree ≤ 1) — then G does not depend on q,
    so the group's joint-space stiffness ``K_q = Gᵀ·diag(k_phi)·G`` is a constant matrix. ``None`` for a
    nonlinear map (finger/thumb/wrist), whose G must be evaluated per pose in the kernel.

    Used by :func:`equal_gain_warnings`: a differential pulley (elbow) has G = [[+a, +b], [−a, +b]], whose
    cross term is ``a·b·(k₁ − k₂)`` — it cancels ONLY while both motors carry the same gain."""
    G = np.zeros((len(models), dim))
    for k, (terms, coeffs) in enumerate(models):
        for e, c in zip(terms, coeffs):
            deg = int(e.sum())
            if deg > 1:
                return None                      # nonlinear → G is pose-dependent
            if deg == 1:
                G[k, int(np.argmax(e))] += c
    return G


def equal_gain_warnings(name: str, models: list, dim: int, k_phi: np.ndarray,
                        k_d: np.ndarray) -> list[str]:
    """Warn when a CONSTANT-Jacobian group's gains break the cancellation that keeps its K_q diagonal.

    The differential elbow is decoupled in joint space only because its two motors share a gain; give them
    different ones and a cross term appears (measured: k_pos [4.0, 3.0] → coupling +0.14, [3.0, 1.5] →
    +0.33). The gains are still applied as written — this only says loudly that the group is no longer
    diagonal, since that is invisible in robot_model.json itself."""
    G = const_jacobian(models, dim)
    if G is None:
        return []                                # pose-dependent coupling is normal here (wrist), not news
    out = []
    for label, gain in (("k_pos", k_phi), ("k_vel", k_d)):
        K = G.T @ np.diag(gain) @ G
        d = np.sqrt(np.outer(np.diag(K), np.diag(K)))
        off = np.abs(np.triu(K / np.where(d > 0, d, 1.0), 1))
        if off.max() > 1e-6:
            out.append(f"{name}: {label}={[float(v) for v in gain]} differ across a differential "
                       f"(constant-Jacobian) "
                       f"transmission → joint-space cross term appears (coupling {off.max():+.3f}, "
                       f"{np.abs(np.triu(K, 1)).max():.1f} N*m/rad). Use ONE gain for both motors to keep "
                       f"K_q diagonal.")
    return out


def _terms_coeffs(node: dict, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """One fitted model json node → (terms (nt,3) int32 zero-padded, coeffs (nt,) f64)."""
    terms = np.asarray(node["model_terms"])
    if terms.ndim == 1:
        assert dim == 1, f"flat model_terms only valid for 1-var models (dim={dim})"
        terms = terms.reshape(-1, 1)
    coeffs = np.asarray(node["coefficients"], dtype=np.float64).reshape(-1)
    assert terms.shape == (coeffs.shape[0], dim), (
        f"model_terms {terms.shape} inconsistent with {coeffs.shape[0]} coefficients, dim={dim}")
    padded = np.zeros((terms.shape[0], 3), dtype=np.int32)
    padded[:, :dim] = terms
    return padded, coeffs


def load_hand_group(params_key: str, joints: list[str], model_file=None, params_file=None,
                    gain_slice: tuple[int, int] | None = None) -> dict:
    """Load one 3-DOF lower-triangular coupled group (finger, thumb, or the diagonal shoulder) from
    json into raw arrays — for the 3-DOF coupled-PD kernel owner (``MotorCoupledPDHand``).

    ``params_key`` keys the motor gains in ``robot_model.json`` (``motor_control_param``: k_phi =
    ``k_pos``, k_d = ``k_vel``). Finger/thumb groups own exactly 3 gain slots (``gain_slice`` None);
    the shoulder's motors are slots (0, 3) of the 7-DOF ``arm_{r,l}`` group — pass ``gain_slice`` to
    slice gains/limits/actuators out of it. ``joints`` are the 3 real joint names in fit-variable
    order. ``model_file`` selects the firmware transmission: a bare basename (e.g. ``"thumb.json"``)
    resolves under ``mj_mapping/``, an absolute path is used as-is, and ``None`` defaults to
    ``finger.json`` (one map backs all eight finger groups; ``thumb.json`` both thumbs;
    ``shoulder.json`` both shoulders). The map is name-agnostic: it declares ``joint_roles`` that
    ``joints`` are checked against by suffix, not exact names.
    """
    model_file = Path(model_file) if model_file else DEFAULT_MODEL_DIR / "finger.json"
    if not model_file.is_absolute():          # bare basename (e.g. "thumb.json") → under mj_mapping/
        model_file = DEFAULT_MODEL_DIR / model_file
    params_file = Path(params_file) if params_file else DEFAULT_PARAMS_FILE
    with open(model_file) as f:
        poly = json.load(f)
    # sliced gains ⇒ params group shared across parts (arm_r) → qualify the label with the map part
    name = f"{params_key}_{poly['part']}" if gain_slice else params_key   # arm_r_shoulder / index_r
    assert poly["joint_unit"] == "rad" and poly["motor_unit"] == "rad", (
        f"{model_file}: units must be rad/rad, got joint={poly['joint_unit']} motor={poly['motor_unit']}")
    roles = list(poly["joint_roles"])
    assert len(roles) == 3 and len(joints) == 3, (
        f"{name}: transmission needs 3 joint_roles + 3 joints, got {roles} / {list(joints)}")
    for jn, role in zip(joints, roles):    # bind roles→real names by suffix (catches wrong order)
        assert jn.endswith(f"_{role}_Joint"), (
            f"{name}: joint '{jn}' does not match transmission role '{role}' (expected order {roles})")

    with open(params_file) as f:
        params = json.load(f)
    assert params_key in params, f"{params_file}: no group '{params_key}'"
    sl = slice(*gain_slice) if gain_slice else slice(None)   # shoulder = 3-slot slice of the 7-DOF arm
    mcp = params[params_key]["motor_control_param"][0]
    k_phi = np.asarray(mcp["k_pos"][sl], dtype=np.float64)
    k_d = np.asarray(mcp["k_vel"][sl], dtype=np.float64)
    assert k_phi.shape == (3,) and k_d.shape == (3,), f"{name}: k_pos/k_vel need 3 values"
    # torque-speed envelope (per motor): stall torque + no-load speed from `actuators`, rated (continuous)
    # torque from motor_control_param. The coupled-PD kernel clips each motor torque to this envelope.
    acts = params[params_key]["actuators"][sl]
    assert len(acts) == 3, f"{name}: need 3 actuators (one per motor), got {len(acts)}"
    tau_stall = np.asarray([a["max_torque"] for a in acts], dtype=np.float64)     # Nm  (envelope x-intercept)
    omega_noload = np.asarray([a["max_speed"] for a in acts], dtype=np.float64)   # rad/s (envelope y-intercept)
    assert (tau_stall > 0).all() and (omega_noload > 0).all(), f"{name}: actuators need +max_torque/+max_speed"
    rated_pos = np.asarray(mcp["torque_limit_pos_abs"][sl], dtype=np.float64)     # Nm  (rated/continuous)
    rated_neg = np.asarray(mcp["torque_limit_neg_abs"][sl], dtype=np.float64)
    assert rated_pos.shape == rated_neg.shape == (3,), f"{name}: torque_limit_pos/neg_abs need 3 values"

    models = [_terms_coeffs(poly["j2m"][key], dim) for key, dim in _MODEL_ORDER_HAND]
    for w in equal_gain_warnings(name, models, 3, k_phi, k_d):
        print(f"[MotorCoupling] WARNING {w}", flush=True)
    lo = np.asarray(poly["q_fit_range_rad"]["lo"], dtype=np.float64)
    hi = np.asarray(poly["q_fit_range_rad"]["hi"], dtype=np.float64)
    assert lo.shape == hi.shape == (3,) and (hi > lo).all(), f"{name}: bad q_fit_range"
    return {"name": name, "joints": list(joints), "models": models,
            "k_phi": k_phi, "k_d": k_d, "q_fit_lo": lo, "q_fit_hi": hi,
            "tau_stall": tau_stall, "omega_noload": omega_noload,
            "rated_pos": rated_pos, "rated_neg": rated_neg,
            # where these gains came from → lets a live run re-read them (read_gains) without a rebuild
            "gain_src": {"file": str(params_file), "params_key": params_key,
                         "slice": list(gain_slice) if gain_slice else None, "n": 3},
            "dim": 3, "part": poly["part"],
            # arm-sliced groups (shoulder) carry the arm's gravcomp share; finger/thumb do not
            # (hands run position-only — their bodies have no gravcomp).
            "gravcomp_fold": gain_slice is not None}


def load_arm_group(params_key: str, joints: list[str], model_file=None, params_file=None,
                     arm_slice: tuple[int, int] = (5, 7)) -> dict:
    """Load a 2-DOF 2×2 arm coupling group — for the 2-DOF coupled-PD kernel owner
    (``MotorCoupledPDArm``). Two instances share it: the wrist Roll/Pitch ballscrew (nonlinear FULL 2×2,
    ``wrist.json``, motors (5, 7)) and the elbow+wristYaw differential pulley (CONSTANT 2×2,
    ``elbow.json``, motors (3, 5)).

    Its motors are a 2-slot slice of the 7-DOF arm (``arm_slice``), so gains (``k_pos``/``k_vel``),
    rated torque, and actuators are SLICED from the arm group ``params_key`` (``"arm_r"``/``"arm_l"``).
    ``joints`` are the 2 real joint names in the map's role order. ``model_file`` (default
    ``mj_mapping/wrist.json``) is the shared firmware map."""
    model_file = Path(model_file) if model_file else DEFAULT_MODEL_DIR / "wrist.json"
    if not model_file.is_absolute():
        model_file = DEFAULT_MODEL_DIR / model_file
    params_file = Path(params_file) if params_file else DEFAULT_PARAMS_FILE
    with open(model_file) as f:
        poly = json.load(f)
    part = poly["part"]                              # "wrist" / "elbow" — labels the group + errors
    name = f"{params_key}_{part}"
    assert poly["joint_unit"] == "rad" and poly["motor_unit"] == "rad", (
        f"{model_file}: units must be rad/rad")
    roles = list(poly["joint_roles"])
    assert len(roles) == 2 and len(joints) == 2, (
        f"{name}: transmission needs 2 joint_roles + 2 joints, got {roles} / {list(joints)}")
    for jn, role in zip(joints, roles):
        assert jn.endswith(f"_{role}_Joint"), (
            f"{name}: joint '{jn}' does not match transmission role '{role}' (expected order {roles})")

    with open(params_file) as f:
        params = json.load(f)
    assert params_key in params, f"{params_file}: no group '{params_key}'"
    sl = slice(*arm_slice)                           # 2-slot slice of the 7-DOF arm motors
    mcp = params[params_key]["motor_control_param"][0]
    k_phi = np.asarray(mcp["k_pos"][sl], dtype=np.float64)
    k_d = np.asarray(mcp["k_vel"][sl], dtype=np.float64)
    assert k_phi.shape == (2,) and k_d.shape == (2,), f"{name}: gain slice needs 2 values"
    rated_pos = np.asarray(mcp["torque_limit_pos_abs"][sl], dtype=np.float64)
    rated_neg = np.asarray(mcp["torque_limit_neg_abs"][sl], dtype=np.float64)
    acts = params[params_key]["actuators"][sl]
    assert len(acts) == 2, f"{name}: actuator slice needs 2"
    tau_stall = np.asarray([a["max_torque"] for a in acts], dtype=np.float64)
    omega_noload = np.asarray([a["max_speed"] for a in acts], dtype=np.float64)
    assert (tau_stall > 0).all() and (omega_noload > 0).all(), f"{name}: actuators need +max_torque/+max_speed"

    models = [_terms_coeffs(poly["j2m"][key], dim) for key, dim in _MODEL_ORDER_ARM]
    for w in equal_gain_warnings(name, models, 2, k_phi, k_d):
        print(f"[MotorCoupling] WARNING {w}", flush=True)
    lo = np.asarray(poly["q_fit_range_rad"]["lo"], dtype=np.float64)
    hi = np.asarray(poly["q_fit_range_rad"]["hi"], dtype=np.float64)
    assert lo.shape == hi.shape == (2,) and (hi > lo).all(), f"{name}: bad q_fit_range"
    return {"name": name, "joints": list(joints), "models": models,
            "k_phi": k_phi, "k_d": k_d, "q_fit_lo": lo, "q_fit_hi": hi,
            "tau_stall": tau_stall, "omega_noload": omega_noload,
            "rated_pos": rated_pos, "rated_neg": rated_neg,
            "gain_src": {"file": str(params_file), "params_key": params_key,
                         "slice": list(arm_slice), "n": 2},                     # see read_gains
            "dim": 2, "part": part,
            "gravcomp_fold": True}   # arm blocks carry the arm's gravcomp share


def _pack(groups: list[dict], models_per_group: int = MODELS_PER_GROUP_HAND) -> dict[str, np.ndarray]:
    """Flatten G groups into the fixed kernel buffers (model order [m1, m2, (m3)])."""
    assert groups, "at least one coupled group required"
    terms_rows, coeffs_rows, beg, end = [], [], [], []
    off = 0
    for grp in groups:
        assert len(grp["models"]) == models_per_group
        for terms, coeffs in grp["models"]:
            terms_rows.append(terms)
            coeffs_rows.append(coeffs)
            beg.append(off)
            off += terms.shape[0]
            end.append(off)
    stack = lambda key: np.stack([g[key] for g in groups]).astype(np.float64)  # noqa: E731
    return {
        "terms": np.concatenate(terms_rows, axis=0).astype(np.int32),
        "coeffs": np.concatenate(coeffs_rows, axis=0).astype(np.float64),
        "model_beg": np.asarray(beg, dtype=np.int32),
        "model_end": np.asarray(end, dtype=np.int32),
        "k_phi": stack("k_phi"), "k_d": stack("k_d"),
        "q_fit_lo": stack("q_fit_lo"), "q_fit_hi": stack("q_fit_hi"),
        "tau_stall": stack("tau_stall"), "omega_noload": stack("omega_noload"),
        "rated_pos": stack("rated_pos"), "rated_neg": stack("rated_neg"),
    }
