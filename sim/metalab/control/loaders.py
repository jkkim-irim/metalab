from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODELS_PER_GROUP_HAND = 3
_MODEL_ORDER_HAND = (("m1", 1), ("m2", 2), ("m3", 3))
MODELS_PER_GROUP_ARM = 2
_MODEL_ORDER_ARM = (("m1", 2), ("m2", 2))


def read_gains(src: dict) -> tuple[np.ndarray, np.ndarray]:
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
    G = np.zeros((len(models), dim))
    x = np.asarray(q, dtype=np.float64)
    for k, (terms, coeffs) in enumerate(models):
        e = terms[:, :dim]
        p = x ** e
        for j in range(dim):
            pj = p.copy()
            pj[:, j] = np.where(e[:, j] > 0, e[:, j] * x[j] ** np.maximum(e[:, j] - 1, 0), 0.0)
            G[k, j] = (coeffs * pj.prod(axis=1)).sum()
    return G


def const_jacobian(models: list, dim: int) -> np.ndarray | None:
    G = np.zeros((len(models), dim))
    for k, (terms, coeffs) in enumerate(models):
        for e, c in zip(terms, coeffs):
            deg = int(e.sum())
            if deg > 1:
                return None
            if deg == 1:
                G[k, int(np.argmax(e))] += c
    return G


def equal_gain_warnings(name: str, models: list, dim: int, k_phi: np.ndarray,
                        k_d: np.ndarray) -> list[str]:
    G = const_jacobian(models, dim)
    if G is None:
        return []
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


def load_hand_group(params_key: str, joints: list[str], model_file, params_file,
                    gain_slice: tuple[int, int] | None = None) -> dict:
    model_file, params_file = Path(model_file), Path(params_file)
    with open(model_file) as f:
        poly = json.load(f)
    name = f"{params_key}_{poly['part']}" if gain_slice else params_key
    assert poly["joint_unit"] == "rad" and poly["motor_unit"] == "rad", (
        f"{model_file}: units must be rad/rad, got joint={poly['joint_unit']} motor={poly['motor_unit']}")
    roles = list(poly["joint_roles"])
    assert len(roles) == 3 and len(joints) == 3, (
        f"{name}: transmission needs 3 joint_roles + 3 joints, got {roles} / {list(joints)}")
    for jn, role in zip(joints, roles):
        assert jn.endswith(f"_{role}_Joint"), (
            f"{name}: joint '{jn}' does not match transmission role '{role}' (expected order {roles})")

    with open(params_file) as f:
        params = json.load(f)
    assert params_key in params, f"{params_file}: no group '{params_key}'"
    sl = slice(*gain_slice) if gain_slice else slice(None)
    mcp = params[params_key]["motor_control_param"][0]
    k_phi = np.asarray(mcp["k_pos"][sl], dtype=np.float64)
    k_d = np.asarray(mcp["k_vel"][sl], dtype=np.float64)
    assert k_phi.shape == (3,) and k_d.shape == (3,), f"{name}: k_pos/k_vel need 3 values"
    acts = params[params_key]["actuators"][sl]
    assert len(acts) == 3, f"{name}: need 3 actuators (one per motor), got {len(acts)}"
    tau_stall = np.asarray([a["max_torque"] for a in acts], dtype=np.float64)
    omega_noload = np.asarray([a["max_speed"] for a in acts], dtype=np.float64)
    assert (tau_stall > 0).all() and (omega_noload > 0).all(), f"{name}: actuators need +max_torque/+max_speed"
    rated_pos = np.asarray(mcp["torque_limit_pos_abs"][sl], dtype=np.float64)
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
            "gain_src": {"file": str(params_file), "params_key": params_key,
                         "slice": list(gain_slice) if gain_slice else None, "n": 3},
            "dim": 3, "part": poly["part"],
            "gravcomp_fold": gain_slice is not None}


def load_arm_group(params_key: str, joints: list[str], model_file, params_file,
                     arm_slice: tuple[int, int]) -> dict:
    model_file, params_file = Path(model_file), Path(params_file)
    with open(model_file) as f:
        poly = json.load(f)
    part = poly["part"]
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
    sl = slice(*arm_slice)
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
                         "slice": list(arm_slice), "n": 2},
            "dim": 2, "part": part,
            "gravcomp_fold": True}


def load_coupled_groups(robot) -> tuple[list[dict], list[dict]]:
    hand, arm = [], []
    for g in robot.coupled_groups():
        model, params = robot.motor.model_path(g), robot.motor.params_path()
        if g.kind == "arm":
            arm.append(load_arm_group(g.params_key, g.joints, model, params, arm_slice=g.gain_slice))
        else:
            hand.append(load_hand_group(g.params_key, g.joints, model, params, gain_slice=g.gain_slice))
    return hand, arm


def _pack(groups: list[dict], models_per_group: int = MODELS_PER_GROUP_HAND) -> dict[str, np.ndarray]:
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
