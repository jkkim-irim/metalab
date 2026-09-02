"""Tests for the motor-to-joint coupling runtime (``motor_coupling``, warp):

- ``_kq_oracle`` ≡ Gᵀ·diag(k_phi)·G (numpy); ``coupled_pd_hand_kernel`` ≡ the method-2 motor-space PD;
  method 1 (k_q·Δq) ≡ method 2 in the small-Δq limit (both use G = ∂m/∂q).
- The shared transmission maps (``mj_mapping/{finger,thumb,wrist,elbow}.json``) each back multiple
  groups, with per-group motor gains from ``robot_model.json``.

The kernels are cross-checked against a self-contained numpy J2M oracle (``_Poly`` below — no runtime
dependency). Runs under pytest, or directly (``python3 sim/metalab/tests/test_motor_coupling.py``);
warp kernel tests self-skip where warp is not installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_FINGER_JSON = _REPO / "sim/metalab/actuation/mj_mapping/finger.json"   # shared firmware J2M
_THUMB_JSON = _REPO / "sim/metalab/actuation/mj_mapping/thumb.json"      # shared thumb J2M
_PARAMS_JSON = _REPO / "sim/metalab/actuation/robot_model.json"
_WRIST_JSON = _REPO / "sim/metalab/actuation/mj_mapping/wrist.json"      # shared wrist J2M
_ELBOW_JSON = _REPO / "sim/metalab/actuation/mj_mapping/elbow.json"      # shared elbow J2M
_SHOULDER_JSON = _REPO / "sim/metalab/actuation/mj_mapping/shoulder.json"  # shared shoulder J2M
_JOINTS = ["R_Index_ABAD_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint"]
_THUMB_JOINTS = ["R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint"]
_WRIST_JOINTS = ["R_Wrist_Roll_Joint", "R_Wrist_Pitch_Joint"]
_ELBOW_JOINTS = ["R_Elbow_Joint", "R_Wrist_Yaw_Joint"]
_SHOULDER_JOINTS = ["R_Shoulder_Pitch_Joint", "R_Shoulder_Roll_Joint", "R_Shoulder_Yaw_Joint"]


# ---------------------------------------------------------------------------
# numpy J2M oracle (mirrors the warp kernels — self-contained, no runtime dep)
# ---------------------------------------------------------------------------

class _Poly:
    """Minimal J2M-model evaluator for the oracle: value Σ_t c_t·Π_j x_j^e_tj and its gradient, over
    the model-json (``model_terms``/``coefficients``) convention the warp kernel packs."""

    def __init__(self, terms, coeffs):
        self.terms = np.asarray(terms, dtype=np.int64)
        self.coeffs = np.asarray(coeffs, dtype=np.float64)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        return float(sum(c * np.prod(x ** e) for e, c in zip(self.terms, self.coeffs)))

    def grad(self, x):
        x = np.asarray(x, dtype=np.float64)
        g = np.zeros(self.terms.shape[1])
        for e, c in zip(self.terms, self.coeffs):
            for j in range(len(e)):
                if e[j] > 0:
                    ej = e.copy()
                    ej[j] -= 1
                    g[j] += c * e[j] * np.prod(x ** ej)
        return g


def _transmission(poly: dict):
    """J2M models + fit envelope from the (firmware) json."""
    j2m = [_Poly(poly["j2m"][k]["model_terms"], poly["j2m"][k]["coefficients"])
           for k in ("m1", "m2", "m3")]
    lo = np.asarray(poly["q_fit_range_rad"]["lo"])
    hi = np.asarray(poly["q_fit_range_rad"]["hi"])
    return j2m, lo, hi


def _motor_pos_np(j2m, q):
    return np.array([j2m[0](q[:1]), j2m[1](q[:2]), j2m[2](q[:3])])


def _G_np(j2m, q):
    """G = ∂m/∂q (lower-triangular), the J2M gradient at joint config q."""
    G = np.zeros((3, 3))
    G[0, :1] = j2m[0].grad(q[:1])
    G[1, :2] = j2m[1].grad(q[:2])
    G[2, :3] = j2m[2].grad(q[:3])
    return G


def _kq_oracle(poly: dict, k_phi: np.ndarray):
    """k_q(q) = Gᵀ·diag(k_phi)·G."""
    j2m, lo, hi = _transmission(poly)

    def kq(q):
        G = _G_np(j2m, np.clip(q, lo, hi))
        return G.T @ np.diag(k_phi) @ G

    return kq


def _coupled_tau_oracle(poly: dict, k_phi: np.ndarray, k_d: np.ndarray,
                        tau_stall=None, omega_noload=None, rated_pos=None, rated_neg=None):
    """Method-2 coupled PD joint torque: τ_q = Gᵀ·τ_m with τ_m = k_phi·(J2M(q*)−J2M(q)) − k_d·(G·q̇),
    optionally clipped to the per-motor torque-speed envelope ∩ ±rated (mirrors ``_clamp_env``)."""
    j2m, lo, hi = _transmission(poly)

    def tau(q, qd, q_tgt):
        qc, qtc = np.clip(q, lo, hi), np.clip(q_tgt, lo, hi)
        G = _G_np(j2m, qc)
        phi_dot = G @ qd
        tau_m = k_phi * (_motor_pos_np(j2m, qtc) - _motor_pos_np(j2m, qc)) - k_d * phi_dot
        if tau_stall is not None:                          # torque-speed envelope (stall droop) ∩ ±rated
            thi = np.minimum(np.minimum(tau_stall * (1.0 - phi_dot / omega_noload), tau_stall), rated_pos)
            tlo = np.maximum(np.maximum(-tau_stall * (1.0 + phi_dot / omega_noload), -tau_stall), -rated_neg)
            tau_m = np.clip(tau_m, tlo, thi)
        return G.T @ tau_m

    return tau


def _tau_q_oracle(poly, grp, q, qd, qt, tau_g):
    """FULL control-law oracle for ONE group (any DOF): τ_q = Gᵀ·clamp(τ_m_PD + G⁻ᵀ·τ_g) — the
    method-2 motor PD + gravcomp fold + torque-speed∩rated clamp. Pins the warp kernels, which BOTH
    engines run (genesis drives them through its backend adapter) — so this is the parity anchor."""
    keys = [k for k in ("m1", "m2", "m3") if k in poly["j2m"]]
    m = [_Poly(poly["j2m"][k]["model_terms"], poly["j2m"][k]["coefficients"]) for k in keys]
    d = len(m)
    qc = np.clip(q, grp["q_fit_lo"], grp["q_fit_hi"])
    qtc = np.clip(qt, grp["q_fit_lo"], grp["q_fit_hi"])
    G = np.stack([np.pad(m[i].grad(qc[: m[i].terms.shape[1]]),   # lower-tri rows → square
                         (0, d - m[i].terms.shape[1])) for i in range(d)])
    mm = np.array([m[i](qc[: m[i].terms.shape[1]]) for i in range(d)])
    mt = np.array([m[i](qtc[: m[i].terms.shape[1]]) for i in range(d)])
    phidot = G @ qd
    tau_m = grp["k_phi"] * (mt - mm) - grp["k_d"] * phidot + np.linalg.inv(G).T @ tau_g
    thi = np.minimum(np.minimum(grp["tau_stall"] * (1.0 - phidot / grp["omega_noload"]),
                                grp["tau_stall"]), grp["rated_pos"])
    tlo = np.maximum(np.maximum(-grp["tau_stall"] * (1.0 + phidot / grp["omega_noload"]),
                                -grp["tau_stall"]), -grp["rated_neg"])
    return G.T @ np.clip(tau_m, tlo, thi)


def _G_and_pd_oracle(poly, grp, q, qd, qt):
    """(G, Gᵀ·τ_m^PD) for ONE group of any DOF — the PD term alone, PRE-clamp and PRE-gravity-fold.
    Same construction as :func:`_tau_q_oracle` (which then folds τ_g in and clamps); this is the oracle
    for the ``tau_pd_torch`` component read."""
    keys = [k for k in ("m1", "m2", "m3") if k in poly["j2m"]]
    m = [_Poly(poly["j2m"][k]["model_terms"], poly["j2m"][k]["coefficients"]) for k in keys]
    d = len(m)
    qc = np.clip(q, grp["q_fit_lo"], grp["q_fit_hi"])
    qtc = np.clip(qt, grp["q_fit_lo"], grp["q_fit_hi"])
    G = np.stack([np.pad(m[i].grad(qc[: m[i].terms.shape[1]]), (0, d - m[i].terms.shape[1]))
                  for i in range(d)])
    mm = np.array([m[i](qc[: m[i].terms.shape[1]]) for i in range(d)])
    mt = np.array([m[i](qtc[: m[i].terms.shape[1]]) for i in range(d)])
    tau_m = grp["k_phi"] * (mt - mm) - grp["k_d"] * (G @ qd)
    return G, G.T @ tau_m


def _index_r_inputs():
    """The shared finger transmission + the index_r group's motor gains (the group under test)."""
    if not _FINGER_JSON.exists():
        return None
    poly = json.loads(_FINGER_JSON.read_text())
    mcp = json.loads(_PARAMS_JSON.read_text())["index_r"]["motor_control_param"][0]
    return poly, np.asarray(mcp["k_pos"], np.float64), np.asarray(mcp["k_vel"], np.float64)


def _thumb_inputs():
    """The shared thumb transmission + the thumb_r group's motor gains (the group under test)."""
    if not _THUMB_JSON.exists():
        return None
    poly = json.loads(_THUMB_JSON.read_text())
    mcp = json.loads(_PARAMS_JSON.read_text())["thumb_r"]["motor_control_param"][0]
    return poly, np.asarray(mcp["k_pos"], np.float64), np.asarray(mcp["k_vel"], np.float64)


# ---------------------------------------------------------------------------
# k_q — numpy oracle symmetry (the small-signal joint stiffness Gᵀ·diag(k_phi)·G)
# ---------------------------------------------------------------------------

def test_kq_oracle_is_symmetric():
    got = _index_r_inputs()
    if got is None:
        print("SKIP test_kq_oracle_is_symmetric (no finger.json)")
        return
    poly, k_phi, _ = got
    kq = _kq_oracle(poly, k_phi)
    lo, hi = np.asarray(poly["q_fit_range_rad"]["lo"]), np.asarray(poly["q_fit_range_rad"]["hi"])
    rng = np.random.default_rng(3)
    for _ in range(50):
        K = kq(rng.uniform(lo, hi))
        assert np.allclose(K, K.T, atol=1e-12)


# ---------------------------------------------------------------------------
# coupled PD — numpy oracle vs the warp kernel, + method-1/2 small-signal agreement
# ---------------------------------------------------------------------------

def _coupled_pd_oracle_check(group_name, joints, model_file, poly, k_phi, k_d, seed, gain_slice=None):
    """coupled_pd_hand_kernel (warp CPU) ≡ numpy method-2 oracle for one transmission, over random
    (q, q̇, q*). Shared by the finger, thumb, and shoulder cases; call only after the warp guard passed."""
    import torch
    import warp as wp

    from sim.metalab.actuation.motor_coupling import MotorCoupledPDHand, load_hand_group
    wp.init()

    n = 256
    lo, hi = np.asarray(poly["q_fit_range_rad"]["lo"]), np.asarray(poly["q_fit_range_rad"]["hi"])
    rng = np.random.default_rng(seed)
    q = rng.uniform(lo, hi, size=(n, 3))
    q_tgt = rng.uniform(lo, hi, size=(n, 3))
    qd = rng.uniform(-2.0, 2.0, size=(n, 3))

    jq = torch.tensor(q.reshape(-1), dtype=torch.float32, device="cpu")
    jqd = torch.tensor(qd.reshape(-1), dtype=torch.float32, device="cpu")
    jtgt = torch.tensor(q_tgt.reshape(-1), dtype=torch.float32, device="cpu")
    jf = torch.zeros(3 * n, dtype=torch.float32, device="cpu")
    idx = torch.arange(3 * n, dtype=torch.int32, device="cpu").reshape(n, 3)

    grp = load_hand_group(group_name, joints, model_file=model_file, params_file=_PARAMS_JSON,
                          gain_slice=gain_slice)
    pd = MotorCoupledPDHand([grp], idx, idx, num_envs=n, device="cpu")
    pd.launch(wp.from_torch(jq), wp.from_torch(jqd), wp.from_torch(jtgt), wp.from_torch(jf))
    wp.synchronize()
    got_tau = jf.cpu().numpy().reshape(n, 3)
    assert np.allclose(got_tau, pd.tau_torch.cpu().numpy().reshape(n, 3))   # joint_f == readout

    q32 = jq.numpy().reshape(n, 3).astype(np.float64)                # match the kernel's f32 inputs
    qd32 = jqd.numpy().reshape(n, 3).astype(np.float64)
    qt32 = jtgt.numpy().reshape(n, 3).astype(np.float64)
    oracle = _coupled_tau_oracle(poly, k_phi, k_d, grp["tau_stall"], grp["omega_noload"],
                                 grp["rated_pos"], grp["rated_neg"])   # kernel clamps → oracle must too
    ref = np.stack([oracle(q32[i], qd32[i], qt32[i]) for i in range(n)])
    assert np.allclose(got_tau, ref, rtol=1e-6, atol=1e-9), np.abs(got_tau - ref).max()


def test_coupled_pd_hand_kernel_matches_oracle():
    """coupled_pd_hand_kernel reproduces the numpy method-2 oracle for the FINGER transmission."""
    try:
        import warp  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP test_coupled_pd_hand_kernel (warp not installed)")
        return
    got = _index_r_inputs()
    if got is None:
        print("SKIP test_coupled_pd_hand_kernel (no finger.json)")
        return
    _coupled_pd_oracle_check("index_r", _JOINTS, _FINGER_JSON, *got, seed=9)


def test_coupled_pd_hand_kernel_matches_oracle_thumb():
    """Same GPU control-law check for the THUMB transmission — 153-term m3, Yaw decoupled, Yaw
    envelope spanning both signs. Validates thumb.json end-to-end through the coupled-PD kernel."""
    try:
        import warp  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP test_coupled_pd_hand_kernel_thumb (warp not installed)")
        return
    got = _thumb_inputs()
    if got is None:
        print("SKIP test_coupled_pd_hand_kernel_thumb (no thumb.json)")
        return
    _coupled_pd_oracle_check("thumb_r", _THUMB_JOINTS, _THUMB_JSON, *got, seed=21)


def test_coupled_pd_shoulder_kernel_matches_oracle():
    """Same hand-kernel check for the SHOULDER (diagonal per-joint scalar ratio, shoulder.json, gains
    sliced from the 7-DOF arm at (0, 3)) — plus the structural invariant that the map is exactly
    DIAGONAL linear: one degree-1 term per model, G = r·I with a single shared positive ratio r."""
    try:
        import warp  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP test_coupled_pd_shoulder (warp not installed)")
        return
    if not _SHOULDER_JSON.exists():
        print("SKIP test_coupled_pd_shoulder (no shoulder.json)")
        return

    poly = json.loads(_SHOULDER_JSON.read_text())
    j2m, _, _ = _transmission(poly)
    G = _G_np(j2m, np.array([-1.1, 0.7, 2.0]))          # constant: config value is arbitrary
    r = G[0, 0]
    assert r > 0 and np.allclose(G, r * np.eye(3)), G   # diagonal, single shared ratio
    assert np.allclose(G, _G_np(j2m, np.zeros(3)))      # config-independent

    mcp = json.loads(_PARAMS_JSON.read_text())["arm_r"]["motor_control_param"][0]
    k_phi = np.asarray(mcp["k_pos"][0:3], np.float64)
    k_d = np.asarray(mcp["k_vel"][0:3], np.float64)
    _coupled_pd_oracle_check("arm_r", _SHOULDER_JOINTS, _SHOULDER_JSON, poly, k_phi, k_d,
                             seed=27, gain_slice=(0, 3))


def _coupled_pd_arm_oracle_check(joints, json_path, arm_slice, seed):
    """coupled_pd_arm_kernel (2-DOF, 2×2) ≡ numpy method-2 oracle for one arm transmission over
    random (q, q̇, q*) incl. the clamp. Shared by the wrist (nonlinear full 2×2) and elbow (constant
    2×2) cases; call only after the warp guard passed."""
    import torch
    import warp as wp

    from sim.metalab.actuation.motor_coupling import MotorCoupledPDArm, load_arm_group
    wp.init()
    grp = load_arm_group("arm_r", joints, model_file=json_path, params_file=_PARAMS_JSON,
                         arm_slice=arm_slice)
    poly = json.loads(json_path.read_text())
    m = [_Poly(poly["j2m"][k]["model_terms"], poly["j2m"][k]["coefficients"]) for k in ("m1", "m2")]
    lo, hi = grp["q_fit_lo"], grp["q_fit_hi"]
    kphi, kd = grp["k_phi"], grp["k_d"]
    ts, w0, rp, rn = grp["tau_stall"], grp["omega_noload"], grp["rated_pos"], grp["rated_neg"]

    def oracle(q, qd, qt):                                       # method-2, 2×2, clamped
        qc, qtc = np.clip(q, lo, hi), np.clip(qt, lo, hi)
        G = np.stack([m[0].grad(qc), m[1].grad(qc)])            # 2×2 (not triangular)
        mm, mt = np.array([m[0](qc), m[1](qc)]), np.array([m[0](qtc), m[1](qtc)])
        phidot = G @ qd
        tau_m = kphi * (mt - mm) - kd * phidot
        thi = np.minimum(np.minimum(ts * (1.0 - phidot / w0), ts), rp)
        tlo = np.maximum(np.maximum(-ts * (1.0 + phidot / w0), -ts), -rn)
        return G.T @ np.clip(tau_m, tlo, thi)

    n = 256
    rng = np.random.default_rng(seed)
    q = rng.uniform(lo, hi, (n, 2))
    qt = rng.uniform(lo, hi, (n, 2))
    qd = rng.uniform(-2.0, 2.0, (n, 2))
    jq = torch.tensor(q.reshape(-1), dtype=torch.float32)
    jqd = torch.tensor(qd.reshape(-1), dtype=torch.float32)
    jtg = torch.tensor(qt.reshape(-1), dtype=torch.float32)
    jf = torch.zeros(2 * n, dtype=torch.float32)
    idx = torch.arange(2 * n, dtype=torch.int32).reshape(n, 2)
    pd = MotorCoupledPDArm([grp], idx, idx, num_envs=n, device="cpu")
    pd.launch(wp.from_torch(jq), wp.from_torch(jqd), wp.from_torch(jtg), wp.from_torch(jf))
    wp.synchronize()
    got = jf.numpy().reshape(n, 2)
    assert np.allclose(got, pd.tau_torch.numpy().reshape(n, 2))     # joint_f == readout
    q32 = jq.numpy().reshape(n, 2).astype(np.float64)
    qd32 = jqd.numpy().reshape(n, 2).astype(np.float64)
    qt32 = jtg.numpy().reshape(n, 2).astype(np.float64)
    ref = np.stack([oracle(q32[i], qd32[i], qt32[i]) for i in range(n)])
    assert np.allclose(got, ref, rtol=1e-5, atol=1e-8), np.abs(got - ref).max()


def test_coupled_pd_wrist_kernel_matches_oracle():
    """coupled_pd_arm_kernel (2-DOF, FULL 2×2 ballscrew — both motors ← both joints) reproduces the
    numpy method-2 oracle over random (q, q̇, q*) incl. the clamp. Validates the wrist GPU control law."""
    try:
        import warp  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP test_coupled_pd_wrist (warp not installed)")
        return
    if not _WRIST_JSON.exists():
        print("SKIP test_coupled_pd_wrist (no wrist.json)")
        return
    _coupled_pd_arm_oracle_check(_WRIST_JOINTS, _WRIST_JSON, (5, 7), seed=15)


def test_coupled_pd_elbow_kernel_matches_oracle():
    """Same arm-kernel check for the ELBOW+WRISTYAW differential pulley (elbow.json) — plus the
    structural invariant that the map is exactly LINEAR: pure degree-1 terms, so G is the constant
    [[a, b], [-a, b]] everywhere (a = ∂m1/∂qElbow = -∂m2/∂qElbow, b = ∂m/∂qYaw shared)."""
    try:
        import warp  # noqa: F401
    except ModuleNotFoundError:
        print("SKIP test_coupled_pd_elbow (warp not installed)")
        return
    if not _ELBOW_JSON.exists():
        print("SKIP test_coupled_pd_elbow (no elbow.json)")
        return

    poly = json.loads(_ELBOW_JSON.read_text())
    m = [_Poly(poly["j2m"][k]["model_terms"], poly["j2m"][k]["coefficients"]) for k in ("m1", "m2")]
    for p in m:                                     # linear: every term degree-1, no constant offset
        assert p.terms.shape == (2, 2) and (p.terms.sum(axis=1) == 1).all(), p.terms
    G = np.stack([p.grad(np.zeros(2)) for p in m])  # constant Jacobian [[a, b], [-a, b]]
    assert np.allclose(G, np.stack([p.grad(np.array([-1.3, 2.1])) for p in m]))   # config-independent
    a, b = G[0]
    assert a > 0 and b > 0 and np.allclose(G, [[a, b], [-a, b]]), G

    _coupled_pd_arm_oracle_check(_ELBOW_JOINTS, _ELBOW_JSON, (3, 5), seed=33)


def test_coupled_pd_gravcomp_folds_into_motor_clamp():
    """Gravity feedforward fold: groups flagged ``gravcomp_fold`` add G⁻ᵀ·τ_g to the motor torque
    BEFORE the envelope clamp — τ_q = Gᵀ·clamp(τ_m_PD + G⁻ᵀ·τ_g) — and write joint_f = τ_q − τ_g
    (cancelling MuJoCo's own passive application) while the readout stays τ_q. Non-fold groups
    (finger) must ignore τ_g entirely. Exercises BOTH kernels: hand (one owner mixing a fold-off
    finger group + a fold-on shoulder group, exactly like production) and arm (wrist)."""
    try:
        import warp as wp
    except ModuleNotFoundError:
        print("SKIP test_coupled_pd_gravcomp (warp not installed)")
        return
    import torch

    from sim.metalab.actuation.motor_coupling import (
        MotorCoupledPDArm,
        MotorCoupledPDHand,
        load_arm_group,
        load_hand_group,
    )
    wp.init()
    n = 128
    rng = np.random.default_rng(41)

    def run(owner_cls, groups, nj_per, tau_g):
        """Launch one owner with an injected fake qfrc_gravcomp == tau_g; return (joint_f, readout)."""
        nj = nj_per * len(groups)
        lo = np.concatenate([g["q_fit_lo"] for g in groups])
        hi = np.concatenate([g["q_fit_hi"] for g in groups])
        q = rng.uniform(lo, hi, (n, nj))
        qt = rng.uniform(lo, hi, (n, nj))
        qd = rng.uniform(-2.0, 2.0, (n, nj))
        jq = torch.tensor(q.reshape(-1), dtype=torch.float32)
        jqd = torch.tensor(qd.reshape(-1), dtype=torch.float32)
        jtg = torch.tensor(qt.reshape(-1), dtype=torch.float32)
        jf = torch.zeros(nj * n, dtype=torch.float32)
        idx = torch.arange(nj * n, dtype=torch.int32).reshape(n, nj)
        gbuf = torch.tensor(tau_g, dtype=torch.float32)               # fake qfrc_gravcomp (n, nj)
        pd = owner_cls(groups, idx, idx, num_envs=n, device="cpu",
                       gravcomp=wp.from_torch(gbuf), gc_dof=torch.arange(nj, dtype=torch.int32))
        pd.launch(wp.from_torch(jq), wp.from_torch(jqd), wp.from_torch(jtg), wp.from_torch(jf))
        wp.synchronize()
        return (jq.numpy().reshape(n, nj).astype(np.float64),        # match the kernel's f32 inputs
                jqd.numpy().reshape(n, nj).astype(np.float64),
                jtg.numpy().reshape(n, nj).astype(np.float64),
                jf.numpy().reshape(n, nj), pd.tau_torch.numpy().reshape(n, nj))

    # --- hand kernel: finger (fold OFF — τ_g ignored) + shoulder (fold ON) in ONE owner ---
    gf = load_hand_group("index_r", _JOINTS, model_file=_FINGER_JSON, params_file=_PARAMS_JSON)
    gs = load_hand_group("arm_r", _SHOULDER_JOINTS, model_file=_SHOULDER_JSON,
                         params_file=_PARAMS_JSON, gain_slice=(0, 3))
    assert not gf["gravcomp_fold"] and gs["gravcomp_fold"]
    tau_g = rng.uniform(-60.0, 60.0, (n, 6))                          # big → clamp is exercised
    q, qd, qt, jf, ro = run(MotorCoupledPDHand, [gf, gs], 3, tau_g)
    pf = json.loads(_FINGER_JSON.read_text())
    ps = json.loads(_SHOULDER_JSON.read_text())
    tg32 = tau_g.astype(np.float32).astype(np.float64)                # kernel reads τ_g as f32
    for i in range(n):
        ref_f = _tau_q_oracle(pf, gf, q[i, :3], qd[i, :3], qt[i, :3], np.zeros(3))   # fold off → τ_g = 0
        ref_s = _tau_q_oracle(ps, gs, q[i, 3:], qd[i, 3:], qt[i, 3:], tg32[i, 3:])
        assert np.allclose(ro[i, :3], ref_f, rtol=1e-5, atol=1e-7)              # readout = τ_q
        assert np.allclose(ro[i, 3:], ref_s, rtol=1e-5, atol=1e-6)
        assert np.allclose(jf[i, :3], ref_f, rtol=1e-5, atol=1e-7)              # fold off → no cancel
        assert np.allclose(jf[i, 3:], ref_s - tg32[i, 3:], rtol=1e-5, atol=1e-6)  # joint_f = τ_q − τ_g

    # --- arm kernel: wrist (fold ON) ---
    gw = load_arm_group("arm_r", _WRIST_JOINTS, model_file=_WRIST_JSON, params_file=_PARAMS_JSON,
                        arm_slice=(5, 7))
    tau_g = rng.uniform(-2.0, 2.0, (n, 2))
    q, qd, qt, jf, ro = run(MotorCoupledPDArm, [gw], 2, tau_g)
    pw = json.loads(_WRIST_JSON.read_text())
    tg32 = tau_g.astype(np.float32).astype(np.float64)
    for i in range(n):
        ref = _tau_q_oracle(pw, gw, q[i], qd[i], qt[i], tg32[i])
        assert np.allclose(ro[i], ref, rtol=1e-5, atol=1e-7)
        assert np.allclose(jf[i], ref - tg32[i], rtol=1e-5, atol=1e-7)


def test_torque_components_split_the_applied_torque():
    """The PRE-clamp component readouts backing ``joint_torque_pd`` / ``joint_torque_gravcomp``:

    * ``tau_pd_torch``  == Gᵀ·τ_m^PD  (the PD term alone, no gravity, no clamp),
    * ``tau_gc_torch``  == τ_g on ``gravcomp_fold`` groups and exactly 0 where the group does not fold,
    * their SUM == the applied τ_q (``tau_torch``) while no motor saturates, and the residual once one
      does is what the envelope clamp removed — so the sum always equals the PRE-clamp joint torque.

    Both kernels, both regimes (gentle = unsaturated everywhere, aggressive = clamp engaged), through
    the production owners — the same buffers the two backend reads slice."""
    try:
        import warp as wp
    except ModuleNotFoundError:
        print("SKIP test_torque_components (warp not installed)")
        return
    import torch

    from sim.metalab.actuation.motor_coupling import (
        MotorCoupledPDArm,
        MotorCoupledPDHand,
        load_arm_group,
        load_hand_group,
    )
    wp.init()
    n = 128
    rng = np.random.default_rng(77)

    def run(owner_cls, groups, nj_per, tau_g, gentle):
        """Launch one owner with fake qfrc_gravcomp == tau_g. gentle=True keeps commands tiny (no
        clamp); False sweeps the full fit range (clamp engaged). → (q, qd, qt, τ_q, pd, gc)."""
        nj = nj_per * len(groups)
        lo = np.concatenate([g["q_fit_lo"] for g in groups])
        hi = np.concatenate([g["q_fit_hi"] for g in groups])
        q = rng.uniform(lo, hi, (n, nj))
        qt = q + rng.uniform(-1e-4, 1e-4, (n, nj)) if gentle else rng.uniform(lo, hi, (n, nj))
        qd = rng.uniform(-1e-3, 1e-3, (n, nj)) if gentle else rng.uniform(-5.0, 5.0, (n, nj))
        jq = torch.tensor(q.reshape(-1), dtype=torch.float32)
        jqd = torch.tensor(qd.reshape(-1), dtype=torch.float32)
        jtg = torch.tensor(qt.reshape(-1), dtype=torch.float32)
        jf = torch.zeros(nj * n, dtype=torch.float32)
        idx = torch.arange(nj * n, dtype=torch.int32).reshape(n, nj)
        gbuf = torch.tensor(tau_g, dtype=torch.float32)
        pd = owner_cls(groups, idx, idx, num_envs=n, device="cpu",
                       gravcomp=wp.from_torch(gbuf), gc_dof=torch.arange(nj, dtype=torch.int32))
        pd.launch(wp.from_torch(jq), wp.from_torch(jqd), wp.from_torch(jtg), wp.from_torch(jf))
        wp.synchronize()
        return (jq.numpy().reshape(n, nj).astype(np.float64),     # match the kernel's f32 inputs
                jqd.numpy().reshape(n, nj).astype(np.float64),
                jtg.numpy().reshape(n, nj).astype(np.float64),
                pd.tau_torch.numpy().reshape(n, nj),
                pd.tau_pd_torch.numpy().reshape(n, nj),
                pd.tau_gc_torch.numpy().reshape(n, nj))

    # --- hand kernel: finger (fold OFF) + shoulder (fold ON) in ONE owner, as in production ---
    gf = load_hand_group("index_r", _JOINTS, model_file=_FINGER_JSON, params_file=_PARAMS_JSON)
    gs = load_hand_group("arm_r", _SHOULDER_JOINTS, model_file=_SHOULDER_JSON,
                         params_file=_PARAMS_JSON, gain_slice=(0, 3))
    assert not gf["gravcomp_fold"] and gs["gravcomp_fold"]
    pf, ps = json.loads(_FINGER_JSON.read_text()), json.loads(_SHOULDER_JSON.read_text())

    for gentle in (True, False):
        tau_g = rng.uniform(-1.0, 1.0, (n, 6)) if gentle else rng.uniform(-60.0, 60.0, (n, 6))
        q, qd, qt, tau, pdc, gcc = run(MotorCoupledPDHand, [gf, gs], 3, tau_g, gentle)
        tg32 = tau_g.astype(np.float32).astype(np.float64)            # kernel reads τ_g as f32
        assert (gcc[:, :3] == 0.0).all(), "fold-off group must report a zero gravity component"
        assert np.allclose(gcc[:, 3:], tg32[:, 3:], rtol=1e-6, atol=1e-6)   # fold-on: component == τ_g
        unsat = 0
        for i in range(n):
            _, ref_f = _G_and_pd_oracle(pf, gf, q[i, :3], qd[i, :3], qt[i, :3])
            _, ref_s = _G_and_pd_oracle(ps, gs, q[i, 3:], qd[i, 3:], qt[i, 3:])
            assert np.allclose(pdc[i, :3], ref_f, rtol=1e-5, atol=1e-7)
            assert np.allclose(pdc[i, 3:], ref_s, rtol=1e-5, atol=1e-5)
            unsat += int(np.allclose(pdc[i] + gcc[i], tau[i], rtol=1e-4, atol=1e-4))
        if gentle:
            assert unsat == n, f"unsaturated regime: components must sum to τ_q ({unsat}/{n})"
        else:
            assert unsat < n // 2, f"clamp barely engaged ({n - unsat}/{n} saturated) — regime too gentle"

    # --- arm kernel (wrist, fold ON): same two regimes ---
    gw = load_arm_group("arm_r", _WRIST_JOINTS, model_file=_WRIST_JSON, params_file=_PARAMS_JSON,
                        arm_slice=(5, 7))
    pw = json.loads(_WRIST_JSON.read_text())
    for gentle, span in ((True, 0.2), (False, 40.0)):
        tau_g = rng.uniform(-span, span, (n, 2))
        q, qd, qt, tau, pdc, gcc = run(MotorCoupledPDArm, [gw], 2, tau_g, gentle)
        tg32 = tau_g.astype(np.float32).astype(np.float64)
        assert np.allclose(gcc, tg32, rtol=1e-6, atol=1e-6)
        unsat = sum(int(np.allclose(pdc[i] + gcc[i], tau[i], rtol=1e-4, atol=1e-4)) for i in range(n))
        for i in range(n):
            assert np.allclose(pdc[i], _G_and_pd_oracle(pw, gw, q[i], qd[i], qt[i])[1],
                               rtol=1e-5, atol=1e-6)
        if gentle:
            assert unsat == n, f"wrist unsaturated regime: {unsat}/{n} summed to τ_q"
        else:
            assert unsat < n, "wrist aggressive regime never saturated — test not exercising the gap"


def test_clamp_bounds_motor_torque_to_envelope():
    """Invariant: the motor torque the kernel actually applied, τ_m = G⁻ᵀ·τ_q recovered from the
    coupled joint torque, never exceeds the per-motor torque-speed envelope ∩ ±rated — for random
    (q, q̇, q*) including huge commands (so the clamp is exercised, not just matched)."""
    try:
        import warp as wp
    except ModuleNotFoundError:
        print("SKIP test_clamp_bounds (warp not installed)")
        return
    import torch

    from sim.metalab.actuation.motor_coupling import MotorCoupledPDHand, load_hand_group

    got = _index_r_inputs()
    if got is None:
        print("SKIP test_clamp_bounds (no finger.json)")
        return
    poly = got[0]
    wp.init()
    grp = load_hand_group("index_r", _JOINTS, model_file=_FINGER_JSON, params_file=_PARAMS_JSON)
    ts, w0, rp, rn = grp["tau_stall"], grp["omega_noload"], grp["rated_pos"], grp["rated_neg"]
    j2m, lo, hi = _transmission(poly)

    n = 256
    rng = np.random.default_rng(31)
    q = rng.uniform(lo, hi, size=(n, 3))
    q_tgt = rng.uniform(lo, hi, size=(n, 3))       # full-range targets → large Δφ → clamp active
    qd = rng.uniform(-5.0, 5.0, size=(n, 3))
    jf = torch.zeros(3 * n, dtype=torch.float32)
    idx = torch.arange(3 * n, dtype=torch.int32).reshape(n, 3)
    pd = MotorCoupledPDHand([grp], idx, idx, num_envs=n, device="cpu")
    pd.launch(wp.from_torch(torch.tensor(q.reshape(-1), dtype=torch.float32)),
              wp.from_torch(torch.tensor(qd.reshape(-1), dtype=torch.float32)),
              wp.from_torch(torch.tensor(q_tgt.reshape(-1), dtype=torch.float32)),
              wp.from_torch(jf))
    wp.synchronize()
    tau_q = jf.numpy().reshape(n, 3).astype(np.float64)

    saturated = 0
    for i in range(n):
        G = _G_np(j2m, np.clip(q[i], lo, hi))
        tau_m = np.linalg.solve(G.T, tau_q[i])                 # recover the applied motor torque
        phi_dot = G @ qd[i]
        thi = np.minimum(np.minimum(ts * (1.0 - phi_dot / w0), ts), rp)
        tlo = np.maximum(np.maximum(-ts * (1.0 + phi_dot / w0), -ts), -rn)
        assert (tau_m <= thi + 1e-6).all() and (tau_m >= tlo - 1e-6).all(), (i, tau_m, tlo, thi)
        saturated += int(np.any(np.isclose(tau_m, thi, atol=1e-6) | np.isclose(tau_m, tlo, atol=1e-6)))
    assert saturated > n // 2, f"clamp barely engaged ({saturated}/{n}) — test not exercising it"


def test_coupled_pd_small_signal_matches_kq():
    """Method 2 (coupled PD, no clamp, q̇=0) → method 1 (k_q·Δq) in the small-Δq limit: the exact
    motor error J2M(q*)−J2M(q) linearizes to G·Δq, so τ_q → Gᵀ·k_phi·G·Δq = k_q·Δq. Both use the
    same G = ∂m/∂q, so the residual shrinks linearly with Δq."""
    got = _index_r_inputs()
    if got is None:
        print("SKIP test_coupled_pd_small_signal (no finger.json)")
        return
    poly, k_phi, _ = got
    kq = _kq_oracle(poly, k_phi)
    tau = _coupled_tau_oracle(poly, k_phi, np.zeros(3))
    lo, hi = np.asarray(poly["q_fit_range_rad"]["lo"]), np.asarray(poly["q_fit_range_rad"]["hi"])
    rng = np.random.default_rng(13)
    for _ in range(30):
        q = rng.uniform(lo + 0.1, hi - 0.1)
        dq = rng.uniform(-1.0, 1.0, size=3)
        dq /= np.linalg.norm(dq)
        prev = None
        for eps in (1e-2, 1e-3, 1e-4):
            resid = np.linalg.norm(tau(q, np.zeros(3), q + eps * dq) - kq(q) @ (eps * dq))
            if prev is not None:
                assert resid <= 0.2 * prev + 1e-12, (eps, resid, prev)   # ~10× shrink per decade
            prev = resid


# ---------------------------------------------------------------------------
# shared finger map — one transmission backs every finger group (control_mode: motor)
# ---------------------------------------------------------------------------

def test_shared_finger_map_backs_multiple_groups():
    """One finger.json serves all finger groups: index_r and middle_r load the SAME transmission
    coefficients + q envelope but their OWN motor gains (robot_model.json[<finger>_<hand>]). Also
    checks the role-order guard fails loud on mis-ordered joints."""
    try:
        from sim.metalab.actuation.motor_coupling import load_hand_group
    except ModuleNotFoundError:
        print("SKIP test_shared_finger_map (torch/warp not installed)")
        return

    gi = load_hand_group("index_r", ["R_Index_ABAD_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint"],
                    model_file=_FINGER_JSON, params_file=_PARAMS_JSON)
    gm = load_hand_group("middle_r", ["R_Middle_ABAD_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint"],
                    model_file=_FINGER_JSON, params_file=_PARAMS_JSON)
    for (ti, ci), (tm, cm) in zip(gi["models"], gm["models"]):        # same shared transmission
        assert np.array_equal(ti, tm) and np.array_equal(ci, cm)
    assert np.array_equal(gi["q_fit_lo"], gm["q_fit_lo"]) and np.array_equal(gi["q_fit_hi"], gm["q_fit_hi"])
    params = json.loads(_PARAMS_JSON.read_text())                      # own per-group motor gains
    assert np.array_equal(gi["k_phi"], np.asarray(params["index_r"]["motor_control_param"][0]["k_pos"]))
    assert np.array_equal(gm["k_phi"], np.asarray(params["middle_r"]["motor_control_param"][0]["k_pos"]))

    try:
        load_hand_group("index_r", ["R_Index_MCP_Joint", "R_Index_ABAD_Joint", "R_Index_PIP_Joint"],
                   model_file=_FINGER_JSON, params_file=_PARAMS_JSON)
        raise AssertionError("mis-ordered joints must fail the role check")
    except AssertionError as e:
        assert "role" in str(e)


def test_shared_thumb_map_backs_both_hands():
    """One thumb.json serves both thumbs: thumb_r and thumb_l (byte-identical firmware) load the SAME
    transmission with their own gains; the Yaw envelope spans both signs (L/R-symmetric union) and
    the role guard is Yaw/CMC/MCP."""
    try:
        from sim.metalab.actuation.motor_coupling import load_hand_group
    except ModuleNotFoundError:
        print("SKIP test_shared_thumb_map (torch/warp not installed)")
        return

    gr = load_hand_group("thumb_r", ["R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint"],
                    model_file=_THUMB_JSON, params_file=_PARAMS_JSON)
    gl = load_hand_group("thumb_l", ["L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint"],
                    model_file=_THUMB_JSON, params_file=_PARAMS_JSON)
    for (tr, cr), (tl, cl) in zip(gr["models"], gl["models"]):        # same shared transmission
        assert np.array_equal(tr, tl) and np.array_equal(cr, cl)
    assert gr["q_fit_lo"][0] < 0.0 < gr["q_fit_hi"][0]               # Yaw union spans both signs

    try:
        load_hand_group("thumb_r", ["R_Thumb_CMC_Joint", "R_Thumb_Yaw_Joint", "R_Thumb_MCP_Joint"],
                   model_file=_THUMB_JSON, params_file=_PARAMS_JSON)
        raise AssertionError("mis-ordered thumb joints must fail the role check")
    except AssertionError as e:
        assert "role" in str(e)


def test_reload_gains_swaps_live_buffers():
    """``reload_gains`` re-reads robot_model.json into the owner's LIVE kernel buffers: new gains land in
    the same arrays (no realloc — that is what keeps a captured CUDA graph valid), the changed groups are
    reported, and the Torque-mode D-gain zeroing survives the swap."""
    try:
        import warp as wp
    except ModuleNotFoundError:
        print("SKIP test_reload_gains (warp not installed)")
        return
    import shutil
    import tempfile

    import torch

    from sim.metalab.actuation.motor_coupling import MotorCoupledPDHand, load_hand_group

    wp.init()
    with tempfile.TemporaryDirectory() as td:
        # a private copy of the params file: the group carries its path, so the edit under test is the
        # copy, never the repo's real robot_model.json
        pf = Path(td) / "robot_model.json"
        shutil.copy(_PARAMS_JSON, pf)
        grp = load_hand_group("index_r", _JOINTS, model_file=_FINGER_JSON, params_file=pf)
        idx = torch.arange(3, dtype=torch.int32).reshape(1, 3)
        own = MotorCoupledPDHand([grp], idx, idx, 1, "cpu")
        k_phi_addr = own.buf.k_phi.ptr
        before = own.buf.k_phi.numpy().copy()

        assert own.reload_gains() == [], "unedited file must report no changed group"

        data = json.loads(pf.read_text())
        data["index_r"]["motor_control_param"][0]["k_pos"] = [v * 0.25 for v in before[0]]
        pf.write_text(json.dumps(data))

        own.set_fold_kd_zero(True)                        # Torque mode on across the reload
        kd_zeroed = own.buf.k_d.numpy().copy()
        assert own.reload_gains() == ["index_r"]
        assert np.allclose(own.buf.k_phi.numpy()[0], before[0] * 0.25), "new k_pos must be in the buffer"
        assert own.buf.k_phi.ptr == k_phi_addr, "buffer must be rewritten in place, not reallocated"
        assert np.array_equal(own.buf.k_d.numpy(), kd_zeroed), "float-mode D zeroing must survive a reload"
        own.set_fold_kd_zero(False)                       # back to Position: k_d = the RELOADED originals
        assert np.allclose(own.buf.k_d.numpy()[0], grp["k_d"])


def test_tau_lim_envelope_contains_the_kernel_torque():
    """``tau_lim`` (the dashboard's Joint Torque Limit readout) is a TRUE and TIGHT bound on the kernel.

    Two claims, both against the production objects — the real ``MotorCoupledPDHand`` and its real
    ``tau_lim`` — rather than a paraphrase of the formula:

    * **true**: for random (q, q̇) and targets large enough to saturate every motor, the torque the kernel
      actually writes stays inside the reported [lo, hi] of every joint;
    * **tight**: the bound is ATTAINED — the sign-matched corner of the motor box maps exactly onto it, so
      the readout is the reachable extreme and not a loose over-estimate.

    Both matter for the dashboard: a bound that the kernel could exceed would read as "saturated" too early,
    and a loose one would hide real headroom."""
    try:
        import torch
        import warp as wp
    except ModuleNotFoundError:
        print("SKIP test_tau_lim_envelope (warp/torch not installed)")
        return
    got = _index_r_inputs()
    if got is None:
        print("SKIP test_tau_lim_envelope (no finger.json)")
        return
    poly = got[0]
    from sim.metalab.actuation.motor_coupling import MotorCoupledPDHand, load_hand_group
    wp.init()

    n = 32
    lo, hi = np.asarray(poly["q_fit_range_rad"]["lo"]), np.asarray(poly["q_fit_range_rad"]["hi"])
    rng = np.random.default_rng(4)
    q = rng.uniform(lo, hi, size=(n, 3))
    qd = rng.uniform(-4.0, 4.0, size=(n, 3))            # spans the torque-speed envelope both ways
    q_tgt = np.where(rng.random((n, 3)) < 0.5, lo, hi)   # extreme targets → every motor saturates

    jq = torch.tensor(q.reshape(-1), dtype=torch.float32)
    jqd = torch.tensor(qd.reshape(-1), dtype=torch.float32)
    jtgt = torch.tensor(q_tgt.reshape(-1), dtype=torch.float32)
    jf = torch.zeros(3 * n, dtype=torch.float32)
    idx = torch.arange(3 * n, dtype=torch.int32).reshape(n, 3)
    grp = load_hand_group("index_r", _JOINTS, model_file=_FINGER_JSON, params_file=_PARAMS_JSON)
    pd = MotorCoupledPDHand([grp], idx, idx, num_envs=n, device="cpu")
    pd.launch(wp.from_torch(jq), wp.from_torch(jqd), wp.from_torch(jtgt), wp.from_torch(jf))
    wp.synchronize()
    tau_q = pd.tau_torch.cpu().numpy()                  # (n, 3) the FINAL joint torque, post-clamp

    j2m, _fit_lo, _fit_hi = _transmission(poly)   # test's own numpy G — independent of the readout's
    saturated = 0
    for e in range(n):
        # the readout evaluates G at the f32 values the kernel saw, so feed it exactly those
        g = pd.tau_lim(jq.numpy()[3 * e:3 * e + 3], jqd.numpy()[3 * e:3 * e + 3])[0]
        assert (tau_q[e] <= g["hi"] + 1e-5).all() and (tau_q[e] >= g["lo"] - 1e-5).all(), \
            f"env {e}: kernel torque {tau_q[e]} escaped the reported envelope {g['lo']}..{g['hi']}"
        if (np.abs(tau_q[e]) > 0.5 * np.maximum(g["hi"], -g["lo"])).any():
            saturated += 1
        # tightness: the corner of the motor box that maximises joint j must land ON the bound
        qe = np.clip(jq.numpy()[3 * e:3 * e + 3].astype(np.float64), grp["q_fit_lo"], grp["q_fit_hi"])
        G = _G_np(j2m, qe)
        phidot = G @ jqd.numpy()[3 * e:3 * e + 3].astype(np.float64)
        ts, w0 = grp["tau_stall"], grp["omega_noload"]
        m_hi = np.minimum(np.minimum(ts * (1.0 - phidot / w0), ts), grp["rated_pos"])
        m_lo = np.maximum(np.maximum(-ts * (1.0 + phidot / w0), -ts), -grp["rated_neg"])
        for j in range(3):
            up = np.where(G[:, j] > 0, m_hi, m_lo) @ G[:, j]
            dn = np.where(G[:, j] > 0, m_lo, m_hi) @ G[:, j]
            assert abs(up - g["hi"][j]) < 1e-9 and abs(dn - g["lo"][j]) < 1e-9, \
                f"env {e} joint {j}: bound {g['lo'][j]}..{g['hi'][j]} is not the reachable corner {dn}..{up}"
    assert saturated > n // 4, \
        f"only {saturated}/{n} envs pushed near the limit — the test is not exercising the clamp"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
