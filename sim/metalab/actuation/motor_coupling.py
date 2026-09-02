from __future__ import annotations

import numpy as np
import torch
import warp as wp

from .loaders import (  # noqa: F401
    MODELS_PER_GROUP_ARM,
    MODELS_PER_GROUP_HAND,
    _pack,
    equal_gain_warnings,
    jacobian,
    load_arm_group,
    load_hand_group,
    read_gains,
)

_F64 = wp.float64


@wp.func
def _ipow(base: wp.float64, e: wp.int32) -> wp.float64:
    r = _F64(1.0)
    for _k in range(e):
        r = r * base
    return r


@wp.func
def _poly_eval(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
               beg: wp.int32, end: wp.int32,
               x0: wp.float64, x1: wp.float64, x2: wp.float64) -> wp.float64:
    s = _F64(0.0)
    for t in range(beg, end):
        s = s + coeffs[t] * _ipow(x0, terms[t, 0]) * _ipow(x1, terms[t, 1]) * _ipow(x2, terms[t, 2])
    return s


@wp.func
def _poly_grad(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
               beg: wp.int32, end: wp.int32,
               x0: wp.float64, x1: wp.float64, x2: wp.float64) -> wp.vec3d:
    g0 = _F64(0.0)
    g1 = _F64(0.0)
    g2 = _F64(0.0)
    for t in range(beg, end):
        c = coeffs[t]
        e0 = terms[t, 0]
        e1 = terms[t, 1]
        e2 = terms[t, 2]
        p0 = _ipow(x0, e0)
        p1 = _ipow(x1, e1)
        p2 = _ipow(x2, e2)
        if e0 > 0:
            g0 = g0 + c * _F64(e0) * _ipow(x0, e0 - 1) * p1 * p2
        if e1 > 0:
            g1 = g1 + c * _F64(e1) * p0 * _ipow(x1, e1 - 1) * p2
        if e2 > 0:
            g2 = g2 + c * _F64(e2) * p0 * p1 * _ipow(x2, e2 - 1)
    return wp.vec3d(g0, g1, g2)


@wp.func
def _motor_pos_hand(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
               model_beg: wp.array(dtype=wp.int32), model_end: wp.array(dtype=wp.int32),
               gm: wp.int32, q0: wp.float64, q1: wp.float64, q2: wp.float64) -> wp.vec3d:
    z = _F64(0.0)
    m0 = _poly_eval(terms, coeffs, model_beg[gm + 0], model_end[gm + 0], q0, z, z)
    m1 = _poly_eval(terms, coeffs, model_beg[gm + 1], model_end[gm + 1], q0, q1, z)
    m2 = _poly_eval(terms, coeffs, model_beg[gm + 2], model_end[gm + 2], q0, q1, q2)
    return wp.vec3d(m0, m1, m2)


@wp.func
def _jac_j2m_hand(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
             model_beg: wp.array(dtype=wp.int32), model_end: wp.array(dtype=wp.int32),
             gm: wp.int32, q0: wp.float64, q1: wp.float64, q2: wp.float64) -> wp.mat33d:
    z = _F64(0.0)
    r0 = _poly_grad(terms, coeffs, model_beg[gm + 0], model_end[gm + 0], q0, z, z)
    r1 = _poly_grad(terms, coeffs, model_beg[gm + 1], model_end[gm + 1], q0, q1, z)
    r2 = _poly_grad(terms, coeffs, model_beg[gm + 2], model_end[gm + 2], q0, q1, q2)
    return wp.mat33d(r0[0], z, z,
                     r1[0], r1[1], z,
                     r2[0], r2[1], r2[2])


@wp.func
def _motor_pos_arm(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
                     model_beg: wp.array(dtype=wp.int32), model_end: wp.array(dtype=wp.int32),
                     gm: wp.int32, qr: wp.float64, qp: wp.float64) -> wp.vec2d:
    z = _F64(0.0)
    m0 = _poly_eval(terms, coeffs, model_beg[gm + 0], model_end[gm + 0], qr, qp, z)
    m1 = _poly_eval(terms, coeffs, model_beg[gm + 1], model_end[gm + 1], qr, qp, z)
    return wp.vec2d(m0, m1)


@wp.func
def _jac_j2m_arm(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
               model_beg: wp.array(dtype=wp.int32), model_end: wp.array(dtype=wp.int32),
               gm: wp.int32, qr: wp.float64, qp: wp.float64) -> wp.mat22d:
    z = _F64(0.0)
    r0 = _poly_grad(terms, coeffs, model_beg[gm + 0], model_end[gm + 0], qr, qp, z)
    r1 = _poly_grad(terms, coeffs, model_beg[gm + 1], model_end[gm + 1], qr, qp, z)
    return wp.mat22d(r0[0], r0[1],
                     r1[0], r1[1])


@wp.func
def _clamp_env(tau: wp.float64, phidot: wp.float64, tau_s: wp.float64, w0: wp.float64,
               rated_pos: wp.float64, rated_neg: wp.float64) -> wp.float64:
    hi = wp.min(wp.min(tau_s * (_F64(1.0) - phidot / w0), tau_s), rated_pos)
    lo = wp.max(wp.max(-tau_s * (_F64(1.0) + phidot / w0), -tau_s), -rated_neg)
    return wp.clamp(tau, lo, hi)


@wp.kernel
def coupled_pd_hand_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_target_q: wp.array(dtype=wp.float32),
    coord_idx: wp.array2d(dtype=wp.int32),
    dof_idx: wp.array2d(dtype=wp.int32),
    model_beg: wp.array(dtype=wp.int32),
    model_end: wp.array(dtype=wp.int32),
    terms: wp.array2d(dtype=wp.int32),
    coeffs: wp.array(dtype=wp.float64),
    k_phi: wp.array2d(dtype=wp.float64),
    k_d: wp.array2d(dtype=wp.float64),
    q_fit_lo: wp.array2d(dtype=wp.float64),
    q_fit_hi: wp.array2d(dtype=wp.float64),
    tau_stall: wp.array2d(dtype=wp.float64),
    omega_noload: wp.array2d(dtype=wp.float64),
    rated_pos: wp.array2d(dtype=wp.float64),
    rated_neg: wp.array2d(dtype=wp.float64),
    gc_fold: wp.array(dtype=wp.int32),
    qfrc_gravcomp: wp.array2d(dtype=wp.float32),
    gc_dof: wp.array(dtype=wp.int32),
    joint_f: wp.array(dtype=wp.float32),
    tau_out: wp.array2d(dtype=wp.float32),
    tau_pd_out: wp.array2d(dtype=wp.float32),
    tau_gc_out: wp.array2d(dtype=wp.float32),
):
    env, g = wp.tid()
    c0 = g * 3
    gm = g * MODELS_PER_GROUP_HAND

    qe0 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 0]]), q_fit_lo[g, 0], q_fit_hi[g, 0])
    qe1 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 1]]), q_fit_lo[g, 1], q_fit_hi[g, 1])
    qe2 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 2]]), q_fit_lo[g, 2], q_fit_hi[g, 2])
    qt0 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 0]]), q_fit_lo[g, 0], q_fit_hi[g, 0])
    qt1 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 1]]), q_fit_lo[g, 1], q_fit_hi[g, 1])
    qt2 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 2]]), q_fit_lo[g, 2], q_fit_hi[g, 2])
    qd = wp.vec3d(_F64(joint_qd[dof_idx[env, c0 + 0]]),
                  _F64(joint_qd[dof_idx[env, c0 + 1]]),
                  _F64(joint_qd[dof_idx[env, c0 + 2]]))

    m = _motor_pos_hand(terms, coeffs, model_beg, model_end, gm, qe0, qe1, qe2)
    m_tgt = _motor_pos_hand(terms, coeffs, model_beg, model_end, gm, qt0, qt1, qt2)
    G = _jac_j2m_hand(terms, coeffs, model_beg, model_end, gm, qe0, qe1, qe2)

    kphi = wp.vec3d(k_phi[g, 0], k_phi[g, 1], k_phi[g, 2])
    kd = wp.vec3d(k_d[g, 0], k_d[g, 1], k_d[g, 2])
    delta_phi = m_tgt - m
    phi_dot = G * qd
    tau_m = wp.cw_mul(kphi, delta_phi) - wp.cw_mul(kd, phi_dot)
    tau_pd_q = wp.transpose(G) * tau_m

    tau_g = wp.vec3d(_F64(0.0), _F64(0.0), _F64(0.0))
    if gc_fold[g] != 0:
        tau_g = wp.vec3d(_F64(qfrc_gravcomp[env, gc_dof[c0 + 0]]),
                         _F64(qfrc_gravcomp[env, gc_dof[c0 + 1]]),
                         _F64(qfrc_gravcomp[env, gc_dof[c0 + 2]]))
        tau_m = tau_m + wp.transpose(wp.inverse(G)) * tau_g

    tau_m = wp.vec3d(
        _clamp_env(tau_m[0], phi_dot[0], tau_stall[g, 0], omega_noload[g, 0], rated_pos[g, 0], rated_neg[g, 0]),
        _clamp_env(tau_m[1], phi_dot[1], tau_stall[g, 1], omega_noload[g, 1], rated_pos[g, 1], rated_neg[g, 1]),
        _clamp_env(tau_m[2], phi_dot[2], tau_stall[g, 2], omega_noload[g, 2], rated_pos[g, 2], rated_neg[g, 2]))
    tau_q = wp.transpose(G) * tau_m

    joint_f[dof_idx[env, c0 + 0]] = wp.float32(tau_q[0] - tau_g[0])
    joint_f[dof_idx[env, c0 + 1]] = wp.float32(tau_q[1] - tau_g[1])
    joint_f[dof_idx[env, c0 + 2]] = wp.float32(tau_q[2] - tau_g[2])
    tau_out[env, c0 + 0] = wp.float32(tau_q[0])
    tau_out[env, c0 + 1] = wp.float32(tau_q[1])
    tau_out[env, c0 + 2] = wp.float32(tau_q[2])
    tau_pd_out[env, c0 + 0] = wp.float32(tau_pd_q[0])
    tau_pd_out[env, c0 + 1] = wp.float32(tau_pd_q[1])
    tau_pd_out[env, c0 + 2] = wp.float32(tau_pd_q[2])
    tau_gc_out[env, c0 + 0] = wp.float32(tau_g[0])
    tau_gc_out[env, c0 + 1] = wp.float32(tau_g[1])
    tau_gc_out[env, c0 + 2] = wp.float32(tau_g[2])


@wp.kernel
def coupled_pd_arm_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_target_q: wp.array(dtype=wp.float32),
    coord_idx: wp.array2d(dtype=wp.int32),
    dof_idx: wp.array2d(dtype=wp.int32),
    model_beg: wp.array(dtype=wp.int32),
    model_end: wp.array(dtype=wp.int32),
    terms: wp.array2d(dtype=wp.int32),
    coeffs: wp.array(dtype=wp.float64),
    k_phi: wp.array2d(dtype=wp.float64),
    k_d: wp.array2d(dtype=wp.float64),
    q_fit_lo: wp.array2d(dtype=wp.float64),
    q_fit_hi: wp.array2d(dtype=wp.float64),
    tau_stall: wp.array2d(dtype=wp.float64),
    omega_noload: wp.array2d(dtype=wp.float64),
    rated_pos: wp.array2d(dtype=wp.float64),
    rated_neg: wp.array2d(dtype=wp.float64),
    gc_fold: wp.array(dtype=wp.int32),
    qfrc_gravcomp: wp.array2d(dtype=wp.float32),
    gc_dof: wp.array(dtype=wp.int32),
    joint_f: wp.array(dtype=wp.float32),
    tau_out: wp.array2d(dtype=wp.float32),
    tau_pd_out: wp.array2d(dtype=wp.float32),
    tau_gc_out: wp.array2d(dtype=wp.float32),
):
    env, g = wp.tid()
    c0 = g * 2
    gm = g * MODELS_PER_GROUP_ARM

    qe0 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 0]]), q_fit_lo[g, 0], q_fit_hi[g, 0])
    qe1 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 1]]), q_fit_lo[g, 1], q_fit_hi[g, 1])
    qt0 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 0]]), q_fit_lo[g, 0], q_fit_hi[g, 0])
    qt1 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 1]]), q_fit_lo[g, 1], q_fit_hi[g, 1])
    qd = wp.vec2d(_F64(joint_qd[dof_idx[env, c0 + 0]]), _F64(joint_qd[dof_idx[env, c0 + 1]]))

    m = _motor_pos_arm(terms, coeffs, model_beg, model_end, gm, qe0, qe1)
    m_tgt = _motor_pos_arm(terms, coeffs, model_beg, model_end, gm, qt0, qt1)
    G = _jac_j2m_arm(terms, coeffs, model_beg, model_end, gm, qe0, qe1)

    kphi = wp.vec2d(k_phi[g, 0], k_phi[g, 1])
    kd = wp.vec2d(k_d[g, 0], k_d[g, 1])
    delta_phi = m_tgt - m
    phi_dot = G * qd
    tau_m = wp.cw_mul(kphi, delta_phi) - wp.cw_mul(kd, phi_dot)
    tau_pd_q = wp.transpose(G) * tau_m

    tau_g = wp.vec2d(_F64(0.0), _F64(0.0))
    if gc_fold[g] != 0:
        tau_g = wp.vec2d(_F64(qfrc_gravcomp[env, gc_dof[c0 + 0]]),
                         _F64(qfrc_gravcomp[env, gc_dof[c0 + 1]]))
        tau_m = tau_m + wp.transpose(wp.inverse(G)) * tau_g

    tau_m = wp.vec2d(
        _clamp_env(tau_m[0], phi_dot[0], tau_stall[g, 0], omega_noload[g, 0], rated_pos[g, 0], rated_neg[g, 0]),
        _clamp_env(tau_m[1], phi_dot[1], tau_stall[g, 1], omega_noload[g, 1], rated_pos[g, 1], rated_neg[g, 1]))
    tau_q = wp.transpose(G) * tau_m

    joint_f[dof_idx[env, c0 + 0]] = wp.float32(tau_q[0] - tau_g[0])
    joint_f[dof_idx[env, c0 + 1]] = wp.float32(tau_q[1] - tau_g[1])
    tau_out[env, c0 + 0] = wp.float32(tau_q[0])
    tau_out[env, c0 + 1] = wp.float32(tau_q[1])
    tau_pd_out[env, c0 + 0] = wp.float32(tau_pd_q[0])
    tau_pd_out[env, c0 + 1] = wp.float32(tau_pd_q[1])
    tau_gc_out[env, c0 + 0] = wp.float32(tau_g[0])
    tau_gc_out[env, c0 + 1] = wp.float32(tau_g[1])


class _KernelBuffers:
    def __init__(self, groups: list[dict], device, models_per_group: int = MODELS_PER_GROUP_HAND):
        self.n_groups = len(groups)
        self.joints = [j for g in groups for j in g["joints"]]
        self.names = [g["name"] for g in groups]
        self.device = str(device)
        packed = _pack(groups, models_per_group)
        self.terms = wp.array(packed["terms"], dtype=wp.int32, device=self.device)
        self.coeffs = wp.array(packed["coeffs"], dtype=wp.float64, device=self.device)
        self.model_beg = wp.array(packed["model_beg"], dtype=wp.int32, device=self.device)
        self.model_end = wp.array(packed["model_end"], dtype=wp.int32, device=self.device)
        self.k_phi = wp.array(packed["k_phi"], dtype=wp.float64, device=self.device)
        self.k_d = wp.array(packed["k_d"], dtype=wp.float64, device=self.device)
        self.q_fit_lo = wp.array(packed["q_fit_lo"], dtype=wp.float64, device=self.device)
        self.q_fit_hi = wp.array(packed["q_fit_hi"], dtype=wp.float64, device=self.device)
        self.tau_stall = wp.array(packed["tau_stall"], dtype=wp.float64, device=self.device)
        self.omega_noload = wp.array(packed["omega_noload"], dtype=wp.float64, device=self.device)
        self.rated_pos = wp.array(packed["rated_pos"], dtype=wp.float64, device=self.device)
        self.rated_neg = wp.array(packed["rated_neg"], dtype=wp.float64, device=self.device)


def _gc_buffers(groups: list[dict], gravcomp, gc_dof, n_joints: int, device):
    if gravcomp is None:
        return (wp.zeros(len(groups), dtype=wp.int32, device=device),
                wp.zeros((1, 1), dtype=wp.float32, device=device),
                wp.zeros(n_joints, dtype=wp.int32, device=device), None)
    dof_t = gc_dof.to(torch.int32).contiguous()
    assert dof_t.shape == (n_joints,), f"gc_dof shape {tuple(dof_t.shape)} != ({n_joints},)"
    fold = np.asarray([1 if g["gravcomp_fold"] else 0 for g in groups], dtype=np.int32)
    return (wp.array(fold, dtype=wp.int32, device=device), gravcomp, wp.from_torch(dof_t), dof_t)


def _reload_gains(owner) -> list[str]:
    kphi = np.stack([read_gains(s)[0] for s in owner._gain_src]).astype(np.float64)
    kd = np.stack([read_gains(s)[1] for s in owner._gain_src]).astype(np.float64)
    changed = [owner.buf.names[i] for i in range(len(owner._gain_src))
               if not (np.array_equal(kphi[i], owner._kphi0[i]) and np.array_equal(kd[i], owner._kd0[i]))]
    owner._kphi0, owner._kd0 = kphi, kd
    owner.buf.k_phi.assign(kphi)
    owner.set_fold_kd_zero(owner._kd_zeroed)
    for w in owner.gain_warnings():
        print(f"[MotorCoupling] WARNING {w}", flush=True)
    return changed


def _kq(owner, q: np.ndarray) -> list[dict]:
    out, d = [], owner._dim
    for i, meta in enumerate(owner._gain_meta):
        qg = np.clip(np.asarray(q[i * d:(i + 1) * d], dtype=np.float64), meta["lo"], meta["hi"])
        G = jacobian(meta["models"], d, qg)
        out.append({"name": meta["name"], "joints": meta["joints"], "part": meta["part"],
                    "K": G.T @ np.diag(owner._kphi0[i]) @ G})
    return out


def _tau_lim(owner, q: np.ndarray, qd: np.ndarray) -> list[dict]:
    out, d = [], owner._dim
    for i, meta in enumerate(owner._gain_meta):
        sl = slice(i * d, (i + 1) * d)
        qg = np.clip(np.asarray(q[sl], dtype=np.float64), meta["lo"], meta["hi"])
        G = jacobian(meta["models"], d, qg)
        phidot = G @ np.asarray(qd[sl], dtype=np.float64)
        ts, w0 = meta["tau_stall"], meta["omega_noload"]
        m_hi = np.minimum(np.minimum(ts * (1.0 - phidot / w0), ts), meta["rated_pos"])
        m_lo = np.maximum(np.maximum(-ts * (1.0 + phidot / w0), -ts), -meta["rated_neg"])
        gp, gn = np.maximum(G, 0.0), np.minimum(G, 0.0)
        out.append({"name": meta["name"], "joints": meta["joints"], "part": meta["part"],
                    "hi": gp.T @ m_hi + gn.T @ m_lo, "lo": gp.T @ m_lo + gn.T @ m_hi})
    return out


def _gain_warnings(owner) -> list[str]:
    return [w for i, meta in enumerate(owner._gain_meta)
            for w in equal_gain_warnings(meta["name"], meta["models"], meta["dim"],
                                         owner._kphi0[i], owner._kd0[i])]


class MotorCoupledPDHand:
    def __init__(self, groups: list[dict], coord_idx: torch.Tensor, dof_idx: torch.Tensor,
                 num_envs: int, device, gravcomp=None, gc_dof: torch.Tensor | None = None):
        assert coord_idx.shape == dof_idx.shape == (num_envs, 3 * len(groups)), (
            f"index shape mismatch: {coord_idx.shape}/{dof_idx.shape} vs (num_envs={num_envs}, 3×{len(groups)})")
        self.num_envs = int(num_envs)
        self.buf = _KernelBuffers(groups, device)
        self._dim = 3
        self.n_groups = self.buf.n_groups
        self.joints = self.buf.joints
        self.device = self.buf.device
        self._coord_idx_t = coord_idx.to(torch.int32).contiguous()
        self._dof_idx_t = dof_idx.to(torch.int32).contiguous()
        self._coord_idx = wp.from_torch(self._coord_idx_t)
        self._dof_idx = wp.from_torch(self._dof_idx_t)
        self._gc_fold, self._qfrc_g, self._gc_dof, self._gc_dof_t = _gc_buffers(
            groups, gravcomp, gc_dof, 3 * self.n_groups, self.device)
        self._kphi0 = np.stack([g["k_phi"] for g in groups]).astype(np.float64)
        self._kd0 = np.stack([g["k_d"] for g in groups]).astype(np.float64)
        self._gain_src = [g["gain_src"] for g in groups]
        self._gain_meta = [{"name": g["name"], "models": g["models"], "dim": g["dim"],
                            "joints": list(g["joints"]), "lo": g["q_fit_lo"], "hi": g["q_fit_hi"],
                            "part": g["part"],
                            "tau_stall": g["tau_stall"], "omega_noload": g["omega_noload"],
                            "rated_pos": g["rated_pos"], "rated_neg": g["rated_neg"]}
                           for g in groups]
        self._kd_zeroed = False
        self._fold_np = np.asarray([bool(g["gravcomp_fold"]) for g in groups])
        self.tau_torch = torch.zeros(self.num_envs, 3 * self.n_groups,
                                     dtype=torch.float32, device=coord_idx.device)
        self._tau = wp.from_torch(self.tau_torch)
        self.tau_pd_torch = torch.zeros_like(self.tau_torch)
        self.tau_gc_torch = torch.zeros_like(self.tau_torch)
        self._tau_pd = wp.from_torch(self.tau_pd_torch)
        self._tau_gc = wp.from_torch(self.tau_gc_torch)

    def set_fold_kd_zero(self, zero: bool) -> None:
        self._kd_zeroed = bool(zero)
        kd = self._kd0.copy()
        if zero:
            kd[self._fold_np] = 0.0
        self.buf.k_d.assign(kd)

    def reload_gains(self) -> list[str]:
        return _reload_gains(self)

    def gain_warnings(self) -> list[str]:
        return _gain_warnings(self)

    def kq(self, q):
        return _kq(self, q)

    def tau_lim(self, q, qd):
        return _tau_lim(self, q, qd)

    def launch(self, joint_q: wp.array, joint_qd: wp.array,
               joint_target_q: wp.array, joint_f: wp.array) -> None:
        b = self.buf
        wp.launch(coupled_pd_hand_kernel, dim=(self.num_envs, self.n_groups),
                  inputs=[joint_q, joint_qd, joint_target_q, self._coord_idx, self._dof_idx,
                          b.model_beg, b.model_end, b.terms, b.coeffs, b.k_phi, b.k_d,
                          b.q_fit_lo, b.q_fit_hi,
                          b.tau_stall, b.omega_noload, b.rated_pos, b.rated_neg,
                          self._gc_fold, self._qfrc_g, self._gc_dof],
                  outputs=[joint_f, self._tau, self._tau_pd, self._tau_gc], device=self.device)


class MotorCoupledPDArm:
    def __init__(self, groups: list[dict], coord_idx: torch.Tensor, dof_idx: torch.Tensor,
                 num_envs: int, device, gravcomp=None, gc_dof: torch.Tensor | None = None):
        assert coord_idx.shape == dof_idx.shape == (num_envs, 2 * len(groups)), (
            f"index shape mismatch: {coord_idx.shape}/{dof_idx.shape} vs (num_envs={num_envs}, 2×{len(groups)})")
        self.num_envs = int(num_envs)
        self.buf = _KernelBuffers(groups, device, models_per_group=MODELS_PER_GROUP_ARM)
        self._dim = 2
        self.n_groups = self.buf.n_groups
        self.joints = self.buf.joints
        self.device = self.buf.device
        self._coord_idx_t = coord_idx.to(torch.int32).contiguous()
        self._dof_idx_t = dof_idx.to(torch.int32).contiguous()
        self._coord_idx = wp.from_torch(self._coord_idx_t)
        self._dof_idx = wp.from_torch(self._dof_idx_t)
        self._gc_fold, self._qfrc_g, self._gc_dof, self._gc_dof_t = _gc_buffers(
            groups, gravcomp, gc_dof, 2 * self.n_groups, self.device)
        self._kphi0 = np.stack([g["k_phi"] for g in groups]).astype(np.float64)
        self._kd0 = np.stack([g["k_d"] for g in groups]).astype(np.float64)
        self._gain_src = [g["gain_src"] for g in groups]
        self._gain_meta = [{"name": g["name"], "models": g["models"], "dim": g["dim"],
                            "joints": list(g["joints"]), "lo": g["q_fit_lo"], "hi": g["q_fit_hi"],
                            "part": g["part"],
                            "tau_stall": g["tau_stall"], "omega_noload": g["omega_noload"],
                            "rated_pos": g["rated_pos"], "rated_neg": g["rated_neg"]}
                           for g in groups]
        self._kd_zeroed = False
        self._fold_np = np.asarray([bool(g["gravcomp_fold"]) for g in groups])
        self.tau_torch = torch.zeros(self.num_envs, 2 * self.n_groups,
                                     dtype=torch.float32, device=coord_idx.device)
        self._tau = wp.from_torch(self.tau_torch)
        self.tau_pd_torch = torch.zeros_like(self.tau_torch)
        self.tau_gc_torch = torch.zeros_like(self.tau_torch)
        self._tau_pd = wp.from_torch(self.tau_pd_torch)
        self._tau_gc = wp.from_torch(self.tau_gc_torch)

    def set_fold_kd_zero(self, zero: bool) -> None:
        self._kd_zeroed = bool(zero)
        kd = self._kd0.copy()
        if zero:
            kd[self._fold_np] = 0.0
        self.buf.k_d.assign(kd)

    def reload_gains(self) -> list[str]:
        return _reload_gains(self)

    def gain_warnings(self) -> list[str]:
        return _gain_warnings(self)

    def kq(self, q):
        return _kq(self, q)

    def tau_lim(self, q, qd):
        return _tau_lim(self, q, qd)

    def launch(self, joint_q: wp.array, joint_qd: wp.array,
               joint_target_q: wp.array, joint_f: wp.array) -> None:
        b = self.buf
        wp.launch(coupled_pd_arm_kernel, dim=(self.num_envs, self.n_groups),
                  inputs=[joint_q, joint_qd, joint_target_q, self._coord_idx, self._dof_idx,
                          b.model_beg, b.model_end, b.terms, b.coeffs, b.k_phi, b.k_d,
                          b.q_fit_lo, b.q_fit_hi,
                          b.tau_stall, b.omega_noload, b.rated_pos, b.rated_neg,
                          self._gc_fold, self._qfrc_g, self._gc_dof],
                  outputs=[joint_f, self._tau, self._tau_pd, self._tau_gc], device=self.device)
