"""Motor-to-joint coupling — warp kernels (off-diagonal joint coupling from a motor-space PD), runtime.

The real ALLEX hand + wrist drive several joints through a nonlinear transmission with PD closed in
MOTOR space, so joint-space stiffness gains off-diagonal terms. The deployed firmware is the source of
truth: it maps joint→motor with m = J2M(q) and uses its analytic Jacobian G = ∂m/∂q. We mirror exactly
that map (extracted to ``mj_mapping/*.json`` — see ``README.md`` for provenance). The position/velocity
path uses **only J2M and its gradient G**; the (optional) gravity-feedforward torque path uses G⁻ᵀ —
both exactly as the firmware does (``cal_motorAngles``/``cal_dmdq`` and ``jointTorqueToMotorTorque``).

Three transmission families — same control law below, differing only in the J2M / G shape:

- **finger** (3-DOF: ABAD/MCP/PIP) — lower-triangular ``m_i = f(q0..qi)``, cubic. One map
  (``finger.json``) backs all 8 finger groups. Loader :func:`load_hand_group`, kernel :func:`coupled_pd_hand_kernel`.
- **thumb** (3-DOF: Yaw/CMC/MCP) — also lower-triangular, but Yaw is decoupled (constant ratio) and
  CMC/MCP are high-order. One map (``thumb.json``) backs both thumbs. Shares the finger loader/kernel.
- **wrist** (2-DOF: Roll/Pitch) — FULL 2×2 ballscrew: both motors depend on both joints (NOT
  triangular). One map (``wrist.json``) backs both wrists. Loader :func:`load_arm_group`, kernel
  :func:`coupled_pd_arm_kernel`; primitives ``_motor_pos_arm`` / ``_jac_j2m_arm``.

The coupled PD control law (:func:`coupled_pd_hand_kernel` for finger/thumb, :func:`coupled_pd_arm_kernel`
for the wrist), evaluated at the current config:

      Δφ = J2M(q*) − J2M(q)                 # exact motor error (firmware: cal_motorAngles)
      φ̇  = G · q̇                            # motor velocity (firmware: dmdq · q̇)
      τ_m = k_phi·Δφ − k_d·φ̇ + G⁻ᵀ·τ_g       # motor PD + joint gravcomp folded to motor space
      τ_m = clamp(τ_m, envelope(φ̇) ∩ ±rated) # ONE clamp on the sum (see _clamp_env)
      τ_q = Gᵀ · τ_m                         # motor → joint torque (firmware: dmdqᵀ · τ_m) — FINAL

τ_g = the group's MuJoCo ``qfrc_gravcomp`` share (zero for hand groups — hands run position-only —
and whenever gravcomp is off). MuJoCo still applies +τ_g through its own passive channel, so the
kernel writes ``joint_f = τ_q − τ_g``: the net torque the joint sees is exactly τ_q, with gravity
feedforward consuming motor-torque budget inside the clamp (like the real motor). ``tau_torch``
readout = τ_q (the motor-delivered torque, gravity included — what the real robot measures).

The same launch also mirrors τ_q's two components, both **PRE-clamp**, for the optional
``joint_torque_pd`` / ``joint_torque_gravcomp`` reads (``tau_pd_torch`` / ``tau_gc_torch``):

      τ_q^PD   = Gᵀ·(k_phi·Δφ − k_d·φ̇)        # the PD term alone, mapped back to joint space
      τ_q^grav = Gᵀ·(G⁻ᵀ·τ_g) = τ_g            # the folded gravity share reads back AS τ_g

Because the clamp acts on the SUM in motor space, ``τ_q^PD + τ_q^grav == τ_q`` exactly while no motor
saturates, and the residual once one does is precisely the torque the envelope removed.

Pure warp over fixed buffers → CUDA-graph capturable (sits in ``NewtonBackend._run_substeps`` with zero
per-substep host work). Internals run in float64 (motor angles reach O(100 rad)); outputs float32.

k_phi = motor P gain (``motor_control_param.k_pos``), k_d = motor D gain (``k_vel``); both from
``robot_model.json`` (the wrist slices indices 5,6 of the 7-DOF arm). The firmware's motor→joint
readback (``cal_jointAngles``/``cal_dqdm``) is telemetry-only and unused here (the sim already knows q).
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

# Group loaders + buffer packing are numpy-only (engine-agnostic; both engines load groups the same
# way) — re-exported here so kernel consumers keep one entry point.
from .loaders import (  # noqa: F401  (re-export)
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


# ---------------------------------------------------------------------------
# warp primitives — polynomial eval/grad + the transmission (J2M and G = ∂m/∂q)
# ---------------------------------------------------------------------------

@wp.func
def _ipow(base: wp.float64, e: wp.int32) -> wp.float64:
    """base^e for small non-negative integer exponents (exact, no transcendental pow)."""
    r = _F64(1.0)
    for _k in range(e):
        r = r * base
    return r


@wp.func
def _poly_eval(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
               beg: wp.int32, end: wp.int32,
               x0: wp.float64, x1: wp.float64, x2: wp.float64) -> wp.float64:
    """polyvaln: Σ_t c_t · x0^e0 · x1^e1 · x2^e2 over term rows [beg, end)."""
    s = _F64(0.0)
    for t in range(beg, end):
        s = s + coeffs[t] * _ipow(x0, terms[t, 0]) * _ipow(x1, terms[t, 1]) * _ipow(x2, terms[t, 2])
    return s


@wp.func
def _poly_grad(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
               beg: wp.int32, end: wp.int32,
               x0: wp.float64, x1: wp.float64, x2: wp.float64) -> wp.vec3d:
    """polyfitn_grad: (∂/∂x0, ∂/∂x1, ∂/∂x2) over term rows [beg, end)."""
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
    """m = J2M(q): the 3 joint→motor fits (m1(q1), m2(q1,q2), m3(q1,q2,q3)). Motor angle [rad]."""
    z = _F64(0.0)
    m0 = _poly_eval(terms, coeffs, model_beg[gm + 0], model_end[gm + 0], q0, z, z)
    m1 = _poly_eval(terms, coeffs, model_beg[gm + 1], model_end[gm + 1], q0, q1, z)
    m2 = _poly_eval(terms, coeffs, model_beg[gm + 2], model_end[gm + 2], q0, q1, q2)
    return wp.vec3d(m0, m1, m2)


@wp.func
def _jac_j2m_hand(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
             model_beg: wp.array(dtype=wp.int32), model_end: wp.array(dtype=wp.int32),
             gm: wp.int32, q0: wp.float64, q1: wp.float64, q2: wp.float64) -> wp.mat33d:
    """G = ∂m/∂q at joint config q — the J2M gradient (firmware ``cal_dmdq``). Lower-triangular
    since m1=f(q1), m2=f(q1,q2), m3=f(q1,q2,q3)."""
    z = _F64(0.0)
    r0 = _poly_grad(terms, coeffs, model_beg[gm + 0], model_end[gm + 0], q0, z, z)   # ∂m1/∂q
    r1 = _poly_grad(terms, coeffs, model_beg[gm + 1], model_end[gm + 1], q0, q1, z)  # ∂m2/∂q
    r2 = _poly_grad(terms, coeffs, model_beg[gm + 2], model_end[gm + 2], q0, q1, q2)  # ∂m3/∂q
    return wp.mat33d(r0[0], z, z,
                     r1[0], r1[1], z,
                     r2[0], r2[1], r2[2])


@wp.func
def _motor_pos_arm(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
                     model_beg: wp.array(dtype=wp.int32), model_end: wp.array(dtype=wp.int32),
                     gm: wp.int32, qr: wp.float64, qp: wp.float64) -> wp.vec2d:
    """m = J2M(q): the 2 wrist ballscrew fits, each FULL f(Roll, Pitch) (both motors ← both joints)."""
    z = _F64(0.0)
    m0 = _poly_eval(terms, coeffs, model_beg[gm + 0], model_end[gm + 0], qr, qp, z)
    m1 = _poly_eval(terms, coeffs, model_beg[gm + 1], model_end[gm + 1], qr, qp, z)
    return wp.vec2d(m0, m1)


@wp.func
def _jac_j2m_arm(terms: wp.array2d(dtype=wp.int32), coeffs: wp.array(dtype=wp.float64),
               model_beg: wp.array(dtype=wp.int32), model_end: wp.array(dtype=wp.int32),
               gm: wp.int32, qr: wp.float64, qp: wp.float64) -> wp.mat22d:
    """G = ∂m/∂q — FULL 2×2 (ballscrew: both motors depend on both joints; not lower-triangular)."""
    z = _F64(0.0)
    r0 = _poly_grad(terms, coeffs, model_beg[gm + 0], model_end[gm + 0], qr, qp, z)   # ∂m_R/∂q
    r1 = _poly_grad(terms, coeffs, model_beg[gm + 1], model_end[gm + 1], qr, qp, z)   # ∂m_L/∂q
    return wp.mat22d(r0[0], r0[1],
                     r1[0], r1[1])


@wp.func
def _clamp_env(tau: wp.float64, phidot: wp.float64, tau_s: wp.float64, w0: wp.float64,
               rated_pos: wp.float64, rated_neg: wp.float64) -> wp.float64:
    """Clip one motor torque to its torque-speed envelope ∩ rated box (motor space).

    Envelope = the DC-motor parallelogram: linear droop τ_s·(1 ∓ φ̇/ω₀) in Q1/Q3 (stall torque τ_s
    x-intercept, no-load speed ω₀ y-intercept), vertical ±τ_s in Q2/Q4; intersected with the rated
    (continuous) limit [−rated_neg, +rated_pos]. So instantaneous headroom is the physical stall
    envelope, additionally capped by the rated torque."""
    hi = wp.min(wp.min(tau_s * (_F64(1.0) - phidot / w0), tau_s), rated_pos)
    lo = wp.max(wp.max(-tau_s * (_F64(1.0) + phidot / w0), -tau_s), -rated_neg)
    return wp.clamp(tau, lo, hi)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

@wp.kernel
def coupled_pd_hand_kernel(
    joint_q: wp.array(dtype=wp.float32),        # current joint pos (coord layout)
    joint_qd: wp.array(dtype=wp.float32),       # current joint vel (dof layout)
    joint_target_q: wp.array(dtype=wp.float32),  # PD target joint pos (coord layout)
    coord_idx: wp.array2d(dtype=wp.int32),      # (num_envs, 3G) joint_q/joint_target_q index
    dof_idx: wp.array2d(dtype=wp.int32),        # (num_envs, 3G) joint_qd/joint_f index
    model_beg: wp.array(dtype=wp.int32),        # (3G,) term-row ranges: [m1,m2,m3]×G
    model_end: wp.array(dtype=wp.int32),
    terms: wp.array2d(dtype=wp.int32),          # (T, 3) exponents (zero-padded columns)
    coeffs: wp.array(dtype=wp.float64),         # (T,)
    k_phi: wp.array2d(dtype=wp.float64),        # (G, 3) motor P gain [Nm/rad]
    k_d: wp.array2d(dtype=wp.float64),          # (G, 3) motor D gain [Nm·s/rad]
    q_fit_lo: wp.array2d(dtype=wp.float64),     # (G, 3) poly-eval clamp envelope
    q_fit_hi: wp.array2d(dtype=wp.float64),
    tau_stall: wp.array2d(dtype=wp.float64),    # (G, 3) stall torque τ_s [Nm] (envelope x-intercept)
    omega_noload: wp.array2d(dtype=wp.float64),  # (G, 3) no-load speed ω₀ [rad/s] (envelope y-intercept)
    rated_pos: wp.array2d(dtype=wp.float64),    # (G, 3) rated torque + [Nm] (torque_limit_pos_abs)
    rated_neg: wp.array2d(dtype=wp.float64),    # (G, 3) rated torque − [Nm] (torque_limit_neg_abs)
    gc_fold: wp.array(dtype=wp.int32),          # (G,) 1 = fold this group's gravcomp into τ_m
    qfrc_gravcomp: wp.array2d(dtype=wp.float32),  # (num_envs, nv) joint gravcomp τ_g (MuJoCo layout)
    gc_dof: wp.array(dtype=wp.int32),           # (3G,) MuJoCo dof index per coupled joint
    joint_f: wp.array(dtype=wp.float32),        # OUT: joint generalized force (dof layout)
    tau_out: wp.array2d(dtype=wp.float32),      # OUT: (num_envs, 3G) joint-torque readout
    tau_pd_out: wp.array2d(dtype=wp.float32),   # OUT: (num_envs, 3G) PD component, PRE-clamp
    tau_gc_out: wp.array2d(dtype=wp.float32),   # OUT: (num_envs, 3G) gravity component, PRE-clamp
):
    """Coupled PD in motor space (method 2), with per-motor torque-speed envelope clamp. See module docstring."""
    env, g = wp.tid()
    c0 = g * 3
    gm = g * MODELS_PER_GROUP_HAND

    # --- read state, clamp eval points to the trusted fit envelope (= firmware joint clip) ---
    qe0 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 0]]), q_fit_lo[g, 0], q_fit_hi[g, 0])
    qe1 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 1]]), q_fit_lo[g, 1], q_fit_hi[g, 1])
    qe2 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 2]]), q_fit_lo[g, 2], q_fit_hi[g, 2])
    qt0 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 0]]), q_fit_lo[g, 0], q_fit_hi[g, 0])
    qt1 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 1]]), q_fit_lo[g, 1], q_fit_hi[g, 1])
    qt2 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 2]]), q_fit_lo[g, 2], q_fit_hi[g, 2])
    qd = wp.vec3d(_F64(joint_qd[dof_idx[env, c0 + 0]]),
                  _F64(joint_qd[dof_idx[env, c0 + 1]]),
                  _F64(joint_qd[dof_idx[env, c0 + 2]]))

    # --- transmission at the current config ---
    m = _motor_pos_hand(terms, coeffs, model_beg, model_end, gm, qe0, qe1, qe2)
    m_tgt = _motor_pos_hand(terms, coeffs, model_beg, model_end, gm, qt0, qt1, qt2)
    G = _jac_j2m_hand(terms, coeffs, model_beg, model_end, gm, qe0, qe1, qe2)   # ∂m/∂q

    # --- motor-space PD → joint torque (the control law) ---
    kphi = wp.vec3d(k_phi[g, 0], k_phi[g, 1], k_phi[g, 2])
    kd = wp.vec3d(k_d[g, 0], k_d[g, 1], k_d[g, 2])
    delta_phi = m_tgt - m                            # motor position error  Δφ = J2M(q*) − J2M(q)
    phi_dot = G * qd                             # motor velocity        φ̇ = G · q̇
    tau_m = wp.cw_mul(kphi, delta_phi) - wp.cw_mul(kd, phi_dot)   # τ_m = k_phi·Δφ − k_d·φ̇
    tau_pd_q = wp.transpose(G) * tau_m           # PD component in JOINT space, PRE-clamp (component read)

    # --- gravity feedforward: fold the joint gravcomp share into the SAME motor-torque budget ---
    tau_g = wp.vec3d(_F64(0.0), _F64(0.0), _F64(0.0))
    if gc_fold[g] != 0:
        tau_g = wp.vec3d(_F64(qfrc_gravcomp[env, gc_dof[c0 + 0]]),      # τ_g (joint space, MuJoCo)
                         _F64(qfrc_gravcomp[env, gc_dof[c0 + 1]]),
                         _F64(qfrc_gravcomp[env, gc_dof[c0 + 2]]))
        tau_m = tau_m + wp.transpose(wp.inverse(G)) * tau_g   # τ_m += G⁻ᵀ·τ_g (firmware jointTorqueToMotorTorque)

    # --- per-motor torque-speed envelope clamp on the SUM (PD + gravity), then map back ---
    tau_m = wp.vec3d(
        _clamp_env(tau_m[0], phi_dot[0], tau_stall[g, 0], omega_noload[g, 0], rated_pos[g, 0], rated_neg[g, 0]),
        _clamp_env(tau_m[1], phi_dot[1], tau_stall[g, 1], omega_noload[g, 1], rated_pos[g, 1], rated_neg[g, 1]),
        _clamp_env(tau_m[2], phi_dot[2], tau_stall[g, 2], omega_noload[g, 2], rated_pos[g, 2], rated_neg[g, 2]))
    tau_q = wp.transpose(G) * tau_m              # τ_q = Gᵀ · τ_m   (the FINAL joint torque)

    # MuJoCo applies +τ_g itself (passive channel); cancel it so the NET applied torque is exactly τ_q.
    joint_f[dof_idx[env, c0 + 0]] = wp.float32(tau_q[0] - tau_g[0])
    joint_f[dof_idx[env, c0 + 1]] = wp.float32(tau_q[1] - tau_g[1])
    joint_f[dof_idx[env, c0 + 2]] = wp.float32(tau_q[2] - tau_g[2])
    tau_out[env, c0 + 0] = wp.float32(tau_q[0])
    tau_out[env, c0 + 1] = wp.float32(tau_q[1])
    tau_out[env, c0 + 2] = wp.float32(tau_q[2])
    # Components of τ_q, both PRE-clamp: Gᵀ·τ_m^PD and Gᵀ·(G⁻ᵀ·τ_g) == τ_g (the fold is norm-preserving
    # in joint space, so the gravity share reads back as the very τ_g that went in). They sum to τ_q
    # exactly while the clamp is inactive; the gap when it bites is what the motor envelope removed.
    tau_pd_out[env, c0 + 0] = wp.float32(tau_pd_q[0])
    tau_pd_out[env, c0 + 1] = wp.float32(tau_pd_q[1])
    tau_pd_out[env, c0 + 2] = wp.float32(tau_pd_q[2])
    tau_gc_out[env, c0 + 0] = wp.float32(tau_g[0])
    tau_gc_out[env, c0 + 1] = wp.float32(tau_g[1])
    tau_gc_out[env, c0 + 2] = wp.float32(tau_g[2])


@wp.kernel
def coupled_pd_arm_kernel(
    joint_q: wp.array(dtype=wp.float32),        # current joint pos (coord layout)
    joint_qd: wp.array(dtype=wp.float32),       # current joint vel (dof layout)
    joint_target_q: wp.array(dtype=wp.float32),  # PD target joint pos (coord layout)
    coord_idx: wp.array2d(dtype=wp.int32),      # (num_envs, 2G) joint_q/joint_target_q index
    dof_idx: wp.array2d(dtype=wp.int32),        # (num_envs, 2G) joint_qd/joint_f index
    model_beg: wp.array(dtype=wp.int32),        # (2G,) term-row ranges: [m1,m2]×G
    model_end: wp.array(dtype=wp.int32),
    terms: wp.array2d(dtype=wp.int32),          # (T, 3) exponents (col2 zero for the 2-var wrist)
    coeffs: wp.array(dtype=wp.float64),         # (T,)
    k_phi: wp.array2d(dtype=wp.float64),        # (G, 2) motor P gain
    k_d: wp.array2d(dtype=wp.float64),          # (G, 2) motor D gain
    q_fit_lo: wp.array2d(dtype=wp.float64),     # (G, 2) poly-eval clamp envelope
    q_fit_hi: wp.array2d(dtype=wp.float64),
    tau_stall: wp.array2d(dtype=wp.float64),    # (G, 2) stall torque τ_s
    omega_noload: wp.array2d(dtype=wp.float64),  # (G, 2) no-load speed ω₀
    rated_pos: wp.array2d(dtype=wp.float64),    # (G, 2) rated torque +
    rated_neg: wp.array2d(dtype=wp.float64),    # (G, 2) rated torque −
    gc_fold: wp.array(dtype=wp.int32),          # (G,) 1 = fold this group's gravcomp into τ_m
    qfrc_gravcomp: wp.array2d(dtype=wp.float32),  # (num_envs, nv) joint gravcomp τ_g (MuJoCo layout)
    gc_dof: wp.array(dtype=wp.int32),           # (2G,) MuJoCo dof index per coupled joint
    joint_f: wp.array(dtype=wp.float32),        # OUT: joint generalized force (dof layout)
    tau_out: wp.array2d(dtype=wp.float32),      # OUT: (num_envs, 2G) joint-torque readout
    tau_pd_out: wp.array2d(dtype=wp.float32),   # OUT: (num_envs, 2G) PD component, PRE-clamp
    tau_gc_out: wp.array2d(dtype=wp.float32),   # OUT: (num_envs, 2G) gravity component, PRE-clamp
):
    """Coupled PD in motor space for the wrist (2-DOF, FULL 2×2 ballscrew), with per-motor
    torque-speed envelope clamp. Same law as coupled_pd_hand_kernel; G = ∂m/∂q is not triangular."""
    env, g = wp.tid()
    c0 = g * 2
    gm = g * MODELS_PER_GROUP_ARM

    qe0 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 0]]), q_fit_lo[g, 0], q_fit_hi[g, 0])      # Roll
    qe1 = wp.clamp(_F64(joint_q[coord_idx[env, c0 + 1]]), q_fit_lo[g, 1], q_fit_hi[g, 1])      # Pitch
    qt0 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 0]]), q_fit_lo[g, 0], q_fit_hi[g, 0])
    qt1 = wp.clamp(_F64(joint_target_q[coord_idx[env, c0 + 1]]), q_fit_lo[g, 1], q_fit_hi[g, 1])
    qd = wp.vec2d(_F64(joint_qd[dof_idx[env, c0 + 0]]), _F64(joint_qd[dof_idx[env, c0 + 1]]))

    m = _motor_pos_arm(terms, coeffs, model_beg, model_end, gm, qe0, qe1)
    m_tgt = _motor_pos_arm(terms, coeffs, model_beg, model_end, gm, qt0, qt1)
    G = _jac_j2m_arm(terms, coeffs, model_beg, model_end, gm, qe0, qe1)          # ∂m/∂q (full 2×2)

    kphi = wp.vec2d(k_phi[g, 0], k_phi[g, 1])
    kd = wp.vec2d(k_d[g, 0], k_d[g, 1])
    delta_phi = m_tgt - m
    phi_dot = G * qd
    tau_m = wp.cw_mul(kphi, delta_phi) - wp.cw_mul(kd, phi_dot)   # τ_m = k_phi·Δφ − k_d·φ̇
    tau_pd_q = wp.transpose(G) * tau_m           # PD component in JOINT space, PRE-clamp (component read)

    # --- gravity feedforward: fold the joint gravcomp share into the SAME motor-torque budget ---
    tau_g = wp.vec2d(_F64(0.0), _F64(0.0))
    if gc_fold[g] != 0:
        tau_g = wp.vec2d(_F64(qfrc_gravcomp[env, gc_dof[c0 + 0]]),      # τ_g (joint space, MuJoCo)
                         _F64(qfrc_gravcomp[env, gc_dof[c0 + 1]]))
        tau_m = tau_m + wp.transpose(wp.inverse(G)) * tau_g   # τ_m += G⁻ᵀ·τ_g (firmware jointTorqueToMotorTorque)

    # --- per-motor torque-speed envelope clamp on the SUM (PD + gravity), then map back ---
    tau_m = wp.vec2d(
        _clamp_env(tau_m[0], phi_dot[0], tau_stall[g, 0], omega_noload[g, 0], rated_pos[g, 0], rated_neg[g, 0]),
        _clamp_env(tau_m[1], phi_dot[1], tau_stall[g, 1], omega_noload[g, 1], rated_pos[g, 1], rated_neg[g, 1]))
    tau_q = wp.transpose(G) * tau_m              # τ_q = Gᵀ · τ_m   (the FINAL joint torque)

    # MuJoCo applies +τ_g itself (passive channel); cancel it so the NET applied torque is exactly τ_q.
    joint_f[dof_idx[env, c0 + 0]] = wp.float32(tau_q[0] - tau_g[0])
    joint_f[dof_idx[env, c0 + 1]] = wp.float32(tau_q[1] - tau_g[1])
    tau_out[env, c0 + 0] = wp.float32(tau_q[0])
    tau_out[env, c0 + 1] = wp.float32(tau_q[1])
    tau_pd_out[env, c0 + 0] = wp.float32(tau_pd_q[0])   # PRE-clamp components (see hand kernel)
    tau_pd_out[env, c0 + 1] = wp.float32(tau_pd_q[1])
    tau_gc_out[env, c0 + 0] = wp.float32(tau_g[0])
    tau_gc_out[env, c0 + 1] = wp.float32(tau_g[1])


class _KernelBuffers:
    """Shared fixed device buffers for the coupling kernels (built once, pre-graph-capture)."""

    def __init__(self, groups: list[dict], device, models_per_group: int = MODELS_PER_GROUP_HAND):
        self.n_groups = len(groups)
        self.joints = [j for g in groups for j in g["joints"]]
        self.names = [g["name"] for g in groups]     # per-group label (reload/log reporting)
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
    """Gravcomp-fold device buffers for an owner: (gc_fold (G,), qfrc_gravcomp, gc_dof (nJ,), keepalive).

    ``gravcomp`` = MuJoCo ``qfrc_gravcomp`` (num_envs, nv) wp array + ``gc_dof`` its per-joint dof
    indices. None (tests / gravcomp off) → every fold flag forced 0 and dummy buffers (never read,
    the kernel's fold branch is skipped)."""
    if gravcomp is None:
        return (wp.zeros(len(groups), dtype=wp.int32, device=device),
                wp.zeros((1, 1), dtype=wp.float32, device=device),
                wp.zeros(n_joints, dtype=wp.int32, device=device), None)
    dof_t = gc_dof.to(torch.int32).contiguous()
    assert dof_t.shape == (n_joints,), f"gc_dof shape {tuple(dof_t.shape)} != ({n_joints},)"
    fold = np.asarray([1 if g["gravcomp_fold"] else 0 for g in groups], dtype=np.int32)
    return (wp.array(fold, dtype=wp.int32, device=device), gravcomp, wp.from_torch(dof_t), dof_t)


def _reload_gains(owner) -> list[str]:
    """Re-read every group's motor gains from ``robot_model.json`` and push them into the LIVE kernel
    buffers — the mechanism behind the standalone runner's "gains apply on reset".

    Writes the fixed buffers' CONTENTS (``assign``), never a new allocation, so a captured CUDA graph keeps
    pointing at the right memory — the same property ``set_fold_kd_zero`` relies on. ``_kd0`` (the D-gain
    originals the Torque-mode toggle restores) is refreshed too, and the float state is re-applied so a
    reload during Torque mode does not silently re-enable damping. Returns the groups whose gains changed.
    """
    kphi = np.stack([read_gains(s)[0] for s in owner._gain_src]).astype(np.float64)
    kd = np.stack([read_gains(s)[1] for s in owner._gain_src]).astype(np.float64)
    changed = [owner.buf.names[i] for i in range(len(owner._gain_src))
               if not (np.array_equal(kphi[i], owner._kphi0[i]) and np.array_equal(kd[i], owner._kd0[i]))]
    owner._kphi0, owner._kd0 = kphi, kd
    owner.buf.k_phi.assign(kphi)
    owner.set_fold_kd_zero(owner._kd_zeroed)       # writes k_d (float state preserved)
    for w in owner.gain_warnings():                # the reloaded gains get the same check as a fresh load
        print(f"[MotorCoupling] WARNING {w}", flush=True)
    return changed


def _kq(owner, q: np.ndarray) -> list[dict]:
    """Joint-space stiffness per group at the CURRENT pose: ``K_q = Gᵀ·diag(k_phi)·G`` (N*m/rad).

    Host-side twin of what the kernel already computes for the control law — G is re-evaluated here rather
    than exported from the kernel so the readout costs the hot loop (and the captured CUDA graph) nothing;
    it runs once per dashboard snapshot on ONE env, not per substep per env. Uses ``_kphi0``, i.e. the gains
    the buffers hold right now, so a hot gain reload shows up immediately.

    ``q`` = this owner's joints in owner order [rad]. Returns one entry per group: name, joints, K (d x d).
    """
    out, d = [], owner._dim
    for i, meta in enumerate(owner._gain_meta):
        qg = np.clip(np.asarray(q[i * d:(i + 1) * d], dtype=np.float64), meta["lo"], meta["hi"])
        G = jacobian(meta["models"], d, qg)
        out.append({"name": meta["name"], "joints": meta["joints"], "part": meta["part"],
                    "K": G.T @ np.diag(owner._kphi0[i]) @ G})
    return out


def _tau_lim(owner, q: np.ndarray, qd: np.ndarray) -> list[dict]:
    """Joint-space TORQUE LIMIT per group at the current pose AND speed: what each joint could produce if its
    group's motors all pushed that way. Host-side twin of the kernel's clamp (:func:`_clamp_env`) + ``Gᵀ``.

    Two things make this live rather than a constant:

    * **pose** — the motor box maps to joint space through ``G = ∂φ/∂q``, so leverage changes with the pose
      (and a joint's limit borrows its group partners' motors: the fingers' MCP is driven by 3 motors);
    * **speed** — each motor's headroom is its torque-speed envelope ∩ rated box evaluated at φ̇ = G·q̇, so a
      joint moving fast toward its own direction of travel has strictly less torque available.

    Per joint j the bound is the support of the motor box mapped through Gᵀ (``τ_q = Gᵀ·τ_m``)::

        τ_j⁺ = Σ_k max(G[k,j]·lo_k, G[k,j]·hi_k)        τ_j⁻ = Σ_k min(G[k,j]·lo_k, G[k,j]·hi_k)

    HONESTY, same caveat as ``_kq``: this is a per-joint PROJECTION of a coupled polytope. Each joint's own
    number is exact — that torque IS reachable — but two joints of one group cannot generally reach their
    respective maxima at the same time, because the same motors serve both. Read it per joint, not as a
    simultaneously achievable vector. It is also the ENVELOPE, not the headroom: with gravcomp folded in, the
    feedforward is already spending part of this budget (which is why the dashboard overlays the live τ_q).

    ``q``/``qd`` = this owner's joints in owner order [rad, rad/s]. Returns per group: name, joints, part,
    ``hi``/``lo`` (d,) in N*m. Uses the gains/limits the buffers hold now, so a hot reload shows up at once.
    """
    out, d = [], owner._dim
    for i, meta in enumerate(owner._gain_meta):
        sl = slice(i * d, (i + 1) * d)
        qg = np.clip(np.asarray(q[sl], dtype=np.float64), meta["lo"], meta["hi"])
        G = jacobian(meta["models"], d, qg)                        # (3|2 motors, d joints) = ∂φ/∂q
        phidot = G @ np.asarray(qd[sl], dtype=np.float64)          # φ̇ = G·q̇ — same as the kernel
        ts, w0 = meta["tau_stall"], meta["omega_noload"]
        # per-motor envelope ∩ rated box — the exact expression _clamp_env applies, in numpy
        m_hi = np.minimum(np.minimum(ts * (1.0 - phidot / w0), ts), meta["rated_pos"])
        m_lo = np.maximum(np.maximum(-ts * (1.0 + phidot / w0), -ts), -meta["rated_neg"])
        gp, gn = np.maximum(G, 0.0), np.minimum(G, 0.0)            # split by sign: which motor bound helps
        out.append({"name": meta["name"], "joints": meta["joints"], "part": meta["part"],
                    "hi": gp.T @ m_hi + gn.T @ m_lo, "lo": gp.T @ m_lo + gn.T @ m_hi})
    return out


def _gain_warnings(owner) -> list[str]:
    """Per-group gain-consistency warnings for whatever gains the buffers hold RIGHT NOW (so a hot reload
    is re-checked, not just the values that were on disk at build). Currently: a differential
    (constant-Jacobian) group whose motors no longer share a gain — see ``loaders.equal_gain_warnings``."""
    return [w for i, meta in enumerate(owner._gain_meta)
            for w in equal_gain_warnings(meta["name"], meta["models"], meta["dim"],
                                         owner._kphi0[i], owner._kd0[i])]


class MotorCoupledPDHand:
    """Coupled PD kernel owner (method 2). ``launch(joint_q, joint_qd, joint_target_q, joint_f)``
    writes the coupled joints' generalized force into ``joint_f`` and mirrors the FINAL joint torque
    τ_q = Gᵀ·τ_m into :attr:`tau_torch` (num_envs, 3G) f32 — the readout the backend overlays.

    ``gravcomp``/``gc_dof`` (optional): MuJoCo ``qfrc_gravcomp`` buffer + per-joint MuJoCo dof indices —
    groups flagged ``gravcomp_fold`` add G⁻ᵀ·τ_g to τ_m before the clamp (see kernel).

    :attr:`tau_pd_torch` / :attr:`tau_gc_torch` mirror the two PRE-clamp components of τ_q (same shape),
    backing the optional ``joint_torque_pd`` / ``joint_torque_gravcomp`` reads."""

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
        self._coord_idx_t = coord_idx.to(torch.int32).contiguous()   # kept alive (wp aliases it)
        self._dof_idx_t = dof_idx.to(torch.int32).contiguous()
        self._coord_idx = wp.from_torch(self._coord_idx_t)
        self._dof_idx = wp.from_torch(self._dof_idx_t)
        self._gc_fold, self._qfrc_g, self._gc_dof, self._gc_dof_t = _gc_buffers(
            groups, gravcomp, gc_dof, 3 * self.n_groups, self.device)
        self._kphi0 = np.stack([g["k_phi"] for g in groups]).astype(np.float64)   # last loaded (reload diff)
        self._kd0 = np.stack([g["k_d"] for g in groups]).astype(np.float64)   # originals (float toggle)
        self._gain_src = [g["gain_src"] for g in groups]   # robot_model.json provenance → reload_gains
        self._gain_meta = [{"name": g["name"], "models": g["models"], "dim": g["dim"],
                            "joints": list(g["joints"]), "lo": g["q_fit_lo"], "hi": g["q_fit_hi"],
                            "part": g["part"],
                            # motor torque envelope (host twin of the kernel's clamp → tau_lim readout)
                            "tau_stall": g["tau_stall"], "omega_noload": g["omega_noload"],
                            "rated_pos": g["rated_pos"], "rated_neg": g["rated_neg"]}
                           for g in groups]
        self._kd_zeroed = False                            # float-mode D-gain state (survives a reload)
        self._fold_np = np.asarray([bool(g["gravcomp_fold"]) for g in groups])
        self.tau_torch = torch.zeros(self.num_envs, 3 * self.n_groups,
                                     dtype=torch.float32, device=coord_idx.device)
        self._tau = wp.from_torch(self.tau_torch)
        self.tau_pd_torch = torch.zeros_like(self.tau_torch)   # PRE-clamp PD component readout
        self.tau_gc_torch = torch.zeros_like(self.tau_torch)   # PRE-clamp gravity component readout
        self._tau_pd = wp.from_torch(self.tau_pd_torch)
        self._tau_gc = wp.from_torch(self.tau_gc_torch)

    def set_fold_kd_zero(self, zero: bool) -> None:
        """Torque/float mode: zero the D gain of the gravcomp-fold groups — the real gravity-comp mode
        sends the feedforward ONLY (τ_m = G⁻ᵀ·τ_g); dissipation comes from joint friction. Restore for
        Position mode. Non-fold groups (fingers, which still position-hold in float) keep their damping.
        Updates the fixed buffer's CONTENTS → CUDA-graph safe (same address, new values)."""
        self._kd_zeroed = bool(zero)
        kd = self._kd0.copy()
        if zero:
            kd[self._fold_np] = 0.0
        self.buf.k_d.assign(kd)

    def reload_gains(self) -> list[str]:
        """Re-read robot_model.json gains into the live buffers — see :func:`_reload_gains`."""
        return _reload_gains(self)

    def gain_warnings(self) -> list[str]:
        """Gain-consistency warnings for the gains currently in the buffers — see :func:`_gain_warnings`."""
        return _gain_warnings(self)

    def kq(self, q):
        """Joint-space stiffness per group at pose ``q`` — see :func:`_kq`."""
        return _kq(self, q)

    def tau_lim(self, q, qd):
        """Joint-space torque limit per group at pose ``q``, speed ``qd`` — see :func:`_tau_lim`."""
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
    """Arm coupled-PD kernel owner — 2-DOF 2×2 (``coupled_pd_arm_kernel``; wrist ballscrew + elbow
    differential pulley). Same interface as :class:`MotorCoupledPDHand` (incl. the optional
    ``gravcomp``/``gc_dof`` fold) but 2 joints/motors per group; ``coord_idx``/``dof_idx`` are
    (num_envs, 2G). Writes ``joint_f`` and mirrors τ_q into :attr:`tau_torch` (num_envs, 2G), with the
    two PRE-clamp components in :attr:`tau_pd_torch` / :attr:`tau_gc_torch`."""

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
        self._coord_idx_t = coord_idx.to(torch.int32).contiguous()   # kept alive (wp aliases it)
        self._dof_idx_t = dof_idx.to(torch.int32).contiguous()
        self._coord_idx = wp.from_torch(self._coord_idx_t)
        self._dof_idx = wp.from_torch(self._dof_idx_t)
        self._gc_fold, self._qfrc_g, self._gc_dof, self._gc_dof_t = _gc_buffers(
            groups, gravcomp, gc_dof, 2 * self.n_groups, self.device)
        self._kphi0 = np.stack([g["k_phi"] for g in groups]).astype(np.float64)   # last loaded (reload diff)
        self._kd0 = np.stack([g["k_d"] for g in groups]).astype(np.float64)   # originals (float toggle)
        self._gain_src = [g["gain_src"] for g in groups]   # robot_model.json provenance → reload_gains
        self._gain_meta = [{"name": g["name"], "models": g["models"], "dim": g["dim"],
                            "joints": list(g["joints"]), "lo": g["q_fit_lo"], "hi": g["q_fit_hi"],
                            "part": g["part"],
                            # motor torque envelope (host twin of the kernel's clamp → tau_lim readout)
                            "tau_stall": g["tau_stall"], "omega_noload": g["omega_noload"],
                            "rated_pos": g["rated_pos"], "rated_neg": g["rated_neg"]}
                           for g in groups]
        self._kd_zeroed = False                            # float-mode D-gain state (survives a reload)
        self._fold_np = np.asarray([bool(g["gravcomp_fold"]) for g in groups])
        self.tau_torch = torch.zeros(self.num_envs, 2 * self.n_groups,
                                     dtype=torch.float32, device=coord_idx.device)
        self._tau = wp.from_torch(self.tau_torch)
        self.tau_pd_torch = torch.zeros_like(self.tau_torch)   # PRE-clamp PD component readout
        self.tau_gc_torch = torch.zeros_like(self.tau_torch)   # PRE-clamp gravity component readout
        self._tau_pd = wp.from_torch(self.tau_pd_torch)
        self._tau_gc = wp.from_torch(self.tau_gc_torch)

    def set_fold_kd_zero(self, zero: bool) -> None:
        """See :meth:`MotorCoupledPDHand.set_fold_kd_zero` — same float-mode D-gain toggle."""
        self._kd_zeroed = bool(zero)
        kd = self._kd0.copy()
        if zero:
            kd[self._fold_np] = 0.0
        self.buf.k_d.assign(kd)

    def reload_gains(self) -> list[str]:
        """See :meth:`MotorCoupledPDHand.reload_gains`."""
        return _reload_gains(self)

    def gain_warnings(self) -> list[str]:
        """See :meth:`MotorCoupledPDHand.gain_warnings`."""
        return _gain_warnings(self)

    def kq(self, q):
        """See :meth:`MotorCoupledPDHand.kq`."""
        return _kq(self, q)

    def tau_lim(self, q, qd):
        """See :meth:`MotorCoupledPDHand.tau_lim`."""
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
