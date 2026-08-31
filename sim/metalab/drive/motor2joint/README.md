# motor2joint — motor-space coupled PD (runtime)

The real ALLEX arm+hand drive several joints through a nonlinear transmission with PD closed in
**motor space** — so joint-space stiffness gains off-diagonal terms. Two structural families:

- **hand** — each finger/thumb's 3 joints via 3 motors: 3-DOF **lower-triangular** `m_i = f(q0..qi)`.
- **arm** — each wrist's 2 joints (Roll/Pitch) via 2 ballscrew motors: 2-DOF **FULL 2×2** (both motors
  depend on both joints). The same family will later cover elbow+wristYaw.

This package reproduces that control law on the newton backend: a warp kernel evaluates the firmware
joint→motor map `m = J2M(q)` and its gradient `G = ∂m/∂q` at physics-substep rate and applies
`τ_q = Gᵀ·τ_m`, where `τ_m = k_φ·Δφ − k_d·φ̇` is clamped to each motor's torque-speed envelope ∩ ±rated
(see `motor_coupling.py` docstring for the full derivation). Enabled by `control_mode: motor` in the
robot YAML.

## Files kept (used directly at runtime)

| File | Role |
|------|------|
| `motor_coupling.py` | warp kernels (`coupled_pd_hand_kernel` 3-DOF, `coupled_pd_arm_kernel` 2-DOF) + loaders (`load_hand_group`, `load_arm_group`) + owners (`MotorCoupledPDHand`, `MotorCoupledPDArm`) |
| `mj_mapping/finger.json` | shared finger transmission (J2M fits + q envelope). One map backs all 8 finger groups (index/middle/ring/little × L/R) |
| `mj_mapping/thumb.json` | shared thumb transmission. One map backs both thumbs (L/R) |
| `mj_mapping/wrist.json` | shared wrist transmission (FULL 2×2 ballscrew J2M + envelope). One map backs both wrists (Roll/Pitch) |
| `mj_mapping/firmware_to_json.py` | extractor: firmware C++ → model json, gradient self-checked (see Provenance) |
| `robot_model.json` | per-group `motor_control_param` (k_φ=`k_pos`, k_d=`k_vel`, torque limits) + `actuators` (rated/stall torque, no-load speed) keyed by `<part>_<hand>`; the wrist slices the 7-DOF `arm_{r,l}` group at indices 5,6 |

`load_hand_group` / `load_arm_group` read the `.json` files by path (no Python import), so `mj_mapping/`
is a data directory, not a package.

## Provenance — how the `.json` maps were built

The maps are **not fits** — they are the *deployed firmware* coefficients (digital twin), extracted
from the robot firmware C++ and self-checked. To regenerate/verify, retrieve the firmware from the
robot package (`irim_robot_pkg/hardware_model/motor_joint/`) and re-run the extractor.

**Firmware sources:**

- `finger.json` ← `janghwan_mj_map_updated.cpp` (`CJanghwanFingerMJMapper`): `cal_motorAngles_janghwan`
  (m = J2M(q), cubic) + `cal_dmdq_janghwan` (G). One coefficient set for every finger; the MJCF gives
  all four fingers, both hands, identical joint ranges + axes → one shared `finger.json`. **Source
  removed** (canonical copy in `irim_robot_pkg`).
- `thumb.json` ← `thumb_{r,l}_mj_map.cpp`, rev `56bf59f3`: the `constexpr` tables
  `kM{1,2,3}Terms`/`kM{1,2,3}Coef` (Yaw linear, CMC and MCP degree 7) that `cal_motorAngles_thumb`
  walks with a generic `evalPolynomial`. `cal_dmdq_thumb` is not a second source — it differentiates
  those same tables — so the extractor's self-check ports `evalPolynomialDerivative` instead of
  comparing two authored expressions. The two files are identical apart from the class name; only the
  Yaw clip range mirrors L/R → one shared `thumb.json` with Yaw = symmetric union [-2.62, 2.62].
  Controlled joints Yaw/CMC/MCP (the thumb IP joint is a kinematic equality follower, not
  motor-coupled). **Source removed.**
- `wrist.json` ← `arm_{r,l}_mj_map.cpp`: `calcWristJoint2MotorAngle` (ballscrew J2M, FULL 2×2) +
  `wristJac_dMdq` (G). arm_r/arm_l wrist maps are identical → one shared `wrist.json` (Roll/Pitch).

**Extractor — `mj_mapping/firmware_to_json.py` (present):**

Self-contained (inline `_Poly`, no fit dependency). Parses a firmware `*J2M` into the model json and
**self-checks** by confirming the parsed map's analytic gradient reproduces the firmware Jacobian at
200 random configs in the joint envelope. Emits `thumb.json` (term counts 2/36/120), `wrist.json`
(66/66), `elbow.json` (2/2) and `shoulder.json` (1/1/1); `finger.json` (4/10/20) was emitted by an
earlier version. Only the J2M side is extracted; the firmware M2J readback (`cal_jointAngles*` /
`cal_dqdm*`) is telemetry-only and unused by control — and measurably so for the thumb, whose
`kQ3` inverse fit diverges in the fully-flexed CMC+MCP corner (round-trip error reaches tens of rad
where `m3` approaches 178 rad), which is why it is not used as a cross-check.

**Superseded fit path (removed):**

- `fitting/` — an offline mocap-fit alternative (capture CSV `pose_joint_all_251120_2.csv` →
  `polyfitn` → model json). Superseded by the firmware extraction above (the sim must run *what the
  robot runs*, not an independent fit). It provided `PolyModel` / `polyfitn` / `buildcompletemodel`.
- `retire/` — the original MATLAB sources `fitting/` was ported from (`polyvaln.m`,
  `polyfitn_grad.m`, `curvefitting_2D.m`, `curvefitting_J2M.m`, `Test_kq_2_3_ratio.m`).

The runtime kernels are cross-checked against a self-contained numpy oracle in
`sim/metalab/tests/test_motor_coupling.py`.
