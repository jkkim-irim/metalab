"""Motor-to-joint coupling (real-HW nonlinear transmission → off-diagonal joint-space stiffness).

- ``loaders`` — group loaders + buffer packing (numpy-only, engine-agnostic).
- ``motor_coupling`` — the coupled PD law: warp kernels (``MotorCoupledPDHand``,
    ``MotorCoupledPDArm``). **BOTH engines run these same kernels** — newton natively, genesis via a
    thin adapter in its backend (warp is a declared dep of both envs; there is no torch fallback,
    a pure-torch evaluation measured ~5 ms/step worse at 2048 envs).
- ``mj_mapping/{finger,thumb}.json`` — the firmware-derived J2M transmission maps (data, loaded by
    path). ``finger.json`` backs all 8 finger groups, ``thumb.json`` both thumbs.
- ``robot_model.json`` — per-group ``motor_control_param`` (motor gains) + actuator specs.

See ``README.md`` for the provenance of the ``.json`` maps (which firmware sources they were
extracted from). ``motor_coupling`` is imported explicitly by its newton/warp consumer, so this
package root pulls in nothing.
"""
