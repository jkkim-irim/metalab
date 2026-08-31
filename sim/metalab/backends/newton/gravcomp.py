"""Gravity compensation for the Newton (MuJoCo-warp) spoke.

Mirrors the real robot: the listed joints are held against gravity so they do not sag. The gravcomp
force is generated via MuJoCo's native per-body ``body_gravcomp`` on each joint's child body. Two
channels (see ``GravCompSpec``):
- ACTUATOR joints — :func:`set_jnt_actgravcomp` routes their gravcomp through the force-limited
  actuator: MuJoCo adds ``qfrc_gravcomp`` to ``qfrc_actuator`` and clamps PD+gravcomp to
  ``jnt_actfrcrange`` (= spec effort), so the joint yields past its max torque and ``joint_torque``
  reads the applied value straight from ``qfrc_actuator``.
- PASSIVE joints/bodies (waist/neck spring) — gravcomp stays a passive force: uncapped, and NOT in
  ``qfrc_actuator`` (mujoco_warp skips actgravcomp joints in the passive pass, so no double count).

MuJoCo-warp pitfall handled in :func:`set_body_gravcomp`: the ``BODY_INERTIAL_PROPERTIES`` re-sync
copies ``mjw_model.body_gravcomp`` from the Newton source ``model.mujoco.gravcomp`` (USD default 0),
so writing only ``mjw_model`` is reverted on the next re-sync (e.g. object mass DR) → also write the
Newton source. ``jnt_actgravcomp`` has no such re-sync (direct device write persists).
"""
from __future__ import annotations

import mujoco as mj
import newton
import torch
import warp as wp


def resolve(solver, joint_names: list[str]) -> tuple[list[int], dict[str, int]]:
    """Map gravcomp joint names → (sorted-unique MuJoCo child-body ids, {name: MuJoCo dof index}).

    Each joint's child body (``jnt_bodyid``) is the segment made weightless; ``jnt_dofadr`` is the dof
    whose ``qfrc_gravcomp`` entry is that joint's gravcomp torque. Fail-loud on an unknown joint."""
    mjm = solver.mj_model
    # MuJoCo joint names are the full body-chain path (underscore-concatenated) ending in the joint name,
    # so match by suffix (mirrors NewtonBackend._local_joint) — mj_name2id needs the exact full name.
    all_names = [(i, mj.mj_id2name(mjm, mj.mjtObj.mjOBJ_JOINT, i) or "") for i in range(mjm.njnt)]
    body_ids: list[int] = []
    dof_of: dict[str, int] = {}
    for name in joint_names:
        hits = [i for i, nm in all_names if nm == name or nm.endswith("_" + name)]
        assert len(hits) == 1, f"gravcomp joint {name!r}: expected exactly 1 MuJoCo match, got {len(hits)}"
        jid = hits[0]
        body_ids.append(int(mjm.jnt_bodyid[jid]))
        dof_of[name] = int(mjm.jnt_dofadr[jid])
    return sorted(set(body_ids)), dof_of


def resolve_bodies(solver, body_names: list[str]) -> list[int]:
    """Extra passive-channel bodies (leaf link names) → MuJoCo body ids. MuJoCo body names are the full
    body-chain path ending in the leaf, so match by suffix (like resolve()). Fail-loud on 0 / >1 matches."""
    if not body_names:
        return []
    mjm = solver.mj_model
    alln = [(i, mj.mj_id2name(mjm, mj.mjtObj.mjOBJ_BODY, i) or "") for i in range(mjm.nbody)]
    out: list[int] = []
    for name in body_names:
        hits = [i for i, nm in alln if nm == name or nm.endswith("_" + name)]
        assert len(hits) == 1, f"gravcomp passive_body {name!r}: expected exactly 1 MuJoCo body, got {len(hits)}"
        out.append(hits[0])
    return out


def _jids(solver, names: list[str]) -> list[int]:
    """joint names → MuJoCo joint ids (suffix match on the full body-chain path). Fail-loud on 0/>1."""
    mjm = solver.mj_model
    alln = [(i, mj.mj_id2name(mjm, mj.mjtObj.mjOBJ_JOINT, i) or "") for i in range(mjm.njnt)]
    out: list[int] = []
    for name in names:
        hits = [i for i, nm in alln if nm == name or nm.endswith("_" + name)]
        assert len(hits) == 1, f"joint {name!r}: expected exactly 1 MuJoCo match, got {len(hits)}"
        out.append(hits[0])
    return out


def resolve_dofs(solver, names: list[str]) -> list[int]:
    """joint names → MuJoCo dof indices (``jnt_dofadr``) for reading ``mjw_data.qfrc_actuator``. 1-DOF joints."""
    mjm = solver.mj_model
    return [int(mjm.jnt_dofadr[j]) for j in _jids(solver, names)]


def set_jnt_actgravcomp(solver, joint_names: list[str]) -> int:
    """Route these joints' gravity-compensation force through the (force-limited) actuator channel.

    MuJoCo then adds ``qfrc_gravcomp`` to ``qfrc_actuator`` for these joints and clamps the sum to
    ``jnt_actfrcrange`` — so gravcomp counts against the motor torque limit (mirrors real HW: the motor supplies
    gravcomp and yields when PD+gravcomp exceeds max torque). Joints left OFF keep gravcomp as an external
    passive force (uncapped, not in ``qfrc_actuator``)."""
    mjw = solver.mjw_model
    jgc = wp.to_torch(mjw.jnt_actgravcomp)                        # (njnt,) int
    for jid in _jids(solver, joint_names):
        jgc[jid] = 1
    return len(joint_names)


def set_body_gravcomp(solver, mjc_body_ids: list[int], scale: float) -> int:
    """Set ``body_gravcomp = scale`` on the given MuJoCo bodies (1 = weightless/ON, 0 = OFF), persist to the
    Newton source, recompute ``ngravcomp``, and notify. Call at build (scale=1) BEFORE the first ``step_n``
    graph capture, or at runtime to toggle (standalone is eager → takes next step; caller drops the graph).
    Returns the resulting ``ngravcomp`` (number of gravcomp'd bodies per world)."""
    mjw = solver.mjw_model
    body_gc = wp.to_torch(mjw.body_gravcomp)                     # (nworld, nbody)
    body_gc[:, mjc_body_ids] = float(scale)
    # (1) persist to the Newton source so the BODY_INERTIAL re-sync reproduces the scales
    src = wp.to_torch(solver.model.mujoco.gravcomp)              # (n_newton_body,)
    m2n = wp.to_torch(solver.mjc_body_to_newton).cpu().numpy()   # (nworld, nbody): mjc_body -> newton_body
    for b in mjc_body_ids:
        nds = m2n[:, b]
        src[torch.from_numpy(nds[nds >= 0]).to(src.device)] = float(scale)
    # (2) recompute ngravcomp (>0 required at capture so the gravcomp kernel is baked into the graph)
    mjw.ngravcomp = int((wp.to_torch(mjw.body_gravcomp).cpu().numpy() > 0.0).any(axis=0).sum())
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    return int(mjw.ngravcomp)
