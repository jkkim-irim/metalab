from __future__ import annotations

import mujoco as mj
import newton
import torch
import warp as wp


def resolve(solver, joint_names: list[str]) -> tuple[list[int], dict[str, int]]:
    mjm = solver.mj_model
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
    mjm = solver.mj_model
    alln = [(i, mj.mj_id2name(mjm, mj.mjtObj.mjOBJ_JOINT, i) or "") for i in range(mjm.njnt)]
    out: list[int] = []
    for name in names:
        hits = [i for i, nm in alln if nm == name or nm.endswith("_" + name)]
        assert len(hits) == 1, f"joint {name!r}: expected exactly 1 MuJoCo match, got {len(hits)}"
        out.append(hits[0])
    return out


def resolve_dofs(solver, names: list[str]) -> list[int]:
    mjm = solver.mj_model
    return [int(mjm.jnt_dofadr[j]) for j in _jids(solver, names)]


def set_jnt_actgravcomp(solver, joint_names: list[str]) -> int:
    mjw = solver.mjw_model
    jgc = wp.to_torch(mjw.jnt_actgravcomp)
    for jid in _jids(solver, joint_names):
        jgc[jid] = 1
    return len(joint_names)


def set_body_gravcomp(solver, mjc_body_ids: list[int], scale: float) -> int:
    mjw = solver.mjw_model
    body_gc = wp.to_torch(mjw.body_gravcomp)
    body_gc[:, mjc_body_ids] = float(scale)
    src = wp.to_torch(solver.model.mujoco.gravcomp)
    m2n = wp.to_torch(solver.mjc_body_to_newton).cpu().numpy()
    for b in mjc_body_ids:
        nds = m2n[:, b]
        src[torch.from_numpy(nds[nds >= 0]).to(src.device)] = float(scale)
    mjw.ngravcomp = int((wp.to_torch(mjw.body_gravcomp).cpu().numpy() > 0.0).any(axis=0).sum())
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    return int(mjw.ngravcomp)
