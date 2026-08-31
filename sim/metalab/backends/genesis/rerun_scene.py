"""genesis → rerun scene logger: static link meshes once, per-link poses every frame.

newton gets this for free — its ``ViewerRerun`` is a ``ViewerBase``, so ``log_state`` already ships geometry
and transforms. genesis has no viewer abstraction, so the spoke logs the scene itself. The framing (``step``
timeline, blueprint, file sink, batcher, ``env<N>`` labels) is shared with newton in
:mod:`sim.metalab.runtime.rerun_recording`; only the traversal below is genesis-specific.

STRUCTURE, and why it is this one. Each LINK is one rerun entity carrying a ``Mesh3D`` logged ``static``, and
every frame that entity gets an ``InstancePoses3D`` with one pose per env. So the geometry is stored ONCE and
all envs instance it — measured on hammer-lift: 80 links / 153 vgeoms / 713 k verts ≈ 25.7 MB of mesh, against
80 × 4 × 7 floats ≈ 9 KB of poses per frame. An entity per (env, link) instead would multiply that 25.7 MB by
the env count for nothing.

A link's several vgeoms are merged into one mesh with their ``init_pos``/``init_quat`` baked into the vertices,
because a vgeom's offset inside its link is fixed — only the LINK moves, so only the link needs a pose. The one
exception is a HETEROGENEOUS link, whose vgeoms belong to different per-env asset variants and must not be
merged together — see :func:`_variant_groups`.

``envs_offset`` is added to the poses: genesis keeps every env's physics near the origin (measured: link poses
differ between envs only by millimetre-scale spawn randomisation) while the recording tiles them apart. Same
situation as newton's ``world_offsets``, reached differently.
"""
from __future__ import annotations

import genesis as gs
import numpy as np
import rerun as rr

from sim.metalab.runtime import rerun_recording

_DEFAULT_COLOR = (170, 172, 178)


def _unwrap(entity):
    """genesis entities reach the backend behind a ByBasename wrapper in some slots; take the real one."""
    return getattr(entity, "_entity", entity)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _is_ground_plane(entity) -> bool:
    """genesis renders ``gs.morphs.Plane`` as a 1000 m x 1000 m quad (measured), and rerun auto-frames the
    camera to fit ALL geometry — so logging it zooms a 1.5 m robot down to a sub-pixel dot. The 3D view draws
    its own ground grid, so the quad is dropped rather than resized.

    ``main_morph``, not ``morph``: the latter raises on a heterogeneous entity (our object has one morph
    variant per hammer asset). A plane is never heterogeneous, so the first variant is the whole truth."""
    return isinstance(getattr(entity, "main_morph", None), gs.morphs.Plane)


def _diffuse(vgeom) -> tuple[int, int, int]:
    """vgeom's diffuse colour as 8-bit RGB, or a neutral grey when it has no texture (plane/table)."""
    surf = getattr(vgeom.vmesh, "surface", None)
    tex = getattr(surf, "diffuse_texture", None) if surf is not None else None
    c = getattr(tex, "color", None)
    if c is None or len(c) < 3:
        return _DEFAULT_COLOR
    return tuple(int(round(float(x) * 255.0)) for x in c[:3])


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth per-vertex normals, area-weighted (the cross product's length IS twice the triangle area).

    Computed rather than skipped because newton's meshes carry normals and genesis' vgeoms do not: an
    unshaded mesh reads as a flat silhouette against the viewer's dark background."""
    n = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-20).astype(np.float32)


def _variant_groups(link, num_envs: int) -> list[tuple[tuple[int, ...] | None, list]]:
    """Split a link's vgeoms into ``(env_idx, vgeoms)`` groups — one per heterogeneous variant.

    A heterogeneous entity keeps EVERY variant's vgeoms on the SAME link (measured: our 4 hammer assets are
    16 vgeoms on one ``object`` link), so merging a link's vgeoms unconditionally draws all 4 hammers
    superimposed in every env. genesis already marks each vgeom with the envs it is active in — the rigid
    solver sets ``active_envs_idx`` from the link's per-variant vgeom range — so the split is read from the
    engine rather than re-derived from its private env→variant mapping.

    ``env_idx`` is ``None`` when the group covers every env (the homogeneous case, where the attribute is
    absent altogether): one instance per env, and the entity path stays unsuffixed."""
    groups: dict[tuple[int, ...] | None, list] = {}
    everywhere = tuple(range(num_envs))
    for vg in link.vgeoms:
        idx = getattr(vg, "active_envs_idx", None)
        key = None if idx is None else tuple(i for i in (int(x) for x in np.asarray(idx)) if i < num_envs)
        groups.setdefault(None if key == everywhere else key, []).append(vg)
    return list(groups.items())


def _merge_vgeoms(vgeoms) -> tuple[np.ndarray, ...] | None:
    """Merge vgeoms into one mesh in LINK coordinates: (verts, faces, colors, normals) or None."""
    vs, fs, cs, base = [], [], [], 0
    for vg in vgeoms:
        v = np.asarray(vg.init_vverts, dtype=np.float32)
        f = np.asarray(vg.init_vfaces, dtype=np.uint32)
        if not len(v) or not len(f):
            continue
        # bake the vgeom's fixed offset inside the link, so only the link needs a per-frame pose
        q = np.asarray(vg.init_quat, dtype=np.float64)          # wxyz
        w, x, y, z = q
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        vs.append((v @ rot.T + np.asarray(vg.init_pos, dtype=np.float32)).astype(np.float32))
        fs.append(f + base)
        cs.append(np.tile(np.asarray(_diffuse(vg), dtype=np.uint8), (len(v), 1)))
        base += len(v)
    if not vs:
        return None
    verts, faces, colors = np.concatenate(vs), np.concatenate(fs), np.concatenate(cs)
    return verts, faces, colors, _vertex_normals(verts, faces.astype(np.int64))


class GenesisRerunScene:
    """Logs a genesis scene into the active rerun recording: geometry once, poses per frame."""

    def __init__(self, scene, entities: dict, num_envs: int):
        """``entities`` = {name: RigidEntity}. Geometry is logged immediately (static), so construct this
        AFTER the recording's sink is open — otherwise the meshes go nowhere."""
        self._offsets = np.asarray(scene.envs_offset, dtype=np.float32).reshape(-1, 3)[:num_envs]
        # (entity, [(entity_path, link_index, env_idx or None)]) — env_idx selects which envs instance a
        # heterogeneous variant's mesh; None means every env.
        self._parts: list[tuple[object, list[tuple[str, int, np.ndarray | None]]]] = []
        n_mesh = n_variant = 0
        for name, ent in entities.items():
            e = _unwrap(ent)
            if _is_ground_plane(e):
                continue
            parts = []
            for li, link in enumerate(e.links):
                groups = _variant_groups(link, num_envs)
                for vi, (envs, vgeoms) in enumerate(groups):
                    m = _merge_vgeoms(vgeoms)
                    if m is None:
                        continue
                    path = f"/model/{name}/link{li:03d}" + (f"/v{vi:02d}" if envs is not None else "")
                    verts, faces, colors, normals = m
                    rr.log(path, rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                           vertex_colors=colors, vertex_normals=normals), static=True)
                    n_mesh += 1
                    n_variant += envs is not None
                    parts.append((path, li, None if envs is None else np.asarray(envs, dtype=np.int64)))
            self._parts.append((e, parts))
        self.n_meshes = n_mesh
        self.n_variants = n_variant
        self._step = 0

    def after_step(self) -> None:
        """One recorded frame per POLICY step — the cadence make_vec_env_handler drives, shared with the
        series log so a frame index equals a series row."""
        rerun_recording.mark_step(self._step)
        self._log_poses()
        self._step += 1

    def _log_poses(self) -> None:
        """One ``InstancePoses3D`` per mesh entity, one pose per env that instances it, at the time cursor."""
        for e, parts in self._parts:
            pos = e.get_links_pos(relative=False).detach().cpu().numpy()      # (N, L, 3)
            quat = e.get_links_quat(relative=False).detach().cpu().numpy()    # (N, L, 4) wxyz
            pos = pos + self._offsets[:, None, :]
            quat = _quat_wxyz_to_xyzw(quat)
            for path, li, envs in parts:
                sel = slice(None) if envs is None else envs
                rr.log(path, rr.InstancePoses3D(translations=pos[sel, li, :].astype(np.float32),
                                                quaternions=quat[sel, li, :].astype(np.float32)))

    def summary(self) -> str:
        n_parts = sum(len(p) for _, p in self._parts)
        return (f"{self.n_meshes} static meshes ({self.n_variants} heterogeneous variants), "
                f"{n_parts} pose entities x up to {len(self._offsets)} envs per frame")
