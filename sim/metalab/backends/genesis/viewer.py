from __future__ import annotations

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import KeyAction, Keybind
import numpy as np
import pyglet
import rerun as rr

from sim.metalab.runtime import rerun_recording


class GenesisViewer:
    def __init__(self, scene):
        self._scene = scene
        self._paused = False
        self._step_once = False
        self._install_pause_keybind()

    @property
    def gl(self):
        return getattr(self._scene, "viewer", None)

    def step_allowed(self) -> bool:
        if self._paused and self._step_once:
            self._step_once = False
            return True
        return not self._paused

    def pump(self) -> None:
        viewer = self.gl
        pv = getattr(viewer, "_pyrender_viewer", None) if viewer is not None else None
        if pv is None or getattr(pv, "run_in_thread", True):
            return
        with viewer.lock:
            pv.refresh()

    def _install_pause_keybind(self) -> None:
        viewer = self.gl
        if viewer is None or not hasattr(viewer, "register_keybinds"):
            return

        def _toggle():
            self._paused = not self._paused

        def _step():
            self._step_once = True

        viewer.register_keybinds(
            Keybind(name="allex_pause", key=pyglet.window.key.SPACE, key_action=KeyAction.PRESS,
                    key_mods=None, callback=_toggle),
            Keybind(name="allex_step", key=pyglet.window.key.PERIOD, key_action=KeyAction.PRESS,
                    key_mods=None, callback=_step),
            overwrite=True)

    def close(self) -> str | None:
        return None

    def focus_env(self, cam, env_idx: int) -> None:
        viewer = self.gl
        if viewer is None or cam is None:
            return
        off = (np.zeros(3, dtype=np.float32) if int(env_idx) < 0
               else self._scene.envs_offset[int(env_idx)])
        eye = np.asarray(cam.eye, dtype=np.float32) + off
        lookat = np.asarray(cam.lookat, dtype=np.float32) + off
        pose = gu.pos_lookat_up_to_T(eye, lookat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        with viewer.lock:
            viewer.set_camera_pose(pose=pose)
        print(f"[telemetry] focus env {env_idx}: eye={eye.round(2).tolist()} "
              f"lookat={lookat.round(2).tolist()}", flush=True)


_DEFAULT_COLOR = (170, 172, 178)


def _unwrap(entity):
    return getattr(entity, "_entity", entity)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _is_ground_plane(entity) -> bool:
    return isinstance(getattr(entity, "main_morph", None), gs.morphs.Plane)


def _diffuse(vgeom) -> tuple[int, int, int]:
    surf = getattr(vgeom.vmesh, "surface", None)
    tex = getattr(surf, "diffuse_texture", None) if surf is not None else None
    c = getattr(tex, "color", None)
    if c is None or len(c) < 3:
        return _DEFAULT_COLOR
    return tuple(int(round(float(x) * 255.0)) for x in c[:3])


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.zeros_like(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-20).astype(np.float32)


def _variant_groups(link, num_envs: int) -> list[tuple[tuple[int, ...] | None, list]]:
    groups: dict[tuple[int, ...] | None, list] = {}
    everywhere = tuple(range(num_envs))
    for vg in link.vgeoms:
        idx = getattr(vg, "active_envs_idx", None)
        key = None if idx is None else tuple(i for i in (int(x) for x in np.asarray(idx)) if i < num_envs)
        groups.setdefault(None if key == everywhere else key, []).append(vg)
    return list(groups.items())


def _merge_vgeoms(vgeoms) -> tuple[np.ndarray, ...] | None:
    vs, fs, cs, base = [], [], [], 0
    for vg in vgeoms:
        v = np.asarray(vg.init_vverts, dtype=np.float32)
        f = np.asarray(vg.init_vfaces, dtype=np.uint32)
        if not len(v) or not len(f):
            continue
        q = np.asarray(vg.init_quat, dtype=np.float64)
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
    def __init__(self, scene, entities: dict, num_envs: int):
        self._offsets = np.asarray(scene.envs_offset, dtype=np.float32).reshape(-1, 3)[:num_envs]
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
        rerun_recording.mark_step(self._step)
        self._log_poses()
        self._step += 1

    def _log_poses(self) -> None:
        for e, parts in self._parts:
            pos = e.get_links_pos(relative=False).detach().cpu().numpy()
            quat = e.get_links_quat(relative=False).detach().cpu().numpy()
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
