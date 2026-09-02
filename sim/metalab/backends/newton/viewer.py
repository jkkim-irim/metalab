from __future__ import annotations

import numpy as np
import rerun as rr
import torch
import warp as wp

from sim.metalab.runtime.rerun_recording import log_world_labels as _log_world_labels
from sim.metalab.runtime.rerun_recording import mark_step

_AXIS_LEN_DEFAULT = 0.1   # [m]
_AXIS_LEN_RANGE = (0.01, 0.5)
_LINES_PER_BODY = 3


@wp.kernel
def _origin_axis_lines(
    body_idx: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_world: wp.array(dtype=wp.int32),
    world_offsets: wp.array(dtype=wp.vec3),
    layer_xform: wp.transform,
    visible_worlds_mask: wp.array(dtype=wp.int32),
    axis_len: float,
    local_offset: wp.vec3,
    color_scale: float,
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = tid // _LINES_PER_BODY
    axis = tid % _LINES_PER_BODY
    nan_line = wp.vec3(wp.nan, wp.nan, wp.nan)

    b = body_idx[i]
    if b < 0:
        line_starts[tid] = nan_line
        line_ends[tid] = nan_line
        line_colors[tid] = wp.vec3(0.0, 0.0, 0.0)
        return

    w = body_world[b]
    if visible_worlds_mask:
        if w >= 0:
            if visible_worlds_mask[w] == 0:
                line_starts[tid] = nan_line
                line_ends[tid] = nan_line
                line_colors[tid] = wp.vec3(0.0, 0.0, 0.0)
                return

    tf = body_q[b]
    pos = wp.transform_point(tf, local_offset)
    rot = wp.transform_get_rotation(tf)
    if world_offsets and w >= 0:
        pos += world_offsets[w]
    pos = wp.transform_point(layer_xform, pos)
    rot = wp.mul(wp.transform_get_rotation(layer_xform), rot)

    if axis == 0:
        vec = wp.quat_rotate(rot, wp.vec3(1.0, 0.0, 0.0))
        color = wp.vec3(1.0, 0.0, 0.0)
    elif axis == 1:
        vec = wp.quat_rotate(rot, wp.vec3(0.0, 1.0, 0.0))
        color = wp.vec3(0.0, 1.0, 0.0)
    else:
        vec = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
        color = wp.vec3(0.0, 0.0, 1.0)

    line_starts[tid] = pos
    line_ends[tid] = pos + vec * axis_len
    line_colors[tid] = color * color_scale


class _Frames:
    def __init__(self, name: str, label: str, offset, color_scale: float, n_lines: int, dev):
        self.name, self.label = name, label
        self.offset = wp.vec3(*(float(c) for c in offset))
        self.color_scale = float(color_scale)
        self.enabled = False
        self.hidden = True
        self.starts = wp.zeros(n_lines, dtype=wp.vec3, device=dev)
        self.ends = wp.zeros(n_lines, dtype=wp.vec3, device=dev)
        self.colors = wp.zeros(n_lines, dtype=wp.vec3, device=dev)


class OriginAxes:
    LINES_NAME = "/metalab/origin_axes"

    def __init__(self, viewer, body_idx: torch.Tensor, *, enabled: bool = False,
                 axis_len: float = _AXIS_LEN_DEFAULT):
        assert body_idx.numel() > 0, "OriginAxes needs at least one body index"
        self.axis_len = float(axis_len)
        self._viewer = viewer
        self._n_lines = int(body_idx.numel()) * _LINES_PER_BODY
        dev = wp.device_from_torch(body_idx.device) if body_idx.is_cuda else str(viewer.device)
        self._body_idx = wp.from_torch(body_idx.to(torch.int32).contiguous(), dtype=wp.int32)
        self._frames = [_Frames(self.LINES_NAME, "Show Origin", (0.0, 0.0, 0.0), 1.0, self._n_lines, dev)]
        self._frames[0].enabled = bool(enabled)
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._ui, position="panel")

    @property
    def enabled(self) -> bool:
        return self._frames[0].enabled

    @enabled.setter
    def enabled(self, on: bool) -> None:
        self._frames[0].enabled = bool(on)

    def _ui(self, imgui) -> None:
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("MetaLab"):
            imgui.separator()
            for f in self._frames:
                _changed, f.enabled = imgui.checkbox(f.label, f.enabled)
            if any(f.enabled for f in self._frames):
                _, self.axis_len = imgui.slider_float("Origin Axis (m)", self.axis_len, *_AXIS_LEN_RANGE)

    def draw(self, state) -> None:
        v = self._viewer
        for f in self._frames:
            if not f.enabled:
                if not f.hidden:
                    v.log_lines(f.name, None, None, None)
                    f.hidden = True
                continue
            wp.launch(
                kernel=_origin_axis_lines,
                dim=self._n_lines,
                inputs=[self._body_idx, state.body_q, v.model.body_world,
                        v.world_offsets, v.layer.xform, getattr(v, "_visible_worlds_mask", None),
                        self.axis_len, f.offset, f.color_scale],
                outputs=[f.starts, f.ends, f.colors],
                device=f.starts.device,
            )
            v.log_lines(f.name, f.starts, f.ends, f.colors)
            f.hidden = False


class InstanceScaleSync:
    def __init__(self, viewer, model, shape_idx: torch.Tensor):
        assert hasattr(viewer, "_shape_instances"), \
            f"{type(viewer).__name__} has no ._shape_instances — the instance-scale hook is stale for this newton"
        self._viewer = viewer
        self._model = model
        self._shape_idx = shape_idx
        self._src = None
        self._map: list[tuple] = []
        self._build()

    def _build(self) -> None:
        batches = self._viewer._shape_instances
        dev = self._shape_idx.device
        model_scale = wp.to_torch(self._model.shape_scale)
        self._map = []
        for b in batches.values():
            ms = torch.as_tensor(list(b.model_shapes), dtype=torch.long, device=dev)
            rows = torch.isin(ms, self._shape_idx).nonzero(as_tuple=True)[0]
            if len(rows):
                shapes = ms[rows]
                scales = wp.to_torch(b.scales)
                self._map.append((scales, rows, shapes, scales[rows].clone(), model_scale[shapes].clone()))
        self._src = batches

    def refresh(self) -> int:
        if self._viewer._shape_instances is not self._src:
            self._build()
        now = wp.to_torch(self._model.shape_scale)
        n = 0
        for scales, rows, shapes, scales0, base in self._map:
            ratio = torch.where(base != 0.0, now[shapes] / base, torch.ones_like(base))
            scales[rows] = scales0 * ratio
            n += len(rows)
        return n


MOUSE_PICK_SCALE = 0.3
_PICK_FIELDS = ("pick_stiffness", "pick_damping", "pick_max_acceleration")


def scale_pick_force(viewer, scale: float = MOUSE_PICK_SCALE) -> dict[str, float]:
    assert scale > 0.0, f"pick force scale must be > 0 (got {scale})"
    picking = getattr(viewer, "picking", None)
    assert picking is not None, \
        "viewer has no .picking — call scale_pick_force() after set_model(), or newton's viewer changed"
    assert hasattr(picking, "pick_state"), \
        "newton Picking has no .pick_state — the mouse-grab force hook is stale for this newton version"
    st = picking.pick_state.numpy()
    out = {}
    for f in _PICK_FIELDS:
        assert f in st.dtype.names, f"newton PickingState has no '{f}' — mouse-grab force hook is stale"
        out[f] = float(st[0][f]) * scale
        st[0][f] = out[f]
        setattr(picking, f, out[f])
    picking.pick_state.assign(st)
    return out


def log_world_labels(viewer, height: float = 0.9) -> int:
    offs = getattr(viewer, "world_offsets", None)
    return 0 if offs is None else _log_world_labels(offs.numpy(), height)


CONTACT_ARROW_PATH = "/metalab/contact_normals"
CONTACT_ARROW_M_PER_N = 0.0015
CONTACT_ARROW_MAX_M = 0.18
_TIP_COLORS = ((230, 74, 60), (46, 204, 113), (52, 120, 246), (240, 150, 40), (160, 90, 220),
               (30, 200, 200), (230, 120, 190), (150, 150, 150))
_ARROW_STATS = {"frames": 0, "arrows": 0, "peak_n": 0.0, "sum_n": 0.0}


def contact_arrow_summary() -> str:
    st = _ARROW_STATS
    if not st["frames"]:
        return "no frames"
    mean = st["sum_n"] / st["arrows"] if st["arrows"] else 0.0
    return (f"{st['arrows']} arrows over {st['frames']} frames ({st['arrows'] / st['frames']:.1f}/frame, "
            f"|normal| mean {mean:.2f} N / peak {st['peak_n']:.2f} N -> "
            f"{mean * CONTACT_ARROW_M_PER_N * 100:.1f} cm typical, cap {CONTACT_ARROW_MAX_M * 100:.0f} cm)")


def log_contact_arrows(arrows) -> int:
    if arrows is None:
        return 0
    origins, forces, tips = arrows
    f = np.asarray(forces, dtype=np.float32)
    _ARROW_STATS["frames"] += 1
    _ARROW_STATS["arrows"] += len(f)
    if len(f):
        m = np.linalg.norm(f, axis=1)
        _ARROW_STATS["peak_n"] = max(_ARROW_STATS["peak_n"], float(m.max()))
        _ARROW_STATS["sum_n"] += float(m.sum())
    v = f * CONTACT_ARROW_M_PER_N
    mag = np.linalg.norm(v, axis=1, keepdims=True)
    over = (mag > CONTACT_ARROW_MAX_M).squeeze(-1)
    if over.any():
        v[over] *= CONTACT_ARROW_MAX_M / mag[over]
    rr.log(CONTACT_ARROW_PATH,
           rr.Arrows3D(origins=np.asarray(origins, dtype=np.float32), vectors=v,
                       colors=[_TIP_COLORS[int(i) % len(_TIP_COLORS)] for i in tips],
                       radii=0.0015))
    return len(v)


class NewtonViewer:
    def __init__(self, gl, rerun, rerun_dt: float, num_envs: int, device, model,
                 obj_body_idx: torch.Tensor | None, obj_shapes: torch.Tensor | None):
        self.gl = gl
        self.rerun = rerun
        self._rerun_dt = float(rerun_dt)
        self._rerun_step = 0
        self._rerun_offsets = None
        self._num_envs = int(num_envs)
        self._device = device
        self.origin_axes = (OriginAxes(gl, obj_body_idx)
                            if gl is not None and obj_body_idx is not None and obj_body_idx.numel() > 0 else None)
        self._scale_syncs = ([InstanceScaleSync(v, model, obj_shapes) for v in (gl, rerun) if v is not None]
                             if obj_shapes is not None else [])

    def step_allowed(self) -> bool:
        return True if self.gl is None else bool(self.gl.should_step())

    def apply_pick_forces(self, state) -> None:
        if self.gl is not None:
            self.gl.apply_forces(state)

    def refresh_scales(self) -> None:
        for sync in self._scale_syncs:
            sync.refresh()

    def rerun_world_offsets(self) -> torch.Tensor:
        if self._rerun_offsets is None:
            o = getattr(self.rerun, "world_offsets", None)
            arr = (np.zeros((self._num_envs, 3), np.float32) if o is None
                   else np.asarray(o.numpy()).reshape(-1, 3)[:self._num_envs])
            self._rerun_offsets = torch.as_tensor(arr, dtype=torch.float32, device=self._device)
        return self._rerun_offsets

    def emit(self, state, sim_time: float, contact_arrows, advance: bool = False) -> None:
        for v, axes in ((self.gl, self.origin_axes), (self.rerun, None)):
            if v is None:
                continue
            v.begin_frame(self._rerun_step * self._rerun_dt if v is self.rerun else sim_time)
            if v is self.rerun:
                mark_step(self._rerun_step)
            v.log_state(state)
            if v is self.rerun:
                log_contact_arrows(contact_arrows())
            if axes is not None:
                axes.draw(state)
            v.end_frame()
        if advance and self.rerun is not None:
            self._rerun_step += 1

    def focus_env(self, cam, env_idx: int) -> None:
        if self.gl is None or cam is None:
            return
        viewer_cam = getattr(self.gl, "camera", None)
        if viewer_cam is None:
            return
        offs = getattr(self.gl, "world_offsets", None)
        off = (np.zeros(3, dtype=np.float32) if offs is None or int(env_idx) < 0
               else np.asarray(offs.numpy()).reshape(-1, 3)[int(env_idx)])
        eye = np.asarray(cam.eye, dtype=np.float32) + off
        lookat = np.asarray(cam.lookat, dtype=np.float32) + off
        viewer_cam.pos = type(viewer_cam.pos)(*(float(v) for v in eye))
        viewer_cam.look_at([float(v) for v in lookat])
        print(f"[telemetry] focus env {env_idx}: eye={eye.round(2).tolist()} "
              f"lookat={lookat.round(2).tolist()}", flush=True)

    def close(self) -> str | None:
        if self.rerun is None:
            return None
        self.rerun.close()
        return contact_arrow_summary()
