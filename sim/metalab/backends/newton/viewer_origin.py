"""``Show Origin`` viewer overlay for the newton spoke — each object's ORIGIN frame as 3 RGB axes.

WHY. newton's viewer can draw JOINT frames ("Show Joints") and centers of mass ("Show Center of Mass"), but
not body-ORIGIN frames — and the origin is the frame every metalab read is expressed in: ``object_pos()`` is
the MJCF body frame origin (measured, NOT the COM: the hammer's mesh bottom sits 1.2 mm below it while its
COM is ~16 mm above), the goal keypoint cage is built around it, and the asset pipeline puts it ON the spot
the hand grasps. Without this, "where exactly is the object's origin and which way do its axes point" was
only answerable by printing numbers.

Deliberately generic: a frame per object body and nothing task-specific. This is the DEBUG aid for checking
that a newly converted asset carries its origin where the task expects — it is what confirmed the grasp-point
alignment — and it is not part of any headless path.

WHERE IT LANDS IN THE UI. newton opens UI hooks at ``panel`` / ``side`` / ``stats`` / ``free`` / ``rendering``
(``ViewerGui.register_ui_callback``); the INSIDE of its own "Visualization" header — where "Show Joints"
lives — has no hook, it is straight-line code in the ~200-line ``ViewerGui._render_left_panel``. Getting the
checkbox literally under "Show Joints" would mean copying that method into our tree and re-syncing it on
every newton bump, so this registers at ``panel``: an "MetaLab" section in the SAME left panel, just above
"Model Information". Nothing in newton is modified — same wrap-don't-patch approach as ``picking.py``.

Geometry/coordinate handling mirrors newton's own ``compute_joint_basis_lines`` exactly (world offset per
world → layer xform → axis rotation), so the axes land on the bodies in multi-world renders and under a
layer transform, and hidden worlds drop out. The kernel reads viewer state newton keeps public for its own
visualizers (``world_offsets`` / ``layer.xform``) plus ``_visible_worlds_mask``.
"""
from __future__ import annotations

import torch
import warp as wp

_AXIS_LEN_DEFAULT = 0.1     # [m] drawn axis length (10 cm — a hammer is ~0.25 m long)
_AXIS_LEN_RANGE = (0.01, 0.5)
_LINES_PER_BODY = 3
_GRASP_COLOR_SCALE = 0.45   # grasp axes at 45% brightness — same RGB convention, visibly the second frame


@wp.kernel
def _origin_axis_lines(
    body_idx: wp.array(dtype=wp.int32),          # which bodies to draw (model body indices)
    body_q: wp.array(dtype=wp.transform),        # state.body_q
    body_world: wp.array(dtype=wp.int32),        # model.body_world
    world_offsets: wp.array(dtype=wp.vec3),      # viewer.world_offsets (may be empty)
    layer_xform: wp.transform,                   # viewer.layer.xform
    visible_worlds_mask: wp.array(dtype=wp.int32),   # viewer._visible_worlds_mask (may be empty)
    axis_len: float,
    local_offset: wp.vec3,                       # frame origin in the BODY frame (0 = the body origin itself)
    color_scale: float,                          # < 1 dims the axes (tells two frames apart)
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    """One thread per LINE: thread t draws axis (t % 3) of body_idx[t // 3]. X=red, Y=green, Z=blue.
    A body in a hidden world emits a NaN line (newton's convention for "skip this segment")."""
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
    pos = wp.transform_point(tf, local_offset)     # body frame → world (== translation when offset is 0)
    rot = wp.transform_get_rotation(tf)
    if world_offsets and w >= 0:                       # multi-world render: same offset the shapes get
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
    """One toggle + one ``log_lines`` batch = one frame drawn per object body."""

    def __init__(self, name: str, label: str, offset, color_scale: float, n_lines: int, dev):
        self.name, self.label = name, label
        self.offset = wp.vec3(*(float(c) for c in offset))
        self.color_scale = float(color_scale)
        self.enabled = False
        self.hidden = True                      # nothing logged yet
        self.starts = wp.zeros(n_lines, dtype=wp.vec3, device=dev)
        self.ends = wp.zeros(n_lines, dtype=wp.vec3, device=dev)
        self.colors = wp.zeros(n_lines, dtype=wp.vec3, device=dev)


class OriginAxes:
    """The ``Show Origin`` / ``Show Grasp`` toggles + their per-frame line batches.

    Owns nothing of newton's: the checkboxes are one registered UI callback and each frame is a ``log_lines``
    batch under our own name, so all of it vanishes if this object is never created. The backend calls
    :meth:`draw` once per rendered frame, between ``log_state`` and ``end_frame``."""

    LINES_NAME = "/metalab/origin_axes"
    GRASP_LINES_NAME = "/metalab/grasp_axes"

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
        # The imgui left panel exists only on the GL/GUI viewers. A viewer without one (ViewerRerun) simply
        # gets no checkbox — `enabled` then stays whatever the caller passed, and `draw` itself is
        # viewer-agnostic (log_lines), so such a viewer can still be driven from code.
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self._ui, position="panel")

    @property
    def enabled(self) -> bool:                  # the origin toggle, kept as the object's own flag
        return self._frames[0].enabled

    @enabled.setter
    def enabled(self, on: bool) -> None:
        self._frames[0].enabled = bool(on)

    # --- UI (left panel, "MetaLab" section) ---------------------------------------------------------
    def _ui(self, imgui) -> None:
        """Rendered by newton's left panel. Same widget pair as its own "Show Joints" + "Joint Scale"."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("MetaLab"):
            imgui.separator()
            for f in self._frames:
                _changed, f.enabled = imgui.checkbox(f.label, f.enabled)
            if any(f.enabled for f in self._frames):
                _, self.axis_len = imgui.slider_float("Origin Axis (m)", self.axis_len, *_AXIS_LEN_RANGE)

    # --- per-frame draw -----------------------------------------------------------------------------
    def draw(self, state) -> None:
        """Log (or clear) the axis lines for this frame. Cheap no-op while the boxes are unchecked."""
        v = self._viewer
        for f in self._frames:
            if not f.enabled:
                if not f.hidden:                       # clear once, then stay quiet
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


def add_origin_axes(viewer, body_idx: torch.Tensor | None, **kw) -> OriginAxes | None:
    """Attach the ``Show Origin`` overlay to a newton viewer. ``None``/empty bodies (a scene with no movable
    object) → no overlay and no checkbox, rather than an empty toggle that does nothing."""
    if viewer is None or body_idx is None or body_idx.numel() == 0:
        return None
    return OriginAxes(viewer, body_idx, **kw)
