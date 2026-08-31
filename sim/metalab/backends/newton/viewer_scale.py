"""Per-instance geometry-scale sync for the newton viewers (``--viz`` GL/RTX and the .rrd recorder).

WHY. newton reads ``model.shape_scale`` exactly ONCE — in ``ViewerBase._populate_shapes`` at ``set_model`` —
and ``ShapeInstances.finalize`` freezes it into that batch's ``scales`` array. The per-frame path only
recomputes transforms (``update_shape_xforms``), so a runtime scale write never reached the picture: with
``randomize_object_scale`` on, every env still drew the BUILD-TIME size in the viewer and in the recording,
which is exactly the domain randomization the recording exists to show (measured: 4 hammers, 4 different
physical sizes, one identical picture).

Nothing needs re-populating, though: both viewers re-send the array every frame — ``log_state`` passes
``shapes.scales`` to ``log_instances`` unconditionally ("Always pass scales"), ViewerGL forwards it to
``MeshInstancerGL.update_from_transforms`` and ViewerRerun puts it in ``InstancePoses3D``. So writing INTO
that array in place is enough, and that is all this module does.

RATIO, not the raw value. ``_populate_shapes`` uses the instance scale differently per geometry type: MESH /
CONVEX_MESH carry ``shape_scale`` as their instance scale (the prototype holds the asset's raw vertices),
while primitives (box, sphere, capsule…) BAKE the size into a prototype hashed on it and take instance scale
1. Copying ``shape_scale`` in verbatim would therefore be right for the hammer meshes and wrong by the base
size for every primitive. Scaling each instance by ``shape_scale_now / shape_scale_at_build`` is correct for
both, and it is also what keeps a rebuilt batch honest — newton re-populates from the CURRENT scale, so the
baseline is re-captured with it.
"""
from __future__ import annotations

import torch
import warp as wp


class InstanceScaleSync:
    """Push ``model.shape_scale`` changes for a fixed shape set into one viewer's instance batches.

    Built once per viewer, right after ``set_model`` (the backend does this), and refreshed by the backend
    immediately after a scale write. The shape → instance-row map is rebuilt whenever the viewer replaces its
    batches (``set_visible_worlds`` assigns a fresh ``_shape_instances`` dict), so it cannot go stale."""

    def __init__(self, viewer, model, shape_idx: torch.Tensor):
        assert hasattr(viewer, "_shape_instances"), \
            f"{type(viewer).__name__} has no ._shape_instances — the instance-scale hook is stale for this newton"
        self._viewer = viewer
        self._model = model
        self._shape_idx = shape_idx                  # (S,) model shape indices this sync owns
        self._src = None                             # the batch dict the current map was built from
        self._map: list[tuple] = []                  # [(scales view, rows, model shapes, scales0, scale0_model)]
        self._build()                                # eager: the baseline must be the pre-DR state

    def _build(self) -> None:
        batches = self._viewer._shape_instances
        dev = self._shape_idx.device
        model_scale = wp.to_torch(self._model.shape_scale)
        self._map = []
        for b in batches.values():
            # model_shapes stays a python list of model shape indices (one per instance, batch order).
            ms = torch.as_tensor(list(b.model_shapes), dtype=torch.long, device=dev)
            rows = torch.isin(ms, self._shape_idx).nonzero(as_tuple=True)[0]
            if len(rows):
                shapes = ms[rows]
                scales = wp.to_torch(b.scales)
                # Baselines captured together, so the ratio below is 1 at this instant whenever it is rebuilt.
                self._map.append((scales, rows, shapes, scales[rows].clone(), model_scale[shapes].clone()))
        self._src = batches

    def refresh(self) -> int:
        """Re-scale every instance of our shapes by shape_scale/shape_scale_at_build. Returns rows written."""
        if self._viewer._shape_instances is not self._src:
            self._build()
        now = wp.to_torch(self._model.shape_scale)   # (nshape, 3) zero-copy view; the model owns the buffer
        n = 0
        for scales, rows, shapes, scales0, base in self._map:
            # A zero component means the axis carries no size to scale (newton encodes infinite planes that
            # way) — leave it alone instead of dividing by it.
            ratio = torch.where(base != 0.0, now[shapes] / base, torch.ones_like(base))
            scales[rows] = scales0 * ratio
            n += len(rows)
        return n


def make_scale_syncs(viewers, model, shape_idx: torch.Tensor) -> list[InstanceScaleSync]:
    """One sync per attached viewer (``None`` entries skipped) — empty list headless, which no-ops the refresh."""
    return [InstanceScaleSync(v, model, shape_idx) for v in viewers if v is not None]
