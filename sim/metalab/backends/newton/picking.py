"""Mouse-grab (viewer picking) force scaling for the newton spoke.

WHY. newton's viewer lets you right-drag a body around; the force is a spring-damper to the mouse ray
(``newton/_src/viewer/picking.py``). Its defaults are tuned for tabletop props, not for a 44-DoF humanoid:

    F = (10 + m_body) * (pick_stiffness * dx - pick_damping * v)        # kernels.py apply_picking_force_kernel
    |F| <= pick_max_acceleration * 9.81 * m_effective                    # m_effective = whole ARTICULATION mass

With the stock ``pick_stiffness=50``, the ``10 + m`` multiplier makes the effective spring ~500 N/m, and the
clamp is computed from ALLEX's full 43.7 kg articulation (~2145 N) so it never engages: dragging a fingertip
10 cm applies ~50 N to that fingertip — orders of magnitude past what the real finger can produce, so the
robot gets thrown around. Scaling all three parameters by the same factor weakens the grab uniformly, for
articulated links (spring-limited) and free objects (clamp-limited) alike.

WHY REWRITE THE STRUCT. ``Picking.__init__`` COPIES the three parameters into the warp struct array
``pick_state`` (picking.py:74-76) and the kernel only ever reads them from there — there is no setter, and
assigning ``picking.pick_stiffness`` alone changes nothing. So the knob has to write ``pick_state`` itself.
The python attributes are updated too, so introspection cannot disagree with the kernel.
"""
from __future__ import annotations

MOUSE_PICK_SCALE = 0.3   # ~3x weaker than newton's defaults: k 50->15, d 5->1.5, clamp 5g->1.5g

_FIELDS = ("pick_stiffness", "pick_damping", "pick_max_acceleration")


def scale_pick_force(viewer, scale: float = MOUSE_PICK_SCALE) -> dict[str, float]:
    """Multiply the viewer's mouse-grab spring, damper and force clamp by ``scale``. Returns the new values.

    Call AFTER ``viewer.set_model(model)`` — that is where ``ViewerGL`` builds its ``Picking``. Fails loud if
    newton's picking surface moved, rather than silently leaving the stock (far too strong) force in place."""
    assert scale > 0.0, f"pick force scale must be > 0 (got {scale})"
    picking = getattr(viewer, "picking", None)
    assert picking is not None, \
        "viewer has no .picking — call scale_pick_force() after set_model(), or newton's viewer changed"
    assert hasattr(picking, "pick_state"), \
        "newton Picking has no .pick_state — the mouse-grab force hook is stale for this newton version"
    st = picking.pick_state.numpy()          # structured array: the kernel's only source for these values
    out = {}
    for f in _FIELDS:
        assert f in st.dtype.names, f"newton PickingState has no '{f}' — mouse-grab force hook is stale"
        out[f] = float(st[0][f]) * scale
        st[0][f] = out[f]
        setattr(picking, f, out[f])          # keep the python-side attribute honest (kernel reads the struct)
    picking.pick_state.assign(st)
    return out
