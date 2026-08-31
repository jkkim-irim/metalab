"""newton-side rerun overlays: what the fingertips are doing, drawn into the recording.

The rerun plumbing itself — the ``step`` timeline, the blueprint, the file sink, the batcher, the ``env<N>``
labels — is engine-agnostic and lives in :mod:`sim.metalab.runtime.rerun_recording`; genesis shares it. Only
the CONTACT arrows are here, because their data comes from newton's ``SensorContact``.
"""
from __future__ import annotations

import numpy as np
import rerun as rr

# Re-exported so the newton parser/backend/server keep one import site for "rerun recording things".
from sim.metalab.runtime.rerun_recording import (  # noqa: F401
    STEP_TIMELINE,
    mark_step,
    no_viewer_client,
    tune_batcher,
)
from sim.metalab.runtime.rerun_recording import (  # noqa: F401
    attach_file_sink as install_rrd_sink,
)
from sim.metalab.runtime.rerun_recording import (
    log_world_labels as _log_world_labels,
)


def log_world_labels(viewer, height: float = 0.9) -> int:
    """``env<N>`` labels for a newton viewer — pulls the tile offsets off it, then defers to the shared
    logger. Returns 0 when the viewer has no offsets (a single-world scene)."""
    offs = getattr(viewer, "world_offsets", None)
    return 0 if offs is None else _log_world_labels(offs.numpy(), height)


CONTACT_ARROW_PATH = "/metalab/contact_normals"
CONTACT_ARROW_M_PER_N = 0.0015
CONTACT_ARROW_MAX_M = 0.18
#: One colour per fingertip, in ``RobotSpec.fingertips`` order (index/middle/ring/little/thumb).
_TIP_COLORS = ((230, 74, 60), (46, 204, 113), (52, 120, 246), (240, 150, 40), (160, 90, 220),
               (30, 200, 200), (230, 120, 190), (150, 150, 150))


#: Rolling tally so a finished recording can say whether contacts were ever drawn — the same reason the
#: recorder prints its clip count. "0 arrows over 600 frames" is a real answer (fingers never closed), and
#: without it that is indistinguishable from a broken read.
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
    """Draw one arrow per touching fingertip: origin at the contact point, direction/length = normal force.

    ``arrows`` is ``NewtonBackend._fingertip_contact_arrows()``'s ``(origins, forces, tip_idx)`` or None.
    Returns how many were drawn.

    ALWAYS logs, even with nothing to draw: rerun is latest-at, so skipping a frame would leave the previous
    frame's arrows on screen for as long as the fingers are open."""
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
    if over.any():                                         # cap the LENGTH, keep the direction
        v[over] *= CONTACT_ARROW_MAX_M / mag[over]
    rr.log(CONTACT_ARROW_PATH,
           rr.Arrows3D(origins=np.asarray(origins, dtype=np.float32), vectors=v,
                       colors=[_TIP_COLORS[int(i) % len(_TIP_COLORS)] for i in tips],
                       radii=0.0015))
    return len(v)


