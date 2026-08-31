"""Engine-agnostic rerun recording plumbing: the timeline, the blueprint, the file sink, the batcher.

Everything here is about how a ``.rrd`` is FRAMED and STORED, not about what is in it — so both spokes share
it. newton drives its recording through newton's own ``ViewerRerun`` (which already logs geometry and poses);
genesis has no viewer abstraction and logs them itself. Only the ``open_*`` entry points differ, and only
because of the sink-replacement trap documented on :func:`attach_file_sink`.

THE AXIS. Recordings are keyed to :data:`STEP_TIMELINE`, a SEQUENCE timeline whose value is the policy-step
index, and the blueprint pins that timeline plus its ``fps``. Both halves are needed. A recording also carries
rerun's automatic ``log_tick`` (one tick per log CALL — measured 51 161 for a 600-step, 4-world rollout) and
``log_time``; whichever timeline is active is what the viewer renders AND what its panel counts, so keying the
report to one axis while the picture ran on another is what kept plots and scene out of step. And a sequence
timeline only replays at the right SPEED once ``fps`` is pinned to the control rate — an earlier attempt
without the blueprint played at rerun's default sequence rate and looked wrong.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import rerun as rr

#: Micro-batching thresholds for RECORDING. rerun's defaults (``RERUN_FLUSH_TICK_SECS`` 0.2 s,
#: ``RERUN_FLUSH_NUM_BYTES`` 1 MiB) assume a fast log loop; a recording rollout is far slower than that
#: (measured ~156 ms/step when the mp4 render still shared it), so the 200 ms tick fires every 1-2 steps and
#: the file fills with tiny
#: chunks. Measured on an identical 120-step / 10 360-row rollout: **7 725 chunks · 27.8 MB** with the
#: defaults vs **670 chunks · 14.9 MB** with these — byte-for-byte the same data, 46 % smaller.
_BATCHER_ENV = {"RERUN_FLUSH_TICK_SECS": "3.0", "RERUN_FLUSH_NUM_BYTES": str(16 * 1024 * 1024)}

STEP_TIMELINE = "step"
APP_ID = "metalab"


def tune_batcher() -> None:
    """Raise rerun's micro-batching thresholds (:data:`_BATCHER_ENV`) for a recording run.

    MUST be called before the ``rr.init`` that creates the recording stream — the batcher belongs to that
    stream and rerun reads these once, there. ``setdefault`` so an operator's own values win. Recording only:
    on a live viewer a 3 s tick would show up as 3 s of lag."""
    for k, v in _BATCHER_ENV.items():
        os.environ.setdefault(k, v)


def mark_step(index: int) -> None:
    """Stamp the current frame with its policy-step index on :data:`STEP_TIMELINE`.

    Call before the frame's data goes out: rerun attaches whatever times are set when a component is logged."""
    rr.set_time(STEP_TIMELINE, sequence=int(index))


def blueprint(fps: float):
    """3D view + a time panel pinned to :data:`STEP_TIMELINE` at the policy rate.

    This is what makes a viewer agree with the report: it pins the timeline (so the panel counts STEPS, not log
    ticks), the sequence ``fps`` (so playback is real time), ``Playing`` (rerun's default is ``Following``,
    which nails the cursor to the newest data and freezes the scene on the last frame) and looping. Panels are
    collapsed/hidden — the report's pane is small and supplies its own controls."""
    import rerun.blueprint as rrb
    return rrb.Blueprint(
        rrb.Spatial3DView(),
        rrb.TimePanel(state="Collapsed", timeline=STEP_TIMELINE, fps=float(fps),
                      playback_speed=1.0, play_state="Playing", loop_mode="All"),
        rrb.BlueprintPanel(state="Hidden"),
        rrb.SelectionPanel(state="Hidden"),
        collapse_panels=True,
    )


def _ensure_dir(rrd_path: str) -> None:
    """The sink is the FIRST writer into the recording dir — the rollout log only opens its data.json at
    shutdown — and rerun fails the open outright ("No such file or directory") if the dir is not there yet."""
    os.makedirs(os.path.dirname(os.path.abspath(rrd_path)), exist_ok=True)


def open_recording(rrd_path: str, *, fps: float) -> list[str]:
    """Start a file-only recording for an engine that has NO rerun viewer of its own (the genesis path).

    Nothing launches a client here, so there is no sink to fight: ``rr.init`` then one ``FileSink``."""
    tune_batcher()                                     # before rr.init — the batcher is read there
    _ensure_dir(rrd_path)
    rr.init(APP_ID)
    rr.set_sinks(rr.FileSink(rrd_path), default_blueprint=blueprint(fps))
    return [f"file:{rrd_path}"]


#: rerun's client launchers, stubbed for the duration of :func:`no_viewer_client`.
_RR_LAUNCHERS = ("serve_grpc", "serve_web_viewer", "spawn")


@contextlib.contextmanager
def no_viewer_client():
    """Build a newton ``ViewerRerun`` WITHOUT it launching a rerun client. Recording paths only.

    ``ViewerRerun.__init__`` always launches one outside Jupyter: with its default ``serve_web_viewer=True``
    it runs ``serve_grpc`` (9876) + ``serve_web_viewer`` (9090), and the latter OPENS A BROWSER TAB at
    ``http://127.0.0.1:9090/?url=rerun+http://127.0.0.1:9876/proxy``; with False it spawns a native viewer
    window instead. A recording wants neither — it wants a file — and the trainer builds one viewer PER
    CHECKPOINT, so every report popped a new tab and re-bound both ports. Stubbing the three launchers for
    the length of the constructor is the smallest hook that leaves the vendored newton unpatched: the viewer
    still ``rr.init``s and logs exactly as before, and :func:`attach_file_sink` then points the stream at the
    ``.rrd``. ``_grpc_server_uri`` comes out None, which the ``keep_live`` tee there already handles.

    NOT for ``--viz rerun``: that viewer IS the client the operator asked for."""
    saved = {n: getattr(rr, n) for n in _RR_LAUNCHERS}
    for n in saved:
        setattr(rr, n, lambda *a, **k: None)
    try:
        yield
    finally:
        for n, fn in saved.items():
            setattr(rr, n, fn)


def attach_file_sink(viewer, rrd_path: str, *, fps: float, keep_live: bool = False) -> list[str]:
    """Re-install a newton ``ViewerRerun``'s sinks so ``rrd_path`` really receives the recording.

    WHY this exists at all: rerun sinks REPLACE one another — ``rr.set_sinks``' own docs say so — and
    ``ViewerRerun.__init__`` calls ``rr.save(record_to_rrd)`` and THEN launches a client
    (``serve_grpc``/``spawn``), which replaces the file sink it just installed. Measured: a 300-step, 4-world
    rollout produced a 28 KB / 8-row ``.rrd``, one row per entity path, nothing to replay. So the viewer is
    constructed as usual and the sinks are re-installed afterwards. Nothing in newton is patched.

    File-only by default: ``keep_live=True`` tees the gRPC sink alongside the file, and a process that does
    BLOCKS ON EXIT (measured: 300 steps wrote the file in ~20 s, then hung >10 min in teardown with no client
    attached). Recording is the point of passing an rrd path, so the file wins until that is understood.

    Fails loud if the viewer is not a rerun one, or if newton stopped exposing the gRPC URI the live half
    needs — a silent fallback here is what produced the empty recordings above."""
    assert hasattr(viewer, "keep_historical_data"), \
        f"attach_file_sink expects a newton ViewerRerun, got {type(viewer).__name__}"
    assert viewer.keep_historical_data, \
        ("ViewerRerun(keep_historical_data=False) logs every transform with static=True, which carries no "
         "timeline — the .rrd would replay as one frozen frame. Construct it with keep_historical_data=True.")
    _ensure_dir(rrd_path)

    sinks: list = [rr.FileSink(rrd_path)]
    names = [f"file:{rrd_path}"]
    if keep_live:
        assert hasattr(viewer, "_grpc_server_uri"), \
            "newton ViewerRerun has no ._grpc_server_uri — the live-tee hook is stale for this version"
        uri = viewer._grpc_server_uri
        if uri:                                        # None when the constructor spawned a native viewer
            sinks.append(rr.GrpcSink(url=uri))
            names.append(f"grpc:{uri}")
    # The blueprint rides the sink so it is stored IN the .rrd: whoever opens the recording gets the step
    # timeline, the policy-rate fps and Playing without having to be told.
    rr.set_sinks(*sinks, default_blueprint=blueprint(fps))
    return names


def log_world_labels(offsets, height: float = 0.9) -> int:
    """Float an ``env<N>`` label over each world. ``offsets`` = (N,3) per-world tile offsets. Returns the count.

    WHY. A recording holds EVERY env at once (both spokes tile them) while the report's plots show one at a
    time, and the envs terminate independently — so without labels you watch one robot reset while another
    env's trace says nothing happened, which reads as a sync bug. Logged ``static`` so the labels stand outside
    the timeline and stay put for the whole replay."""
    if offsets is None:
        return 0
    pos = np.asarray(offsets, dtype=np.float32).reshape(-1, 3).copy()
    pos[:, 2] += float(height)                         # lift clear of the table so the text is readable
    rr.log("/metalab/env_labels",
           rr.Points3D(positions=pos, labels=[f"env{i}" for i in range(len(pos))],
                       radii=0.0, show_labels=True),
           static=True)
    return len(pos)
