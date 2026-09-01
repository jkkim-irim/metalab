"""Standalone sim runner — build ONE task env and step it with the GUI, no policy / no learning.

Drives the engine **backend directly**, NOT the training ``EnvDriver.step`` —
so there is **no termination, no domain randomization, no reward/curriculum, and no automatic reset**
(those are training/eval concerns that live inside ``EnvDriver.step``). The robot holds its init pose
(a zero policy: PD target = contract init pose), OR follows a **cubic-Hermite CSV trajectory** driven from
the SIM tab, while the GL viewer renders the physics (gravity / contact / equality).

Control channel: the runner starts a small ``TrajControlServer`` (``drive/control_server.py``) that the
Launchpad's SIM tab embeds. Browser → runner commands (``play/pause/resume/stop``) are drained each loop; the
runner publishes a live **monitor-channel** snapshot back over SSE for the plots — one value vector per
plot tab (joint position / joint torque, its PD+grav components where the backend separates them / palm
pose), each produced by the very obs term factory the train/eval contracts use (``drive/monitor.py``), so
Standalone plots exactly what a policy observes.
This is SEPARATE from ``telemetry.py`` (env_driver's train/eval RL dashboard) — standalone bypasses env_driver.

Recording: when a trajectory plays to the end (``finished``), the runner auto-saves the whole playback under
``_logs/standalone/<group>/<group>_<YYYYMMDDHHmm>_<engine>/`` as CSVs (first column ``timestamp`` = sim time [s]):
``joint_position.csv`` [deg] (actual joint state read back), ``target_position.csv`` [deg] (commanded PD target
sent to the robot), ``joint_torque.csv`` [Nm] — columns = the robot's driven joints in report order — and
``contact_force.csv`` [N] (per-fingertip net contact-force magnitude, columns = the 5 right fingertips,
measured on their distal collision links). Both engines (genesis/newton) log identically. Stopping/resetting
(or switching to Torque) before the end discards the partial recording.

Reset is **on demand only**: SIGUSR1 (Launchpad's "Reset Simulator") or the SIM tab's Stop → ``backend.reset_idx``
(contract init pose, NO domain randomization). Ctrl-C exits cleanly. Does NOT import ``learning``.
Single-env (num_envs=1) on the default GPU with the GL viewer on.

  # in the engine's uv venv (sim/standalone.sh does the env provisioning + activation)
  python -m sim.metalab.runtime.standalone --engine newton  --task dumbbell-test
  python -m sim.metalab.runtime.standalone --engine genesis --task dumbbell-test \
      --trajectory sim/metalab/assets/data/trajectory/dumbbell/demo2_dumbbell_group   # auto-play on start
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from sim.metalab.drive import ALLEX_CSV_JOINT_NAMES, CsvTrajectory, monitor
from sim.metalab.drive.control_server import TrajControlServer
from sim.metalab.drive.dashboard_page import DESCRIBE_CACHE_REL
from sim.metalab.drive.motor2joint.loaders import DEFAULT_PARAMS_FILE
from sim.metalab.runtime.signals import restore_default_sigint

_RAD2DEG = 180.0 / math.pi
_DEG2RAD = math.pi / 180.0   # dashboard/CSV side is degrees, the engine side is radians
_SIM = Path(__file__).resolve().parents[2]          # <repo>/sim  (this file: sim/metalab/runtime/)
_REPO = _SIM.parent                                 # <repo>
_TRAJ_ROOT = _SIM / "metalab" / "assets" / "data" / "trajectory"

# Right-hand fingertip contact force: (column label, collision link measured). The MJCF ``R_*_Fingertip``
# bodies are visual-only (no collision geom), so each fingertip's net contact force is read on its distal
# collision link — matching the repo's fingertip convention (hammer_lift_teacher `_FINGERTIPS`).
_R_FINGERTIP_CONTACT = [
    ("R_Thumb_Fingertip", "R_Thumb_Distal_Link"),
    ("R_Index_Fingertip", "R_Index_Distal_Link"),
    ("R_Middle_Fingertip", "R_Middle_Distal_Link"),
    ("R_Ring_Fingertip", "R_Ring_Distal_Link"),
    ("R_Little_Fingertip", "R_Little_Distal_Link"),
]

_reset_requested = False   # set by the SIGUSR1 handler; consumed at the top of the step loop (main thread)


def _on_sigusr1(_signum, _frame) -> None:
    global _reset_requested
    _reset_requested = True


def _build(engine: str, task: str):
    """Engine spoke ``build_env`` → EnvDriver, single-env on the default GPU with the GL viewer. Lazy
    import so only the selected engine (which lives in its own uv venv) is imported.
    We use only ``env.backend``/``env.spec`` below (never ``env.step``)."""
    if engine == "genesis":
        from sim.metalab.backends.genesis.server import build_env
    elif engine == "newton":
        from sim.metalab.backends.newton.server import build_env
    else:
        raise SystemExit(f"engine must be genesis|newton (got {engine!r})")
    # viz="gl" → GL 3D viewer ON; telemetry=False → do NOT start env_driver's RL SSE dashboard. Standalone
    # has its own control_server; otherwise the Launchpad would scrape its "[telemetry] live dashboard" line and
    # light the RL tab too (this runner never calls EnvDriver.step, so that dashboard would be dead anyway).
    return build_env(task=task.replace("-", "_"), num_envs=1, device="cuda:0", viz="gl", telemetry=False)


def _discover_groups() -> list[dict]:
    """Via-point CSV groups (``*_group`` dirs) for the SIM tab combobox — ``[{label, path}]`` with
    repo-relative POSIX path (mirrors the Launchpad's ``discover_traj_groups``; served via /describe)."""
    if not _TRAJ_ROOT.is_dir():
        return []
    return [{"label": f"{d.parent.name}/{d.name}", "path": d.relative_to(_REPO).as_posix()}
            for d in sorted(_TRAJ_ROOT.rglob("*_group")) if d.is_dir()]


_ROM_FALLBACK_DEG = 180.0   # span given to a joint the model declares no usable limit for (continuous axis)
_GAIN_WATCH_S = 1.0         # how often the runner re-fingerprints robot_model.json (edit detection)


def _gain_fingerprint() -> str:
    """Content hash of ``robot_model.json``. Content, not mtime: editors touch/rewrite files without
    changing anything, and a false "gains edited" banner is worse than a 1 Hz 16 KB read. Missing file
    (a task with no motor coupling) → empty string, i.e. never dirty."""
    p = Path(DEFAULT_PARAMS_FILE)
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else ""


def _rom_deg(lo_rad: float, hi_rad: float) -> dict:
    """One joint's slider range in degrees, from the engine's (rad) limits. A model may declare no bound on
    an axis (±inf, or the ±1e6 sentinel engines use for "free"): those get a ±180° span and are flagged, so
    the row still has a usable slider instead of one whose extremes are unreachable. Never silently clamps a
    real limit — the flag is what tells the UI the number is ours, not the model's."""
    lo, hi = lo_rad * _RAD2DEG, hi_rad * _RAD2DEG
    bounded = math.isfinite(lo) and math.isfinite(hi) and hi > lo and max(abs(lo), abs(hi)) <= 1e4
    if bounded:
        return {"lo": lo, "hi": hi, "bounded": True}
    return {"lo": -_ROM_FALLBACK_DEG, "hi": _ROM_FALLBACK_DEG, "bounded": False}


def _report_joints(spec) -> list[str]:
    """Stable joint set driven + reported (targets, plots): CSV-map joints the robot owns, in map order.
    = ALLEX_CSV_JOINT_NAMES ∩ active_joints (allex → 48, allex_right → 22)."""
    active = spec.robot.active_joints()
    return [j for names in ALLEX_CSV_JOINT_NAMES.values() for j in names if j in active]


def _write_joint_csv(path: Path, joints: list[str], times: list[float], rows: list[list[float]]) -> None:
    """One CSV: header ``timestamp,<joint...>`` (joints in report order), then one row per control step —
    sim time [s] followed by that step's per-joint values (position rows are [deg], torque rows [Nm])."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", *joints])
        for t, row in zip(times, rows):
            w.writerow([f"{t:.4f}", *(f"{v:.4f}" for v in row)])   # round to 4 decimal places


def _save_trajectory_log(repo: Path, group_path: str, engine: str, times: list[float],
                         series: dict[str, tuple[list[str], list[list[float]]]]) -> Path:
    """Auto-save a finished playback under
    ``_logs/standalone/<group>/<group>_<YYYYMMDDHHmm>_<engine>/`` (``<group>`` = played CSV-group dir name with
    a trailing ``_group`` trimmed; ``<engine>`` = genesis|newton). ``series`` maps each CSV filename to its
    ``(column_labels, rows)`` — written as ``timestamp`` + those columns. Returns the run directory."""
    gname = Path(group_path).name if group_path else ""
    if gname.endswith("_group"):
        gname = gname[: -len("_group")]
    gname = gname or "trajectory"
    out_dir = repo / "_logs" / "standalone" / gname / f"{gname}_{datetime.now().strftime('%Y%m%d%H%M')}_{engine}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, (cols, rows) in series.items():
        _write_joint_csv(out_dir / fname, cols, times, rows)
    return out_dir


def _maybe_autoplot(run_dir: Path, group_path: str) -> None:
    """Right-hand finger logs (group ``hand_<finger>_...``) → auto-render the overlay PNG for the finger's
    TESTED joint: a ``cmc``/``mcp``/``pip`` token in the group name picks ``R_<Finger>_<CMC|MCP|PIP>``
    (e.g. ``hand_ring_mcp_test`` → ``R_Ring_MCP``); no token → PIP (the original convention), so MCP runs
    get their own ``*_MCP_overlay.png`` separate from the PIP ones. The THUMB is special: its driven chain
    is Yaw/CMC/MCP (no PIP) — its groups name the real joint (``hand_thumb_cmc_test`` → ``R_Thumb_CMC``,
    ``hand_thumb_mcp_test`` → ``R_Thumb_MCP``), and a legacy ``pip`` token / no token maps to MCP (the
    thumb's PIP-analog joint), so thumb runs always plot instead of skipping. Best-effort: a plotting error
    (missing column, matplotlib absent, …) is logged and never fails the sim run.
    matplotlib is imported lazily here (optional post-processing dep, not needed by the core runner)."""
    gname = Path(group_path).name if group_path else ""
    if gname.endswith("_group"):
        gname = gname[: -len("_group")]
    parts = gname.split("_")
    if len(parts) < 2 or parts[0] != "hand":
        return                                       # only auto-plot hand_<finger> logs
    jtype = next((p for p in parts[2:] if p in ("cmc", "mcp", "pip")), "pip")
    if parts[1] == "thumb" and jtype == "pip":
        jtype = "mcp"                                # thumb has no PIP; its PIP-analog joint is MCP
    joint = f"R_{parts[1].capitalize()}_{jtype.upper()}"   # e.g. hand_index_mcp_* → R_Index_MCP
    try:
        from sim.metalab.drive.hand_log_plot import plot_hand_log
        png = plot_hand_log(run_dir, joint)          # fingertip defaults to R_<Finger>_Fingertip
        print(f"[standalone] auto-plot ({joint}) → {png}", flush=True)
    except Exception as exc:
        print(f"[standalone] auto-plot skipped ({joint}): {exc}", flush=True)


def run(engine: str, task: str, trajectory_dir: str | None = None) -> None:
    """Step the backend directly until Ctrl-C, driven by the SIM tab's control channel.

    Holds the init pose by default. The dashboard's Play/Pause/Stop drive the SIMULATOR: ``pause`` halts the
    physics loop outright (nothing steps; the viewer keeps getting frames via ``backend.render_frame``),
    ``play`` unfreezes and — if a group is selected and nothing is mid-playback — starts that CsvTrajectory,
    ``stop`` resets to the init pose and leaves the sim frozen. SIGUSR1 (Launchpad 'Reset Simulator') resets in
    place without touching the frozen/running state. Publishes a live joint snapshot each loop, paused too.

    ``joint_target`` (Joint Control tab) hands individual joints to the operator: those columns of the PD
    target are written from the dashboard each step, overriding the trajectory where the two overlap. The
    engine is asked for the joint ROM once (``joint_limits``) so the sliders carry the model's real range.

    The dense trajectory is uploaded to the GPU once on Play and indexed on-device each step
    (``dense_gpu[advance()]``), so the hot loop does no per-step CPU↔GPU copy."""
    global _reset_requested
    env = _build(engine, task)
    b, spec = env.backend, env.spec
    all_mask = torch.ones(b.num_envs, dtype=torch.bool, device=b.device)
    decim = spec.physics.decimation
    control_hz = spec.physics.hz / max(1, decim)   # trajectory is sampled + targets set once per control step
    step_n = getattr(b, "step_n", None)            # newton: whole control step in one CUDA-graph launch

    report = _report_joints(spec)                  # driven + reported joints (stable across play/idle)
    assert report, "no drivable joints (CSV map ∩ active_joints is empty)"
    available = spec.robot.active_joints()

    b.reset_idx(all_mask)                          # spawn at the contract init pose (NO domain randomization)
    cur = b.joint_pos(report).clone()              # (N, k) current PD target [rad]; idle = init pose

    # Position (default) vs Torque mode (SIM-tab toggle). Gravcomp is always on underneath both. Position =
    # gravcomp + PD position control (tracks targets; trajectory plays here). Torque = PD neutralized on the
    # gravcomp joints (their target follows the current pose each step → zero position error → no PD drive =
    # float on gravity feedforward alone). NOTE the float is NOT an ideal joint-level cancellation: on
    # control_mode:motor robots the coupled joints' gravcomp share flows THROUGH the motor-space fold
    # (τ_m += G⁻ᵀ·τ_g, clamped to the real torque-speed envelope — see motor_coupling.py), so the float
    # droops where the motors lack budget; only passive joints (waist) keep plain joint-level gravcomp.
    # Trajectory playback is allowed ONLY in Position mode.
    _gc = getattr(spec.robot, "gravcomp", None)     # gravcomp contract absent on this branch → torque mode hidden
    _gc_joints = _gc.joints() if _gc is not None else []
    gc_cols = torch.tensor([report.index(j) for j in _gc_joints if j in report],
                           device=b.device, dtype=torch.long)   # columns of `report` that float in torque mode
    torque_mode = False

    settle_s = float(os.environ.get("METALAB_SETTLE_S", "0.4"))            # reset settle cap [s]; 0 = off
    settle_first_s = float(os.environ.get("METALAB_SETTLE_FIRST_S", "2.0"))  # initial-spawn cap [s]
    _SETTLE_VEL_EPS = 0.05                          # rad/s: robot "at rest" → stop settling early

    def _settle(cap_s: float) -> None:
        """Advance physics with the viewer frame suppressed so the reset transient (contact settling +
        uncompensated fingers/hand) decays OFF-SCREEN — the first shown frame is already at rest. Holds the
        init target ``cur``, and stops early once the robot is at rest (max joint speed < eps for a few steps)
        or ``cap_s`` elapses. The FIRST spawn needs a longer cap: the solver starts fully cold (no warm-start /
        contact islands), so its transient is larger and longer-lived than a warm reset's."""
        rest = 0
        for _ in range(round(cap_s * control_hz)):
            b.set_joint_targets(report, cur)
            if step_n is not None:
                step_n(decim, render=False)
            else:
                for _ in range(decim):
                    b.step(render=False)
            rest = rest + 1 if float(b.joint_vel(report).abs().max()) < _SETTLE_VEL_EPS else 0
            if rest >= 3:                          # settled for 3 consecutive control steps → done
                break

    _settle(settle_first_s)                        # initial spawn: longer cap (cold solver → worst transient)

    # gravcomp initial state for the SIM-tab toggle: bool if the task has a gravcomp contract, else None (hide it)
    gravcomp0 = getattr(b, "_gc_on", False) if _gc is not None else None

    # Dashboard plot channels — one per tab, each a bound obs term from sim/metalab/terms/obs (the SAME
    # factories the train/eval contracts reference), evaluated with the backend as `state`. Adding a plot
    # tab = adding a channel in drive/monitor.py; nothing else here changes.
    channels = monitor.build_channels(spec, report, b)

    # Joint Control rows: the SAME joint set the loop drives (`report`), with each joint's ROM read off the
    # engine (both spokes return (J,) rad limits) and its init pose as the starting value. Degrees on the
    # wire and in the UI, radians everywhere inside — the one conversion boundary is here and in the
    # joint_target command below, matching how the CSV trajectories and the joint_pos channel already work.
    _lo, _hi = b.joint_limits(report)
    controls = [{"name": j, **_rom_deg(float(_lo[i]), float(_hi[i])),
                 "init": float(cur[0, i]) * _RAD2DEG} for i, j in enumerate(report)]

    # standalone control + telemetry server (the Launchpad SIM tab embeds this URL; browser talks to it directly)
    describe = {"engine": engine, "task": task, "joints": report, "channels": monitor.describe(channels),
                "groups": _discover_groups(), "control_hz": control_hz, "gravcomp": gravcomp0,
                "controls": controls}
    # Cache it for the Launchpad's OFFLINE Standalone preview (/simui): with no run alive the Launchpad
    # serves the same dashboard page with this channel set, so its tabs/series are real (just no data).
    cache = _REPO / DESCRIBE_CACHE_REL
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(describe))
    srv = TrajControlServer(describe)

    # Right-hand fingertip contact-force channel: net contact force is read on each fingertip's distal collision
    # link (the fingertip bodies are visual-only). Bodies exist for any ALLEX load regardless of joint mask.
    fingertip_labels = [lbl for lbl, _ in _R_FINGERTIP_CONTACT]
    fingertip_links = [lnk for _, lnk in _R_FINGERTIP_CONTACT]

    # Trajectory recording: accumulate (sim time; actual joint pos [deg]; commanded target [deg]; joint torque
    # [Nm]; fingertip contact-force magnitude [N]) each control step while a trajectory plays; on `finished` write
    # the CSVs (_save_trajectory_log). Stop/reset/Torque before the end discards it. `rec` is a dict so the nested
    # play/reset helpers mutate it without a `nonlocal` for each field.
    dt_ctrl = 1.0 / control_hz                     # sim time advanced per control step [s] (CSV timestamp delta)
    rec = {"active": False, "saved": False, "t": 0.0, "group": None,
           "time": [], "pos": [], "tgt": [], "trq": [], "cf": []}

    def _rec_start(group_path: str) -> None:       # begin capturing a fresh playback
        rec.update(active=True, saved=False, t=0.0, group=group_path,
                   time=[], pos=[], tgt=[], trq=[], cf=[])

    def _rec_discard() -> None:                    # abandon a partial (unfinished) recording
        rec.update(active=False, saved=False, t=0.0, time=[], pos=[], tgt=[], trq=[], cf=[])

    traj: CsvTrajectory | None = None
    traj_cols: torch.Tensor | None = None          # columns of `report` the trajectory drives (may be a subset)
    dense_gpu: torch.Tensor | None = None          # (n_samples, len(traj_cols)) rad, GPU-resident → indexed per step
    paused = False
    group_label: str | None = None
    # Joint Control (dashboard tab): the joints the operator has taken over by hand. Rebuilt as GPU tensors
    # on each joint_target command — NOT per step — so the hot loop stays two index writes, not a dict walk.
    man_cols = torch.zeros(0, device=b.device, dtype=torch.long)
    man_vals = torch.zeros(0, device=b.device, dtype=cur.dtype)
    # Motor-gain hot reload: gains are edited in robot_model.json while a run is up (wrist/finger tuning).
    # Applying them mid-flight would step the control law under a moving robot, so the file is only WATCHED
    # here — the dashboard says "edited, pending" and the next reset is what swaps them in.
    gain_hash = _gain_fingerprint()                # fingerprint of what the buffers currently hold
    gain_dirty = False                             # file differs from it → banner on the dashboard
    gain_next_check = 0.0
    # Gain-consistency warnings for what the buffers hold (e.g. a differential group whose two motors no
    # longer share a gain → a cross term appears in K_q). Cached: only a reload can change them.
    gain_warn = b.motor_gain_warnings()

    def _snapshot() -> dict:
        """One dashboard frame. Published from BOTH loop branches (stepping and frozen) — a paused sim must
        keep reporting or the page cannot tell 'frozen' from 'the run died'. ``paused`` is the SIMULATOR's
        state (physics halted), not a trajectory-only flag."""
        return {
            "t": traj.elapsed_s if traj is not None else 0.0,
            "duration": traj.duration_s if traj is not None else 0.0,
            "playing": traj is not None, "paused": paused,
            "finished": bool(traj.finished) if traj is not None else False,
            "group": group_label,
            "gains_dirty": gain_dirty,                 # robot_model.json edited → applies on the next reset
            "gain_warn": gain_warn,                    # gains loaded but inconsistent (see motor_gain_warnings)
            "ch": monitor.sample(channels, b),         # {channel key: values in display units}
        }

    def _seed_dict() -> dict:
        pos = b.joint_pos(report)                  # (N, k) current pose → ramp-in seed per joint
        return {j: float(pos[0, i]) for i, j in enumerate(report)}

    def _reset() -> None:
        # Leaves `paused` alone: reset is about STATE (pose + playback), the freeze is transport. SIGUSR1
        # during a Pause resets in place and stays frozen; `stop` re-freezes on its own after calling this.
        nonlocal traj, dense_gpu, cur, group_label, gain_hash, gain_dirty, gain_warn
        if gain_dirty:                             # edited robot_model.json lands HERE — see the watcher
            changed = b.reload_motor_gains()
            gain_hash, gain_dirty, gain_warn = _gain_fingerprint(), False, b.motor_gain_warnings()
            print(f"[standalone] motor gains reloaded from {Path(DEFAULT_PARAMS_FILE).name}: "
                  f"{', '.join(changed) if changed else 'no gain group changed'}", flush=True)
        b.reset_idx(all_mask)
        traj, dense_gpu, group_label = None, None, None
        _rec_discard()                             # drop any partial (unfinished) recording
        cur = b.joint_pos(report).clone()
        _settle(settle_s)                          # decay the reset transient off-screen before resuming render

    def _play(group_path: str, hz) -> None:
        nonlocal traj, traj_cols, dense_gpu, paused, group_label
        gdir = group_path if Path(group_path).is_absolute() else str(_REPO / group_path)
        traj = CsvTrajectory(gdir, available, control_hz, seed_pose=_seed_dict(), ramp_s=1.0)
        # map the trajectory's (possibly partial, e.g. one finger = 3 joints) joint set → columns of `report`;
        # the rest of `report` keeps holding the init pose while the trajectory drives only its own joints.
        traj_cols = torch.tensor([report.index(j) for j in traj.joint_names], device=b.device)
        # upload the whole dense buffer to the GPU once → the step loop indexes it on-device (no CPU→GPU copy).
        dense_gpu = torch.as_tensor(traj.dense, device=b.device, dtype=torch.float32)
        paused, group_label = False, group_path    # a fresh playback always runs (unfreezes a stopped sim)
        _rec_start(group_path)                     # begin capturing this playback → CSV on finish
        note = "" if not hz or float(hz) == control_hz else f" (hz knob={hz} → M5; playing @ {control_hz:.0f}Hz)"
        print(f"[standalone] play {group_path}: {len(traj.joint_names)} joints, "
              f"{traj.duration_s:.2f}s @ {control_hz:.0f}Hz{note}", flush=True)

    if trajectory_dir is not None:                 # --trajectory → auto-play on start
        _play(trajectory_dir, control_hz)

    signal.signal(signal.SIGUSR1, _on_sigusr1)     # Launchpad 'Reset Simulator' → SIGUSR1 → reset on demand
    print(f"[standalone] {engine} · {task} — env up (num_envs={b.num_envs}, GL viewer). "
          f"SIM tab · Reset=SIGUSR1 · Ctrl-C to stop.", flush=True)
    try:
        while True:
            # Gain-edit watch (1 Hz, both branches — a tuning edit while the sim is paused must show too).
            if time.monotonic() >= gain_next_check:
                gain_next_check = time.monotonic() + _GAIN_WATCH_S
                dirty = _gain_fingerprint() != gain_hash
                if dirty != gain_dirty:              # log the transition once, not every poll
                    print(f"[standalone] {Path(DEFAULT_PARAMS_FILE).name} "
                          f"{'edited — reset to apply the new motor gains' if dirty else 'back to the loaded gains'}",
                          flush=True)
                gain_dirty = dirty

            if _reset_requested:
                _reset_requested = False
                _reset()
                print("[standalone] reset to initial state", flush=True)

            for c in srv.drain_commands():         # browser → runner (dashboard transport + mode)
                cmd = c.get("cmd")
                # Play/Pause/Stop are the SIMULATOR's transport, not the trajectory's: Pause freezes physics
                # itself (below), so Play's first job is to unfreeze. Starting the selected group is its
                # second job, and only when nothing is already mid-playback (Play on a paused run resumes
                # that playback where it stopped instead of restarting it).
                if cmd in ("play", "resume"):
                    paused = False
                    group = c.get("group") or ""
                    live = traj is not None and not traj.finished
                    if group and not live:
                        if torque_mode:            # torque/float mode can't take a target → no playback
                            print("[standalone] play: resumed (no playback — Torque mode)", flush=True)
                        else:
                            try:
                                _play(group, c.get("hz"))
                            except Exception as exc:   # bad group / parse error → stay idle, report
                                traj, dense_gpu, group_label = None, None, None
                                _rec_discard()
                                print(f"[standalone] play failed: {exc}", flush=True)
                    else:
                        print(f"[standalone] play → running{' (resumed playback)' if live else ''}", flush=True)
                elif cmd == "pause":
                    paused = True
                    print("[standalone] pause → sim frozen (no physics steps)", flush=True)
                elif cmd == "stop":
                    _reset()                       # drop playback + back to the init pose, then hold frozen
                    paused = True
                    print("[standalone] stop → reset to init, sim frozen", flush=True)
                elif cmd == "joint_target":
                    # Joint Control tab. `targets` = {joint name: DEGREES} for EVERY joint currently under
                    # manual control — the page always sends the whole checked set, so a joint dropping out
                    # of the map IS the release (it simply stops being written and holds where it was).
                    tgt = c.get("targets") or {}
                    cols = [report.index(j) for j in tgt if j in report]
                    man_cols = torch.tensor(cols, device=b.device, dtype=torch.long)
                    man_vals = torch.tensor([float(tgt[report[i]]) * _DEG2RAD for i in cols],
                                            device=b.device, dtype=cur.dtype)
                elif cmd == "mode":                # Position ↔ Torque (float). Entering Torque stops playback.
                    torque_mode = bool(c.get("torque"))
                    if torque_mode:
                        traj, dense_gpu, group_label = None, None, None
                        _rec_discard()
                    # newton coupled-PD: real gravity-comp mode = feedforward ONLY — zero the coupled
                    # D gain too (dissipation = joint friction); restored on Position. genesis: no-op.
                    b.set_coupled_float_damping(torque_mode)
                    print(f"[standalone] {'TORQUE (float)' if torque_mode else 'POSITION'} mode", flush=True)

            if paused:
                # Sim frozen: no targets, no physics, no recording — the whole state stands still. The viewer
                # still gets a frame (newton pumps window input only inside one; genesis' viewer thread
                # repaints itself) and the dashboard still gets snapshots, so the page can show the held
                # values instead of going silent. Idle at frame rate rather than spinning the CPU.
                b.render_frame()
                if srv.has_clients():
                    srv.publish(_snapshot())
                time.sleep(1.0 / 60)
                continue

            if traj is not None:                    # advance the trajectory (idle → hold `cur`)
                cur[:, traj_cols] = dense_gpu[traj.advance()]   # GPU-resident buffer → GPU→GPU (no per-step CPU copy)
            if man_cols.numel():                    # hand-driven joints win over the trajectory on their columns
                cur[:, man_cols] = man_vals
            if torque_mode and gc_cols.numel():     # Torque mode: gravcomp joints float (target = current → 0 PD drive)
                tgt = cur.clone()
                tgt[:, gc_cols] = b.joint_pos(report)[:, gc_cols]
                b.set_joint_targets(report, tgt)
            else:
                b.set_joint_targets(report, cur)

            # Live snapshot for the dashboard: EVERY channel every step (not just the open tab), so the
            # browser buffers all tabs on one timeline and Pause freezes them together.
            if srv.has_clients():
                srv.publish(_snapshot())

            if traj is not None and rec["active"] and not rec["saved"]:
                pos, trq = b.joint_pos(report), b.joint_torque(report)       # one row per recorded CSV series
                rec["time"].append(rec["t"])
                rec["pos"].append((pos[0] * _RAD2DEG).tolist())              # actual joint state [deg]
                rec["tgt"].append((cur[0] * _RAD2DEG).tolist())              # commanded PD target [deg]
                rec["trq"].append(trq[0].tolist())                           # joint torque [Nm]
                if fingertip_links:                                          # fingertip net contact-force magnitude [N]
                    cf = b.contact_force(fingertip_links)                    # (N, K, 3) world, on distal collision links
                    rec["cf"].append(cf[0].norm(dim=-1).tolist())            # per-finger |force| [N]
                rec["t"] += dt_ctrl
                if traj.finished:                   # played to the end → write the CSVs once
                    series = {"joint_position.csv": (report, rec["pos"]),
                              "target_position.csv": (report, rec["tgt"]),
                              "joint_torque.csv": (report, rec["trq"])}
                    if fingertip_links:
                        series["contact_force.csv"] = (fingertip_labels, rec["cf"])
                    out = _save_trajectory_log(_REPO, rec["group"], engine, rec["time"], series)
                    rec["saved"] = True
                    print(f"[standalone] trajectory finished — saved {len(rec['time'])} samples "
                          f"({' / '.join(series)}) → {out}", flush=True)
                    _maybe_autoplot(out, rec["group"])   # hand_<finger> logs → auto overlay PNG (tested joint)

            if step_n is not None:
                step_n(decim)
            else:
                for _ in range(decim):
                    b.step()
    except KeyboardInterrupt:
        print("[standalone] stopped", flush=True)
    finally:
        srv.close()


def main() -> None:
    # The Launchpad starts us from a `nohup … &` server, which hands down an ignored SIGINT — restore
    # it so Stop's SIGINT reaches `run`'s KeyboardInterrupt (control server closed, recording saved)
    # instead of being discarded until the SIGKILL backstop.
    restore_default_sigint()
    ap = argparse.ArgumentParser(
        description="MetaLab standalone sim runner (env only, GL viewer, default GPU, no policy/learning).")
    ap.add_argument("--engine", required=True, choices=["genesis", "newton"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--trajectory", default=None,
                    help="Directory of via-point CSVs (e.g. sim/metalab/assets/data/trajectory/dumbbell/"
                         "demo2_dumbbell_group) to AUTO-PLAY on start. Otherwise drive it live from the SIM tab.")
    args = ap.parse_args()
    run(args.engine, args.task, trajectory_dir=args.trajectory)


if __name__ == "__main__":
    main()
