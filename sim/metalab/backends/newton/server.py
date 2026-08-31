"""newton sim-service server — builds a Newton env from an EnvSpec contract and serves the VecEnv over RPC.

Newton counterpart of `sim/metalab/backends/genesis/server.py`. Contract (`sim.metalab.contract.tasks.rl.<task>`) → `parser`
(newton.Model + SolverMuJoCo) → `NewtonBackend` → `EnvDriver` (VecEnv) → `RpcServer`. The sim-service is
**RPC + CUDA-IPC as one set** (`sim/service/transport.py`): a localhost socket carries the control
channel + one-shot CUDA-IPC handle bootstrap (the team-required RPC boundary), and the hot-path obs/action
payload lives in shared GPU buffers (never leaves the GPU). GPU-only (same-GPU client, single-venv).

``main`` is the sim-service — MetaLab training/eval go through it. ``build_env`` builds the EnvDriver from
contract's TASK defaults.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))                  # sim/metalab/backends/newton
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))    # repo root
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sim.metalab.runtime.rollout_log import PerEnvRolloutLog  # noqa: E402
from sim.metalab.runtime.rollout_report import build_report  # noqa: E402
from sim.metalab.runtime.vec_env_handler import (  # noqa: E402
    install_sigterm_unwind,
    make_vec_env_handler,
    shield_shutdown_from_sigterm,
)
from sim.service.transport import (  # noqa: E402  (unified RPC + CUDA-IPC transport)
    RpcServer,
    serve_vec_env,
)


def build_env(task: str, recipe: str | None = None, num_envs: int | None = None,
              device: str = "cuda:0", viz: str = "none",
              telemetry: bool | None = None, rrd_path: str | None = None):
    """Contract → Newton scene → EnvDriver (VecEnv duck-typed).

    If ``num_envs`` is unset (None), use the task contract value = single source. Override only when explicit.
    ``rrd_path`` records the session to that ``.rrd`` for later replay — the record path, independent of
    ``viz`` (a dedicated ViewerRerun does it; see parser.build_scene)."""
    from sim.metalab.backends.newton import parser
    from sim.metalab.backends.newton.backend import NewtonBackend
    from sim.metalab.contract.loader import load_task
    from sim.metalab.runtime.env_driver import EnvDriver
    spec = load_task(task.replace("-", "_"), recipe, num_envs=num_envs)
    num_envs = spec.num_envs   # None→contract, override→that value. Concrete int from here on.
    handles = parser.build_scene(
        spec, num_envs=num_envs, viz=(viz not in (None, "none")),
        viewer_kind=(viz if viz in ("gl", "rtx", "rerun") else "gl"), device=device,
        rrd_path=rrd_path)
    backend = NewtonBackend(spec, handles, num_envs=num_envs)
    max_ep = max(1, round(spec.episode_length_s / (spec.physics.dt * spec.physics.decimation)))
    # telemetry: None (default) → follow viz (viewer on = dashboard on); explicit bool overrides (the
    # standalone runner passes False — it has its own control server and never calls EnvDriver.step).
    tele = (viz not in (None, "none")) if telemetry is None else telemetry
    return EnvDriver(spec, backend, max_episode_length=max_ep, telemetry=tele)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hammer-lift-teacher")
    ap.add_argument("--recipe", default="", help="which recipe of --task to run; a task FAMILY "
                                                "(tasks/rl/<task>/) requires one, a single-file contract takes none.")
    ap.add_argument("--num_envs", type=int, default=None,
                    help="if unset, uses num_envs from the task contract (.py) = single source.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--port_file", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--viz", default="none")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--record_envs", type=int, default=1,
                    help="how many envs (0..N-1) get a series + a tab in the report (needs --rrd)")
    ap.add_argument("--rrd", default="", help="record the rollout here; data.json and report.html are written "
                                             "beside it, from the SAME rollout. This is the record path.")
    args, _ = ap.parse_known_args()

    record = bool(args.rrd)
    rec_idxs = list(range(args.record_envs)) if record else None
    out_dir = os.path.dirname(os.path.abspath(args.rrd)) if record else ""
    env = build_env(args.task, args.recipe or None, args.num_envs, args.device, args.viz,
                    rrd_path=(args.rrd or None))
    assert str(env.device).startswith("cuda"), \
        f"MetaLab sim-service is GPU-only (RPC + CUDA-IPC); got device {env.device!r}"
    if args.seed is not None:
        env.seed(args.seed)
    data_log = None
    if record:
        ph = env.spec.physics
        fps = max(1, round(1.0 / (ph.dt * ph.decimation)))         # control_hz → real-time playback
        # Per-step series beside the .rrd (obs/reward/action/joint state) → the report's plot half. Same env
        # set and same step hook as the .rrd scene, so one series row is one recorded frame.
        data_log = PerEnvRolloutLog(env, rec_idxs, out_dir, fps=fps)
        print(f"[newton-sim] recording {len(rec_idxs)} env series -> {out_dir}", flush=True)
    srv = RpcServer(args.host, args.port, map_location=args.device)
    if args.port_file:
        with open(args.port_file, "w") as f:
            f.write(str(srv.port))
    print(f"[newton-sim] {args.task} num_envs={env.num_envs} device={env.device} "
          f"serving on {srv.host}:{srv.port} (RPC + CUDA-IPC)", flush=True)
    srv.accept()
    print("[newton-sim] client connected", flush=True)
    # The client SIGTERMs us when the rollout is done — without this the `finally` below never runs and the
    # record path loses its finalization (the .rrd tail and the rollout log's data.json).
    install_sigterm_unwind()
    try:
        serve_vec_env(srv, make_vec_env_handler(env, data_log))   # obs/action stay on the GPU (shared buffers); socket = control only
    finally:
        # The client sends `close` and SIGTERMs us right after — shield these two so they complete.
        shield_shutdown_from_sigterm()
        if getattr(env.backend, "_rerun", None) is not None:
            env.backend._rerun.close()   # flush the .rrd tail — an unclosed stream loses the last frames
            # Lazy for the same reason build_env defers `parser`: importing this module must not pull the
            # engine (or rerun) in, so that a non-newton caller can import the server side of it.
            from sim.metalab.backends.newton.viewer_rerun_sink import contact_arrow_summary
            print(f"[newton-sim] rerun recording finalized -> {args.rrd} "
                  f"[contact normals: {contact_arrow_summary()}]", flush=True)
        if data_log is not None:
            # data.json FIRST: the page is rebuildable from it offline (`python -m
            # sim.metalab.runtime.rollout_report <dir>`), so if the client's 30 s kill budget ever runs out
            # here, the series still survive. Both recording callers — a standalone eval and the trainer's
            # per-checkpoint val hook — come through this same shutdown, so neither needs its own wiring.
            if data_log.finish():
                build_report(out_dir)
    print("[newton-sim] closed", flush=True)


if __name__ == "__main__":
    main()
