from __future__ import annotations

import argparse
import os

from sim.metalab.dashboard.rollout_log import PerEnvRolloutLog
from sim.metalab.dashboard.rollout_report import build_report
from sim.metalab.runtime.vec_env_handler import (
    install_sigterm_unwind,
    make_vec_env_handler,
    shield_shutdown_from_sigterm,
)
from sim.metalab.transport import RpcServer, serve_vec_env


def main(build_env, tag: str) -> None:
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
        fps = max(1, round(1.0 / (ph.dt * ph.decimation)))
        data_log = PerEnvRolloutLog(env, rec_idxs, out_dir, fps=fps)
        print(f"[{tag}] recording {len(rec_idxs)} env series -> {out_dir}", flush=True)
    srv = RpcServer(args.host, args.port, map_location=args.device)
    if args.port_file:
        with open(args.port_file, "w") as f:
            f.write(str(srv.port))
    print(f"[{tag}] {args.task} num_envs={env.num_envs} device={env.device} "
          f"serving on {srv.host}:{srv.port} (RPC + CUDA-IPC)", flush=True)
    srv.accept()
    print(f"[{tag}] client connected", flush=True)
    install_sigterm_unwind()
    try:
        serve_vec_env(srv, make_vec_env_handler(env, data_log, rerun_scene=getattr(env, "rerun_scene", None)))
    finally:
        shield_shutdown_from_sigterm()
        summary = env.backend.viewer.close()
        if summary is not None:
            print(f"[{tag}] rerun recording finalized -> {args.rrd} [contact normals: {summary}]", flush=True)
        if data_log is not None:
            if data_log.finish():
                build_report(out_dir)
    print(f"[{tag}] closed", flush=True)
