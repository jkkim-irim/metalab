"""VLA closed-loop sim-eval runner — the model-agnostic machinery for evaluating an action-chunking
policy (a ``policies.vla.VlaEvalPolicy``) in the sim, over the sim-service RPC boundary. Deliberately
NOT tied to any one model: it drives the ``VlaEvalPolicy`` adapter, which wraps any wir_v1 chunk-policy
(GR00T today — ``policies/groot.py`` loads it into the adapter). This file is only the harness.

What lives here:
  * the closed-loop runner (``run_task``) — spawns the sim server for one task, drives the policy over
    the ``attrs``/``reset``/``step``/``close`` RPC, executes each predicted chunk (or just its first
    ``--replan-steps``), freezes each env's success on its first ``done``, optionally captures rollout
    MP4s, and returns the per-task SR.
  * ``run_eval`` — evaluate a built policy over a task set + aggregate SR + (optionally) log to wandb.
  * sim-server selection/spawn (``server_for``), video encoding, wandb logging, and the shared CLI
    args (``add_vla_args``) — all model-agnostic.

The ONE runner drives EITHER sim server (they share the ``attrs``/``reset``/``step``/``close`` +
``{top,wrist,state}`` obs contract), selected by ``--suite`` (which also fixes the obs/state layout;
``--server-script`` optionally overrides just the script path):
  * ``--suite mikasa``   -> ``sim/maniskill/server.py`` (vectorized, ``num_envs`` envs in lockstep).
  * ``--suite libero_*`` -> ``sim/libero/server.py`` (single-env; ``num_envs==1``, episodes
    sequential with a per-episode seed).
Obs cross the wire as ``{top,wrist: (N,H,W,3) uint8, state: (N,state_dim) f32}``; the policy adapter
maps them onto its model's inputs.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import wandb

from learning.eval.policies.vla import VlaEvalPolicy
from learning.eval.sim_eval_helpers import (
    replan_cap,
    safe_name,
    video_filename,
)

# learning/eval/vla_runner.py -> up 3 dirs to the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MIKASA_SERVER = os.path.join(_REPO_ROOT, "sim", "maniskill", "server.py")
_LIBERO_SERVER = os.path.join(_REPO_ROOT, "sim", "libero", "server.py")
sys.path.insert(0, os.path.join(_REPO_ROOT, "sim", "service"))  # the unified wire protocol
from transport import K_ACTION, RpcClient, obs_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval.vla")


def server_for(modality: str, override) -> str:
    """The sim-service server script for this modality; ``--server-script`` overrides the path."""
    if override:
        return override
    return _LIBERO_SERVER if modality == "libero" else _MIKASA_SERVER


def _build_server_args(modality: str, args) -> list[str]:
    """Translate the client CLI args into the selected server's own (task-LESS) arg list. The specific
    task is loaded per-task over the wire via ``load_task`` on the persistent server, so one server
    process serves the whole task set (env rebuilt in-process per task) instead of one server per task."""
    if modality == "libero":
        server_args = ["--suite", args.suite, "--num-envs", str(args.num_envs),
                       "--resolution", str(args.resolution)]
    else:  # mikasa
        server_args = ["--num-envs", str(args.num_envs),
                       "--min-episode-steps", str(args.min_episode_steps),
                       "--sim-backend", args.sim_backend]
        if getattr(args, "transport", "socket") == "ipc":
            server_args += ["--transport", "ipc"]  # zero-copy shared GPU obs buffers
    if args.max_episode_steps is not None:
        server_args += ["--max-episode-steps", str(args.max_episode_steps)]
    return server_args


def _spawn_server(sim_python: str, server_script: str, server_args: list[str]
                  ) -> tuple[subprocess.Popen, int]:
    """Launch the sim server in the SIM venv; return (proc, port) once it publishes SIM_SERVICE_PORT.
    The server runs with a NON-expandable CUDA allocator (``PYTORCH_CUDA_ALLOC_CONF`` stripped from its
    env): the trainer sets ``expandable_segments:True`` for the GR00T model, but under it the IPC
    transport's ``share()`` cannot export CUDA-IPC handles — and the sim server does not need it."""
    cmd = [sim_python, server_script] + server_args
    logger.info("spawning sim server: %s", " ".join(cmd))
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # keep the sim server's allocator IPC-capable
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    port = None
    for line in proc.stdout:  # server prints its startup log, then SIM_SERVICE_PORT=<port>
        line = line.rstrip()
        logger.info("[server] %s", line)
        if line.startswith("SIM_SERVICE_PORT="):
            port = int(line.split("=", 1)[1])
            break
        if proc.poll() is not None:
            raise RuntimeError("sim server exited before publishing its port")
    assert port is not None, "sim server closed stdout without publishing SIM_SERVICE_PORT"
    return proc, port


def _frame_hwc(top: torch.Tensor, j: int) -> np.ndarray:
    """One env's ``top`` camera frame as an HWC uint8 numpy array (copied off the shared obs buffer)."""
    return top[j].detach().cpu().numpy().astype(np.uint8, copy=True)


def _encode_video(path: str, frames: list[np.ndarray], fps: int) -> None:
    """Encode a captured episode's frames to an MP4 at ``path`` via ``imageio`` + ``imageio-ffmpeg``,
    imported at module top so a missing video dependency fails at import, not mid-rollout."""
    assert frames, f"no frames captured for {path}"
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def _log_wandb(args, results: list[dict], mean_sr: float, succ_total: int, ep_total: int,
               fps: int) -> None:
    """Attach eval SR + rollout videos to the (resumed) wandb run ``--wandb-run-id``.

    ``resume="allow"`` re-opens the existing run with that id (typically the training run) so eval
    metrics/videos land on it; if no run with that id exists yet, wandb creates one with that id.
    ``results`` is one dict per evaluated task: ``eval/SR`` is the micro-average over ALL episodes,
    and each task also logs ``eval/<env_id>/SR`` + its videos (namespaced by env_id so a set doesn't
    collide).
    """
    wandb.init(id=args.wandb_run_id, project=args.wandb_project, entity=args.wandb_entity,
               resume="allow")
    log = {
        "eval/SR": mean_sr,                 # micro-average SR over all episodes of all tasks
        "eval/n_episodes": ep_total,
        "eval/n_success": succ_total,
        "eval/n_tasks": len(results),
    }
    table = wandb.Table(columns=["env_id", "idx", "success", "ep_len", "prompt"])
    for res in results:
        env_id = res["env_id"]
        log[f"eval/{env_id}/SR"] = res["sr"]   # per-task SR key so a set logs one each, no collision
        for rec in res["episodes"]:
            log[f"eval/videos/{env_id}/ep{rec['idx']}"] = wandb.Video(rec["path"], fps=fps, format="mp4")
            table.add_data(env_id, rec["idx"], rec["success"], rec["ep_len"], rec["prompt"])
    log["eval/episodes"] = table
    wandb.log(log)
    wandb.finish()


class _SocketClient:
    """v1 socket transport: obs + success/done ride each RPC round-trip (torch-serialized over the
    socket). Drives LIBERO (CPU obs) and MIKASA ``--transport socket``. The rollout loop calls
    ``load_task`` / ``reset`` / ``step`` on this uniform interface, transport-agnostic."""

    def __init__(self, rpc: RpcClient):
        self._rpc = rpc

    def load_task(self, task) -> dict:
        return self._rpc.call("load_task", task)

    def reset(self, seeds: list) -> dict:
        return self._rpc.call("reset", seeds)["obs"]

    def step(self, action):
        r = self._rpc.call("step", action)
        return r["obs"], r["terminated"], r["truncated"], r["success"]

    def close(self) -> None:
        self._rpc.close()


class _IpcClient:
    """CUDA-IPC transport (MIKASA ``--transport ipc``): obs + action live in shared GPU buffers exchanged
    once; per step the client writes the action buffer + sends a tiny control message, reads obs back
    from the buffers (never leaving the GPU), and gets the small success/done tensors over the socket.
    The server ``share()``s the buffers on the FIRST reset (MIKASA obs shapes are fixed across env-ids)."""

    def __init__(self, rpc: RpcClient, device: str):
        self._rpc = rpc
        self._device = device
        self._buf: dict | None = None

    def _ctl(self, method: str, payload=None):
        self._rpc.send_ctl((method, payload))
        r = self._rpc.recv_ctl()
        if not r.get("ok"):
            raise RuntimeError(f"sim service error: {r.get('error')}\n{r.get('tb', '')}")
        return r.get("result")

    def load_task(self, task) -> dict:
        return self._ctl("load_task", task)

    def reset(self, seeds: list) -> dict:
        self._rpc.send_ctl(("reset", seeds))
        if self._buf is None:  # server share()s on the FIRST reset -> open the IPC handles now
            self._buf = self._rpc.recv_shared()
        r = self._rpc.recv_ctl()
        if not r.get("ok"):
            raise RuntimeError(f"sim service error: {r.get('error')}\n{r.get('tb', '')}")
        return self._read_obs()

    def step(self, action):
        self._buf[K_ACTION].copy_(action.to(self._device))
        torch.cuda.synchronize()  # the action write must land before the server reads it
        res = self._ctl("step")
        return self._read_obs(), res["terminated"], res["truncated"], res["success"]

    def _read_obs(self) -> dict:
        # clone off the shared buffers so the returned obs is decoupled from the next step's overwrite
        return {k: self._buf[obs_key(k)].clone() for k in ("top", "wrist", "state")}

    def close(self) -> None:
        self._rpc.close()
        self._buf = None


def start_service(sim_python: str, server_script: str, modality: str, args):
    """Spawn ONE persistent sim server (task-LESS) and connect a transport client, reused across every
    task via ``load_task`` — one env process for the whole eval, not one spawn per task (the dominant
    per-task setup cost). MIKASA with ``--transport ipc`` gets the zero-copy GPU client; everything else
    (LIBERO, MIKASA socket) gets the v1 socket client. Pair with ``stop_service`` in a ``finally``."""
    proc, port = _spawn_server(sim_python, server_script, _build_server_args(modality, args))
    rpc = RpcClient("127.0.0.1", port, map_location=args.device)
    if modality == "mikasa" and getattr(args, "transport", "socket") == "ipc":
        return proc, _IpcClient(rpc, args.device)
    return proc, _SocketClient(rpc)


def stop_service(proc: subprocess.Popen, client) -> None:
    """Close the client and reap the persistent sim server."""
    client.close()
    proc.wait(timeout=30)


def run_task(policy: VlaEvalPolicy, client, args, task, video_root: str) -> dict:
    """Load ``task`` on the ALREADY-RUNNING service, roll out ``--num-episodes`` closed-loop, print its
    RESULT, and return ``{env_id, sr, succ, total, episodes}``. ``client`` is a transport proxy
    (``_SocketClient`` / ``_IpcClient``) exposing ``load_task`` / ``reset`` / ``step``; ``load_task``
    rebuilds the server's env in-process for this task (the process is reused across the whole set).
    Rollout videos (first ``--video-episodes``) go under ``video_root/<env_id>/``."""
    n_video = max(0, args.video_episodes)
    episodes: list[dict] = []  # per captured episode: {idx, success, ep_len, prompt, path}
    attrs = client.load_task(task)  # (re)build the server env for this task -> its attrs
    N, action_dim = attrs["num_envs"], attrs["action_dim"]
    max_steps, prompt = attrs["max_episode_steps"], attrs["prompt"]
    env_id = attrs["env_id"]  # the REAL env from the server (not our --tasks token)
    policy.set_task(prompt, action_dim)  # bind this task's instruction + action_dim
    logger.info("env=%s N=%d action_dim=%d max_steps=%d prompt=%r",
                env_id, N, action_dim, max_steps, prompt)
    video_dir = os.path.join(video_root, safe_name(env_id))
    if n_video:
        os.makedirs(video_dir, exist_ok=True)  # fail loud if unwritable

    n_batches = math.ceil(args.num_episodes / N)
    succ_total, ep_total = 0, 0
    for b in range(n_batches):
        seed_list = [args.base_seed + b * N + j for j in range(N)]
        obs = client.reset(seed_list)
        success_once = torch.zeros(N, dtype=torch.bool)
        done_frozen = torch.zeros(N, dtype=torch.bool)
        # envs in this batch whose GLOBAL episode index falls in the first K -> record their frames.
        tracked = {j: b * N + j for j in range(N) if b * N + j < n_video}
        frames = {gidx: [_frame_hwc(obs["top"], j)] for j, gidx in tracked.items()}  # seed w/ reset
        t = 0
        while t < max_steps and not bool(done_frozen.all()):
            chunk = policy.act(obs)  # (N, chunk_size, action_dim)
            for k in range(replan_cap(chunk.shape[1], args.replan_steps)):
                if t >= max_steps or bool(done_frozen.all()):
                    break
                obs, terminated, truncated, success = client.step(chunk[:, k, :].contiguous())
                live = ~done_frozen
                success_once |= (success.cpu() & live)
                for j, gidx in tracked.items():
                    if bool(live[j]):  # still running this step -> keep its frame (incl. terminal)
                        frames[gidx].append(_frame_hwc(obs["top"], j))
                done_frozen |= (terminated.cpu() | truncated.cpu())
                t += 1
        batch_succ = int(success_once.sum())
        succ_total += batch_succ
        ep_total += N
        logger.info("batch %d/%d: %d/%d success (running SR %.1f%%)",
                    b + 1, n_batches, batch_succ, N, 100.0 * succ_total / ep_total)
        for j, gidx in sorted(tracked.items()):
            succ = bool(success_once[j])
            path = os.path.join(video_dir, video_filename(gidx, succ))
            _encode_video(path, frames[gidx], args.video_fps)
            episodes.append({"idx": gidx, "success": succ, "ep_len": len(frames[gidx]),
                             "prompt": prompt, "path": path})
            logger.info("captured ep%d -> %s (%d frames, %s)",
                        gidx, path, len(frames[gidx]), "success" if succ else "fail")

    sr = succ_total / ep_total if ep_total else 0.0
    logger.info("=== closed-loop sim eval [%s]: SR = %d/%d = %.1f%% ===",
                env_id, succ_total, ep_total, 100.0 * sr)
    print(f"RESULT env_id={env_id} episodes={ep_total} successes={succ_total} SR={sr:.4f}")
    return {"env_id": env_id, "sr": sr, "succ": succ_total, "total": ep_total, "episodes": episodes}


def add_vla_args(p: argparse.ArgumentParser) -> None:
    """Register the model-agnostic VLA sim-eval args on ``p`` (suite/tasks + sim/rollout/video/wandb
    knobs). A policy adapter's own ``add_args`` adds its model-specific flags (e.g. ``--checkpoint``)
    and then calls this."""
    p.add_argument("--suite", required=True,
                   help="benchmark family + obs/state layout: 'mikasa', or a LIBERO suite "
                        "('libero_90', 'libero_10', 'libero_spatial', ...). Selects the sim server too.")
    p.add_argument("--tasks", required=True,
                   help="task(s) within the suite, comma-separated. LIBERO: integer task-ids "
                        "(e.g. '11' or '0,11,42'); MIKASA: env-ids (e.g. 'ShellGameTouch-VLA-v0'). "
                        "A set runs task-by-task and reports per-task + a mean eval/SR.")
    p.add_argument("--server-script", default=None,
                   help="override the sim-service server path (default: per --suite)")
    p.add_argument("--num-episodes", type=int, default=32, help="episodes PER task")
    p.add_argument("--max-episode-steps", type=int, default=None)
    # MIKASA server knobs.
    p.add_argument("--num-envs", type=int, default=8,
                   help="parallel envs per task (MIKASA: vectorized GPU envs; LIBERO: parallel CPU workers)")
    p.add_argument("--min-episode-steps", type=int, default=60, help="MIKASA horizon floor")
    p.add_argument("--sim-backend", default="gpu", help="MIKASA sim backend")
    p.add_argument("--transport", choices=("socket", "ipc"), default="socket",
                   help="MIKASA obs transport: 'socket' (serialize each step) or 'ipc' (zero-copy shared "
                        "GPU buffers, same-GPU only). LIBERO is CPU-obs and always uses socket.")
    # LIBERO server knobs.
    p.add_argument("--resolution", type=int, default=256, help="LIBERO square camera render size")
    # Shared rollout knobs.
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--replan-steps", type=int, default=0,
                   help="execute only the first N steps of each predicted chunk before re-observing "
                        "(0 = execute the full chunk = original behaviour; the reference LIBERO eval used 5)")
    p.add_argument("--sim-python", default="/root/mikasa/venv312/bin/python")
    p.add_argument("--device", default="cuda")
    # Rollout video capture + optional wandb logging.
    p.add_argument("--video-episodes", type=int, default=4,
                   help="capture the first K episodes' top-camera rollouts as MP4s (0 = disable)")
    p.add_argument("--video-dir", default=None,
                   help="root for the rollout MP4s; each task writes an <env_id>/ subdir "
                        "(default: policy-specific)")
    p.add_argument("--video-fps", type=int, default=15, help="MP4 frame rate (~10-20)")
    p.add_argument("--wandb-run-id", default=None,
                   help="resume this wandb run and log eval SR + videos onto it (unset = print-only)")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)


def run_eval(policy: VlaEvalPolicy, args, *, modality: str, tasks: list, server_script: str,
             video_root: str) -> list[dict]:
    """Evaluate a BUILT ``policy`` over ``tasks`` closed-loop, aggregate the micro-average SR, and
    (given ``--wandb-run-id``) log SR + videos. The policy adapter builds ``policy`` from its own args
    and hands off here; this loop/aggregation/logging is model-agnostic. ONE persistent sim server
    serves the whole set (each task loaded in-process via ``load_task``), not a spawn per task."""
    proc, client = start_service(args.sim_python, server_script, modality, args)
    try:
        results = [run_task(policy, client, args, task, video_root) for task in tasks]
    finally:
        stop_service(proc, client)

    succ_total = sum(r["succ"] for r in results)
    ep_total = sum(r["total"] for r in results)
    mean_sr = succ_total / ep_total if ep_total else 0.0
    if len(results) > 1:  # a task set -> one aggregate line (single-task already printed its RESULT)
        logger.info("=== SUMMARY suite=%s tasks=%d: SR = %d/%d = %.1f%% (micro-avg over episodes) ===",
                    args.suite, len(results), succ_total, ep_total, 100.0 * mean_sr)
        print(f"SUMMARY suite={args.suite} tasks={len(results)} episodes={ep_total} "
              f"successes={succ_total} SR={mean_sr:.4f}")

    if args.wandb_run_id:
        _log_wandb(args, results, mean_sr, succ_total, ep_total, args.video_fps)
    else:
        logger.info("no --wandb-run-id -> skipping wandb (videos under %s)", video_root)
    return results
