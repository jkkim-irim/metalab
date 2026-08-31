# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Collect WBT reference trajectories by rolling out a trained (blind) lift checkpoint over the sim service.

Rolls out a dexblind hammer-lift policy (e.g. ``model_999``) in the WBT COLLECT env (``--wbt --collect``)
and keeps the SUCCESSFUL episodes as reference motions to track. The WBT COLLECT env adds a dedicated
``tracking_state`` obs group — the flat per-step WBT tracked state (object pose+vel, palm pose+vel,
fingertip positions, joints; layout in ``sim/isaaclab/envs/hammer_lift/mdp/wbt.py``) — which the collector
records alongside the per-episode env ``setup`` (for exact RSI). ``learning/rl/tracking/reference.py`` then
repackages ``tracking_state`` (+ ``setup``) into ``ref_*.npz``.

Inference-only (rebuilds the actor from POLICY, loads ``actor_state_dict``); Isaac-free — runs in the
isaaclab conda env. Driven reproducibly by ``learning/scripts/aws/wbt_collect.sh``.

Lives in ``learning.rl.tracking`` (a subpackage; not a module directly under ``learning.rl``, so it does
not trip rsl_rl's ``resolve_callable`` module scan)."""
from __future__ import annotations

import argparse
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from learning.rl.dexblind.hammer_lift.experiment import POLICY
from learning.rl.models import MLPModel, RNNModel
from learning.rl.service import ensure_transport_importable, sim_server
from learning.rl.utils import resolve_obs_groups

ensure_transport_importable()  # put the sim-service dir (transport.py) on sys.path before the client
from learning.rl.client import NaNSafeVecEnv, SimServiceVecEnv  # noqa: E402

# Mirrors sim/isaaclab/envs/hammer_lift/scene_cfg.HAMMER_USD_FILENAMES (round-robin by env index).
# Learning side stays sim-import-free, so the count is pinned here; the reference attach validates
# variants against the live scene at load.
NUM_HAMMER_VARIANTS = 3

_ACTOR_CLASSES = {"MLPModel": MLPModel, "RNNModel": RNNModel}


def _load_policy(train_cfg: dict, obs, num_actions: int, checkpoint: str, device: str):
    """Rebuild the actor from POLICY and load its checkpoint weights (inference only)."""
    cfg = copy.deepcopy(train_cfg)
    obs_groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor"])
    actor_kwargs = cfg["actor"]
    actor_class = _ACTOR_CLASSES[actor_kwargs.pop("class_name")]
    actor = actor_class(obs, obs_groups, "actor", num_actions, **actor_kwargs).to(device)
    ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    return actor


def main() -> int:
    p = argparse.ArgumentParser(description="Collect WBT success trajectories over the sim-service client.")
    p.add_argument("--checkpoint", required=True, help="path to a model_*.pt checkpoint (on the node)")
    p.add_argument("--checkpoint_s3", default="", help="S3 source of the checkpoint (recorded in meta)")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min_trajectories", type=int, default=1000, help="collect at least this many successes")
    p.add_argument("--max_steps", type=int, default=40000, help="safety cap on total rollout steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--play", action=argparse.BooleanOptionalAction, default=True,
                   help="roll out the play (eval) env variant — the successful full-lift regime")
    p.add_argument("--out_dir", default="/home/ubuntu/sim_trajectories", help="where to write traj_*.npz + meta.json")
    # provenance passthrough — recorded verbatim into meta.json
    p.add_argument("--experiment", default="tracking")
    p.add_argument("--train", default="")
    p.add_argument("--eval_sha", default="")
    p.add_argument("--eval_date", default="")
    args, _ = p.parse_known_args()

    # wbt=True → server builds HammerLiftEnvCfg_WBTCOLLECT (PLAY + setup capture + tracking_state obs group).
    srv_args = SimpleNamespace(num_envs=args.num_envs, device=args.device, seed=args.seed, play=args.play,
                               collect=True, wbt=True, video=False, video_length=0, video_dir="")
    with sim_server(srv_args) as port:
        print(f"[wbt-collect] sim service on port {port}", flush=True)
        env = NaNSafeVecEnv(SimServiceVecEnv("127.0.0.1", port, device=args.device))
        obs, _ex0 = env.reset()  # reset() carries the initial per-episode env setup in extras (--collect)
        policy = _load_policy(POLICY, obs, env.num_actions, args.checkpoint, args.device)
        # pending_setup[i] = env i's CURRENT episode setup (captured at its reset); saved with the traj so
        # the WBT env can reproduce it exactly (RSI). None if the server didn't emit it.
        _su0 = _ex0.get("setup") if isinstance(_ex0, dict) else None
        pending_setup = _su0.detach().cpu().numpy() if _su0 is not None else None
        if pending_setup is None:
            print("[wbt-collect] WARN: no extras['setup'] — env setup NOT recorded (RSI will be unavailable)", flush=True)

        n = env.num_envs
        groups = [k for k in obs.keys() if torch.is_tensor(obs[k])]  # obs groups on the wire (actor, privileged, tracking_state)
        shapes = {k: tuple(obs[k].shape[1:]) for k in groups}
        print(f"[wbt-collect] obs groups={shapes} num_envs={n} target={args.min_trajectories}", flush=True)
        if "tracking_state" not in groups:
            print("[wbt-collect] WARN: no 'tracking_state' obs group — is the server built with --wbt --collect?", flush=True)

        # Per-env running buffers (reset on episode end). We keep only SUCCESSFUL episodes.
        buf = [{k: [] for k in groups} for _ in range(n)]
        act_buf: list[list] = [[] for _ in range(n)]
        rew_buf: list[list] = [[] for _ in range(n)]
        trajectories: list[dict] = []
        n_done = n_succ = step = 0

        while len(trajectories) < args.min_trajectories and step < args.max_steps:
            with torch.inference_mode():
                actions = policy(obs)
                nobs, rew, dones, extras = env.step(actions)
                policy.reset(dones)
            obs_cpu = {k: obs[k].detach().cpu().numpy() for k in groups}
            act_cpu = actions.detach().cpu().numpy()
            rew_cpu = rew.detach().reshape(-1).cpu().numpy()
            for i in range(n):
                for k in groups:
                    buf[i][k].append(obs_cpu[k][i])
                act_buf[i].append(act_cpu[i])
                rew_buf[i].append(float(rew_cpu[i]))
            step += 1

            d = dones.bool()
            if int(d.sum()):
                succ = extras.get("task_success") if isinstance(extras, dict) else None
                succ = succ.bool() if succ is not None else None
                new_setup = extras.get("setup") if isinstance(extras, dict) else None
                new_setup = new_setup.detach().cpu().numpy() if new_setup is not None else None
                for i in d.nonzero(as_tuple=True)[0].tolist():
                    n_done += 1
                    is_succ = bool(succ[i]) if succ is not None else False
                    n_succ += int(is_succ)
                    if is_succ and len(trajectories) < args.min_trajectories:
                        traj = {k: np.stack(buf[i][k]).astype(np.float32) for k in groups}  # [T, dim] per group
                        traj["action"] = np.stack(act_buf[i]).astype(np.float32)
                        traj["reward"] = np.asarray(rew_buf[i], dtype=np.float32)
                        if pending_setup is not None:
                            traj["setup"] = pending_setup[i].astype(np.float32)  # this episode's env setup (RSI)
                        # Which hammer SHAPE demonstrated this episode: the scene assigns the USD
                        # variants round-robin by env index (scene_cfg sequential clone plan), so
                        # variant = env_idx mod NUM_HAMMER_VARIANTS. Without it, a reference's setup
                        # pose gets replayed onto a different shape at tracking time (a pose that is
                        # table-flat for one head geometry tilts/intersects for another) and the
                        # tracker is scored against a grasp demonstrated on different geometry.
                        traj["variant"] = np.asarray(i % NUM_HAMMER_VARIANTS, dtype=np.int64)
                        trajectories.append(traj)
                    buf[i] = {k: [] for k in groups}  # reset (env auto-resets to the next episode)
                    act_buf[i] = []
                    rew_buf[i] = []
                    if pending_setup is not None and new_setup is not None:
                        pending_setup[i] = new_setup[i]  # env i's NEW episode setup (captured post auto-reset)
            obs = nobs
            if step % 50 == 0:
                print(f"[wbt-collect] step={step} episodes_done={n_done} successes={n_succ} "
                      f"kept={len(trajectories)}/{args.min_trajectories}", flush=True)

        os.makedirs(args.out_dir, exist_ok=True)
        lengths = [int(t["action"].shape[0]) for t in trajectories]
        for j, traj in enumerate(trajectories):
            np.savez_compressed(os.path.join(args.out_dir, f"traj_{j:04d}.npz"), **traj)
        meta = {
            "experiment": args.experiment, "eval_date": args.eval_date, "eval_sha": args.eval_sha,
            "train": args.train, "checkpoint": args.checkpoint_s3 or args.checkpoint, "play": bool(args.play),
            "num_envs": n, "seed": args.seed, "num_trajectories": len(trajectories),
            "episodes_done": n_done, "successes_seen": n_succ, "steps": step,
            "obs_groups": shapes, "num_actions": int(env.num_actions),
            "traj_len": {"min": min(lengths), "max": max(lengths),
                         "mean": round(sum(lengths) / len(lengths), 1)} if lengths else {},
            "layout": "each traj_*.npz holds [T, dim] per obs group (tracking_state = the WBT tracked "
                      "state) + action [T, num_actions] + reward [T] + setup [D] (env init for RSI)",
        }
        with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"COLLECT_OK trajectories={len(trajectories)} episodes_done={n_done} successes={n_succ} "
              f"steps={step} out={args.out_dir}", flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
