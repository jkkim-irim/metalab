"""In-training closed-loop sim-eval for a GR00T (BC) policy.

Drives the LIVE, in-memory policy through the *same* closed-loop rollout as the standalone eval
(``learning.eval.vla_runner.run_task``), over the sim-service RPC boundary, and returns the
closed-loop success rate as a flat ``sim_eval/`` dict for wandb. Called periodically from the BC
trainer (``learning/trainer/bc_trainer.py``) so training curves carry a real task-success signal, not
just the offline prediction-error metrics.

Optionally captures the first ``video_episodes`` rollouts as MP4s and returns their
``(wandb_key, mp4_path)`` pairs SEPARATELY from the SR dict (``run_sim_eval`` returns
``(metrics, videos)``). The trainer's wandb wrapper ``log_dict`` is scalar-only and silently drops
media, so the caller logs each clip via the media path (``wandb.Video``) — see
``learning.metrics.validation.log_fk_video`` — onto the same step as the SR scalars.

Reuses the standalone runner verbatim — the only difference from ``policies/groot.run`` is that the
policy is the training model (wrapped in ``VlaEvalPolicy``), NOT rebuilt from a checkpoint. The sim
still runs in its OWN venv subprocess (``sim_python``): the trainer (GR00T venv) never imports the sim
stack.

This module pulls the heavy eval/sim boundary (torch, wandb, the RPC transport, the sim server path),
so the trainer imports it LAZILY — only when in-training sim-eval is enabled — keeping a normal train
run (and CI) free of the sim dependency.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from learning.eval.policies.vla import VlaEvalPolicy
from learning.eval.sim_eval_helpers import aggregate_sim_metrics, parse_tasks, resolve_suite
from learning.eval.vla_runner import run_task, server_for, start_service, stop_service


def run_sim_eval(policy, *, suite: str, tasks: str, sim_python: str, chunk_size: int, device: str,
                 num_episodes: int, base_seed: int = 0, resolution: int = 256,
                 max_episode_steps: int | None = None, replan_steps: int = 0,
                 num_inference_timesteps: int | None = None, video_episodes: int = 0,
                 video_fps: int = 15, video_dir: str | None = None,
                 server_script: str | None = None) -> tuple[dict, list[tuple[str, str]]]:
    """Closed-loop eval of the LIVE ``policy`` over ``suite``/``tasks``; returns ``(metrics, videos)``.

    ``policy`` is the unwrapped training model (has ``predict_action_chunk``); it is wrapped in a
    ``VlaEvalPolicy`` and driven task-by-task. Put in eval mode for the rollout (the trainer only sets
    eval when a val pass runs first, so we can't assume it) and — with ``num_inference_timesteps`` — a
    lower denoising step count for a faster reading. Both are restored on the SUCCESS path so a later
    checkpoint keeps the deployment defaults; on error the rollout raises and the run crashes — no
    restore (fail loud, no fallback), only the sim subprocess is torn down. Mirrors the trainer's
    ``--val_flow_matching_steps``.

    ``metrics`` is the flat ``sim_eval/`` scalar dict. ``video_episodes > 0`` captures the first N
    rollouts per task as MP4s under ``video_dir``; their ``(wandb_key, mp4_path)`` pairs
    (``sim_eval/videos/<env_id>/ep<idx>``) are returned as ``videos`` for the caller to log via the
    media path, so the SAME wandb step carries both the SR scalars and the rollout clips.
    """
    assert suite and tasks and sim_python, \
        "in-training sim-eval needs suite, tasks, and sim_python (the sim venv's python)"
    assert video_episodes <= 0 or video_dir, "video_episodes>0 requires a video_dir to write the MP4s"
    modality, task_kind = resolve_suite(suite)
    task_list = parse_tasks(task_kind, tasks)
    script = server_for(modality, server_script)
    eval_policy = VlaEvalPolicy(policy, chunk_size, device)   # wrap the LIVE model (no checkpoint)

    # run_task / _build_server_args read these off one namespace (see vla_runner). num_envs/
    # min_episode_steps/sim_backend are the MIKASA-only server knobs, present so _build_server_args
    # never KeyErrors; video_episodes/video_fps drive the (optional) rollout capture.
    args = SimpleNamespace(
        suite=suite, resolution=resolution, max_episode_steps=max_episode_steps,
        num_episodes=num_episodes, base_seed=base_seed, replan_steps=replan_steps,
        chunk_size=chunk_size, device=device, sim_python=sim_python,
        video_episodes=max(0, video_episodes), video_fps=video_fps,
        num_envs=1, min_episode_steps=60, sim_backend="gpu",
    )
    video_root = video_dir or ""

    # eval mode + (optional) a lower denoise count for the rollout; restored straight-line on success so
    # a later checkpoint keeps the deployment defaults. NO outer try/finally: on error the rollout
    # raises and the run crashes (fail loud, no restore-on-failure fallback).
    was_training = policy.training
    policy.eval()
    old_steps = (policy.set_inference_steps(num_inference_timesteps)
                 if num_inference_timesteps and hasattr(policy, "set_inference_steps") else None)
    with torch.no_grad():
        proc, client = start_service(sim_python, script, modality, args)
        try:
            results = [run_task(eval_policy, client, args, task, video_root)
                       for task in task_list]
        finally:
            stop_service(proc, client)  # tear down the spawned sim subprocess (not error-recovery)
    if old_steps is not None:
        policy.set_inference_steps(old_steps)
    if was_training:
        policy.train()

    metrics = aggregate_sim_metrics(results)
    # Rollout MP4s are returned SEPARATELY (not folded into `metrics`): the trainer's wandb wrapper
    # log_dict is scalar-only and silently drops media, so the caller logs these as wandb.Video via the
    # same media path as log_fk_video. One (wandb_key, mp4_path) per captured episode.
    videos = [(f"sim_eval/videos/{r['env_id']}/ep{ep['idx']}", ep["path"])
              for r in results for ep in r["episodes"]] if args.video_episodes else []
    return metrics, videos
