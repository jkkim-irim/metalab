"""GR00T checkpoint loader + ``--policy groot`` CLI for closed-loop sim-eval.

NOT an eval-policy adapter — the adapter is the model-agnostic ``policies.vla.VlaEvalPolicy``. This
module does the one GR00T-specific thing standalone eval needs: reconstruct a fine-tuned GR00T policy
from a checkpoint (a config — model architecture + pretrained weights + modality) via the SAME
``groot_policy.build`` the trainer uses, then hand the model to ``VlaEvalPolicy`` and run the shared
closed-loop runner. In-training sim-eval skips this entirely — it wraps the LIVE model directly.

Runs in the GR00T venv (Isaac-GR00T + learning). ``run`` spawns the sim-service server in the SIM venv
(``--sim-python``) via the runner; see ``vla_runner`` for the rollout + obs contract.
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

from learning.data.lerobot_dataset import LeRobotDatasetMetadata
from learning.eval import vla_runner
from learning.eval.policies.vla import VlaEvalPolicy
from learning.eval.sim_eval_helpers import parse_tasks, resolve_suite
from learning.model import groot_policy
from learning.model.groot.configuration import GrootConfig


def load_policy(checkpoint: str, dataset_root: str, modality: str, chunk_size: int,
                num_inference_timesteps, device: str):
    """Reconstruct the fine-tuned GR00T policy from a checkpoint via ``groot_policy.build``
    (base_model_path = the checkpoint dir; save_pretrained wrote the model+processor there). Returns
    the raw GR00T policy in eval mode — the caller wraps it in ``VlaEvalPolicy``."""
    meta = LeRobotDatasetMetadata(os.path.basename(dataset_root.rstrip("/")), root=dataset_root)
    pcfg = GrootConfig(modality=modality, base_model_path=checkpoint, device=device,
                       chunk_size=chunk_size, num_inference_timesteps=num_inference_timesteps)
    # build() reads only cfg.policy + cfg.steps; steps is irrelevant at eval (the optimizer is unused).
    policy, *_ = groot_policy.build(SimpleNamespace(policy=pcfg, steps=1), SimpleNamespace(meta=meta))
    policy.eval()
    return policy


def add_args(p: argparse.ArgumentParser) -> None:
    """Register the ``--policy groot`` flag set: the GR00T checkpoint-load flags, plus the shared VLA
    sim-eval args from ``vla_runner.add_vla_args``."""
    p.add_argument("--checkpoint", required=True, help="fine-tuned GR00T checkpoint dir")
    p.add_argument("--dataset-root", required=True, help="wir_v1 dataset dir (for stats/cameras)")
    p.add_argument("--num-inference-timesteps", type=int, default=None, help="fast profile (e.g. 2)")
    vla_runner.add_vla_args(p)


def run(argv=None) -> None:
    """Closed-loop GR00T eval over one or more tasks in a suite (the ``--policy groot`` entrypoint)."""
    p = argparse.ArgumentParser(description="Closed-loop sim-eval of a GR00T VLA policy.")
    add_args(p)
    args = p.parse_args(argv)

    modality, task_kind = resolve_suite(args.suite)      # suite -> server/obs layout + how to read --tasks
    tasks = parse_tasks(task_kind, args.tasks)
    server_script = vla_runner.server_for(modality, args.server_script)

    # The model depends only on the modality (obs/state layout), not the task -> load it ONCE and reuse
    # across the whole task set, wrapped in the model-agnostic eval adapter.
    policy = load_policy(args.checkpoint, args.dataset_root, modality, args.chunk_size,
                         args.num_inference_timesteps, args.device)
    eval_policy = VlaEvalPolicy(policy, args.chunk_size, args.device)

    video_root = args.video_dir or os.path.join(args.checkpoint, "sim_eval_videos")
    vla_runner.run_eval(eval_policy, args, modality=modality, tasks=tasks, server_script=server_script,
                        video_root=video_root)
