"""``VlaEvalPolicy`` — the one closed-loop sim-eval adapter for an action-chunking policy.

Wraps ANY policy exposing the shared BC contract ``predict_action_chunk(batch) -> (N, K, action_dim)``
— the wir_v1 LeRobot interface that both the ACT and GR00T policies implement — and drives it as an
``EvalPolicy``: it maps the sim wire obs ``{top,wrist,state}`` onto the wir_v1 batch, runs the model,
and returns the predicted action chunk. Model-agnostic: it never names a model. GR00T is the only such
model today — ``policies/groot.py`` reconstructs it from a checkpoint and hands it here; the in-training
eval hands its LIVE model here. A private VLA plugs in the same way (or, if it does not speak
``predict_action_chunk``, as its own ``EvalPolicy`` subclass).

The machinery that DRIVES this policy — the closed-loop runner, sim-server selection/spawn, video
capture, wandb logging, and the shared CLI args — lives in ``learning/eval/vla_runner.py``, not here.
"""
from __future__ import annotations

import torch

from learning.eval.policies.eval_policy import EvalPolicy
from learning.model.act.constants import ACTION, OBS_STATE


def _make_batch(obs: dict, prompt: str, action_dim: int, chunk_size: int, device: str) -> dict:
    """{top,wrist: (N,H,W,3) uint8, state: (N,7)} -> the wir_v1 batch predict_action_chunk consumes."""
    batch = {}
    for cam in ("top", "wrist"):
        img = obs[cam].to(device)                      # (N,H,W,3) uint8
        batch[f"observation.images.{cam}"] = img.permute(0, 3, 1, 2).float().div(255.0)  # (N,C,H,W)
    n = obs["state"].shape[0]
    batch[OBS_STATE] = obs["state"].to(device).float()  # (N, state_dim)
    # predict_action_chunk only reads batch[ACTION].device — a device carrier, not a target.
    batch[ACTION] = torch.zeros(n, chunk_size, action_dim, device=device)
    batch["task"] = [prompt] * n
    return batch


class VlaEvalPolicy(EvalPolicy):
    """Drives any wir_v1 chunk-policy (``predict_action_chunk``) as a closed-loop ``EvalPolicy``.

    Built ONCE around a policy object — the live training model, or one reconstructed from a checkpoint
    by ``policies/groot.py``. The per-task language instruction + action_dim come from the sim server's
    ``attrs`` and are bound via ``set_task`` before each task's rollout (the policy itself is
    task-agnostic). Stateless per step — the prompt rides each batch — so ``reset`` is a no-op."""

    def __init__(self, policy, chunk_size: int, device: str):
        self._policy = policy                          # any model exposing predict_action_chunk
        self.chunk_size = chunk_size
        self.device = device
        self.prompt: str | None = None
        self.action_dim: int | None = None

    def set_task(self, prompt: str, action_dim: int) -> None:
        """Bind the per-task language instruction + action_dim (from the server ``attrs``)."""
        self.prompt = prompt
        self.action_dim = action_dim

    def reset(self, done=None) -> None:
        pass  # stateless per step; the prompt rides each batch

    def act(self, obs: dict) -> torch.Tensor:
        """obs {top,wrist,state} -> (N, chunk_size, action_dim) predicted action chunk."""
        batch = _make_batch(obs, self.prompt, self.action_dim, self.chunk_size, self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._policy.predict_action_chunk(batch)  # (N, chunk_size, action_dim)
