"""``EvalPolicy`` — the policy contract a closed-loop sim-eval runner drives.

A runner steps a policy over the sim-service RPC boundary without knowing what the policy *is*:
``act(obs) -> (num_envs, K, action_dim)``. One model-agnostic adapter implements it — the single
``policies/vla.py`` ``VlaEvalPolicy``, which wraps any wir_v1 chunk-policy (GR00T today), driven by
``learning/eval/vla_runner.py``; a private VLA that does not speak that contract plugs in the same
way, as its own ``EvalPolicy`` subclass in its own repo.

This is the RUNNER's seam, not the dispatcher's: ``eval_service`` selects a ``--policy`` module and
calls its module-level ``run(argv)`` — that is the only cross-policy contract. Whether a module also
uses an ``EvalPolicy`` is up to it: the VLA path does (the shared ``VlaEvalPolicy``, to reuse the
runner); the RL ``actor`` eval has its own bespoke rollout and does not.

Deliberately dependency-free (no torch / gr00t / RL stack) so the contract can be imported and
unit-tested off-node, and so importing it never pulls a policy-specific stack.
"""
from __future__ import annotations

import abc
from typing import Any


class EvalPolicy(abc.ABC):
    """Maps a batch of observations to an action chunk, for closed-loop rollout.

    ``act`` returns a ``(num_envs, K, action_dim)`` chunk:
      * ``K == 1`` for a per-step policy,
      * ``K == chunk_size`` for an action-chunking policy (a VLA), of which the runner executes
        the first ``replan_cap(K, replan_steps)`` steps before re-observing.

    The observation type is whatever the paired runner supplies (an RL obs TensorDict, or the
    ``{top, wrist, state}`` wire obs) — obs schemas converge once the sim RPC is canonicalised; until
    then each adapter owns its own obs→model translation.
    """

    #: action dimensionality the policy emits (set by the concrete adapter at build time).
    action_dim: int

    @abc.abstractmethod
    def reset(self, done: Any = None) -> None:
        """Reset per-episode policy state. Recurrent actors reset hidden state for the envs in
        ``done``; stateless policies (chunking VLAs carry no cross-step state) treat this as a no-op."""

    @abc.abstractmethod
    def act(self, obs: Any) -> Any:
        """Map ``obs`` to an action chunk of shape ``(num_envs, K, action_dim)``."""
