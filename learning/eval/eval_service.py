"""One closed-loop sim-eval entrypoint, over the sim-service RPC boundary.

Selects a policy adapter by ``--policy`` and hands off to it. Today there is one:

  * ``--policy actor`` (default) — a trained RL MLP/RNN actor on any sim served over the sim-service
    (``learning.eval.policies.actor``; the sim is selected by the ``SIM`` env var, its spawn recipe by
    ``sim/<SIM>/launch.py``). Driven by ``learning/scripts/local/metalab_eval.sh``.

Each ``--policy`` module exposes a module-level ``run(argv)`` — the only contract this dispatcher
needs, so a new policy plugs in as its own ``policies/<name>.py`` without touching this generic path.
The chosen module is imported *dynamically*, so a policy whose stack is not installed never gets
imported just because this dispatcher loaded.

Run:
    python -m learning.eval.eval_service --policy actor --checkpoint <model.pt> [--num_envs 64 ...]
Pass ``--policy <name> --help`` to see that policy's full arg set.
"""
from __future__ import annotations

import argparse
import importlib

from learning.utils.signals import restore_default_sigint

_POLICIES = ("actor",)


def main():
    # A detached launcher (Launchpad `nohup … &`) hands us an ignored SIGINT — restore it so the
    # Stop button's SIGINT unwinds the rollout (and finishes its W&B run) instead of being discarded.
    restore_default_sigint()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--policy", choices=_POLICIES, default="actor",
                     help="policy to eval: 'actor' (RL actor on any sim-service sim). Selects the adapter.")
    known, rest = pre.parse_known_args()
    # Import ONLY the chosen policy module (dynamic, so the other stack is never imported).
    mod = importlib.import_module(f"learning.eval.policies.{known.policy}")
    return mod.run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
