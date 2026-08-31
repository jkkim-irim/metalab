"""One closed-loop sim-eval entrypoint for any policy, over the sim-service RPC boundary.

Selects a policy adapter by ``--policy`` and hands off to it:
  * ``--policy actor`` (default) — a trained RL MLP/RNN actor on ANY sim served over the sim-service
    (``learning.eval.policies.actor``; the sim is selected by the ``SIM`` env var, its spawn recipe by
    ``sim/<SIM>/launch.py``). Driven by ``learning/scripts/aws/rl_eval.sh`` and
    ``learning/scripts/local/rl_eval.sh``.
  * ``--policy groot`` — a fine-tuned GR00T VLA on the LIBERO / MIKASA sim
    (``learning.eval.policies.groot``).

Each ``--policy`` module exposes a module-level ``run(argv)`` — the only contract this dispatcher
needs. The VLA path additionally shares a model-agnostic ``EvalPolicy`` adapter + runner
(``policies/vla.py`` + ``vla_runner.py``) — ``policies/groot.py`` only loads the GR00T model into it;
the RL ``actor`` eval has its own rollout. A new policy — including a private model — plugs in as its
own ``policies/<name>.py`` exposing ``run`` without touching this generic path.

The selected policy module is imported *dynamically*: the actor path pulls the RL / Isaac stack and the
groot path pulls the GR00T stack, and the two run in different venvs — importing both at once would
fail. Dynamic dispatch keeps ``python -m learning.eval.eval_service`` loadable in either env.

Run:
    python -m learning.eval.eval_service --policy actor --checkpoint <model.pt> [--num_envs 64 ...]
    python -m learning.eval.eval_service --policy groot --checkpoint <ckpt_dir> --dataset-root <wir> \
        --suite libero_90 --tasks 0,11,42 [...]
Pass ``--policy <name> --help`` to see that policy's full arg set.
"""
from __future__ import annotations

import argparse
import importlib

from learning.utils.signals import restore_default_sigint

_POLICIES = ("actor", "groot", "metalab")


def main():
    # A detached launcher (Launchpad `nohup … &`) hands us an ignored SIGINT — restore it so the
    # Stop button's SIGINT unwinds the rollout (and finishes its W&B run) instead of being discarded.
    restore_default_sigint()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--policy", choices=_POLICIES, default="actor",
                     help="policy to eval: 'actor' (RL actor on any sim-service sim) or 'groot' "
                          "(GR00T VLA on LIBERO/MIKASA). Selects the adapter.")
    known, rest = pre.parse_known_args()
    # Import ONLY the chosen policy module (dynamic, so the other stack is never imported).
    mod = importlib.import_module(f"learning.eval.policies.{known.policy}")
    return mod.run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
