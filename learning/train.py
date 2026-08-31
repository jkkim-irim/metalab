#!/usr/bin/env python
"""MetaLab — training entrypoint.

Dispatches to the RL trainer, which runs the engine sim-service + PPO/SAPG stack behind the RPC
boundary (``learning/trainer/rl_trainer.py``). Every run knob is re-parsed there from the remaining
argv (--sim / --task / --recipe / --num_envs / --max_iterations / --seed / --device / …).

Run as a module::

    python -m learning.train --trainer rl --task hammer-lift-teacher --recipe privileged

In practice this is reached through ``learning/scripts/local/metalab_train.sh``, which activates the
engine's uv venv first.
"""
import argparse
import sys

# restore_default_sigint: a detached launcher hands us an IGNORED SIGINT, which would make Stop a silent
# no-op followed by SIGKILL — and the W&B run "crashed" instead of "killed".
from learning.trainer.rl_trainer import RLTrainer
from learning.utils.signals import restore_default_sigint


def main() -> int:
    restore_default_sigint()
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--trainer", choices=["rl"], default="rl",
                   help="which trainer to run (rl — the sim-service + PPO stack)")
    known, rest = p.parse_known_args(sys.argv[1:])
    assert known.trainer == "rl", f"unknown trainer: {known.trainer!r}"
    return RLTrainer(rest).run()


if __name__ == "__main__":
    raise SystemExit(main())
