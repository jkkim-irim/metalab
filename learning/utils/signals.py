# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Signal fixups every ``learning`` entrypoint applies before it starts work.

One thing lives here today: undoing an *inherited* ignored SIGINT — see
:func:`restore_default_sigint`.
"""
from __future__ import annotations

import signal


def restore_default_sigint() -> None:
    """Re-arm Ctrl-C when we were started with SIGINT already set to ``SIG_IGN``.

    POSIX requires a shell to ignore SIGINT (and SIGQUIT) in a command it runs ASYNCHRONOUSLY
    (``cmd &``) while job control is off — i.e. in every non-interactive shell. ``SIG_IGN`` survives
    ``exec`` and is inherited by every descendant, and CPython installs its ``default_int_handler``
    only when it finds SIGINT at ``SIG_DFL``. A process started down such a chain therefore never
    raises ``KeyboardInterrupt``: SIGINT is discarded, silently.

    BOTH detached launch paths do exactly that, so every run they started was uninterruptible:
      * the Launchpad server (``sim/metalab/launchpad.sh``: ``nohup … &``) — its ignored SIGINT is
        inherited by every run the Stop button later tries to interrupt;
      * ``learning/scripts/aws/metalab_train.sh`` (``setsid bash … &`` in the SSM shell).
    Stop SIGINTs the trainer and SIGKILLs it once the grace elapses, so an ignored SIGINT turned
    every Stop into a hard kill: the trainer never reached ``wandb.finish(exit_code=255)`` and W&B
    timed the run out as **crashed** instead of marking it **killed**.

    Call this first thing in an entrypoint. No-op when SIGINT is already at its default (a
    foreground run), and it must run on the main thread (``signal.signal``'s own constraint).

    ``sim`` keeps its own copy in ``sim/metalab/runtime/signals.py`` — the two packages are
    deliberately import-independent (``learning`` never imports ``sim``).
    """
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        print("[signals] SIGINT was inherited as ignored (detached launch) — restored, so Stop/Ctrl-C "
              "unwinds cleanly (W&B 'killed', not 'crashed')", flush=True)
