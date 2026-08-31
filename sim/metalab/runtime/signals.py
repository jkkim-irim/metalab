# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Signal fixups the MetaLab runtime entrypoints apply before they start work.

One thing lives here today: undoing an *inherited* ignored SIGINT — see
:func:`restore_default_sigint`. (Signal handling that belongs to the SIM SERVER's shutdown race —
``install_sigterm_unwind`` / ``shield_shutdown_from_sigterm`` — stays in ``vec_env_handler.py``.)
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

    The Launchpad server is started detached — ``launchpad.sh``: ``nohup … &`` — so it runs with
    SIGINT ignored and hands that disposition to every run it launches. Stop (``_kill_roots``)
    SIGINTs the trainer/runner and SIGKILLs it once the grace elapses, so an ignored SIGINT turned
    every Stop into a hard kill: no ``finally`` (the standalone runner's control server + trajectory
    save), and for a training run no ``wandb.finish`` — W&B timed it out as **crashed** rather than
    marking it **killed**.

    Call this first thing in an entrypoint. No-op when SIGINT is already at its default (a
    foreground run), and it must run on the main thread (``signal.signal``'s own constraint).

    ``learning`` keeps its own copy in ``learning/utils/signals.py`` — the two packages are
    deliberately import-independent.
    """
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        print("[signals] SIGINT was inherited as ignored (detached launch) — restored, so Stop/Ctrl-C "
              "unwinds cleanly", flush=True)
