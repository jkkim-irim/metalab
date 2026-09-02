# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Adapt a duck-typed MetaLab VecEnv (``EnvDriver``) to the ``(method, payload)`` handler that
:func:`sim.metalab.transport.serve_vec_env` drives — shared by the genesis + newton spokes' ``server.py``.

The unified transport's ``serve_vec_env`` is handler-based (one callback for step/reset/get_observations
+ the cold methods) so isaaclab, metalab, libero, and maniskill all serve over the SAME loop. MetaLab's
server owns a duck-typed VecEnv rather than a handler, so this wraps it: the hot methods return the dicts
the loop writes into the shared CUDA-IPC buffers, and the cold methods pass straight through. The
per-step ``data_log`` / ``rerun_scene`` hooks (the record path) fire here since they need the just-computed
dones + task_success — and firing both from this ONE call site is what keeps a recorded frame and a plotted
series row on the same policy step (see dashboard/rollout_log.py).
"""
from __future__ import annotations

import signal
import sys

import torch

from sim.metalab.transport import K_TASK_SUCCESS


def install_sigterm_unwind() -> None:
    """Turn SIGTERM into a normal unwind so the server's ``finally`` block runs instead of being skipped.

    Python's default SIGTERM handling exits the process on the spot — no ``finally``, so everything the
    record path finalizes there would be lost if the client kills us while we sit in the blocking control
    read. Raising ``SystemExit`` from the handler propagates out of that read (PEP 475) and unwinds normally.

    SIGINT needs no handler — ``KeyboardInterrupt`` already unwinds, PROVIDED the process did not
    inherit an ignored SIGINT from a detached launcher (see ``runtime/signals.restore_default_sigint``).
    Pair this with :func:`shield_shutdown_from_sigterm`, which covers the other half of the race.
    """
    def _unwind(_sig, _frame):
        print("[sim-service] SIGTERM → unwinding (finalize record artifacts)", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _unwind)


def shield_shutdown_from_sigterm() -> None:
    """Ignore SIGTERM from here on, so a shutdown ALREADY IN PROGRESS runs to completion.

    The client says goodbye and then immediately kills us: it sends the transport's ``close`` (which returns
    the serve loop normally) and then, on leaving its ``sim_server`` context, calls ``Popen.terminate()``.
    Measured: that SIGTERM lands *while* the record path is still finalizing — flushing the ``.rrd`` tail and
    writing the rollout log — and aborted it half way, which is why a recording never carried its last frames.
    The client then waits 30 s before SIGKILL
    (``learning/rl/service.py``), so ignoring SIGTERM for the duration of the finalization is both safe and
    sufficient. Call it as the FIRST statement of the shutdown path.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def make_vec_env_handler(env, data_log=None, rerun_scene=None):
    """Wrap ``env`` (an EnvDriver VecEnv) as the handler ``serve_vec_env`` expects.

    Hot path: ``step``/``reset`` return ``{"obs", ...}`` dicts (the loop copies obs/rew/dones plus
    ``extras``' time_outs/task_success into the shared GPU buffers and ships the rest over the control
    channel); ``get_observations`` returns the obs TensorDict. Cold path (attrs/get_ep_len/set_ep_len/
    seed/apply_curriculum_end) returns plain results over the control channel.

    ``rerun_scene`` (a spoke's .rrd scene logger) rides the SAME per-policy-step hook as the other two, which
    is what keeps one .rrd frame equal to one series row. It exists for spokes with no rerun viewer of their
    own — genesis, whose ``step()`` the driver calls ``decimation`` times per policy step, so logging from
    inside the backend would emit that many frames per sample. newton needs none of this: its ViewerRerun is
    driven from ``step_n``, which IS one control step.

    ``data_log`` (per-step series for the HTML report) is optional and rides the same ``step`` branch.
    """

    def handler(method, payload):
        if method == "step":
            obs, rew, dones, extras = env.step(payload)
            # task_success is optional on the env; the buffer layout always carries it, so default to False.
            if K_TASK_SUCCESS not in extras:
                extras[K_TASK_SUCCESS] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            if rerun_scene is not None:
                rerun_scene.after_step()
            if data_log is not None:
                data_log.after_step(dones, extras[K_TASK_SUCCESS])
            return {"obs": obs, "rew": rew, "dones": dones, "extras": extras}
        if method == "get_observations":
            return env.get_observations()
        if method == "reset":
            obs, extras = env.reset()
            return {"obs": obs, "extras": extras}
        if method == "attrs":
            return {"num_envs": int(env.num_envs), "num_actions": int(env.num_actions),
                    "device": str(env.device), "max_episode_length": int(env.max_episode_length),
                    # the contract's tuning knobs — the trainer lifts them into the W&B run config, which
                    # is the only place a finished run's recipe can be read back from.
                    "cfg_dict": {"recipe": env.spec.recipe}}
        if method == "get_ep_len":
            return env.episode_length_buf
        if method == "set_ep_len":
            env.episode_length_buf = payload
            return True
        if method == "seed":
            return env.seed(int(payload) if payload is not None else 0)
        if method == "set_num_steps_per_env":
            env.set_num_steps_per_env(int(payload))
            return True
        if method == "curriculum_level":
            return int(env.curriculum_level())
        if method == "apply_curriculum_end":
            # payload = the TRAINING level whose world to reproduce (None/absent = snap everything to the
            # end). See EnvDriver.apply_curriculum_end — a mid-training checkpoint must be judged at the
            # GATE but run in the world it learned in.
            env.apply_curriculum_end(None if payload is None else int(payload))
            return True
        raise ValueError(f"unknown method: {method!r}")

    return handler
