# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Sim-service client — the trainer-side proxy VecEnv (learning venv; no Isaac Lab).

`SimServiceVecEnv` implements the rsl_rl `VecEnv` interface by driving `server.py` over the
RPC + CUDA-IPC transport (`sim/metalab/transport.py`): a localhost socket carries the control
channel (plus a one-shot CUDA-IPC handle bootstrap), and the hot-path obs/action payload lives in
shared GPU buffers exchanged once at connect — each step only writes the action buffer, sends a tiny
"step" signal, and reads obs/rew/dones back from the buffers (the tensors never leave the GPU).
A consumer (the eval in `learning/eval`, and the PPO trainer) drives it exactly as an in-process
VecEnv, so the only change is that the env↔consumer boundary is now a process boundary — the
Isaac Lab / Newton stack lives in the server; the policy lives client-side. GPU-only, same-GPU
(the launch scripts pin trainer + server to one physical GPU).
"""
from __future__ import annotations

import logging

from tensordict import TensorDict
import torch

from learning.rl.vec_env import VecEnv

# The sim service's wire protocol is the single source in sim/metalab/transport.py — the same module
# the server side imports, so the two ends cannot drift.
from sim.metalab.transport import (
    K_ACTION,
    K_DONES,
    K_REW,
    K_TASK_SUCCESS,
    K_TIME_OUTS,
    OBS_PREFIX,
    RpcClient,
    obs_group,
)

logger = logging.getLogger(__name__)


class _CfgShim:
    """Stand-in for ``env.cfg`` — rsl_rl only calls ``.to_dict()`` (wandb config)."""

    def __init__(self, d: dict):
        self._d = d

    def to_dict(self) -> dict:
        return self._d


class SimServiceVecEnv(VecEnv):
    """Proxy VecEnv that drives a `server.py` sim service over RPC + CUDA-IPC.

    On connect the server ships the CUDA-IPC handles for its fixed-shape shared GPU buffers; from then
    on each step writes the ``action`` buffer, sends a tiny "step" control message, and reads obs/rew/
    dones/time_outs/task_success back from the buffers — no per-step serialization, the payload stays
    on the GPU. Everything else the server puts in ``extras`` (term_reason, wbt_metrics, setup, episode
    logs, ...) arrives over the control channel and is passed through unchanged. Cold-path calls
    (attrs/seed/ep_len) ride the control channel too. Same VecEnv interface as before, so the trainer
    and eval are transport-agnostic.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, device: str | None = None):
        self._rpc = RpcClient(host, port, map_location=device)
        # the server ships the CUDA-IPC handles right after accept (before its serve loop), so open
        # them first — every subsequent hot-path call reads/writes these same-GPU buffers.
        self._buffers = self._rpc.recv_shared()
        a = self._ctl("attrs")
        self.num_envs = a["num_envs"]
        self.num_actions = a["num_actions"]
        self.max_episode_length = a["max_episode_length"]
        self.device = device or a["device"]
        self.cfg = _CfgShim(a["cfg_dict"])
        # gym Spaces stay server-side (a sim/env concern, not the trainer's) — keeping them off the
        # wire means the trainer needs no gymnasium. rsl_rl infers dims from num_actions + the obs.
        self.observation_space = None
        self.action_space = None

    def _ctl(self, method: str, payload=None):
        """One control-channel round-trip (cold-path result, or a hot-path signal whose tensor payload
        rides the shared buffers). Raises on a server-side error."""
        self._rpc.send_ctl((method, payload))
        r = self._rpc.recv_ctl()
        if not r.get("ok"):
            raise RuntimeError(f"sim service error: {r.get('error')}\n{r.get('tb', '')}")
        return r.get("result")

    def _read_obs(self) -> TensorDict:
        """Rebuild the obs TensorDict from the shared obs buffers (clone → decouple from the next step)."""
        groups = {obs_group(k): v.clone() for k, v in self._buffers.items() if k.startswith(OBS_PREFIX)}
        return TensorDict(groups, batch_size=[self.num_envs])

    def get_observations(self):  # noqa: D102
        self._ctl("get_observations")   # signal; server writes the obs buffers + syncs before replying
        return self._read_obs()

    def reset(self):  # noqa: D102
        r = self._ctl("reset")
        return self._read_obs(), r["extras"]

    def step(self, actions):  # noqa: D102
        self._buffers[K_ACTION].copy_(actions)
        torch.cuda.synchronize()            # the action write must land before the server reads it
        r = self._ctl("step")               # server steps, writes the buffers, syncs, replies
        extras = dict(r["extras"])          # ctl-channel extras (term_reason, logs, ...) pass through
        extras["time_outs"] = self._buffers[K_TIME_OUTS].clone()
        extras["task_success"] = self._buffers[K_TASK_SUCCESS].clone()
        return (self._read_obs(), self._buffers[K_REW].clone(),
                self._buffers[K_DONES].clone(), extras)

    @property
    def episode_length_buf(self):
        return self._ctl("get_ep_len")

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._ctl("set_ep_len", value)

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self._ctl("seed", seed)

    def set_num_steps_per_env(self, n: int) -> None:
        """Publish this run's rollout length to the env (see EnvDriver.set_num_steps_per_env). The
        experiment config is the single source; a curriculum gating on iterations reads it back."""
        self._ctl("set_num_steps_per_env", int(n))

    def curriculum_level(self) -> int:
        """The env's CURRENT curriculum training level (0 if it has none) — the trainer asks its TRAINING env
        for this and hands it to the per-checkpoint recorder, so the clip runs in that checkpoint's world."""
        return int(self._ctl("curriculum_level") or 0)

    def apply_curriculum_end(self, train_level: int | None = None):
        """Freeze the env's curriculum for eval/recording (the server has no
        such method and this is never called on that path).

        ``train_level=None`` snaps to the END world. ``train_level=L`` runs the WORLD of level L — what a
        mid-training checkpoint needs once the curriculum ramps physics (gravity/mass), since the end-world is
        a different task from the one it learned in. Success is the GATE's either way."""
        return self._ctl("apply_curriculum_end", train_level)

    def close(self) -> None:  # noqa: D102
        self._rpc.close()
        self._buffers = {}   # drop IPC tensor refs so the client's CUDA-IPC handles close on teardown


class NaNSafeVecEnv(VecEnv):
    """Client-side NaN/inf sanitizer that composes any VecEnv.

    Mirrors allex_rl's NaNSafeRslRlVecEnvWrapper but wraps (rather than subclasses) the
    base env, so it works over the SimServiceVecEnv boundary: unconditionally on-GPU
    ``nan_to_num`` on every obs group + reward (branchless, no per-step host sync), with a
    deferred circuit-breaker read every ``abort_check_interval`` steps.
    """

    def __init__(self, env: VecEnv, abort_dead_ratio: float = 0.5, abort_check_interval: int = 256):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.cfg = env.cfg
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self._abort_ratio = abort_dead_ratio
        self._interval = max(1, int(abort_check_interval))
        self._acc = None
        self._step_count = 0
        self._total_nan = 0

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    @staticmethod
    def _sanitize_obs(obs):
        for k in list(obs.keys()):
            v = obs[k]
            if torch.is_tensor(v):
                obs[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def get_observations(self):  # noqa: D102
        return self._sanitize_obs(self.env.get_observations())

    def reset(self):  # noqa: D102
        obs, extras = self.env.reset()
        return self._sanitize_obs(obs), extras

    def step(self, actions):  # noqa: D102
        obs, rew, dones, extras = self.env.step(actions)
        bad = ~torch.isfinite(rew)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        for k in list(obs.keys()):
            v = obs[k]
            if torch.is_tensor(v):
                bad = bad | (~torch.isfinite(v)).reshape(self.num_envs, -1).any(dim=1)
                obs[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        step_bad = bad.sum()
        if self._acc is None:
            self._acc = torch.zeros(2, dtype=torch.long, device=step_bad.device)
        self._acc[0] += step_bad
        self._acc[1] = torch.maximum(self._acc[1], step_bad)
        self._step_count += 1
        if self._step_count % self._interval == 0:
            total, worst = self._acc.tolist()
            self._acc.zero_()
            if total:
                self._total_nan += int(total)
                ratio = worst / self.num_envs
                logger.warning("[NaNSafe] %d non-finite env-events over %d steps (worst %d/%d=%.1f%%; lifetime %d)",
                               int(total), self._interval, int(worst), self.num_envs, ratio * 100.0, self._total_nan)
                if ratio > self._abort_ratio:
                    raise RuntimeError(f"non-finite env ratio {ratio:.1%} > {self._abort_ratio:.0%} — training broken.")
        return obs, rew, dones, extras

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.env.seed(seed)

    def close(self) -> None:  # noqa: D102
        return self.env.close()
