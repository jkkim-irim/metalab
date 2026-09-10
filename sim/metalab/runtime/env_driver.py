from __future__ import annotations

import math
import time

import torch

from sim.metalab.api.backend import assert_backend
from sim.metalab.dashboard.telemetry import NO_DASHBOARD, LiveDashboard
from sim.metalab.runtime import snapshot as snap
from sim.metalab.runtime.obs_pipeline import ObsPipeline
from sim.metalab.terms.gate import hold_and_pass


class EnvDriver:
    def __init__(self, spec, backend, max_episode_length: int, telemetry: bool = False):
        self.spec = spec
        self.backend = backend
        self.num_envs = int(backend.num_envs)
        self.device = backend.device
        self.cfg = {"task": spec.name}
        self.max_episode_length = int(max_episode_length)
        self.step_dt = spec.physics.dt * spec.physics.decimation   # [s]
        self.max_episode_length_s = self.max_episode_length * self.step_dt   # [s]
        self.capabilities = assert_backend(backend)
        self.fingertips = spec.robot.fingertips
        self.physics_gravity_z = float(spec.physics.gravity[2])   # [m/s^2]

        self.action_groups = list(spec.action.items())
        self.num_actions = sum(g.term.action_dim for _, g in self.action_groups)
        assert self.num_actions > 0, "action groups are empty (env_driver requires an action)"

        self._init_action_pipeline()
        self._init_obs_pipeline()
        self._init_task_state()
        self._init_registries()
        self._init_term_buffers()
        self._telem = LiveDashboard.start(self) if telemetry else NO_DASHBOARD

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "backend"), name)

    def buffer(self, key: str, shape=(), fill: float = 0.0, dtype=torch.float32) -> torch.Tensor:
        assert self._cur_term is not None, "env.buffer() is only callable from inside a term"
        k = (*self._cur_term, key)
        buf = self._term_bufs.get(k)
        if buf is None:
            buf = torch.full((self.num_envs, *shape), fill, dtype=dtype, device=self.device)
            self._term_bufs[k] = buf
            self._term_fill[k] = fill
        return buf

    def run_term(self, kind: str, t, *args):
        assert self._cur_term is None, f"run_term({kind}/{t.name}) while {self._cur_term} is running"
        self._cur_term = (kind, t.name)
        try:
            return t.fn(self, *args, **t.params)
        finally:
            self._cur_term = None

    def curriculum_success_rate(self) -> float:
        return float(self._last_ep_curr_success.float().mean().item())

    def set_curriculum_bars(self, hold_steps: int, **bars) -> None:
        self._curr_hold_steps = int(hold_steps)
        self._curr_bars = bars

    def set_num_steps_per_env(self, n: int) -> None:
        assert n >= 1, f"num_steps_per_env must be >= 1 — got {n}"
        self._num_steps_per_env = int(n)

    @property
    def num_steps_per_env(self) -> int:
        assert self._num_steps_per_env is not None, (
            "num_steps_per_env was never published to this env, and a curriculum term is asking for it to "
            "count iterations. The trainer publishes it before the first reset (learning/rl/client.py); an "
            "eval or standalone run has no rollout length, so freeze the curriculum first "
            "(apply_curriculum_end).")
        return self._num_steps_per_env

    def set_dr_value(self, key: str, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        buf = self.dr_values.get(key)
        if buf is None:
            buf = torch.zeros(self.num_envs, device=self.device)
            self.dr_values[key] = buf
        buf[env_ids] = values.to(dtype=buf.dtype)

    def dr_value(self, key: str) -> torch.Tensor:
        buf = self.dr_values.get(key)
        if buf is not None:
            return buf
        assert not self._reset_done, (
            f"dr_value asks for {key!r}, which no DR event of this contract publishes — published: "
            f"{sorted(self.dr_values)}.")
        return torch.zeros(self.num_envs, device=self.device)

    def _init_action_pipeline(self) -> None:
        spec, d = self.spec, self.device
        self.last_action = torch.zeros(self.num_envs, self.num_actions, device=d)
        self._ema = [None] * len(self.action_groups)
        self._ema_alpha = [None if g.ema_tau is None else 1.0 - math.exp(-self.step_dt / g.ema_tau)
                           for _, g in self.action_groups]
        self._ema_order = [g.ema_order for _, g in self.action_groups]
        self._default = [self.backend.joint_pos(g.joints).clone() for _, g in self.action_groups]
        self._limits = [self.backend.joint_limits(g.joints) for _, g in self.action_groups]
        for (_, g), lim, dflt in zip(self.action_groups, self._limits, self._default):
            g.term.check(default=dflt, limits=lim)
            g.term.bind(self)
        self.prev_action_targets = torch.cat([x.clone() for x in self._default], dim=-1)   # [rad]
        sa = torch.cat([g.term.spawn_action(default=self._default[i], limits=self._limits[i])
                        for i, (_, g) in enumerate(self.action_groups)], dim=-1)
        assert bool(torch.allclose(sa, sa[:1].expand_as(sa))), "spawn pose differs across envs at build time"
        self.spawn_action = sa[0].clone()
        self._delay_min = spec.action_delay.min_delay
        self._delay_max = spec.action_delay.max_delay
        self._delay_size = self._delay_max + 1
        self.action_delay_lag = (torch.randint(self._delay_min, self._delay_max + 1,
                                         (self.num_envs,), device=d)
                           if self._delay_max > 0 else None)
        self._delay_q = ([x.unsqueeze(0).repeat(self._delay_size, 1, 1) for x in self._default]
                         if self.action_delay_lag is not None else None)

    def _init_obs_pipeline(self) -> None:
        spec = self.spec
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_reward = torch.zeros(self.num_envs, device=self.device)
        self.last_reward_terms = torch.zeros(self.num_envs, len(spec.reward), device=self.device)
        self.obs = ObsPipeline(spec, self.num_envs)

    def _init_task_state(self) -> None:
        spec = self.spec
        g = spec.goal
        if g is not None:
            self.goal_pos = torch.tensor(g.pos, dtype=torch.float32, device=self.device).unsqueeze(0).expand(self.num_envs, 3).contiguous()
            self.goal_quat = torch.tensor(g.quat, dtype=torch.float32, device=self.device).unsqueeze(0).expand(self.num_envs, 4).contiguous()
            self.goal_half_extent = tuple(g.keypoint_half_extent)
        else:
            self.goal_pos = self.goal_quat = self.goal_half_extent = None
        self.gate = spec.gate
        self.gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_bars = {} if spec.gate is None else spec.gate.bars()
        self._curr_bars = dict(self._gate_bars)
        self._curr_hold_steps = 1 if spec.gate is None else spec.gate.hold_steps
        self.curriculum_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.curriculum_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_ep_curr_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.curriculum_values: dict[str, torch.Tensor] = {}
        self._curr_key_owner: dict[str, str] = {}
        self._last_ep_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_attempts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ep_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dr_values: dict[str, torch.Tensor] = {}
        self._dropped_events = {t.name: miss for t in spec.events
                                if (miss := tuple(c for c in t.requires if c not in self.capabilities))}
        for name, miss in self._dropped_events.items():
            print(f"[metalab] {type(self.backend).__name__}: event {name!r} dropped — this engine does not "
                  f"have {list(miss)}", flush=True)
        self._events = [t for t in spec.events if t.name not in self._dropped_events]
        self._reset_events = [t for t in self._events if t.mode == "reset"]
        self._interval_events = [t for t in self._events if t.mode == "interval"]

    def _init_registries(self) -> None:
        spec = self.spec
        self._ep_rew_sums = {t.name: torch.zeros(self.num_envs, device=self.device) for t in spec.reward}
        self._ep_log: dict = {}
        self.reward_terms = {t.name: t for t in spec.reward}
        self.event_terms = {t.name: t for t in self._events}
        self._term_names = [*(t.name for t in spec.terminate), "nan_world"]
        assert len(set(self._term_names)) == len(self._term_names), \
            f"duplicate terminate term name (nan_world is reserved): {self._term_names}"
        self._last_ep_done = {n: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                              for n in self._term_names}
        self.common_step_counter = 0
        self._num_steps_per_env: int | None = None
        self._curriculum_frozen = False
        self._reset_done = False
        self._all_ids = torch.arange(self.num_envs, device=self.device)

    def _init_term_buffers(self) -> None:
        self._term_bufs: dict = {}
        self._term_fill: dict = {}
        self._cur_term: tuple[str, str] | None = None
        for t in self.spec.reward:
            self.run_term("reward", t)
        self._fill_term_buffers(self._all_ids)
        if self.spec.curriculum:
            self._run_curriculum()

    def _fill_term_buffers(self, env_ids: torch.Tensor) -> None:
        for k, buf in self._term_bufs.items():
            buf[env_ids] = self._term_fill[k]

    def reward_best(self, name: str) -> torch.Tensor:
        buf = self._term_bufs.get(("reward", name, "best"))
        assert buf is not None, (
            f"reward term {name!r} has no 'best' episode buffer — progress terms only, and only after the "
            f"first step. present: {sorted(k[1:] for k in self._term_bufs if k[0] == 'reward')}"
        )
        return buf

    def _apply_actions(self, actions: torch.Tensor) -> None:
        self.last_action = actions.detach().clone()
        off = 0
        targets = []
        for i, (_, g) in enumerate(self.action_groups):
            k = g.term.action_dim
            a = actions[:, off:off + k]
            off += k
            tgt = g.term.decode(a, default=self._default[i], limits=self._limits[i])
            alpha = self._ema_alpha[i]
            if alpha is not None:
                prev = self._ema[i]
                if prev is None:
                    self._ema[i] = [tgt.clone() for _ in range(self._ema_order[i])]
                else:
                    stages = []
                    for p in prev:
                        tgt = alpha * tgt + (1.0 - alpha) * p
                        stages.append(tgt)
                    self._ema[i] = stages
            targets.append(tgt)
            send = tgt
            if self.action_delay_lag is not None:
                q = self._delay_q[i]
                slot = self.common_step_counter % self._delay_size
                q[slot] = tgt
                send = q[(slot - self.action_delay_lag) % self._delay_size, self._all_ids]
            self.backend.set_joint_targets(g.joints, send)
        self.prev_action_targets = torch.cat(targets, dim=-1)

    def get_observations(self):
        return self.obs.observe(self)

    def _rewards(self) -> torch.Tensor:
        rew = torch.zeros(self.num_envs, device=self.device)
        for i, t in enumerate(self.spec.reward):
            term = t.weight * self.run_term("reward", t)
            self.last_reward_terms[:, i] = term
            self._ep_rew_sums[t.name] += term
            rew = rew + term
        return rew

    def _all_mask(self) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _gate_eval(self, name: str, bars: dict, hold_steps: int):
        assert self._cur_term is None, f"gate {name} while {self._cur_term} is running"
        self._cur_term = ("gate", name)
        try:
            return hold_and_pass(self, self.gate.predicate, bars, hold_steps, self.gate.hold_mode)
        finally:
            self._cur_term = None

    def _advance_curriculum(self) -> None:
        if self.gate is None:
            return
        _, self.curriculum_passed, self.curriculum_hold = self._gate_eval(
            "curriculum", self._curr_bars, self._curr_hold_steps)

    def _advance_gate(self) -> None:
        if self.gate is None:
            return
        _, self.gate_passed, _ = self._gate_eval("eval", self._gate_bars, self.gate.hold_steps)

    def _post_reset(self, env_ids: torch.Tensor) -> None:
        self._fill_term_buffers(env_ids)
        for t in self._reset_events:
            self.run_term("event", t, env_ids)
        for i, (_, g) in enumerate(self.action_groups):
            self._default[i][env_ids] = self.backend.joint_pos(g.joints)[env_ids]
            g.term.reset(env_ids)
        if self.action_delay_lag is not None:
            self.action_delay_lag[env_ids] = torch.randint(
                self._delay_min, self._delay_max + 1, (int(env_ids.numel()),), device=self.device)
            for i, q in enumerate(self._delay_q):
                q[:, env_ids] = self._default[i][env_ids]
        self.prev_action_targets[env_ids] = torch.cat([d[env_ids] for d in self._default], dim=-1)
        self.last_action[env_ids] = 0.0

    def _log_curriculum(self) -> None:
        for name, vals in self._run_curriculum().items():
            for k, v in vals.items():
                key = f"Curriculum/{k}"
                assert key not in self._ep_log or self._curr_key_owner.get(key) == name, (
                    f"two curriculum terms both publish {k!r} ({self._curr_key_owner.get(key)} and {name}) — "
                    f"the log key Curriculum/{k} cannot say which. Rename one of them.")
                self._curr_key_owner[key] = name
                self._ep_log[key] = float(v)

    def _run_curriculum(self) -> dict[str, dict[str, float]]:
        out = {}
        for t in self.spec.curriculum:
            vals = self.run_term("curriculum", t)
            out[t.name] = vals
            for k, v in vals.items():
                buf = self.curriculum_values.get(k)
                if buf is None:
                    self.curriculum_values[k] = torch.tensor(float(v), dtype=torch.float32, device=self.device)
                else:
                    buf.fill_(float(v))
        return out

    def curriculum_level(self) -> int:
        levels = [int(t.fn.level) for t in self.spec.curriculum if hasattr(t.fn, "level")]
        return max(levels) if levels else 0

    def apply_curriculum_end(self, train_level: int | None = None) -> None:
        dropped = [t.name for t in self._reset_events + self._interval_events if t.train_only]
        if dropped:
            self._reset_events = [t for t in self._reset_events if not t.train_only]
            self._interval_events = [t for t in self._interval_events if not t.train_only]
            print(f"[env] train-only events dropped for eval: {dropped}", flush=True)
        if not self.spec.curriculum:
            return
        assert not self._reset_done, (
            "apply_curriculum_end() after the env was reset: reset-time curriculum knobs (friction "
            "mu_scale, mass_scale) already bound at the old level, so episode 0 would run a different "
            "world than the rest of the eval. Freeze the curriculum BEFORE reset().")
        for t in self.spec.curriculum:
            if train_level is not None and hasattr(t.fn, "freeze_for_eval"):
                t.fn.freeze_for_eval(train_level)
            elif hasattr(t.fn, "jump_to_end"):
                t.fn.jump_to_end()
        how = "END" if train_level is None else f"level-{train_level} world"
        for name, vals in self._run_curriculum().items():
            print(f"[env] curriculum '{name}' frozen at {how}: {vals}", flush=True)
        self._curriculum_frozen = True

    def reset(self):
        self._reset_done = True
        self.backend.reset_idx(self._all_mask())
        self.episode_length_buf.zero_()
        self.last_action = torch.zeros_like(self.last_action)
        self.last_reward = torch.zeros_like(self.last_reward)
        for s in self._ep_rew_sums.values():
            s.zero_()
        self._ep_log = {}
        self._post_reset(self._all_ids)
        self._ema = [[d.clone() for _ in range(o)] for d, o in zip(self._default, self._ema_order)]
        if self.spec.curriculum and not self._curriculum_frozen:
            self._log_curriculum()
        self.obs.reset()
        return self.get_observations(), {"log": dict(self._ep_log)}

    def step(self, actions: torch.Tensor):
        self._telem.drain(self)
        self._apply_actions(actions)
        self._hold_while_paused()
        for t in self._interval_events:
            self.run_term("event", t, self._all_ids)
        if self.goal_pos is not None:
            self.backend.set_goal_markers(self.goal_pos)
        if "batched_step" in self.capabilities:
            self.backend.step_n(self.spec.physics.decimation)
        else:
            for _ in range(self.spec.physics.decimation):
                self.backend.step()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._advance_curriculum()
        rew = self._rewards()
        self.last_reward = rew.detach()
        self._advance_gate()
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(dones)
        term_dones = []
        for t in self.spec.terminate:
            m = self.run_term("terminate", t)
            term_dones.append((t.name, m))
            dones = dones | m
            if t.time_out:
                truncated = truncated | m
        nan_m = self.backend.nan_world_detected()
        term_dones.append(("nan_world", nan_m))
        dones = dones | nan_m

        d = None
        if bool(dones.any()):
            d = dones.nonzero(as_tuple=False).flatten()
            self._reset_done = True
            self._ep_attempts[d] += 1
            reached = next((m for n, m in term_dones if n == "curriculum_passed"), None)
            if reached is not None:
                self._ep_successes[d] += reached[d].long()
            if self.gate is not None:
                self._last_ep_success[d] = self.gate_passed[d]
            self._last_ep_curr_success[d] = self.curriculum_passed[d]
            self.backend.reset_idx(dones)
            self._post_reset(d)
            for name, m in term_dones:
                self._last_ep_done[name][d] = m[d]
            rnames = list(self._ep_rew_sums)
            tnames = [n for n, _ in term_dones]
            vals = torch.stack([self._ep_rew_sums[n][d].mean() / self.max_episode_length for n in rnames]
                               + [self._last_ep_done[n].float().mean() for n in tnames]).tolist()
            for n, v in zip(rnames, vals[:len(rnames)]):
                self._ep_log[f"Reward/{n}"] = v
            for n, v in zip(tnames, vals[len(rnames):]):
                self._ep_log[f"Termination/{n}"] = v
            for n in rnames:
                self._ep_rew_sums[n][d] = 0.0
            self.episode_length_buf[dones] = 0
            for i in range(len(self.action_groups)):
                if self._ema[i] is not None and self._default[i] is not None:
                    self._ema[i] = [s.clone() for s in self._ema[i]]
                    for s in self._ema[i]:
                        s[dones] = self._default[i][dones]
            if self.spec.curriculum and not self._curriculum_frozen:
                self._log_curriculum()
            if self.gate is not None:
                self._ep_log["val/SR"] = float(self._last_ep_success.float().mean().item())

        frames = self.obs.capture(self)
        self.obs.advance_history(frames, d)
        obs = self.obs.stack(frames)
        extras = {"time_outs": truncated, "log": dict(self._ep_log)}
        if self.gate is not None:
            extras["task_success"] = self._last_ep_success
        self._telem.publish(self, obs)
        return obs, rew, dones, extras

    def seed(self, seed: int) -> int:
        torch.manual_seed(seed)
        if self.action_delay_lag is not None:
            self.action_delay_lag = torch.randint(self._delay_min, self._delay_max + 1,
                                                  (self.num_envs,), device=self.device)
        return seed

    def snapshot_describe(self) -> dict:
        return snap.describe(self, self)

    def _hold_while_paused(self) -> None:
        held = False
        while self._telem.paused or not self.backend.viewer_step_allowed():
            if not held:
                held = True
                self._telem.note_pause(self, True)
            self.backend.pump_viewer()
            self._telem.drain(self)
            time.sleep(0.01)
        if held:
            self._telem.note_pause(self, False)

    def snapshot_rows(self, idxs, extra=None) -> dict:
        return snap.rows(self, idxs, extra)
