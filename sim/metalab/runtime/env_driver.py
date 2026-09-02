from __future__ import annotations

import math
import time

from tensordict import TensorDict
import torch

from sim.metalab.api.backend import assert_backend
from sim.metalab.dashboard.telemetry import NO_DASHBOARD, LiveDashboard
from sim.metalab.runtime import snapshot as snap


def _hold_count(count: torch.Tensor, ok: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cumulative":
        return count + ok.to(count.dtype)
    assert mode == "consecutive", f"hold_count: unknown mode {mode!r} (consecutive | cumulative)"
    return torch.where(ok, count + 1, torch.zeros_like(count))


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
        self.num_actions = sum(len(g.joints) for _, g in self.action_groups)
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
        if self._cur_rew_term is not None:
            return self._reward_buffer(key, shape, fill, dtype)
        assert self._cur_evt_term is not None, \
            "env.buffer() is only callable from inside a reward or event term"
        return self._event_buffer(key, shape, fill, dtype)

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

    def set_object_seen_steps(self, steps: int) -> None:
        assert steps >= 1, f"object seen window must be >= 1 step (1 = the spawn pose) — got {steps}"
        self._obj_seen_steps = int(steps)

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
        self._default = [self.backend.joint_pos(g.joints).clone() for _, g in self.action_groups]
        self._limits = [self.backend.joint_limits(g.joints) for _, g in self.action_groups]
        self.prev_action_targets = torch.cat([x.clone() for x in self._default], dim=-1)   # [rad]
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
        self._prev_vel: dict = {}   # [rad/s]
        self._joint_acc_buf: dict = {}   # [rad/s^2]
        self._acc_dt = self.step_dt
        self._obs_hist_len = dict(spec.obs_history_length)
        self._obs_hist: dict = {}
        self._obs_noise_groups = set(spec.obs_noise_groups)

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
        self._gate_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contact_steps: dict[tuple, torch.Tensor] = {}
        self.palm_body = spec.robot.frames.get("palm") if spec.gate is not None else None
        if spec.gate is not None and spec.gate.palm_distance > 0.0:
            assert self.palm_body is not None, (
                f"GATE.palm_distance={spec.gate.palm_distance} needs a robot frame named 'palm' — "
                f"robot({spec.robot.asset.get('mjcf')}) declares frames {sorted(spec.robot.frames)}. "
                f"On a two-handed robot name the hand the task uses.")
        self._gate_bars = {} if spec.gate is None else {
            "goal_dist_tol": spec.gate.goal_dist_tol,
            "palm_distance": spec.gate.palm_distance, "contact_count": spec.gate.contact_count,
            "contact_fingers": tuple(spec.gate.contact_fingers),
            "force_threshold": spec.gate.force_threshold, "joint_pose": spec.gate.joint_final_pose,
            "joint_pose_tolerance": spec.gate.joint_pose_tolerance,
        }
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
        self.object_init_z = (self.backend.object_pos()[:, 2].clone() if spec.movable_objects
                               else torch.zeros(self.num_envs, device=self.device))
        self.object_seen_pose_w = (
            torch.cat([self.backend.object_pos(), self.backend.object_quat()], dim=-1).clone() if spec.movable_objects
            else torch.zeros(self.num_envs, 7, device=self.device))
        self._obj_seen_steps = 1
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
        self._term_names = ["time_out", *(t.name for t in spec.terminate), "nan_world"]
        assert len(set(self._term_names)) == len(self._term_names), \
            f"duplicate terminate term name (time_out is reserved): {self._term_names}"
        self._last_ep_done = {n: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                              for n in self._term_names}
        self.common_step_counter = 0
        self._num_steps_per_env: int | None = None
        self._curriculum_frozen = False
        self._reset_done = False
        self._all_ids = torch.arange(self.num_envs, device=self.device)

    def _init_term_buffers(self) -> None:
        self._rew_bufs: dict = {}
        self._rew_buf_fill: dict = {}
        self._cur_rew_term: str | None = None
        self._evt_bufs: dict = {}
        self._evt_buf_fill: dict = {}
        self._cur_evt_term: str | None = None
        _warm = self
        for t in self.spec.reward:
            self._cur_rew_term = t.name
            t.fn(_warm, **t.params)
        self._cur_rew_term = None
        for _k, _buf in self._rew_bufs.items():
            _buf[:] = self._rew_buf_fill[_k]

        if self.spec.curriculum:
            self._run_curriculum()

    def _reward_buffer(self, key: str, shape, fill: float, dtype) -> torch.Tensor:
        assert self._cur_rew_term is not None, "EnvDriver.buffer() is only callable from inside a reward term"
        k = (self._cur_rew_term, key)
        buf = self._rew_bufs.get(k)
        if buf is None:
            buf = torch.full((self.num_envs, *shape), fill, dtype=dtype, device=self.device)
            self._rew_bufs[k] = buf
            self._rew_buf_fill[k] = fill
        return buf

    def _event_buffer(self, key: str, shape, fill: float, dtype) -> torch.Tensor:
        assert self._cur_evt_term is not None, "EnvDriver.buffer() is only callable from inside an event term"
        k = (self._cur_evt_term, key)
        buf = self._evt_bufs.get(k)
        if buf is None:
            buf = torch.full((self.num_envs, *shape), fill, dtype=dtype, device=self.device)
            self._evt_bufs[k] = buf
            self._evt_buf_fill[k] = fill
        return buf

    def reward_best(self, name: str) -> torch.Tensor:
        buf = self._rew_bufs.get((name, "best"))
        assert buf is not None, (
            f"reward term {name!r} has no 'best' episode buffer — progress terms only, and only after the "
            f"first step. present: {sorted(k for k in self._rew_bufs)}"
        )
        return buf

    def joint_acc(self, names: list[str]) -> torch.Tensor:
        key = tuple(names)
        buf = self._joint_acc_buf.get(key)
        if buf is None:
            buf = torch.zeros(self.num_envs, len(names), device=self.device)
            self._joint_acc_buf[key] = buf
            self._prev_vel[key] = self.backend.joint_vel(names).detach().clone()
        return buf

    def _advance_joint_acc(self) -> None:
        for key, buf in self._joint_acc_buf.items():
            cur = self.backend.joint_vel(list(key))
            buf.copy_((cur - self._prev_vel[key]) / self._acc_dt)
            self._prev_vel[key].copy_(cur)

    def _apply_actions(self, actions: torch.Tensor) -> None:
        self.last_action = actions.detach().clone()
        off = 0
        targets = []
        for i, (_, g) in enumerate(self.action_groups):
            k = len(g.joints)
            a = actions[:, off:off + k]
            off += k
            default = self._default[i]
            tgt = a * g.scale if default is None else default + a * g.scale
            alpha = self._ema_alpha[i]
            if alpha is not None:
                prev = self._ema[i]
                tgt = tgt if prev is None else alpha * tgt + (1.0 - alpha) * prev
            lim = self._limits[i]
            if lim is not None:
                tgt = torch.clamp(tgt, lim[0], lim[1])
            if alpha is not None:
                self._ema[i] = tgt
            targets.append(tgt)
            send = tgt
            if self.action_delay_lag is not None:
                q = self._delay_q[i]
                slot = self.common_step_counter % self._delay_size
                q[slot] = tgt
                send = q[(slot - self.action_delay_lag) % self._delay_size, self._all_ids]
            self.backend.set_joint_targets(g.joints, send)
        self.prev_action_targets = torch.cat(targets, dim=-1)

    @staticmethod
    def _add_obs_noise(v: torch.Tensor, ns) -> torch.Tensor:
        if ns.std is not None:
            return v if ns.std == 0.0 else v + torch.randn_like(v) * ns.std
        w = v.shape[-1]
        assert w % 7 == 0, f"ObsNoise(pos=/rot=) needs a pos3+quat4 layout, term is {w} wide"
        b = v.reshape(v.shape[0], w // 7, 7)
        p, q = b[..., :3], b[..., 3:7]
        if ns.pos:
            p = p + torch.randn_like(p) * ns.pos
        if ns.rot:
            q = q + torch.randn_like(q) * (0.5 * ns.rot)
            q = q / q.norm(dim=-1, keepdim=True)
        return torch.cat([p, q], dim=-1).reshape(v.shape)

    def _capture_frames(self) -> dict:
        env = self
        frames = {}
        for gname, terms in self.spec.obs.items():
            noisy = gname in self._obs_noise_groups
            cols = []
            for t in terms:
                v = t.fn(env, **t.params)
                if noisy and t.noise is not None:
                    v = self._add_obs_noise(v, t.noise)
                cols.append(v * t.scale)
            frames[gname] = torch.cat(cols, dim=-1)
        return frames

    def _stack(self, frames: dict) -> TensorDict:
        groups = {}
        for gname, f in frames.items():
            if self._obs_hist_len.get(gname, 1) > 1:
                groups[gname] = self._obs_hist[gname].reshape(self.num_envs, -1)
            else:
                groups[gname] = f
        return TensorDict(groups, batch_size=[self.num_envs])

    def _advance_history(self, frames: dict, reset_ids) -> None:
        for gname, f in frames.items():
            h = self._obs_hist_len.get(gname, 1)
            if h <= 1:
                continue
            buf = self._obs_hist.get(gname)
            if buf is None:
                buf = f.unsqueeze(1).repeat(1, h, 1)
            else:
                buf = torch.roll(buf, shifts=-1, dims=1)
                buf[:, -1] = f
            if reset_ids is not None and reset_ids.numel() > 0:
                buf[reset_ids] = f[reset_ids].unsqueeze(1)
            self._obs_hist[gname] = buf

    def get_observations(self) -> TensorDict:
        frames = self._capture_frames()
        for gname, f in frames.items():
            h = self._obs_hist_len.get(gname, 1)
            if h > 1 and gname not in self._obs_hist:
                self._obs_hist[gname] = f.unsqueeze(1).repeat(1, h, 1)
        return self._stack(frames)

    def _rewards(self) -> torch.Tensor:
        env = self
        rew = torch.zeros(self.num_envs, device=self.device)
        for i, t in enumerate(self.spec.reward):
            self._cur_rew_term = t.name
            term = t.weight * t.fn(env, **t.params)
            self.last_reward_terms[:, i] = term
            self._ep_rew_sums[t.name] += term
            rew = rew + term
        self._cur_rew_term = None
        return rew

    def _all_mask(self) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def contact_steps(self, force_threshold: float, target: str | None = None) -> torch.Tensor:
        key = (float(force_threshold), target)
        buf = self._contact_steps.get(key)
        if buf is None:
            buf = torch.zeros(self.num_envs, len(self.spec.robot.fingertips), device=self.device)
            self._contact_steps[key] = buf
        return buf

    def _advance_contact_steps(self) -> None:
        if not self._contact_steps:
            return
        tips = self.spec.robot.fingertips
        for (thr, target), buf in self._contact_steps.items():
            f = (self.backend.contact_force(tips) if target is None
                 else self.backend.contact_force_with(tips, target))
            pressing = f.norm(dim=-1) > thr
            buf.copy_(torch.where(pressing, buf + 1.0, torch.zeros_like(buf)))

    def _advance_object_seen(self) -> None:
        if self._obj_seen_steps <= 1 or not self.spec.movable_objects:
            return
        now = torch.cat([self.backend.object_pos(), self.backend.object_quat()], dim=-1)
        seeing = (self.episode_length_buf < self._obj_seen_steps).unsqueeze(-1)
        self.object_seen_pose_w = torch.where(seeing, now, self.object_seen_pose_w)

    def _advance_curriculum(self) -> None:
        if self.gate is None:
            return
        near = self.gate.predicate(self, **self._curr_bars)
        self.curriculum_hold = _hold_count(self.curriculum_hold, near, self.gate.hold_mode)
        self.curriculum_passed |= self.curriculum_hold >= self._curr_hold_steps

    def _advance_gate(self) -> None:
        if self.gate is None:
            return
        near = self.gate.predicate(self, **self._gate_bars)
        self._gate_hold = _hold_count(self._gate_hold, near, self.gate.hold_mode)
        self.gate_passed = self.gate_passed | (self._gate_hold >= self.gate.hold_steps)

    def _post_reset(self, env_ids: torch.Tensor) -> None:
        for k, buf in self._evt_bufs.items():
            buf[env_ids] = self._evt_buf_fill[k]
        if self._reset_events:
            env = self
            for t in self._reset_events:
                self._cur_evt_term = t.name
                t.fn(env, env_ids, **t.params)
            self._cur_evt_term = None
        if self.spec.movable_objects:
            self.object_init_z[env_ids] = self.backend.object_pos()[env_ids, 2]
            self.object_seen_pose_w[env_ids] = torch.cat(
                [self.backend.object_pos()[env_ids], self.backend.object_quat()[env_ids]], dim=-1)
        for i, (_, g) in enumerate(self.action_groups):
            self._default[i][env_ids] = self.backend.joint_pos(g.joints)[env_ids]
        if self.action_delay_lag is not None:
            self.action_delay_lag[env_ids] = torch.randint(
                self._delay_min, self._delay_max + 1, (int(env_ids.numel()),), device=self.device)
            for i, q in enumerate(self._delay_q):
                q[:, env_ids] = self._default[i][env_ids]
        self.prev_action_targets[env_ids] = torch.cat([d[env_ids] for d in self._default], dim=-1)
        self.last_action[env_ids] = 0.0
        for key, buf in self._prev_vel.items():
            buf[env_ids] = self.backend.joint_vel(list(key))[env_ids]
            self._joint_acc_buf[key][env_ids] = 0.0
        if self.gate is not None:
            self._gate_hold[env_ids] = 0
            self.gate_passed[env_ids] = False
        self.curriculum_hold[env_ids] = 0
        self.curriculum_passed = self.curriculum_passed.clone()
        self.curriculum_passed[env_ids] = False
        for buf in self._contact_steps.values():
            buf[env_ids] = 0.0
        for k, buf in self._rew_bufs.items():
            buf[env_ids] = self._rew_buf_fill[k]

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
        env = self
        out = {}
        for t in self.spec.curriculum:
            vals = t.fn(env)
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
        self._prev_vel = {}
        self._joint_acc_buf = {}
        for s in self._ep_rew_sums.values():
            s.zero_()
        self._ep_log = {}
        self._post_reset(self._all_ids)
        self._ema = [d.clone() for d in self._default]
        if self.spec.curriculum and not self._curriculum_frozen:
            self._log_curriculum()
        self._obs_hist = {}
        return self.get_observations(), {"log": dict(self._ep_log)}

    def step(self, actions: torch.Tensor):
        self._telem.drain(self)
        self._apply_actions(actions)
        self._hold_while_paused()
        if self._interval_events:
            env = self
            for t in self._interval_events:
                self._cur_evt_term = t.name
                t.fn(env, self._all_ids, **t.params)
            self._cur_evt_term = None
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
        self._advance_contact_steps()
        self._advance_joint_acc()
        self._advance_object_seen()
        self._advance_gate()
        time_out = self.episode_length_buf >= self.max_episode_length
        dones = time_out.clone()
        truncated = time_out.clone()
        term_dones = [("time_out", time_out)]
        tctx = self
        for t in self.spec.terminate:
            m = t.fn(tctx, **t.params)
            term_dones.append((t.name, m))
            dones = dones | m
            if t.truncation:
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
                    self._ema[i] = self._ema[i].clone()
                    self._ema[i][dones] = self._default[i][dones]
            if self.spec.curriculum and not self._curriculum_frozen:
                self._log_curriculum()
            if self.gate is not None:
                self._ep_log["val/SR"] = float(self._last_ep_success.float().mean().item())

        frames = self._capture_frames()
        self._advance_history(frames, d)
        obs = self._stack(frames)
        extras = {"time_outs": truncated, "log": dict(self._ep_log)}
        _reached = next((m for n, m in term_dones if n == "curriculum_passed"), None)
        if _reached is not None:
            extras["task_success"] = _reached
        self._telem.publish(self, obs)
        return obs, rew, dones, extras

    def seed(self, seed: int) -> int:
        torch.manual_seed(seed)
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
