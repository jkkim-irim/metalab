"""EnvDriver — EnvSpec + SimBackend → VecEnv (engine-agnostic RL runtime).

Runs each step via the contract's action/obs/reward/terminate, satisfying the VecEnv interface the trainer
expects (num_envs·num_actions·device·max_episode_length·episode_length_buf·get_observations·step·reset·seed)
in a **duck-typed** way. Does not import ``learning.rl`` — the trainer consumes this EnvDriver directly as its VecEnv (in-process).
The engine hides behind :class:`sim.metalab.api.backend.SimBackend`.

Terms (obs/reward/terminate) are pure functions that read via ``backend`` as ``env`` (composed by the contract).
Includes events (DR)/curriculum/action-delay hooks: reset events run in the order
``backend.reset_idx → event → recapture init``, interval events run every step, and curriculum runs at each
done-reset boundary (its gate is inside the term). Also manages per-term termination tracking
(Termination/<name> log + curriculum termination_rate) and per-env random command delay
(ring buffer, delaying the post-EMA value).
"""
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
        """Build the driver from a contract + an engine spoke.

        The six ``_init_*`` blocks below run in an order that matters, in one direction only: each reads what
        the ones above it set. Task state must exist before the term buffers, because warming a reward term
        runs it against a ``EnvDriver`` that reads the goal, the lift latch and the fingertips; the registries
        must exist before the curriculum's first call; and telemetry is last because describing a snapshot
        evaluates every obs term once."""
        self.spec = spec
        self.backend = backend
        self.num_envs = int(backend.num_envs)
        self.device = backend.device
        self.cfg = {"task": spec.name}
        self.max_episode_length = int(max_episode_length)
        self.step_dt = spec.physics.dt * spec.physics.decimation   # POLICY step [s]
        self.max_episode_length_s = self.max_episode_length * self.step_dt   # horizon [s]
        self.capabilities = assert_backend(backend)
        self.fingertips = spec.robot.fingertips
        self.physics_gravity_z = float(spec.physics.gravity[2])   # [m/s^2, signed]

        self.action_groups = list(spec.action.items())   # [(name, ActionGroupSpec), ...]
        self.num_actions = sum(len(g.joints) for _, g in self.action_groups)
        assert self.num_actions > 0, "action groups are empty (env_driver requires an action)"

        self._init_action_pipeline()
        self._init_obs_pipeline()
        self._init_task_state()
        self._init_registries()
        self._init_term_buffers()
        self._telem = LiveDashboard.start(self) if telemetry else NO_DASHBOARD

    def __getattr__(self, name: str):
        """Backend reads reach a TERM through here — a term is ``fn(env, **knobs)`` and ``env`` IS this driver
        (isaaclab's shape: one env object, no per-family projection). Underscored names never delegate, so a
        miss during __init__ raises instead of recursing through ``self.backend``."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "backend"), name)

    def buffer(self, key: str, shape=(), fill: float = 0.0, dtype=torch.float32) -> torch.Tensor:
        """Per-env episode state for the reward or event term currently running — ``(num_envs, *shape)``.

        Allocated filled with ``fill`` on first use and restored to ``fill`` for every env the driver resets,
        so a term needs no init/reset hook and stays a flat function. Mutate it IN PLACE (``buf[:] =``) —
        rebinding the local name writes to nothing. Keys are namespaced by the calling term, so two terms may
        both use ``"best"``; that key is additionally what the ``closest_*`` obs terms read through
        :meth:`reward_best`. Which store it lands in follows from which term loop is running."""
        if self._cur_rew_term is not None:
            return self._reward_buffer(key, shape, fill, dtype)
        assert self._cur_evt_term is not None, \
            "env.buffer() is only callable from inside a reward or event term"
        return self._event_buffer(key, shape, fill, dtype)

    def termination_rate(self, name: str) -> float:
        """Fraction of envs [0,1] whose last episode ended on terminate term ``name`` (or ``"time_out"``).
        A GPU sync — a curriculum term must call it inside its own iteration gate."""
        buf = self._last_ep_done.get(name)
        assert buf is not None, f"unknown termination term {name!r} — valid: {sorted(self._last_ep_done)}"
        return float(buf.float().mean().item())

    def curriculum_success_rate(self) -> float:
        """Fraction of envs [0,1] whose last episode met the success conditions at the CURRICULUM's OWN bars.

        The level-up signal, latched per episode by ``_advance_curriculum`` — the curriculum term's OWN bar,
        not what ``val/SR`` measures (the GATE). The same latch ends the episode
        (``terminate.curriculum_passed``) and pays the success bonus, so all three say one thing.
        A GPU sync, like :meth:`termination_rate` — call it inside the term's own iteration gate."""
        return float(self._last_ep_curr_success.float().mean().item())

    def set_curriculum_bars(self, hold_steps: int, **bars) -> None:
        """Set the bar THIS LEVEL is judged at — the knobs of ``GATE.predicate``, plus how many steps they
        must hold for. ``_advance_curriculum`` counts and latches it into the one flag that promotes a level,
        ends the episode and pays the success bonus; ``val/SR`` stays on the GATE's own fixed bar."""
        self._curr_hold_steps = int(hold_steps)
        self._curr_bars = bars

    def set_num_steps_per_env(self, n: int) -> None:
        """Publish the trainer's ROLLOUT LENGTH — env steps per policy update.

        An RL hyperparameter, so the experiment config owns it (``EXP["num_steps_per_env"]``) and the env is
        TOLD, once, before the first reset. A curriculum that gates on iterations has no other way to turn
        ``common_step_counter`` into one, and stating it in the contract as well would be a second source for
        one number."""
        assert n >= 1, f"num_steps_per_env must be >= 1 — got {n}"
        self._num_steps_per_env = int(n)

    @property
    def num_steps_per_env(self) -> int:
        """The trainer's rollout length — see :meth:`set_num_steps_per_env`."""
        assert self._num_steps_per_env is not None, (
            "num_steps_per_env was never published to this env, and a curriculum term is asking for it to "
            "count iterations. The trainer publishes it before the first reset (learning/rl/client.py); an "
            "eval or standalone run has no rollout length, so freeze the curriculum first "
            "(apply_curriculum_end).")
        return self._num_steps_per_env

    def set_object_seen_steps(self, steps: int) -> None:
        """Set how many opening policy steps of an episode the object is still PERCEIVED for — the window
        ``_advance_object_seen`` refreshes the latch over. 1 = the spawn pose only. Measured from each env's
        own episode start, so it takes effect mid-episode too."""
        assert steps >= 1, f"object seen window must be >= 1 step (1 = the spawn pose) — got {steps}"
        self._obj_seen_steps = int(steps)

    def set_dr_value(self, key: str, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Record the per-env DRAW a DR event just made, so ``obs.dr_params`` can hand it to the critic.

        The event samples the value and writes it into the world; nothing keeps the number itself — the
        backend folds it into model arrays it never reads back — so a critic asking "how slippery is THIS
        env" has no channel to read. Stated at the call site of each DR term rather than intercepted here,
        for the same reason the ``exclude`` knob is: what a term publishes should be visible in the term.

        Allocated on the first write, and NOT cleared at reset — the draw is the episode's, so it must stand
        until that env's next reset overwrites it."""
        buf = self.dr_values.get(key)
        if buf is None:
            buf = torch.zeros(self.num_envs, device=self.device)
            self.dr_values[key] = buf
        buf[env_ids] = values.to(dtype=buf.dtype)

    def dr_value(self, key: str) -> torch.Tensor:
        """The per-env DR draw published under ``key`` → (N,).

        Asking for a channel no event of this contract publishes fails loud, the same way a curriculum key
        that is not published does — a DR that is off must not read as a live one. The pre-reset zeros are
        for the telemetry describe pass alone (it evaluates every obs term once, before any event has run);
        after the first reset a missing channel is an error, never a default."""
        buf = self.dr_values.get(key)
        if buf is not None:
            return buf
        assert not self._reset_done, (
            f"dr_value asks for {key!r}, which no DR event of this contract publishes — published: "
            f"{sorted(self.dr_values)}.")
        return torch.zeros(self.num_envs, device=self.device)

    def _init_action_pipeline(self) -> None:
        """Action-decode state: the per-group EMA, the decode BASE, the joint-limit clamp, the delay queues.

        ``_default`` is captured HERE, not lazily. reset() recaptures it, but a trainer that calls step()
        before reset() would otherwise decode against ``None`` — target = action*scale ~ 0, which drags the
        arm to zero. The delay queues pre-fill with that same pose, so an episode opens holding the spawn pose
        for ``lag`` steps instead of replaying the previous episode's targets. ONE lag per env, shared by every
        group: it models a single controller link (spec.ActionDelaySpec); ``max_delay=0`` is off.

        The EMA coefficient is DERIVED here, not authored: a contract states the filter as a time constant
        (``ema_tau`` [s]) and this turns it into the per-step ``alpha = 1 - exp(-step_dt/tau)`` of THIS run's
        policy rate. So changing ``physics.hz``/``decimation`` leaves the filter where it was in the time
        domain, and hardware running at another command rate reproduces it from the same number."""
        spec, d = self.spec, self.device
        self.last_action = torch.zeros(self.num_envs, self.num_actions, device=d)
        self._ema = [None] * len(self.action_groups)
        self._ema_alpha = [None if g.ema_tau is None else 1.0 - math.exp(-self.step_dt / g.ema_tau)
                           for _, g in self.action_groups]
        self._default = [self.backend.joint_pos(g.joints).clone() for _, g in self.action_groups]
        self._limits = [self.backend.joint_limits(g.joints) for _, g in self.action_groups]
        self.prev_action_targets = torch.cat([x.clone() for x in self._default], dim=-1)   # (N, num_actions) [rad]
        self._delay_min = spec.action_delay.min_delay
        self._delay_max = spec.action_delay.max_delay
        self._delay_size = self._delay_max + 1
        self.action_delay_lag = (torch.randint(self._delay_min, self._delay_max + 1,
                                         (self.num_envs,), device=d)
                           if self._delay_max > 0 else None)
        self._delay_q = ([x.unsqueeze(0).repeat(self._delay_size, 1, 1) for x in self._default]
                         if self.action_delay_lag is not None else None)

    def _init_obs_pipeline(self) -> None:
        """Obs-side buffers the backend cannot supply: the joint-acceleration finite difference, the per-group
        history ring, and which groups get sensor noise.

        The acceleration difference divides by the POLICY step dt (the same one IsaacLab's RewardManager
        scales every reward term by). History is a ring per group, built lazily and advanced once per step;
        noise applies to the contract's noise groups only, and stays on in eval/play."""
        spec = self.spec
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_reward = torch.zeros(self.num_envs, device=self.device)
        # Per-term contributions of the step just scored, in CONTRACT order — the columns the
        # ``instantaneous_reward`` obs publishes after its total. Written every step, never reallocated.
        self.last_reward_terms = torch.zeros(self.num_envs, len(spec.reward), device=self.device)
        self._prev_vel: dict = {}   # names tuple -> (N,k) [rad/s]
        self._joint_acc_buf: dict = {}   # names tuple -> (N,k) [rad/s^2]
        self._acc_dt = self.step_dt
        self._obs_hist_len = dict(spec.obs_history_length)   # group -> H (1 = off)
        self._obs_hist: dict = {}   # group -> (N, H, D)
        self._obs_noise_groups = set(spec.obs_noise_groups)

    def _init_task_state(self) -> None:
        """Per-episode task state: the fixed goal, the eval GATE's counters, the object latches and the
        val/SR tallies.

        No tolerance is kept here: every bar is a knob of the predicate call that tests
        it (``_curr_bars`` / ``_gate_bars``), so there is no runtime value a term could read behind the
        contract's back."""
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
        # The CURRICULUM's own bar — what a LEVEL-UP is judged at. Separate from what the reward pays for and
        # from the GATE, so the three may demand different grips. It starts AT the gate and a curriculum term
        # relaxes it every call (_run_curriculum): a contract with no curriculum term never does, which is
        # what "level 0 already is the END criteria" means for an eval recipe.
        self._curr_bars = dict(self._gate_bars)
        self._curr_hold_steps = 1 if spec.gate is None else spec.gate.hold_steps
        self.curriculum_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Episode latch at the CURRICULUM's bar: the level-up SR, and what ends the episode
        # (``terminate.curriculum_passed`` reads it).
        self.curriculum_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_ep_curr_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.curriculum_values: dict[str, torch.Tensor] = {}
        self._curr_key_owner: dict[str, str] = {}   # Curriculum/<key> -> the term that owns it
        self._last_ep_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_attempts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ep_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_init_z = (self.backend.object_pos()[:, 2].clone() if spec.movable_objects
                               else torch.zeros(self.num_envs, device=self.device))
        self.object_seen_pose_w = (
            torch.cat([self.backend.object_pos(), self.backend.object_quat()], dim=-1).clone() if spec.movable_objects
            else torch.zeros(self.num_envs, 7, device=self.device))
        self._obj_seen_steps = 1
        # DR draw store: channel -> (N,) the value the reset event actually sampled for that env.
        self.dr_values: dict[str, torch.Tensor] = {}
        # An event whose `requires` capability this engine lacks is dropped HERE, once, out loud — the term
        # never fires and never appears in `event_terms`, so a curriculum that tries to ramp it fails loud
        # instead of tuning a term that does nothing.
        self._dropped_events = {t.name: miss for t in spec.events
                                if (miss := tuple(c for c in t.requires if c not in self.capabilities))}
        for name, miss in self._dropped_events.items():
            print(f"[metalab] {type(self.backend).__name__}: event {name!r} dropped — this engine does not "
                  f"have {list(miss)}", flush=True)
        self._events = [t for t in spec.events if t.name not in self._dropped_events]
        self._reset_events = [t for t in self._events if t.mode == "reset"]
        self._interval_events = [t for t in self._events if t.mode == "interval"]

    def _init_registries(self) -> None:
        """Name → term maps (the curriculum's tuning surface) and per-term termination-cause tracking.

        ``time_out`` and ``nan_world`` are reserved names: the first is the horizon, the second the always-on
        divergence termination every backend must provide. A contract that reuses either fails loud here."""
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
        self._num_steps_per_env: int | None = None   # published by the trainer, not the contract
        self._curriculum_frozen = False
        self._reset_done = False
        self._all_ids = torch.arange(self.num_envs, device=self.device)

    def _init_term_buffers(self) -> None:
        """The reward/event episode-state stores, then a WARM-UP pass so their keysets exist from step 1.

        A flat term allocates its buffer on first use, but the trainer's logger reads ``extras["log"]`` from
        the first step's keyset (rsl_rl reads ep_extras[0]) — so every reward term is run once here against a
        throwaway EnvDriver and the buffers are then reset to their fill. The curriculum is seeded for the
        same reason: its difficulty keys must be publishable before the first real step."""
        self._rew_bufs: dict = {}   # (term, key) -> (N, ...)
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
        """Backing store behind :meth:`buffer` when a REWARD term is running — allocate-on-first-use, by the term
        currently being evaluated (``_cur_rew_term``, stamped by :meth:`_rewards`). ``_post_reset`` restores
        every entry to its ``fill`` for the reset envs, which is what replaces per-term init/reset hooks."""
        assert self._cur_rew_term is not None, "EnvDriver.buffer() is only callable from inside a reward term"
        k = (self._cur_rew_term, key)
        buf = self._rew_bufs.get(k)
        if buf is None:
            buf = torch.full((self.num_envs, *shape), fill, dtype=dtype, device=self.device)
            self._rew_bufs[k] = buf
            self._rew_buf_fill[k] = fill
        return buf

    def _event_buffer(self, key: str, shape, fill: float, dtype) -> torch.Tensor:
        """Backing store behind :meth:`buffer` when an EVENT term is running.

        Allocate-on-first-use, namespaced by the event term currently running (``_cur_evt_term``). Restored
        to ``fill`` for the reset envs at the TOP of ``_post_reset`` — before the reset events run, so a term
        that reads its buffer during a reset event sees this episode's fill and never last episode's leftovers."""
        assert self._cur_evt_term is not None, "EnvDriver.buffer() is only callable from inside an event term"
        k = (self._cur_evt_term, key)
        buf = self._evt_bufs.get(k)
        if buf is None:
            buf = torch.full((self.num_envs, *shape), fill, dtype=dtype, device=self.device)
            self._evt_bufs[k] = buf
            self._evt_buf_fill[k] = fill
        return buf

    def reward_best(self, name: str) -> torch.Tensor:
        """The ``"best"`` episode buffer of progress reward term ``name``, whatever the contract calls it —
        state, not a metric, which is why this one stays here. inf at reset.

        Names no term: the caller passes one and a name with no such buffer fails loud with the list that
        does exist. Allocated lazily, so it exists only after the term has run once; the obs terms read it on
        the same step the reward ran, and the driver computes rewards before obs."""
        buf = self._rew_bufs.get((name, "best"))
        assert buf is not None, (
            f"reward term {name!r} has no 'best' episode buffer — progress terms only, and only after the "
            f"first step. present: {sorted(k for k in self._rew_bufs)}"
        )
        return buf

    def joint_acc(self, names: list[str]) -> torch.Tensor:
        """The (N, k) joint acceleration for these joints — created on first read (an obs term asking for it),
        then advanced once per policy step by :meth:`_advance_joint_acc`. READ-ONLY, like every other obs
        context: this used to difference velocities and advance ``_prev_vel`` inside the read itself, so the
        SECOND evaluation within one step (``get_observations()`` — documented idempotent — the report's
        ``snapshot_rows``, the dashboard) saw cur == prev and read exactly 0. joint_acc plotted as all-zeros
        for that reason while the value inside ``step()``'s own obs was correct."""
        key = tuple(names)
        buf = self._joint_acc_buf.get(key)
        if buf is None:
            buf = torch.zeros(self.num_envs, len(names), device=self.device)
            self._joint_acc_buf[key] = buf
            self._prev_vel[key] = self.backend.joint_vel(names).detach().clone()
        return buf

    def _advance_joint_acc(self) -> None:
        """acc = (current vel - previous vel) / policy dt, for every joint set an obs term asked for. Exactly
        once per policy step (buffers updated in place, so the obs term's tensor is always the current
        value). Reset envs are handled in _post_reset, which re-seeds _prev_vel to the POST-reset velocity so
        the reset's velocity jump is not differentiated into a non-physical spike."""
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
        self.prev_action_targets = torch.cat(targets, dim=-1)   # (N, num_actions)

    @staticmethod
    def _add_obs_noise(v: torch.Tensor, ns) -> torch.Tensor:
        """Sensor noise on one term's raw value (pre-scale), in the term's own units. See spec.ObsNoise.

        Out-of-place on purpose: `v` may be a view into a backend buffer."""
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
        """Current single-frame obs per group (name → (N, D)). Backend read + driver context via EnvDriver.

        Terms are FLAT — `fn(env, **params)`, the same call shape as a reward term — so a knob stays live in
        `t.params` instead of being frozen into a closure at load.

        Noise is applied HERE, per group: the same ObsTerm object is shared by the actor and the critic
        (critic = ACTOR + CRITIC blocks), and only groups in `obs_noise_groups` get corrupted — which is why
        it cannot live on the term's own value. `fn` is already evaluated once per (group, term)."""
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
        """Build the obs TensorDict: H>1 groups return the flattened ring buffer (N, H*D), else the live frame."""
        groups = {}
        for gname, f in frames.items():
            if self._obs_hist_len.get(gname, 1) > 1:
                groups[gname] = self._obs_hist[gname].reshape(self.num_envs, -1)   # (N, H*D)
            else:
                groups[gname] = f
        return TensorDict(groups, batch_size=[self.num_envs])

    def _advance_history(self, frames: dict, reset_ids) -> None:
        """Advance every stacked group's ring buffer by one policy step (drop oldest, append current frame).
        ``reset_ids`` (or None) = envs whose episode just ended → fill all H slots with the post-reset frame so
        no pre-reset frame leaks across the episode boundary. Called exactly once per step()."""
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
        """Read-only obs (idempotent): never advances the history ring buffer — that happens once per step().
        Lazily initializes buffers if called before the first step/reset (runner-construction path)."""
        frames = self._capture_frames()
        for gname, f in frames.items():
            h = self._obs_hist_len.get(gname, 1)
            if h > 1 and gname not in self._obs_hist:
                self._obs_hist[gname] = f.unsqueeze(1).repeat(1, h, 1)
        return self._stack(frames)

    def _rewards(self) -> torch.Tensor:
        env = self
        # The publish slots go back to their defaults every step: a contract whose reward terms stop setting
        # one must read as "nothing published", not as last step's value.
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
        """The ``(N, K)`` consecutive pad-press step counter over ``spec.robot.fingertips`` at this threshold
        and counterpart — created on first read (an obs term asking for it), then advanced once per policy
        step by _advance_contact_steps. Both knobs are part of the key, so a contract may count against the
        object and against everything at once and get two independent counters.

        WHICH bodies is not a knob: the tips are the hand's pad shells, a robot fact."""
        key = (float(force_threshold), target)
        buf = self._contact_steps.get(key)
        if buf is None:
            buf = torch.zeros(self.num_envs, len(self.spec.robot.fingertips), device=self.device)
            self._contact_steps[key] = buf
        return buf

    def _advance_contact_steps(self) -> None:
        """+1 per fingertip pressing with its PAD, back to 0 the step that press breaks — the "how long has
        THIS fingertip been holding" signal. Exactly once per policy step (buffers updated in place, so the
        obs term's returned tensor is always the current count). Reset envs are zeroed in _post_reset.

        The DRIVER owns only the counter; the predicate is
        ``force.norm(dim=-1) > threshold``, the same call the grip reward and the GATE make — so
        "5 steps of contact" in the critic's input means what the reward pays for, not merely "touching
        something". ``target`` picks what must be pressed: a counterpart (``"object"``) scopes the read to
        that asset, ``None`` takes the NET force over everything, which on a table-top task is dominated by
        the table during the approach."""
        if not self._contact_steps:
            return
        tips = self.spec.robot.fingertips
        for (thr, target), buf in self._contact_steps.items():
            f = (self.backend.contact_force(tips) if target is None
                 else self.backend.contact_force_with(tips, target))
            pressing = f.norm(dim=-1) > thr
            buf.copy_(torch.where(pressing, buf + 1.0, torch.zeros_like(buf)))

    def _advance_object_seen(self) -> None:
        """Refresh the perception latch in every env still inside its SEEING window, freeze it in the rest.

        ``_obj_seen_steps`` is how many opening policy steps of an episode the actor keeps tracking the object
        for: 1 = it saw the spawn pose and nothing after, ``max_episode_length`` = it never loses sight of it.
        The window is a curriculum knob (``set_object_seen_steps``), so a run can start fully sighted and end
        blind after the first frame — the deployable case, where the hand covers the object as it closes.
        Masked rather than indexed: the window is per-env only because episodes reset at different steps, and
        a nonzero() here would sync the GPU every step."""
        if self._obj_seen_steps <= 1 or not self.spec.movable_objects:
            return
        now = torch.cat([self.backend.object_pos(), self.backend.object_quat()], dim=-1)
        seeing = (self.episode_length_buf < self._obj_seen_steps).unsqueeze(-1)
        self.object_seen_pose_w = torch.where(seeing, now, self.object_seen_pose_w)

    def _advance_curriculum(self) -> None:
        """Count and latch the CURRICULUM's bar — the level-up signal — exactly once per policy step.

        Same shape as ``_advance_gate`` one bar down: the contract's ``GATE.predicate`` again, at the knobs the
        curriculum term published for THIS level (``set_curriculum_bars``), counted with the GATE's
        ``hold_mode`` and latched per episode into ``curriculum_passed``, which ``curriculum_success_rate`` reads.

        ONE latch, three consumers — the level-up SR, the episode end (``terminate.curriculum_passed``) and
        the success bonus (``reward.object_goal_reach_bonus`` pays for exactly this, having no bars of its
        own). So "solved at this level" is judged once per step and cannot mean three things. ``val/SR`` is
        the separate measurement (``_advance_gate``), which is why a level may demand something the GATE does
        not (a named-fingertip grip, say) without moving what the run is scored at."""
        if self.gate is None:
            return
        near = self.gate.predicate(self, **self._curr_bars)
        self.curriculum_hold = _hold_count(self.curriculum_hold, near, self.gate.hold_mode)
        self.curriculum_passed |= self.curriculum_hold >= self._curr_hold_steps

    def _advance_gate(self) -> None:
        """Count and latch the eval GATE on the post-physics state, exactly once per policy step.

        The JUDGMENT is the contract's own ``GATE.predicate`` (``sim.metalab.terms.gate``) — the driver does
        not know what solved means. What it owns is the counting: +1 while the predicate holds, and the
        episode latch ``gate_passed`` once the count reaches ``hold_steps``. ``GATE.hold_mode`` decides what a
        break does — ``consecutive`` zeroes the counter (a HOLD), ``cumulative`` merely stops it (a TOTAL).

        The bars are the GATE's own (``_gate_bars``), never the curriculum's, so val/SR is comparable across
        the whole run and equals what an eval judges at END criteria. It is a pure MEASUREMENT: an episode
        that meets the gate at any level is counted, and nothing here ends the episode or pays anything —
        ``_advance_curriculum`` does both, at the level's bar. Pure tensor ops, no DtoH sync."""
        if self.gate is None:
            return
        near = self.gate.predicate(self, **self._gate_bars)
        self._gate_hold = _hold_count(self._gate_hold, near, self.gate.hold_mode)
        self.gate_passed = self.gate_passed | (self._gate_hold >= self.gate.hold_steps)

    def _post_reset(self, env_ids: torch.Tensor) -> None:
        """Common handling right after backend.reset_idx — ★ordering contract (events/common.py):
        clear event-term buffers → reset events (randomization) → recapture init state
        (object_init_z·default·delay) → clear the remaining driver-owned per-episode buffers (reward-term
        state, contact durations, latches)."""
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
        # REBOUND, not cleared in place: `terminate.curriculum_passed` returns this very tensor, and step()
        # still reads that mask after the reset (Termination/<name>, extras["task_success"]). Zeroing the
        # object under it would report every success as a non-event.
        self.curriculum_passed = self.curriculum_passed.clone()
        self.curriculum_passed[env_ids] = False
        for buf in self._contact_steps.values():
            buf[env_ids] = 0.0
        for k, buf in self._rew_bufs.items():
            buf[env_ids] = self._rew_buf_fill[k]

    def _log_curriculum(self) -> None:
        """Run the curriculum terms and publish their difficulty under ``Curriculum/<key>``.

        The TERM's name is deliberately not in the key: a contract normally declares one curriculum, and
        ``Curriculum/success_curriculum/level`` puts a segment in every dashboard path that says nothing.
        Two terms publishing the SAME key would then collide, so that fails loud here rather than letting
        whichever ran last win — the reader would have no way to tell which curve they were looking at."""
        for name, vals in self._run_curriculum().items():
            for k, v in vals.items():
                key = f"Curriculum/{k}"
                assert key not in self._ep_log or self._curr_key_owner.get(key) == name, (
                    f"two curriculum terms both publish {k!r} ({self._curr_key_owner.get(key)} and {name}) — "
                    f"the log key Curriculum/{k} cannot say which. Rename one of them.")
                self._curr_key_owner[key] = name
                self._ep_log[key] = float(v)

    def _run_curriculum(self) -> dict[str, dict[str, float]]:
        """Run every curriculum term once and cache what it reports → ``{term: values}``.

        The cache (``curriculum_values``) is what the ``curriculum_state`` obs reads: 0-dim GPU scalars written ONLY
        here (a level-up is rare) and read every step, so the obs costs a stack of scalars rather than a
        host→device copy per step. Names collide across terms deliberately — a task with two curriculum terms
        ramping the same knob has a contract bug, and last-writer-wins makes it visible instead of averaging."""
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
        """The curriculum's CURRENT training level, or 0 if this task has no levelled curriculum.

        Exists so a SEPARATE process — the trainer's per-checkpoint recorder — can ask the training env what
        world a checkpoint learned in and reproduce it (see :meth:`apply_curriculum_end`). Max over terms; in
        practice a task has one levelled term."""
        levels = [int(t.fn.level) for t in self.spec.curriculum if hasattr(t.fn, "level")]
        return max(levels) if levels else 0

    def apply_curriculum_end(self, train_level: int | None = None) -> None:
        """Freeze the curriculum for eval/recording and apply it once, and drop the ``train_only`` events.
        After this the step loop stops re-running the curriculum, so the values stick.

        This is the ONE call every eval and recording path makes before its first reset, so it is also where
        a disturbance that only exists to harden the policy (an ``Event(..., train_only=True)``) stops firing
        — a recording then measures the task, not the task plus a tug the deployed robot will never feel.

        ``train_level=None`` — snap to the final level (``jump_to_end``). For a gate-ramp curriculum the END
        *is* the contract's GATE. Right for evaluating a FINISHED run.

        ``train_level=L`` — run the WORLD of level L (``freeze_for_eval``). This is what a mid-training
        checkpoint needs: with a curriculum that ramps friction and a tug on the object, snapping the world
        to the end rolls a level-3 policy (a grippy world, no tug) at the real friction under a full pull — a
        different task, so the clip shows nothing about that checkpoint.
        Terms without ``freeze_for_eval`` fall back to ``jump_to_end``.

        Neither choice moves the SUCCESS bar: the episode end and ``val/SR`` are the GATE's, latched by
        ``_advance_gate`` outside the curriculum, so every recording stays comparable across checkpoints.

        MUST be called BEFORE the first reset. Some ramped knobs (friction ``mu_scale``, object
        ``mass_scale``) are consumed by the RESET EVENTS, so they only bind at the *next* reset —
        freezing afterwards leaves episode 0 running the level-0 world (on hammer_lift, 3x friction)
        while every later episode runs the frozen one. That silently made the first episode of every
        eval fail, which a short recording window shows as "no successes ever"."""
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
        """One POLICY step: decode + apply the action, advance physics, then score what happened.

        The post-physics order is a contract, not a preference:

        1. ``_advance_curriculum`` — the level's success latch. BEFORE the rewards, because the success bonus
           is paid FOR it: judged after them, the latch would flip on a step whose reward was already scored
           and the episode would reset before the next one, so the bonus could never be paid at all.
        2. rewards, which read that latch (bonus) and the world (everything else).
        3. the remaining per-step advances — contact duration, joint acceleration, the perception latch, and
           the GATE's hold counter + episode latch (``val/SR``), which nothing else this step depends on.
        4. terminations — the success one reads the latch from 1 — then the reset of whatever finished, then
           observations.

        Each ``_advance_*`` ticks EXACTLY once here, which is the reason those counters live in the driver at
        all: an obs term is evaluated once per obs GROUP, and actor and critic share most terms, so a counter
        advanced inside one would tick several times per step.

        Physics goes through ``step_n`` when the engine offers the ``batched_step`` capability (newton runs the
        whole control step as one CUDA-graph launch) and through a decimation loop otherwise. Interval events
        fire BEFORE physics, after the action is applied; the pause hold sits between the two, so a pause only
        DELAYS a step and never reorders one — the counters, rewards and dones still run exactly once per
        advance, which is why a paused eval cannot skew its own SR.
        """
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
            # Per-term reward is logged PER STEP (the episode sum over max_episode_length), not as the raw
            # sum the trainer optimises: the raw sum scales with how long an episode is allowed to run, so
            # two tasks with different episode_length_s could not be read off the same axis.
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
        """Static description of what a snapshot row holds — :func:`sim.metalab.runtime.snapshot.describe`.

        Kept as a method because it is the published surface: the live dashboard adapter (``rl_monitor``) and
        the eval report (``rollout_log``) are both keyed to this name."""
        return snap.describe(self, self)

    def _hold_while_paused(self) -> None:
        """Block until the sim may advance — the dashboard toggle OR the viewer's own Pause holds it.

        Called once per policy step, right after ``_apply_actions`` and before physics, so the pause only
        DELAYS a step and never reorders one: the episode counters, rewards and dones all still run exactly
        once per advance, which is why a paused eval cannot skew its own SR. The RPC transport sets no socket
        timeout, so a client blocked in ``env.step`` simply waits (verified in sim/metalab/transport.py).

        The two sources are independent and either one holds. `_paused` is checked first and short-circuits,
        so the viewer's single-step request is not consumed (hence not lost) while the dashboard is pausing —
        release the dashboard toggle to use the viewer's Step.
        """
        allowed = getattr(self.backend, "viewer_step_allowed", None)
        pump = getattr(self.backend, "pump_viewer", None)
        held = False
        while self._telem.paused or (allowed is not None and not allowed()):
            if not held:
                held = True
                self._telem.note_pause(self, True)
            if pump is not None:
                pump()
            self._telem.drain(self)
            time.sleep(0.01)
        if held:
            self._telem.note_pause(self, False)

    def snapshot_rows(self, idxs, extra=None) -> dict:
        """Per-env snapshot rows keyed by GLOBAL env id → ``{"<id>": {step, max_step, obs, action, state}}``.

        All GPU reads (obs terms + action + ``extra``) are concatenated into ONE tensor and moved
        to CPU in a SINGLE sync (+1 tiny sync for the int step), so sampling every policy step (~60 Hz) stays
        cheap — that single-sync property is the reason both consumers go through here instead of reading
        piecemeal.

        ``extra`` = ``[(name, fn, params, scale)]`` extra channels (the rollout log's CUSTOM tabs) —
        called like any obs term (``fn(env, **params)``), evaluated inside that same sync
        and returned under ``state``.
        """
        sel = torch.as_tensor([int(i) for i in idxs], dtype=torch.long, device=self.device)
        n = int(sel.numel())
        st = self
        obs_names, parts, obs_w = [], [], []
        for gname, terms in self.spec.obs.items():
            noisy = gname in self._obs_noise_groups
            for t in terms:
                if t.name in obs_names:
                    continue
                v = t.fn(st, **t.params)
                if noisy and t.noise is not None:
                    v = self._add_obs_noise(v, t.noise)
                v = (v * t.scale)[sel]
                obs_names.append(t.name); parts.append(v); obs_w.append(int(v.shape[-1]))
        parts.append(self.last_action[sel])
        ex_names, ex_w = [], []
        for name, fn, params, scale in (extra or ()):
            v = fn(st, **params)[sel] * scale
            ex_names.append(name); parts.append(v); ex_w.append(int(v.shape[-1]))
        flat = torch.cat(parts, dim=-1).detach().cpu()
        step_c = self.episode_length_buf[sel].cpu()
        rows = {}
        for i in range(n):
            row, off, obs_i = flat[i], 0, {}
            for name, w in zip(obs_names, obs_w):
                obs_i[name] = row[off:off + w].tolist(); off += w
            act_i = row[off:off + self.num_actions].tolist(); off += self.num_actions
            state_i = {}
            for name, w in zip(ex_names, ex_w):
                state_i[name] = row[off:off + w].tolist(); off += w
            rows[str(int(sel[i]))] = {"step": int(step_c[i]), "max_step": self.max_episode_length,
                                      "obs": obs_i, "action": act_i, "state": state_i}
        return rows

