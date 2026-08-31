"""Common curriculum term factories — difficulty progression (engine-agnostic).

Term contract (fixed signature):
  factory ``f(*args, **kwargs)`` -> ``fn(env) -> dict[str, float]``
  - ``env`` = env_driver's :class:`EnvDriver`:
      .common_step_counter        — cumulative policy steps (iteration = counter // num_steps_per_env;
                                    num_steps_per_env comes in as a term kwarg — legacy style)
      .termination_rate(name)     — fraction of recent episodes ended by terminate term <name> in [0,1]
                                    (incl. time_out; for SR gating — GPU sync, so call only inside gates)
      .reward_terms[name]         — RewardTerm (.weight mutable — driver live-reads each step)
      .event_terms[name]          — EventTerm (.params mutable — retune a knob via .params[k] = v; the
                                    driver splats it into the flat term on the next call)
  - Returned dict logged to extras["log"] as ``Curriculum/<key>``.
  - Called at each done reset boundary (legacy "compute per reset"). Iteration gating (e.g. evaluate
    once per 50 iters) lives **inside** the term — legacy task_success_difficulty pattern.
  - Progress state (level etc.) is held in term instance attributes (class term). Curriculum gets no
    .init/.reset lifecycle (unlike reward/event), so it grabs a lazy baseline on the first ``__call__``.


Reference: sim/isaaclab/envs/hammer_lift/mdp/curriculum.py (task_success_difficulty),
sim/isaaclab/envs/perceptive_dexdeepmimic/mdp/curriculums.py (goal_tolerance_curriculum).
"""
from __future__ import annotations


def _advance(start: float, end: float, step: float, level: int) -> float:
    """start + step*level, clamped toward end (step sign sets direction)."""
    val = start + step * level
    return min(end, val) if step >= 0.0 else max(end, val)


class task_success_difficulty:
    """SR-based single-difficulty curriculum (ports legacy hammer_lift task_success_difficulty).

    Converts to PPO iteration via ``common_step_counter // num_steps_per_env``, evaluates success
    termination rate at most once per ``eval_interval_iterations`` iters, and level+1 when it exceeds
    ``level_up_threshold``. Reading SR every reset is noisy -> the iteration gate emits the sync only at
    eval boundaries (legacy semantics). As level advances, shaping reward weight is decayed via
    :func:`_advance` so the sparse signal takes over.

    """

    def __init__(
        self,
        success_term: str = "task_success",
        level_up_threshold: float = 0.5,
        eval_interval_iterations: int = 50,
        num_steps_per_env: int = 24,
        contact_reward_term: str = "fingertip_object_contact",
        contact_weight_start: float = 0.5,
        contact_weight_end: float = 0.0,
        contact_weight_step: float = -0.05,
        palm_reward_term: str = "palm_object_proximity",
        palm_weight_start: float = 0.5,
        palm_weight_end: float = 0.0,
        palm_weight_step: float = -0.05,
    ):
        self.success_term = success_term
        self.level_up_threshold = float(level_up_threshold)
        self.eval_interval_iterations = int(eval_interval_iterations)
        self.num_steps_per_env = int(num_steps_per_env)
        self.contact_reward_term = contact_reward_term
        self.contact_weight_start = float(contact_weight_start)
        self.contact_weight_end = float(contact_weight_end)
        self.contact_weight_step = float(contact_weight_step)
        self.palm_reward_term = palm_reward_term
        self.palm_weight_start = float(palm_weight_start)
        self.palm_weight_end = float(palm_weight_end)
        self.palm_weight_step = float(palm_weight_step)
        # Persistent state (legacy used env attributes, we use term instance attributes). last_iter=None -> first-call baseline.
        self._level = 0
        self._last_iter: int | None = None
        self._success_rate = 0.0

    def __call__(self, env) -> dict[str, float]:
        current_iter = env.common_step_counter // self.num_steps_per_env
        if self._last_iter is None:
            self._last_iter = current_iter                 # baseline — no bump on first call

        # --- gate: evaluate/bump at most once per eval_interval_iterations iters ---
        # Read SR's .item() (GPU->CPU sync) only at eval boundaries (~every N iters), not every reset.
        # Boundary check uses common_step_counter (CPU int) so no sync -> removes per-reset sync (legacy semantics).
        if current_iter - self._last_iter >= self.eval_interval_iterations:
            self._last_iter = current_iter
            self._success_rate = env.termination_rate(self.success_term)   # sync — only inside the gate
            if self._success_rate > self.level_up_threshold:
                self._level += 1
        level = self._level

        # --- shaping weight decay (applied via reward_terms[name].weight — the only live mutate surface) ---
        contact_weight = _advance(self.contact_weight_start, self.contact_weight_end,
                                  self.contact_weight_step, level)
        palm_weight = _advance(self.palm_weight_start, self.palm_weight_end,
                               self.palm_weight_step, level)
        env.reward_terms[self.contact_reward_term].weight = contact_weight
        env.reward_terms[self.palm_reward_term].weight = palm_weight

        # --- # NEEDS: external-load ramp (not wired into THIS curriculum) ---
        # Legacy advanced hammer_force_when_lifted's force_range by level. The surface exists — the pull is a
        # per-axis knob of apply_object_external_force_when_lifted, so a ramp writes
        # env.event_terms[name].params["z_range"] — but this curriculum does not drive it yet -> deferred.

        return {
            "level": float(level),
            "success_rate": self._success_rate,
            "contact_reward_weight": contact_weight,
            "palm_reward_weight": palm_weight,
        }


class goal_tolerance_curriculum:
    """SR-gated shrink of goal success-tolerance from loose(start) toward floor
    (ports legacy perceptive goal_tolerance_curriculum, adapted for fixed-goal).

    Every ``interval`` [policy steps], if success termination rate >= ``level_up_threshold`` then
    tol *= increment, clamped at floor. **Rearm only when a shrink actually happens** (SimToolReal
    semantics — otherwise it re-checks at every reset boundary). Boundary check uses common_step_counter
    (CPU int) so no sync; SR is read only when firing.

    """

    def __init__(
        self,
        success_term: str = "task_success",
        start: float = 0.15,
        floor: float = 0.01,
        increment: float = 0.9,
        interval: int = 3000,
        level_up_threshold: float = 0.5,
    ):
        assert 0.0 < increment < 1.0, f"increment must be in (0,1) — got {increment}"
        assert start >= floor > 0.0, f"need start>=floor>0 — start={start} floor={floor}"
        self.success_term = success_term
        self.start = float(start)
        self.floor = float(floor)
        self.increment = float(increment)
        self.interval = int(interval)
        self.level_up_threshold = float(level_up_threshold)
        # Persistent state (term instance). last_shrink_step=None -> first-call baseline.
        self._tol = float(start)
        self._last_shrink_step: int | None = None
        self._success_rate = 0.0

    def __call__(self, env) -> dict[str, float]:
        step = env.common_step_counter
        if self._last_shrink_step is None:
            self._last_shrink_step = step                  # baseline — no shrink on first call

        if step - self._last_shrink_step >= self.interval:
            self._success_rate = env.termination_rate(self.success_term)   # sync — only inside the gate
            if self._success_rate >= self.level_up_threshold:
                self._tol = max(self._tol * self.increment, self.floor)
                self._last_shrink_step = step              # rearm only on shrink (SimToolReal)


        return {"goal_dist_tol": self._tol, "success_rate": self._success_rate}
