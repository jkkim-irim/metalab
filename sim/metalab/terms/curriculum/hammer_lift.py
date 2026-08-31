"""hammer_lift task-specific curriculum terms — kept out of ``common.py`` so that stays task-agnostic.
Reference: sim/isaaclab/envs/hammer_lift/mdp/curriculum.py.
"""
from __future__ import annotations

import inspect
import math


def _assert_takes_knob(env, term_name: str, knob: str, why: str):
    """Fail at the first curriculum call if the event term this ramp drives cannot take ``knob``.

    A ramp retunes an event by writing ``EventTerm.params[knob]``; unchecked, a knob the signature does not
    name would surface as a TypeError on the next episode reset instead of here."""
    ev = env.event_terms.get(term_name)
    assert ev is not None, \
        f"{why} needs an event term named {term_name!r} — have: {sorted(env.event_terms)}"
    ps = inspect.signature(ev.fn).parameters
    assert knob in ps or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values()), (
        f"{why}: event {term_name!r} takes no {knob!r} knob — its signature takes {list(ps)[2:]}")
    return ev


class hammer_lift_success_curriculum:
    """SR-gated difficulty ramp: level 0 = the ``*_start`` values, level ``levels`` = the ``*_end`` ones,
    linear in between. An ``*_end`` left unset pins that bar to its ``*_start``.

    INDEPENDENT of the contract's ``GATE``. ``GATE.predicate`` is judged twice per step:

    * at THIS term's bar — ONE latch that promotes a level, ENDS the episode
      (``terminate.curriculum_passed``) and pays the success bonus (``reward.object_goal_reach_bonus``,
      which has no bars of its own), so the three cannot disagree.
    * at the GATE's own, fixed for the whole run — what ``val/SR`` reports.

    Level +1 above ``level_up_threshold``, -1 below ``demote_threshold`` (unset = one-way), checked at most
    once per ``eval_interval_iterations`` iters (the SR read is a GPU sync).

    Ramped per level: ``goal_dist_tol`` / ``hold_steps`` / ``force_threshold`` / ``palm_distance`` linearly;
    ``contact_count`` and the ``contact_fingers`` PREFIX by their own stated steps; and — OPT-IN, unset =
    the bar does not exist — ``joint_pose_tolerance``, friction ``mu_scale``, the actor's object-perception
    window, and the OBJECT's gravity.

    ``contact_step`` is STATED, not derived from ``levels``: at one tip per level the full grasp was demanded
    by level 4 and training collapsed (PR #108).

    Every term in ``dense_reward_terms`` fades to 0 at ``dense_reward_off_level``, so the task ends as the
    sparse bonus alone. The list is EXPLICIT so regularizers keep paying, and a name that is not a reward
    term fails loud."""

    def __init__(
        self,
        # Every bar is a start → end PAIR; `*_end` None = that bar is its `*_start` at every level.
        goal_dist_tol_start: float = 0.10,
        goal_dist_tol_end: float | None = None,   # [m]
        steps_start: int = 1,
        steps_end: int | None = None,             # steps
        contact_start: int = 0,
        contact_step: float | None = None,   # tips added PER LEVEL; required once contact_end > 0
        contact_end: int = 0,                # tips the count ramp clamps at (0 = no count condition)
        # The tips a LEVEL-UP demands a grip from, in the order the ramp adds them. This term's bar, not the GATE's.
        contact_fingers: tuple[str, ...] = (),
        contact_fingers_start: int = 0,      # how many of them are demanded at level 0
        contact_fingers_step: float | None = None,   # named tips added PER LEVEL; required once any is named
        force_start: float = 0.1,          # [N] pad-normal force that counts as gripping
        force_end: float | None = None,
        palm_start: float = 0.0,
        palm_end: float | None = None,            # [m]
        joint_pose_start: float | None = None,   # [rad] worst-joint bound at level 0; None = pose not ramped
        joint_pose_end: float | None = None,     # [rad] and at the last level
        joint_final_pose: dict | None = None,    # {joint: rad} the posture those bounds are measured against
        seen_start: float | None = None,   # object visible for this FRACTION of the episode at level 0; None = no ramp
        grav_start: float | None = None,   # [m/s^2] MAGNITUDE at level 0 (0 = weightless); None = no ramp
        # Friction MULTIPLIER at level 0 → 1.0 (the authored friction) at the last. >1 = a grippier start.
        friction_scale_start: float | None = None,
        friction_events: tuple[str, ...] = (),           # the friction-DR events the ramp scales
        levels: int = 20,
        dense_reward_off_level: int | None = None,   # from this level on, dense_reward_terms weights -> 0
        dense_reward_terms: tuple[str, ...] = (      # the DENSE shaping terms (regularizers stay on)
            "object_goal_keypoint_progress",
            "fingertip_object_distance_delta",
            "fingertip_object_contact",
        ),
        level_up_threshold: float = 0.5,   # SR at the CURRENT difficulty that promotes to the next level
        demote_threshold: float | None = None,   # SR below this DEMOTES one level (None = one-way curriculum)
        eval_interval_iterations: int = 50,
    ):
        assert levels >= 1, f"levels must be >=1 — got {levels}"
        assert goal_dist_tol_start > 0.0, f"goal_dist_tol_start must be > 0 — got {goal_dist_tol_start}"
        assert force_start > 0.0, f"force_start must be > 0 — got {force_start}"
        assert dense_reward_off_level is None or 0 <= dense_reward_off_level <= levels, \
            f"dense_reward_off_level must be within [0, levels={levels}] — got {dense_reward_off_level}"
        assert demote_threshold is None or 0.0 <= demote_threshold < level_up_threshold, \
            f"demote_threshold must be in [0, level_up_threshold={level_up_threshold}) — got {demote_threshold}"
        assert joint_pose_end is None or joint_pose_start is not None, (
            f"joint_pose_end={joint_pose_end} without joint_pose_start — a bar cannot have an end and no "
            f"start (drop it to leave the posture condition off)")
        self.goal_dist_tol_start = float(goal_dist_tol_start)
        self.goal_dist_tol_end = float(goal_dist_tol_start if goal_dist_tol_end is None else goal_dist_tol_end)
        self.steps_start = int(steps_start)
        self.steps_end = int(steps_start if steps_end is None else steps_end)
        self.contact_start = int(contact_start)
        self.contact_step = None if contact_step is None else float(contact_step)
        self.contact_end = int(contact_end)
        self.contact_fingers = tuple(contact_fingers)
        self.contact_fingers_start = int(contact_fingers_start)
        self.contact_fingers_step = None if contact_fingers_step is None else float(contact_fingers_step)
        self.force_start = float(force_start)
        self.force_end = float(force_start if force_end is None else force_end)
        self.palm_start = float(palm_start)
        self.palm_end = float(palm_start if palm_end is None else palm_end)
        self.joint_pose_start = None if joint_pose_start is None else float(joint_pose_start)
        self.joint_pose_end = (None if joint_pose_start is None else
                               float(joint_pose_start if joint_pose_end is None else joint_pose_end))
        self.joint_final_pose = dict(joint_final_pose or {})
        self.seen_start = None if seen_start is None else float(seen_start)
        self.grav_start = None if grav_start is None else float(grav_start)
        self._grav_applied: float | None = None            # last magnitude pushed to the sim (write on change)
        self.friction_scale_start = None if friction_scale_start is None else float(friction_scale_start)
        self.friction_events = tuple(friction_events)
        self.levels = int(levels)
        self.dense_reward_off_level = None if dense_reward_off_level is None else int(dense_reward_off_level)
        self.dense_reward_terms = tuple(dense_reward_terms)
        self._dense_w0: dict[str, float] | None = None     # original weights, captured on the first call
        self.level_up_threshold = float(level_up_threshold)
        self.demote_threshold = None if demote_threshold is None else float(demote_threshold)
        self.eval_interval_iterations = int(eval_interval_iterations)
        self._level = 0
        self._last_iter: int | None = None
        self._success_rate = 0.0
        self._checked = False

    @property
    def level(self) -> int:
        """The TRAINING level — read by the driver so a recording can reproduce that checkpoint's world."""
        return self._level

    def jump_to_end(self) -> None:
        """Snap to the final level — the eval path for a finished run (``--curriculum_end``, no level)."""
        self._level = self.levels

    def freeze_for_eval(self, train_level: int) -> None:
        """Run the world of ``train_level`` — a mid-training checkpoint's recording then reproduces the task
        it actually learned, while ``val/SR`` still measures the GATE and stays comparable across the run."""
        assert 0 <= train_level <= self.levels, \
            f"freeze_for_eval: train_level={train_level} outside [0, levels={self.levels}]"
        self._level = int(train_level)

    def _check(self, env) -> None:
        """Fail loud, once, on a start/end pair that does not ramp the way the contract implies."""
        gate = env.gate
        assert gate is not None, (
            "hammer_lift_success_curriculum judges its level-up with the contract's GATE.predicate, but this "
            "task declares no GATE — add one (between EVENTS and TERMINATE). Its BARS are not read here; "
            "this term states its own start/end pairs.")
        assert self.goal_dist_tol_start >= self.goal_dist_tol_end, (
            f"goal_dist_tol_start={self.goal_dist_tol_start} must be >= "
            f"goal_dist_tol_end={self.goal_dist_tol_end} (tightens)")
        assert self.steps_start <= self.steps_end, \
            f"steps_start={self.steps_start} must be <= steps_end={self.steps_end} (grows)"
        assert self.contact_start <= self.contact_end, \
            f"contact_start={self.contact_start} must be <= contact_end={self.contact_end} (grows)"
        assert (self.contact_step is not None) == (self.contact_end > 0), (
            f"contact_step={self.contact_step} and contact_end={self.contact_end} must be set "
            f"together (both to ramp the grip-count condition, neither to disable it)")
        if self.contact_step is not None:
            assert self.contact_step > 0.0, f"contact_step must be > 0 — got {self.contact_step}"
            assert self.contact_start + self.contact_step * self.levels >= self.contact_end, (
                f"contact_step={self.contact_step} never reaches contact_end={self.contact_end} "
                f"from contact_start={self.contact_start} in levels={self.levels} — the last level would "
                f"not be the end this ramp states")
        assert (self.contact_fingers_step is not None) == bool(self.contact_fingers), (
            f"contact_fingers_step={self.contact_fingers_step} and contact_fingers="
            f"{list(self.contact_fingers)} must be set together (both to ramp the named-tip level-up "
            f"condition, neither to disable it)")
        if self.contact_fingers:
            unknown = [b for b in self.contact_fingers if b not in env.fingertips]
            assert not unknown, (
                f"contact_fingers names {unknown}, which the robot does not declare as fingertips: "
                f"{env.fingertips}")
            assert self.contact_fingers_step > 0.0, \
                f"contact_fingers_step must be > 0 — got {self.contact_fingers_step}"
            assert 0 <= self.contact_fingers_start <= len(self.contact_fingers), (
                f"contact_fingers_start={self.contact_fingers_start} must be within "
                f"[0, len(contact_fingers)={len(self.contact_fingers)}] (grows)")
            assert (self.contact_fingers_start + self.contact_fingers_step * self.levels
                    >= len(self.contact_fingers)), (
                f"contact_fingers_step={self.contact_fingers_step} never reaches all "
                f"{len(self.contact_fingers)} of contact_fingers from contact_fingers_start="
                f"{self.contact_fingers_start} in levels={self.levels} — the last level would never demand "
                f"the full grip this ramp is built around")
        assert self.force_start <= self.force_end, \
            f"force_start={self.force_start} must be <= force_end={self.force_end} (grows)"
        assert (self.palm_end <= 0.0) == (self.palm_start <= 0.0), (
            f"palm_start={self.palm_start} and palm_end={self.palm_end} must be set together "
            f"(both > 0 to ramp the palm condition, both 0 to disable it)")
        assert self.palm_start >= self.palm_end, \
            f"palm_start={self.palm_start} must be >= palm_end={self.palm_end} (tightens)"
        assert not self.joint_final_pose or self.joint_pose_start is not None, (
            "joint_final_pose without joint_pose_start — the posture is stated but no bound on it is ramped, "
            "so nothing would ever test it")
        if self.joint_pose_start is not None:
            assert self.joint_final_pose, (
                "joint_pose_start ramps how exactly the FINAL POSTURE must be held, so state the posture "
                "itself here too (joint_final_pose={joint: rad}). The GATE's own posture, if it has one, is "
                "what val/SR measures — a different bar.")
            assert self.joint_pose_start >= self.joint_pose_end, (
                f"joint_pose_start={self.joint_pose_start} must be >= joint_pose_end={self.joint_pose_end} "
                f"(tightens). Both are [rad].")
        if self.seen_start is not None:
            assert 0.0 < self.seen_start <= 1.0, (
                f"seen_start={self.seen_start} must be in (0, 1] — it is the FRACTION of the episode the "
                f"object stays visible for at level 0 (1.0 = the whole episode), not a step count. The ramp "
                f"ends at 1 step (the spawn pose); to skip it entirely pass None.")
        if self.grav_start is not None:
            assert "object_gravity" in env.capabilities, (
                "grav_start ramps the OBJECT's gravity, which this engine cannot do — it has "
                f"{sorted(env.capabilities)}. newton does it with per-body gravity compensation.")
            g_end = abs(env.physics_gravity_z)
            assert 0.0 <= self.grav_start <= g_end, (
                f"grav_start={self.grav_start} must be in [0, |physics.gravity|={g_end}] — the ramp GROWS up "
                f"to the contract's own gravity. It is a MAGNITUDE; the sign comes from physics.gravity. "
                f"0 starts WEIGHTLESS; to skip the ramp entirely pass None (not 0).")
        if self.friction_scale_start is not None:
            assert self.friction_events, \
                "friction_scale_start needs friction_events — name the friction-DR events the ramp scales"
            assert self.friction_scale_start > 0.0, \
                f"friction_scale_start={self.friction_scale_start} must be > 0 (it is a multiplier)"
            for name in self.friction_events:
                _assert_takes_knob(env, name, "mu_scale", f"friction_events names {name!r}, which")
        self._checked = True

    def __call__(self, env) -> dict[str, float]:
        if not self._checked:
            self._check(env)
        # Iteration = policy steps / the TRAINER's rollout length, which the trainer publishes on the env
        # (EnvDriver.set_num_steps_per_env) rather than the contract restating it. At counter 0 — the driver's
        # construction-time warm-up, and every eval/recording call, which all run before the first step — the
        # iteration is 0 for EVERY divisor, so the value is not read and those paths need none.
        it = 0 if env.common_step_counter == 0 else env.common_step_counter // env.num_steps_per_env
        if self._last_iter is None:
            self._last_iter = it                           # baseline — no bump on first call
        if it - self._last_iter >= self.eval_interval_iterations:
            self._last_iter = it
            self._success_rate = env.curriculum_success_rate()   # sync — only inside the gate
            if self._success_rate > self.level_up_threshold and self._level < self.levels:
                self._level += 1
            elif self.demote_threshold is not None and self._success_rate < self.demote_threshold \
                    and self._level > 0:
                # A level whose SR collapsed falls back to a difficulty the policy can still solve, instead
                # of dwelling on a zero-reward bar forever.
                self._level -= 1

        frac = self._level / self.levels                   # 0 → 1 over `levels` level-ups
        # start*(1-f)+end*f, NOT start+(end-start)*f: at f=1 this lands EXACTLY on the stated `*_end`.
        def _lerp(start: float, end: float) -> float:
            return start * (1.0 - frac) + end * frac
        tol = _lerp(self.goal_dist_tol_start, self.goal_dist_tol_end)
        steps = round(_lerp(self.steps_start, self.steps_end))
        force = _lerp(self.force_start, self.force_end)
        palm = _lerp(self.palm_start, self.palm_end)
        contact = (0 if self.contact_step is None else
                   min(self.contact_end, round(self.contact_start + self.contact_step * self._level)))
        fingers = () if self.contact_fingers_step is None else self.contact_fingers[:min(
            len(self.contact_fingers),
            round(self.contact_fingers_start + self.contact_fingers_step * self._level))]
        # Posture ramp (opt-in). The POSTURE never moves; only how exactly it must be held does.
        joint_tol = None
        if self.joint_pose_start is not None:
            joint_tol = float(_lerp(self.joint_pose_start, self.joint_pose_end))
        # THE bar of this level, counted and latched by _advance_curriculum. The level moves the bar; HOW a
        # hold is counted stays the GATE's word.
        env.set_curriculum_bars(hold_steps=int(steps), goal_dist_tol=tol, palm_distance=float(palm),
                                contact_count=contact, contact_fingers=fingers,
                                force_threshold=float(force),
                                joint_pose=self.joint_final_pose if joint_tol is not None else None,
                                joint_pose_tolerance=0.0 if joint_tol is None else joint_tol)
        # Perception ramp (opt-in): the ACTOR's view shrinks toward the spawn pose; the critic is never
        # blinded. Follows `frac` in a recording too, or a mid-training checkpoint would be replayed on an
        # input it never trained on. Floor 1 step, never 0 — without the spawn pose the task is unsolvable.
        seen = None
        if self.seen_start is not None:
            seen = max(1, round(_lerp(self.seen_start * env.max_episode_length, 1.0)))
            env.set_object_seen_steps(seen)
        # Gravity ramp (opt-in) — OBJECT only, so the arm always holds its own weight. Written only on a
        # CHANGE: the write re-syncs inertial properties in the solver and this term runs every step.
        grav = None
        if self.grav_start is not None:
            grav = _lerp(self.grav_start, abs(env.physics_gravity_z))
            if grav != self._grav_applied:
                env.set_object_gravity(math.copysign(grav, env.physics_gravity_z))
                self._grav_applied = grav
        # Friction ramp: a MULTIPLIER, so the authored mu_range keeps randomizing proportionally, and it
        # lands at the NEXT reset — changing mu mid-episode is a discontinuity the policy cannot see coming.
        if self.friction_scale_start is not None:
            fric = _lerp(self.friction_scale_start, 1.0)
            for name in self.friction_events:
                env.event_terms[name].params["mu_scale"] = fric
        # Dense-reward FADE: 1.0 at level 0 → 0.0 at dense_reward_off_level, linear so the return scale never
        # moves in one jump. Recomputed from the captured originals, so a level DOWN restores the weight.
        dense_on = 1.0
        if self.dense_reward_off_level is not None:
            if self._dense_w0 is None:
                missing = [n for n in self.dense_reward_terms if n not in env.reward_terms]
                assert not missing, (
                    f"dense_reward_terms names no such reward term: {missing} — valid: "
                    f"{sorted(env.reward_terms)}. A renamed term must be updated here or its weight would "
                    f"never be cut off.")
                self._dense_w0 = {n: float(env.reward_terms[n].weight) for n in self.dense_reward_terms}
            off = self.dense_reward_off_level
            dense_on = max(0.0, 1.0 - self._level / off) if off > 0 else 0.0
            for n, w0 in self._dense_w0.items():
                env.reward_terms[n].weight = w0 * dense_on
        # A ramp that is OFF publishes nothing — an always-present key reads as a live difficulty, and the
        # obs term naming it would plot a constant. Knobs no obs term reads are left out: each is a pure
        # function of `level`, which is already published.
        log = {"level": float(self._level), "success_rate": self._success_rate,
               "goal_dist_tol": tol, "hold_steps": float(steps)}
        if self.contact_end > 0:
            log["contact_count"] = float(contact)
        if self.contact_fingers:
            log["contact_fingers"] = float(len(fingers))
        if self.palm_end > 0.0 or self.palm_start > 0.0:
            log["palm_distance"] = palm
        if self.dense_reward_off_level is not None:
            log["dense_on"] = dense_on
        if joint_tol is not None:
            log["joint_pose_tolerance"] = joint_tol   # [rad]
        if seen is not None:
            log["seen_steps"] = float(seen)   # [policy steps] the actor still perceives the object for
        if grav is not None:
            log["gravity"] = grav          # [m/s^2] magnitude actually in the sim right now
        return log
