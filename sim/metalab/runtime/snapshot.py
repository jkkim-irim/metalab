"""Snapshot SCHEMA — what a per-env snapshot row holds, described once for both dashboards.

:meth:`sim.metalab.runtime.env_driver.EnvDriver.snapshot_describe` delegates here, and two consumers key
their time series to the result: ``rl_monitor`` (the live --viz tab) and ``rollout_log`` (the eval report).
The ROWS stay in the driver — batching every obs/reward/action read into one device-to-host sync is
orchestration over driver-owned buffers — but naming the columns is a dashboard concern with no business in
an RL runtime, so it lives here where it can be read and tested on its own.

Column naming is the whole problem this module exists for. A term that takes no joint- or body-list knob
leaves the loader nothing to derive ``dim_labels`` from, and the card used to fall back to bare indices:
a 5-fingertip force VECTOR plotted as "0".."14", which is how a mislabelled contact column stayed invisible,
and enough to break any pairing with a joint-named channel (the overlay cards intersect on label). The sets
below are matched on the obs FUNCTION's name, not the contract's term name, so a contract that aliases the
same function (the student critic's ``prev_action_targets_clean``) is labelled like its twin instead of
sitting on "0","1",... beside it.
"""
from __future__ import annotations

#: Component suffixes for a term that emits several numbers PER named body/joint, keyed by dims-per-entry.
#: Quaternions are wxyz — the one order every pose obs term emits (envs/obs/common.py header).
_COMPONENTS = {
    3: ("x", "y", "z"),
    7: ("x", "y", "z", "qw", "qx", "qy", "qz"),
}

#: Single-pose terms: one pos3+quat4 and no body/joint-list knob, so there is nothing to name the columns with.
_POSE_TERMS = frozenset({"object_state_world", "object_pose",
                         "body_pose_in_chest", "palm_pose_in_chest"})

#: Terms whose columns ARE the action vector, in action-group order.
_ACTION_VECTOR_TERMS = frozenset({"prev_action_targets"})

#: Terms whose columns are the ROBOT's fingertips, in ``RobotSpec.fingertips`` order. They take no ``bodies``
#: knob ON PURPOSE — which bodies are pads is a robot fact, not a term knob (api/contact.py).
_FINGERTIP_TERMS = frozenset({"fingertip_contact_steps"})

#: Terms whose columns are the total then one per REWARD term, in contract order. Labelled from the spec
#: rather than a contract ``labels=`` list, which would be a second copy of the REWARD block to keep in sync.
_REWARD_VECTOR_TERMS = frozenset({"instantaneous_reward"})


def dim_labels(names: list[str], width: int, fn_name: str = "") -> list[str]:
    """Per-dim labels for a card: the term's own names when they match 1:1, expanded ``name·x`` style when the
    term emits a known component group per name, the pose components for a nameless single-pose term
    (:data:`_POSE_TERMS`), else index fallback."""
    if len(names) == width:
        return list(names)
    if names and width % len(names) == 0 and (width // len(names)) in _COMPONENTS:
        return [f"{n}·{c}" for n in names for c in _COMPONENTS[width // len(names)]]
    if not names and fn_name in _POSE_TERMS and width == 7:
        return list(_COMPONENTS[7])
    return [str(i) for i in range(width)]


def describe(driver, state) -> dict:
    """Static description of what a snapshot row holds → the dict both dashboards consume.

    ``cards`` = one plottable group per name (each obs term, plus a single ``action`` card),
    with per-dim labels. An obs card also carries ``obs_groups`` = the obs groups that term belongs to
    (``actor``/``critic``/``privileged``): terms are deduped by NAME here, since actor and critic share most
    of them, so this is the only record of which policy input a term actually feeds.

    ``state`` is the driver's ``EnvDriver`` — each term is evaluated once, purely to learn its width. That is
    why a term which is only evaluable after a step (``curriculum_state`` on a curriculum that has published
    nothing yet) forces its consumer to describe late rather than at construction.
    """
    spec = driver.spec
    engine = type(driver.backend).__name__.replace("Backend", "").lower()   # genesis / newton
    obs_groups: dict[str, list[str]] = {}
    for gname, terms in spec.obs.items():
        for t in terms:
            obs_groups.setdefault(t.name, []).append(gname)
    act_labels = [j for _n, g in driver.action_groups for j in g.joints]    # the action vector's columns
    tip_labels = list(spec.robot.fingertips)                                # the pad-indexed terms' columns
    cards: list[dict] = []
    for _gname, terms in spec.obs.items():
        for t in terms:
            if any(c["name"] == t.name for c in cards):
                continue
            w = int((t.fn(state, **t.params) * t.scale).shape[-1])
            names = t.dim_labels
            if t.fn.__name__ in _ACTION_VECTOR_TERMS:
                names = act_labels
            elif t.fn.__name__ in _FINGERTIP_TERMS:
                names = tip_labels
            elif t.fn.__name__ in _REWARD_VECTOR_TERMS:
                names = ["total"] + [r.name for r in spec.reward]
            cards.append({"group": "obs", "name": t.name,
                          "labels": dim_labels(names, w, t.fn.__name__),
                          "obs_groups": obs_groups[t.name], "unit": t.unit, "digits": t.digits})
    cards.append({"group": "action", "name": "action", "labels": list(act_labels)})
    goal = spec.goal
    if goal is not None:      # val/SR: per-env success/attempt counter (a table, not a plot)
        cards.append({"group": "eval", "name": "val/SR", "kind": "eval_sr",
                      "labels": ["success", "attempts"]})
    return {
        "task": spec.name, "engine": engine, "num_envs": driver.num_envs,
        "hz": round(1.0 / driver.step_dt), "dt": driver.step_dt,
        "episode_length_s": spec.episode_length_s, "max_step": driver.max_episode_length,
        "cards": cards,
        "goal": ({"pos": list(goal.pos), "quat": list(goal.quat),
                  "goal_dist_tol": goal.goal_dist_tol} if goal is not None else None),
    }
