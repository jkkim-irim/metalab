"""RL monitor adapter — env_driver snapshots in the STANDALONE dashboard's schema.

Standalone already has the finished plot viewer (``dashboard/page.py``, fed by
``drive/control_server.py`` + ``drive/monitor.py``): one plot tab per channel over a shared timeline,
per-series checkboxes, hover crosshair, freeze-on-pause. The RL tab grew its own page in ``telemetry.py``
and drifted. Rather than maintain two plotters, this module re-shapes what env_driver already publishes into
the schema that page consumes, so the RL tab can serve the very same HTML.

Pure dict -> dict on purpose: no torch, no driver, no engine. env_driver keeps producing
``snapshot_describe()`` / ``snapshot_rows()`` unchanged (the eval rollout log is the other consumer and is
keyed to those shapes), and everything below is a translation of that output. That also makes the mapping
directly testable without a GPU.

Two shapes, standalone's on the left::

    /describe   {engine, task, channels: [{key, title, unit, labels, digits, section}], groups, controls, ...}
                                ^-- from `cards`: key = "<group>.<name>", title = name

    /stream     {t, ..., ch: {key: [values]}}          standalone, one env
                {t, ..., envs: {"<id>": {key: [values]}}}   RL, several envs

`envs` is the one addition: standalone runs a single env and has no selector, RL publishes a few. The page
normalizes ``envs`` and a bare ``ch`` to the same per-env buffer, so a single ingest path covers both.

RL leaves ``groups`` (trajectory CSVs) and ``controls`` (joint take-over rows) empty — the policy drives, and
there is nothing to play back — so the page hides those sub-tabs and opens on Monitor.

On top of the driver's own cards it synthesizes **overlay view cards** (:data:`OVERLAYS`): one channel that
pairs two existing ones per joint, so target and measured position share an axis instead of sitting in two
tabs. They reuse the page's ``rows`` grouping (one pane per joint) — the same mechanism standalone's Joint Kp
channel uses.
"""
from __future__ import annotations

from typing import Any

#: Cards the page renders as something OTHER than a plot. ``eval_sr`` is a per-env success/attempt counter
#: table (all envs at once, so it ignores the env selector) — it still needs a combo entry to be reachable,
#: so it is kept as a channel and tagged with its ``kind`` instead of dropped.
_TABLE_KINDS = frozenset({"eval_sr"})

#: Fallbacks for display metadata an obs term does not declare. ``monitor.Channel`` carries a real unit and
#: digit count per channel; a card without them falls back to these.
_DEFAULT_DIGITS = 3

#: Tab sections, in display order. This is the RL tab's whole reason for existing next to standalone: what
#: matters is WHICH policy input a value feeds, and an obs term is only actor-visible if it is in that group.
#: A term in both groups is ACTOR (the actor sees it); privileged = critic-only.
SECTIONS = ("actor obs", "critic obs", "reward", "action", "state", "custom")

#: Non-plot channel kinds live under ``custom`` too, so everything that is not a raw obs/reward/action series
#: is found in one place.

#: Overlay view cards — a SYNTHETIC channel built by pairing two existing channels per joint, so two series
#: that belong together land on one axis instead of in two tabs you have to flip between.
#:
#: ``target_vs_pos`` is the command-delay / tracking view: `prev_action_targets` is the PRE-delay commanded
#: target and `joint_pos` is what the joint actually did, so their gap is transport lag + PD tracking. Only
#: honest to share an axis because both are rad. `marker` asks the page for a dot per sample, which is what
#: makes a delay countable (with min=max=10 the queue holds the spawn pose, so pos stays flat for exactly 10
#: points before it moves).
#: The position source is always a CLEAN read, never an actor copy carrying ``noise=``. What this view
#: measures — transport lag and PD tracking — is physical, and sensor noise is added ON TOP of it: at the
#: student's 0.1 rad encoder std the "N discrete flat samples" that make a delay countable are buried, so the
#: one thing the card exists to show stops being readable. Read the noisy copy in its own ACTOR OBS tab.
#:
#: One entry PER position-read term name a contract might use, because a task is free to split that read by
#: limb (a split one keeps its clean copies under ``*_clean`` in the critic group; a merged one has a single
#: un-noised ``joint_pos``). ``overlay_plan`` drops an entry whose source card is
#: absent, so listing all the spellings is what makes this work for either shape without the RL tab having to
#: know which task it is looking at. ``prev_action_targets`` needs no clean twin — a commanded target is not a
#: sensor read, so no contract puts noise on it.
OVERLAYS = (
    #: ``line: False`` = SCATTER only. A connecting line interpolates between samples, which is the one thing
    #: that must not be read here: a transport delay is N discrete flat samples, and a line through them looks
    #: like motion that never happened.
    {"key": "custom.target_vs_pos", "title": "target vs pos", "unit": "rad", "marker": True, "line": False,
     "sources": (("prev_action_targets", "tgt"), ("joint_pos", "pos"))},
    {"key": "custom.target_vs_pos_arm", "title": "target vs pos (arm)", "unit": "rad", "marker": True,
     "line": False, "sources": (("prev_action_targets", "tgt"), ("arm_joint_pos_clean", "pos"))},
    {"key": "custom.target_vs_pos_hand", "title": "target vs pos (hand)", "unit": "rad", "marker": True,
     "line": False, "sources": (("prev_action_targets", "tgt"), ("hand_joint_pos_clean", "pos"))},
)


def _card(driver_describe: dict, name: str) -> dict | None:
    return next((c for c in driver_describe.get("cards", ()) if c["name"] == name), None)


def overlay_plan(driver_describe: dict) -> list[dict]:
    """Resolve each :data:`OVERLAYS` entry against the run's actual cards → channel def + value plan.

    Joints are the INTERSECTION of the sources' labels, in the first source's order: a target exists only for
    a commanded joint, so `prev_action_targets` (the action joints) is the narrower list. Empty intersection,
    or a missing source card, drops the overlay rather than emitting a channel with no series.

    Each entry carries ``rows`` (the page's existing per-joint grouping — one PANE per joint, holding that
    joint's series with its own y autoscale) and ``pick`` = ``[(source name, source index)]`` in label order,
    which is all :func:`snapshot` needs.
    """
    out = []
    for ov in OVERLAYS:
        cards = [(_card(driver_describe, n), suffix) for n, suffix in ov["sources"]]
        if any(c is None for c, _ in cards):
            continue
        first = cards[0][0]["labels"]
        common = [j for j in first if all(j in c["labels"] for c, _ in cards)]
        if not common:
            continue
        labels, pick, rows = [], [], []
        for j in common:
            items = []
            for card, suffix in cards:
                items.append([len(labels), f"{j} {suffix}"])
                labels.append(f"{j} {suffix}")
                pick.append((card["name"], card["labels"].index(j)))
            rows.append({"joint": j, "items": items})
        out.append({"key": ov["key"], "title": ov["title"], "unit": ov["unit"],
                    "digits": int(ov.get("digits", _DEFAULT_DIGITS)), "section": "custom",
                    "marker": bool(ov.get("marker")), "line": bool(ov.get("line", True)),
                    "labels": labels, "rows": rows, "pick": pick})
    return out


def _section(card: dict) -> str:
    """Which tab section a card belongs to (see :data:`SECTIONS`)."""
    g = card["group"]
    if g != "obs":
        return g if g in SECTIONS else "state"
    return "actor obs" if "actor" in (card.get("obs_groups") or ()) else "critic obs"


def channel_key(group: str, name: str) -> str:
    """Snapshot key / tab id for one card. Group-qualified so ``reward`` and an obs term named ``reward``
    cannot collide."""
    return f"{group}.{name}"


def channels(driver_describe: dict) -> list[dict]:
    """``snapshot_describe()["cards"]`` -> standalone ``/describe`` channel dicts, in card order.

    Non-plot cards are dropped (see ``_NON_PLOT_KINDS``). ``obs_groups`` rides along on obs channels: it is
    the only record of which policy input a term feeds, and the RL tab labels its tabs with it.
    """
    out = []
    for c in driver_describe.get("cards", ()):
        kind = c.get("kind")
        ch: dict[str, Any] = {
            "key": channel_key(c["group"], c["name"]),
            "title": c["name"],
            "unit": c.get("unit", ""),
            "labels": list(c["labels"]),
            "digits": int(c.get("digits", _DEFAULT_DIGITS)),
            "section": "custom" if kind in _TABLE_KINDS else _section(c),
        }
        if kind:
            ch["kind"] = kind                 # the page switches on this instead of plotting
        if c.get("obs_groups"):
            ch["obs_groups"] = list(c["obs_groups"])
        out.append(ch)
    # Grouped by section, stable within one: the tab row then reads actor -> critic -> reward -> action
    # instead of contract declaration order. Standalone sets no section, so its tab order is untouched.
    out += [{k: v for k, v in ov.items() if k != "pick"} for ov in overlay_plan(driver_describe)]
    # Unknown sections sort last rather than raising, so a new one needs no edit here.
    return sorted(out, key=lambda c: (SECTIONS.index(c["section"]) if c["section"] in SECTIONS
                                      else len(SECTIONS)))


def describe(driver_describe: dict) -> dict:
    """Full standalone-shaped ``/describe`` for the RL tab.

    Carries the RL-only fields (``num_envs``, ``max_step``, ``goal``) alongside the standalone ones; the page
    ignores what it does not know, and the env selector needs ``num_envs``.
    """
    hz = driver_describe.get("hz")
    return {
        "engine": driver_describe.get("engine", ""),
        "task": driver_describe.get("task", ""),
        "channels": channels(driver_describe),
        "groups": [],          # no trajectory playback under RL -> the page hides that sub-tab
        "controls": [],        # the policy drives every joint -> no Joint Control rows
        "control_hz": hz,
        "gravcomp": None,      # not a run-level toggle here (the contract owns it)
        "num_envs": driver_describe.get("num_envs"),
        "max_step": driver_describe.get("max_step"),
        "goal": driver_describe.get("goal"),
    }


def snapshot(rows: dict, t: float | None = None, overlays: list[dict] | None = None) -> dict:
    """``snapshot_rows()`` -> standalone ``/stream`` payload with a per-env ``envs`` map.

    Every channel of every published env is present in every snapshot, which is what lets the page buffer all
    tabs (and all envs) at once and show the same frozen instant on each after a pause.

    A reward row is ``{term: float}`` while a channel is a value LIST, so reward is re-flattened in the
    describe's label order — the page pairs values to labels positionally.
    """
    envs: dict[str, dict[str, list[float]]] = {}
    step = max_step = None
    for env_id, row in rows.items():
        ch: dict[str, list[float]] = {}
        for name, vals in row.get("obs", {}).items():
            ch[channel_key("obs", name)] = list(vals)
        for name, vals in row.get("state", {}).items():
            ch[channel_key("state", name)] = list(vals)
        rew = row.get("reward") or {}
        if rew:
            ch[channel_key("reward", "reward")] = [float(v) for v in rew.values()]
        act = row.get("action")
        if act is not None:
            ch[channel_key("action", "action")] = list(act)
        obs = row.get("obs", {})
        for ov in (overlays or ()):               # synthetic overlays, gathered from the same obs row
            vals = [obs[src][i] if src in obs and i < len(obs[src]) else float("nan")
                    for src, i in ov["pick"]]
            ch[ov["key"]] = vals
        envs[str(env_id)] = ch
        if step is None:                     # every env shares max_step; step is per env (first wins)
            step, max_step = row.get("step"), row.get("max_step")
    out: dict[str, Any] = {"envs": envs, "step": step, "max_step": max_step}
    if t is not None:
        out["t"] = t
    return out


def reward_labels(driver_describe: dict) -> list[str]:
    """The reward channel's per-dim labels — the term order ``snapshot`` flattens a reward row into."""
    for c in driver_describe.get("cards", ()):
        if c["group"] == "reward":
            return list(c["labels"])
    return []
