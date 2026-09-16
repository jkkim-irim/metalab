"""Standalone monitor channels — the dashboard's plot tabs, built from the **obs term library**.

One channel = one plot tab. A channel holds a *bound* obs term (the driver's ``EnvDriver`` in, ``(N, d)`` out) — the exact
object a task contract's ``ObsTerm`` holds — plus display metadata (tab title, unit, per-dim labels,
display scale). The factories come from ``sim.metalab.terms.obs``, the same symbols the train/eval
contract references (``hammer_lift_teacher``: ``joint_positions`` / ``joint_torque_obs`` /
``body_pose_in_chest``), so a value plotted in Standalone is the value a policy would observe —
one api, no re-implementation. ``state`` is the backend (obs terms read through it), so standalone
evaluates them directly without env_driver.

Standalone contracts carry **no obs list** (all learning stripped — see tasks/standalone/README of
intent in each contract's docstring), so the channel set is derived here from the robot's joint set,
its ``fingertips`` and declared ``frames`` vocabulary instead of a ``TermRef`` list.

The runner samples EVERY channel on each published step (:func:`sample`) — the browser buffers all of
them continuously, so switching tabs while paused shows the same frozen instant on every plot.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Callable

from sim.metalab.terms.obs import (
    body_pose_in_chest,
    hand_contact_force,
    joint_gravcomp_torque_obs,
    joint_pd_torque_obs,
    joint_positions,
    joint_torque_obs,
    palm_pose_in_chest,
)

_RAD2DEG = 180.0 / math.pi

# Pose channels → pos3 + quat4, wxyz (the one obs quaternion order — envs/obs/common.py header). A PALM
# frame uses palm_pose_in_chest, whose FRAME is the real robot's, so the plot is directly comparable with
# the robot's FK readout and with what the policy observes; any other frame keeps the generic body term.
_POSE_LABELS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
_PALM_PREFIX = "palm"          # robot.frames keys that are hands (palm_r / palm_l / palm)

_CHEST_FRAME = "chest_origin"   # robot.frames key the pose channels are expressed relative to


@dataclass(frozen=True)
class Channel:
    """One plot tab: a bound obs term + how to display it."""

    key: str                # snapshot key / tab id
    title: str              # tab label
    unit: str               # shown next to the title
    labels: list[str]       # per-dim series names (the left-column checkboxes)
    fn: Callable            # bound obs term: EnvDriver -> (N, d)
    scale: float = 1.0      # display scale applied to fn's output (e.g. rad → deg)
    digits: int = 2         # decimals the dashboard prints for this channel's values
    # Optional checkbox grouping: [{joint, items: [[series index, entry label], ...]}, ...]. Channels whose
    # series are not simply "one per joint" (Fingertip Contact Force = x/y/z per tip) use it so the selector
    # can nest the entries under their heading instead of listing them flat.
    rows: tuple = ()


def _contact_force_channel(spec, state) -> Channel | None:
    """``Fingertip Contact Force`` — per-fingertip object contact force in THAT TIP's own frame [N].

    The force ON the tip, both engines agreeing on the sign (newton's SensorContact measures "the contact
    force on each sensing object"; genesis sums the queried link's own side of each contact). A press
    therefore reads OPPOSITE the surface it lands on: the pads face **+x on the four fingers and −z on the
    thumb**, so a fingerprint press is −x / +z and a fingernail press is the other way. Reading that sign is
    the whole reason the vector is plotted per axis in the tip's OWN frame instead of as a magnitude —
    ``Hand Object Force Mag`` cannot tell pad from nail.

    Three series per tip in one pane (the grid overlays a heading's entries). ``None`` when the robot
    declares no fingertips or the task has no movable object to scope the contact against.
    """
    tips = spec.robot.fingertips
    if not tips or not spec.movable_objects:
        return None
    labels, rows = [], []
    for t in tips:
        me = t.replace("_Distal_Link", "")
        items = []
        for ax in ("x", "y", "z"):
            labels.append(f"{me}·{ax}")
            items.append([len(labels) - 1, ax])
        rows.append({"joint": me, "items": items})
    return Channel(key="contact_force", title="Fingertip Contact Force",
                   unit="N · force ON the tip, tip's own frame · pad press = fingers −x, thumb +z",
                   labels=labels, digits=4, rows=tuple(rows),
                   fn=partial(hand_contact_force, bodies=tips, target="object", ref_body="self"))


def build_channels(spec, joints: list[str], state) -> list[Channel]:
    """Channel set for one standalone run: joint position + torque over ``joints`` (the runner's report
    set), plus one chest-relative pose channel per non-chest frame the robot declares (``allex_right`` →
    palm; ``allex`` → palm_r, palm_l).

    Two channel sets are capability-gated on what the backend can actually read, rather than shipping tabs
    that would plot a quantity the engine never computes (the same style of check as the runner's
    gravcomp/Torque-mode row):

    * pose channels need the ``chest_origin`` frame;
    * **PD / Grav torque** split the applied torque into its two components (PD = applied − gravcomp).
    """
    chans = [
        Channel(key="joint_pos", title="Joint Position", unit="deg", labels=list(joints),
                fn=partial(joint_positions, names=joints), scale=_RAD2DEG),
        Channel(key="joint_torque", title="Joint Torque", labels=list(joints),
                unit="N·m · applied: PD+grav, post-clamp",
                fn=partial(joint_torque_obs, names=joints)),
        Channel(key="pd_torque", title="PD Torque", unit="N·m · pre-clamp component",
                labels=list(joints), fn=partial(joint_pd_torque_obs, names=joints)),
        Channel(key="grav_torque", title="Grav Torque", unit="N·m · pre-clamp component",
                labels=list(joints), fn=partial(joint_gravcomp_torque_obs, names=joints)),
    ]
    contact = _contact_force_channel(spec, state)      # needs fingertips + a movable object to scope against
    if contact is not None:
        chans.append(contact)
    frames = dict(spec.robot.frames)
    chest = frames.pop(_CHEST_FRAME, None)
    if chest is not None:
        for name, body in frames.items():
            palm = name.startswith(_PALM_PREFIX)
            chans.append(Channel(
                key=f"{name}_pose", title=f"{name.replace('_', ' ').title()} Pose",
                unit=(f"m · quat wxyz · real-robot frame · rel {_CHEST_FRAME}" if palm
                      else f"m · quat wxyz · rel {_CHEST_FRAME}"),
                labels=list(_POSE_LABELS),
                fn=(partial(palm_pose_in_chest, chest_body=chest, palm_body=body) if palm
                    else partial(body_pose_in_chest, chest_body=chest, target_body=body)), digits=4))
    return chans


def describe(channels: list[Channel]) -> list[dict]:
    """Static channel metadata for ``/describe`` — the browser builds one tab (+ its checkbox list) per entry."""
    return [{"key": c.key, "title": c.title, "unit": c.unit, "labels": c.labels, "digits": c.digits,
             **({"rows": [dict(r) for r in c.rows]} if c.rows else {})}
            for c in channels]


def sample(channels: list[Channel], state, env: int = 0) -> dict[str, list[float]]:
    """Evaluate every channel for ``env`` → ``{key: [values...]}``, already in display units (``scale``
    applied) so the browser plots what it receives. Dim count must match the declared labels."""
    out = {}
    for c in channels:
        v = c.fn(state)[env]
        assert v.shape[-1] == len(c.labels), \
            f"channel {c.key!r}: obs term returned {v.shape[-1]} dims but {len(c.labels)} labels are declared"
        out[c.key] = (v * c.scale).tolist() if c.scale != 1.0 else v.tolist()
    return out
