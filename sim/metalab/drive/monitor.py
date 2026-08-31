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

Two channels are deliberately NOT obs terms: :func:`_kp_channel` (``Joint Kp``) and
:func:`_tau_lim_channel` (``Joint Torque Limit``). Both read the coupled transmission — joint-space
stiffness ``Gᵀ·diag(k_phi)·G`` and the joint torque the motors can actually deliver ``Gᵀ·(envelope ∩
rated)`` — controller properties rather than something a policy observes. They are computed host-side per
snapshot from the backend (``coupled_kq`` / ``coupled_tau_lim``), so they stay out of the step loop and out
of the obs library until a task actually wants to observe them.

The runner samples EVERY channel on each published step (:func:`sample`) — the browser buffers all of
them continuously, so switching tabs while paused shows the same frozen instant on every plot.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Callable

import numpy as np

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
    # series are not simply "one per joint" (Joint Kp = a row of K per joint) use it so the selector can
    # nest the entries under their joint instead of listing 124 flat names.
    rows: tuple = ()


# Which transmissions get cross-term checkboxes. finger (MCP×PIP ~0.97 normalised) and wrist (≤0.24,
# pose-dependent) are the ones where a joint's stiffness genuinely depends on its neighbour's error.
# The rest are diagonal in practice — thumb ~0.03 (Yaw exactly 0), shoulder exactly 0 (direct drive),
# elbow exactly 0 while its two motors share a gain, which `equal_gain_warnings` polices instead of a
# flat-zero plot line: break that symmetry and the dashboard says so in the header.
_OFF_DIAG_PARTS = frozenset({"finger", "wrist"})


def _partner(owner: str, other: str) -> str:
    """Label a cross term by what it couples TO, shortened against the joint that owns the row:
    R_Index_MCP's partner R_Index_PIP reads "PIP"; R_Elbow's partner R_Wrist_Yaw reads "Wrist_Yaw"."""
    a, b = owner.split("_"), other.split("_")
    k = 0
    while k < min(len(a), len(b)) - 1 and a[k] == b[k]:
        k += 1
    return "_".join(b[k:])


def _kp_channel(state, order: list[str]) -> Channel | None:
    """``Joint Kp`` — the coupled groups' joint-space stiffness ``K_q = Gᵀ·diag(k_phi)·G`` [N·m/rad], live.

    Motor-space PD makes joint stiffness a function of POSE (through the transmission Jacobian G) and it is
    not diagonal: what a joint feels depends on which other joints of its group are off target too. So the
    channel plots matrix ENTRIES, not one scalar per joint — no reduction to a single number can carry the
    coupling honestly (the same K row can read 9.3 or −1.3 N·m/rad for the same joint depending on the error
    pattern).

    Laid out **one row of K per joint**, joints in ``order`` (the runner's report order, so the checkbox
    list reads like the Joint Position tab): the joint's own ``diag``, plus one entry per group partner for
    the transmissions that actually couple (``_OFF_DIAG_PARTS``). A symmetric pair therefore appears under
    BOTH joints — that redundancy is the point, each joint's row is readable on its own. Joints outside a
    coupled group (waist/neck, native PD) have no K_q and no row.

    Note the scales when picking series: a shoulder diagonal is 4930 N·m/rad against a finger's 0.8, so
    those belong in different panes (which is what the Monitor grid does — one pane per joint).

    ``None`` when the robot has no coupled transmission (control_mode != motor) — no tab rather than a flat
    zero one. Evaluated host-side per snapshot (see backend.coupled_kq), so the step loop pays nothing."""
    groups = state.coupled_kq()
    if not groups:
        return None
    rank = {j: k for k, j in enumerate(order)}
    owned = sorted(((gi, i, j) for gi, g in enumerate(groups) for i, j in enumerate(g["joints"])),
                   key=lambda e: rank.get(e[2], len(rank)))
    labels, plan, rows = [], [], []
    for gi, i, jname in owned:
        g = groups[gi]
        names = g["joints"]
        me = jname.replace("_Joint", "")
        cross = g["part"] in _OFF_DIAG_PARTS
        items = []
        for j in [i] + ([k for k in range(len(names)) if k != i] if cross else []):
            tag = "diag" if i == j else _partner(jname, names[j]).replace("_Joint", "")
            labels.append(f"{me}·{tag}" if i == j else f"{me}×{tag}")
            items.append([len(labels) - 1, tag])
            plan.append((gi, i, j))
        rows.append({"joint": me, "items": items})

    def fn(st):
        gs = st.coupled_kq()
        return np.asarray([gs[gi]["K"][i][j] for gi, i, j in plan], dtype=np.float32).reshape(1, -1)

    return Channel(key="joint_kp", title="Joint Kp", unit="N·m/rad · Gᵀ·diag(k_phi)·G, pose-dependent",
                   labels=labels, fn=fn, digits=3, rows=tuple(rows))


def _tau_lim_channel(state, order: list[str]) -> Channel | None:
    """``Joint Torque Limit`` — how much torque each coupled joint can actually produce, right now [N·m].

    Not a constant per joint: motor-space PD means a joint's torque comes from its group's motors through
    ``τ_q = Gᵀ·τ_m``, so the bound moves with the POSE (leverage) and with the SPEED (each motor's
    torque-speed envelope). See :func:`motor_coupling._tau_lim` for the expression and its honesty caveat
    (per-joint projection of a coupled polytope — exact per joint, not simultaneously achievable).

    Three series per joint, in ONE pane (the grid overlays a joint's entries): ``+lim`` / ``−lim`` = the
    envelope, and ``τ`` = the applied joint torque — the same post-clamp value the Joint Torque tab plots,
    repeated here because the limit is only meaningful next to what is being drawn against it. τ riding on a
    boundary IS motor saturation; the distance to it is the headroom the PD has left (gravcomp fold spends
    part of that budget, which is exactly why it belongs in the same pane).

    Joints outside a coupled group (waist/neck, native PD) have no G and no row. ``None`` when the robot has
    no coupled transmission at all (control_mode != motor) — no tab rather than a flat zero one."""
    groups = state.coupled_tau_lim()
    if not groups:
        return None
    rank = {j: k for k, j in enumerate(order)}
    owned = sorted(((gi, i, j) for gi, g in enumerate(groups) for i, j in enumerate(g["joints"])),
                   key=lambda e: rank.get(e[2], len(rank)))
    labels, plan, rows, jnames = [], [], [], []
    for gi, i, jname in owned:
        me = jname.replace("_Joint", "")
        items = []
        for tag, kind in (("+lim", "hi"), ("−lim", "lo"), ("τ", None)):
            labels.append(f"{me}·{tag}")
            items.append([len(labels) - 1, tag])
            plan.append((gi, i, kind, len(jnames)))   # kind None → read it from the applied-torque term
        rows.append({"joint": me, "items": items})
        jnames.append(jname)
    def fn(st):
        gs = st.coupled_tau_lim()
        # the applied τ_q, from the same obs term the Joint Torque tab uses
        tau = joint_torque_obs(st, jnames)[0].detach().cpu().numpy()   # (len(jnames),) — one read per snapshot
        row = [(gs[gi][kind][i] if kind else tau[t]) for gi, i, kind, t in plan]
        return np.asarray(row, dtype=np.float32).reshape(1, -1)

    return Channel(key="joint_tau_lim", title="Joint Torque Limit",
                   unit="N·m · Gᵀ·(envelope ∩ rated), pose+speed-dependent · τ = applied",
                   labels=labels, fn=fn, digits=3, rows=tuple(rows))


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
    * **PD / Grav torque** need a backend that separates the components — i.e. the motor-level control
      path (joint↔motor Jacobian: PD + gravity feedforward summed in motor space, ONE torque-speed/rated
      clamp on that sum, then ``Gᵀ`` back to joint space). Where it exists you get three torque tabs:
      the applied total (post-clamp, exact) and the two PRE-clamp components. They only add up while
      unsaturated — the gap IS what the clamp removed, which is the point of watching all three.
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
    # Coupled groups only — joint stiffness and joint torque limit are both transmission properties (pose-,
    # and for the limit speed-dependent), so they exist exactly where a motor-space PD does.
    for extra in (_kp_channel(state, joints), _tau_lim_channel(state, joints)):
        if extra is not None:
            chans.append(extra)
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
