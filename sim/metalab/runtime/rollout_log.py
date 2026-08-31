"""Per-step rollout data log for the record path — the PLOT half of the eval HTML report.

Runs in the same sim-service process, on the same per-step hook, and over the same env set as the recording
itself (a spoke's rerun scene; see runtime/rerun_recording), so **one series row IS one recorded frame** by
construction rather than by post-hoc matching. Series are indexed by position in the recorded WINDOW (samples
since recording began), and ``meta.fps`` is the policy rate, so the page turns an index into a time with
``t = index / fps`` and into a recording frame with the identity.

The separate ``step`` array is the per-EPISODE step counter, which resets mid-window (the recorded window
deliberately spans resets): use ``done`` for episode boundaries, not a step jump.

WHAT IT WRITES. One ``data.json`` in ``out_dir``, beside the ``.rrd``::

    {"meta": {task, engine, hz, dt, max_step, fps,
              tabs:  [{key, title, unit, labels, section}],     # the report's channel tabs
              envs:  [{id, success, steps}]},                   # the report's top-level env tabs
     "data": {"<env id>": {step: [...], done: [0|1...], success: [0|1...],
                           series: {"<tab key>": [[dim0 over time], [dim1 over time], ...]}}}}

Channels come from :meth:`EnvDriver.snapshot_describe` / :meth:`EnvDriver.snapshot_rows` — the same pair
the ``--viz`` telemetry dashboard publishes live, so actor/critic obs, per-term reward and action are the
policy's real inputs/outputs, not a re-derivation. On top of those it adds joint-state channels (position/
velocity/torque, plus the PD and gravcomp components where the backend separates them) built from the obs
term library, mirroring the standalone Monitor's tabs. Those are MEASUREMENTS and have to be sampled here;
the report's other synthetic channels (``target vs pos``, val/SR) are views of what is already written and
are built at report time instead. Both land in the report's ``custom`` section.

Errors are NOT swallowed: a bad row fails the recording, which the train-time caller already catches at its
system boundary (``rl_trainer._record_and_upload``) and reports without killing the run.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

from sim.metalab.terms.obs import (
    joint_gravcomp_torque_obs,
    joint_pd_torque_obs,
    object_state_world,
)

# Joint-state channels: (key, title, unit, obs term, display scale). Terms are flat functions called as
# ``fn(env, **knobs)`` (envs/obs/common.py) — the knob here is ``names``. Torques in N*m.
#
# CUSTOM carries what the contract's obs groups do NOT: position/velocity/applied-torque used to live here
# too and were dropped once the critic declared them, so this section stays the place to look for a quantity
# no obs term publishes. These two are the applied torque's pre-clamp COMPONENTS, which no obs group asks
# for; they need a backend with the motor-level control path (same gate as drive/monitor.py).
_STATE_CHANNELS_SPLIT = [
    ("joint_torque_pd", "PD Torque", "N·m · pre-clamp component", joint_pd_torque_obs, 1.0),
]
# Same gate, but NOT over the whole action set: gravcomp is a robot HW fact that covers the arm only (hands
# are left loaded on purpose — held by wrist PD, see RobotSpec.gravcomp), so every finger column would be a
# flat zero. Scoped to what the contract declares, intersected with the joints the policy drives.
_GRAVCOMP_CHANNEL = ("joint_torque_gravcomp", "Grav Torque", "N·m · pre-clamp component",
                     joint_gravcomp_torque_obs, 1.0)


def _nacon(env) -> torch.Tensor:
    """Both naconmax consumers over ALL envs vs their shared cap, ``[nacon, ncollision, naconmax]`` → (N, 3).
    Batch-wide → broadcast to every env."""
    b = env.contact_budget_t()        # (5,) [nacon, ncollision, nefc] this step, then [naconmax, njmax]
    return b[[0, 1, 3]].float().unsqueeze(0).expand(env.num_envs, 3)


def _nefc(env) -> torch.Tensor:
    """Constraint rows in the WORST env vs its cap, ``[nefc, njmax]`` → (N, 2). Broadcast, as above."""
    b = env.contact_budget_t()
    return b[[2, 4]].float().unsqueeze(0).expand(env.num_envs, 2)


def _num(x):
    """4 significant digits — halves the JSON with no visible difference at plot resolution.

    Non-finite → ``None`` (JSON ``null``): ``JSON.parse`` rejects ``NaN``/``Infinity``, and a gap in the
    line is the honest rendering of a diverged step anyway."""
    v = float(x)
    return float(f"{v:.4g}") if math.isfinite(v) else None


class PerEnvRolloutLog:
    """Per-step series for the recorded envs, written as one ``data.json`` beside the ``.rrd``.

    ``idxs`` = the GLOBAL env ids that get a series and a tab in the report. The recording itself holds every
    env, so these select which ones are plotted, in this order."""

    def __init__(self, env, idxs, out_dir: str, *, fps: int):
        self._env = env
        self._idxs = [int(i) for i in idxs]
        assert self._idxs, "PerEnvRolloutLog needs at least one env index"
        self._dir = out_dir
        self._fps = int(fps)
        self._described = False                      # channel set resolved on the first sample (_ensure_tabs)
        self._tabs: list[dict] = []
        self._extra: list = []
        self._meta: dict = {}
        self._rows: dict[str, list] = {str(i): [] for i in self._idxs}
        self._succ: dict[str, bool] = {str(i): False for i in self._idxs}
        self._secs = 0.0                             # cumulative sampling time, reported at finish

    def _ensure_tabs(self) -> None:
        """Resolve the channel set on the FIRST sample, not at construction.

        ``snapshot_describe`` evaluates every obs term once to learn its width, and some terms are only
        evaluable after a step has run — e.g. ``curriculum_state`` asserts on a curriculum that has published
        nothing yet. Doing this in ``__init__`` therefore killed the sim-service at boot (taking the whole
        recording with it) for a contract that steps perfectly well. Deferring it to the first ``after_step`` makes the
        invariant airtight instead: the step whose row we are about to record has ALREADY captured every obs
        term successfully, so describing them cannot fail here for a reason the rollout itself would survive.
        """
        if self._described:                          # a flag, not `if self._tabs` — a contract with no
            return                                   # channels at all must not re-describe every step
        self._described = True
        desc = self._env.snapshot_describe()
        self._extra, state_tabs = self._state_channels(self._env, desc)
        object_extra, object_tabs = self._object_channels(self._env)
        budget_extra, budget_tabs = self._budget_channels(self._env)
        self._extra = self._extra + object_extra + budget_extra
        self._tabs = self._channel_tabs(desc) + state_tabs + object_tabs + budget_tabs
        self._meta = {
            "task": desc["task"], "engine": desc["engine"], "hz": desc["hz"], "dt": desc["dt"],
            "max_step": desc["max_step"],
            "fps": self._fps,                        # policy rate: the page's time axis is index / fps
            "tabs": self._tabs,
        }

    # --- channel/tab construction -----------------------------------------------------------------
    @staticmethod
    def _channel_tabs(desc: dict) -> list[dict]:
        """``snapshot_describe`` cards → report tabs. The ``eval`` card is a cumulative success/attempt
        counter, not a per-step series, so it is not a tab."""
        tabs = []
        for c in desc["cards"]:
            if c["group"] == "obs":
                # SECTION = the privilege partition, not raw group membership. An asymmetric contract wires
                # the critic as ACTOR + CRITIC, so every actor term is in both groups and joining the names
                # labelled the whole deployable half "actor + critic obs" — no plain "actor obs" section
                # existed at all. What a reader needs from the heading is the boundary: is this something the
                # DEPLOYED policy sees, or is it privileged (critic-only)? Membership in the actor group
                # answers that, so it wins. The exact groups stay visible in the tab's unit line.
                groups = list(c.get("obs_groups") or ["obs"])
                section = ("actor" if "actor" in groups else " + ".join(groups)) + " obs"
                tabs.append({"key": f"obs.{c['name']}", "title": c["name"],
                             "unit": "obs · " + " + ".join(groups),
                             "labels": list(c["labels"]), "section": section})
            elif c["group"] == "reward":
                tabs.append({"key": "reward", "title": "Reward", "unit": "Σ weight·value·dt, per term",
                             "labels": list(c["labels"]), "section": "reward"})
            elif c["group"] == "action":
                tabs.append({"key": "action", "title": "Action", "unit": "policy output, raw (pre-decode)",
                             "labels": list(c["labels"]), "section": "action"})
        return tabs

    @staticmethod
    def _state_channels(env, desc: dict):
        """Joint-state extras → ``([(name, fn, params, scale)], [tab])``. Joints = the action set (what the policy
        drives), taken from the action card so this needs none of the driver's internals. A contract with no
        action groups (scene-only) gets no state channels rather than zero-width ones.

        Filed under ``custom``: they are not contract channels, and the report groups everything that is not a
        raw obs/reward/action series in one section (same place the live RL tab puts its synthetic cards).

        Each channel carries its OWN joint list — every one is the action set except Grav Torque, which is the
        robot's gravcomp joints (see :data:`_GRAVCOMP_CHANNEL`)."""
        joints = next((list(c["labels"]) for c in desc["cards"] if c["group"] == "action"), [])
        if not joints:
            return [], []
        chans = [(c, joints) for c in _STATE_CHANNELS_SPLIT]
        gc = env.spec.robot.gravcomp
        gc_joints = [j for j in gc.joints() if j in joints] if gc is not None else []
        if gc_joints:                              # no gravcomp in the contract → no all-zero tab
            chans.append((_GRAVCOMP_CHANNEL, gc_joints))
        extra = [(key, fn, {"names": names}, scale) for (key, _t, _u, fn, scale), names in chans]
        tabs = [{"key": f"state.{key}", "title": title, "unit": unit, "labels": list(names),
                 "section": "custom"} for (key, title, unit, _f, _s), names in chans]
        return extra, tabs

    @staticmethod
    def _object_channels(env):
        """Object world pose → ``([(name, fn, params, scale)], [tab])``, filed under ``custom``.

        WORLD frame, and that is the point: every height bar this task family judges at is stated in world z
        (the below-height termination, the lift gates), while the obs groups carry the object chest-relative.
        Reading the two against each other is what this section is for. A scene with no movable object gets
        no tab rather than a read that would fail.
        """
        if not env.spec.movable_objects:
            return [], []
        extra = [("object_pose", object_state_world, {}, 1.0)]
        tabs = [{"key": "state.object_pose", "title": "Object Pose (world)", "unit": "m · quat wxyz",
                 "labels": ["x", "y", "z", "qw", "qx", "qy", "qz"], "section": "custom"}]
        return extra, tabs

    @staticmethod
    def _budget_channels(env):
        """mjwarp buffer pressure → ``([(name, fn, params, scale)], [tab])``, filed under ``custom``.

        WHY IT IS WORTH A CHANNEL: both caps are pre-sized (a GPU has no arena) and mjwarp drops the excess
        in SILENCE — a dropped contact generates no constraint, so the solver never pushes those bodies apart
        and they interpenetrate, while the force reads say 0. Each panel plots the live count AGAINST ITS CAP,
        so the crossing is the overflow: a flat line and a curve, and where the curve goes over it is where
        contacts started being thrown away. (The backend also asserts on it, but the panel is what says how
        much headroom was left when it did not.)

        Each counter is measured in the unit ITS OWN cap is judged in, and those units differ. The ``nacon``
        panel carries TWO counters because two stages draw from the same ``naconmax`` pool: the broadphase
        writes candidate pairs (``ncollision``) and bails before the narrowphase ever runs, then the
        narrowphase writes contacts (``nacon``). Candidates outnumber contacts several times over, so
        ``ncollision`` is usually the one that reaches the line first. Both are summed over every env —
        the pool is shared through one global atomic, so busy envs starve quiet ones and only the total says
        whether anything was dropped; a per-env count could not even identify who lost a contact. ``nefc`` is
        the worst single env instead, because constraint rows are walled off per env and an overflow stays
        inside the env that caused it. All series are batch-wide, so every env carries the same numbers.

        Newton-only — genesis sizes its own buffers and halts on overflow itself — so it is gated on the
        engine-partial capability rather than on whether the read happens to exist.
        """
        if "contact_budget" not in env.capabilities:
            return [], []
        chans = [
            ("nacon", "nacon", f"contacts + broadphase candidates, all {env.num_envs} envs · vs naconmax",
             _nacon, ["nacon", "ncollision", "naconmax"]),
            ("nefc", "nefc", "constraint rows, worst env · vs njmax", _nefc, ["nefc", "njmax"]),
        ]
        extra = [(key, fn, {}, 1.0) for key, _t, _u, fn, _l in chans]
        tabs = [{"key": f"state.{key}", "title": title, "unit": unit,
                 "labels": list(labels), "section": "custom"} for key, title, unit, _f, labels in chans]
        return extra, tabs

    # --- per-step sampling ------------------------------------------------------------------------
    def after_step(self, dones, task_success) -> None:
        """Sample one row per recorded env. Called from the same step hook as the recording (see
        vec_env_handler), so row *k* and recorded frame *k* are the same policy step.

        ``dones``/``task_success`` are (num_envs,) tensors indexed by GLOBAL env id; both are pulled to the
        host ONCE here rather than per env."""
        t0 = time.perf_counter()
        self._ensure_tabs()
        rows = self._env.snapshot_rows(self._idxs, extra=self._extra)
        d = dones.detach().cpu()
        s = None if task_success is None else task_success.detach().cpu()
        for k, out in self._rows.items():
            e, r = int(k), rows[k]
            r["done"] = bool(d[e])
            r["success"] = bool(s[e]) if s is not None else False
            if r["done"] and r["success"]:
                self._succ[k] = True          # latched over the whole window (the clip's ✓/✗ tag)
            out.append(r)
        self._secs += time.perf_counter() - t0

    # --- output -----------------------------------------------------------------------------------
    def finish(self) -> str:
        """Transpose the rows to per-dim series and write ``data.json``. Returns its path.

        Column-major (one array per dim over time) is what the report plots directly, and compresses far
        better than a dict per step.

        The elapsed time is logged because this runs inside the server's shutdown, and the client SIGKILLs
        after 30 s (see vec_env_handler.shield_shutdown_from_sigterm) — a long recording creeping toward
        that budget should be visible in the log, not discovered as a missing file.

        A session that never stepped (client died during boot) writes nothing rather than an empty file that
        looks like a recording."""
        if not any(self._rows.values()):
            print("[rollout-log] no steps sampled — nothing to write", flush=True)
            return ""
        t0 = time.perf_counter()
        data = {}
        for k, rows in self._rows.items():
            series = {}
            for tab in self._tabs:
                series[tab["key"]] = self._series(tab, rows)
            data[k] = {"step": [int(r["step"]) for r in rows],
                       "done": [int(r["done"]) for r in rows],
                       "success": [int(r["success"]) for r in rows],
                       "series": series}
        meta = dict(self._meta)
        # env tabs, in the order given. `success` is latched here over the whole window (not read back off a
        # filename), so the ✓/✗ on a tab comes from the rollout itself.
        meta["envs"] = [{"id": i, "success": self._succ[str(i)], "steps": len(self._rows[str(i)])}
                        for i in self._idxs]
        path = os.path.join(self._dir, "data.json")
        os.makedirs(self._dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"meta": meta, "data": data}, f, separators=(",", ":"))
        n_steps = len(self._rows[str(self._idxs[0])])
        print(f"[rollout-log] wrote {len(self._idxs)} envs × {n_steps} steps × {len(self._tabs)} channels "
              f"({os.path.getsize(path) / 1e6:.1f} MB; sampling {self._secs:.1f}s + write "
              f"{time.perf_counter() - t0:.1f}s) -> {path}", flush=True)
        return path

    @staticmethod
    def _series(tab: dict, rows: list) -> list:
        """One tab → ``[[dim0 over time], [dim1 over time], ...]``."""
        key = tab["key"]
        if key == "reward":       # dict of per-term scalars → one series per term, in label order
            return [[_num(r["reward"].get(lab, float("nan"))) for r in rows] for lab in tab["labels"]]
        if key == "action":
            vecs = [r["action"] for r in rows]
        else:                     # "obs.<name>" / "state.<name>"
            group, name = key.split(".", 1)
            vecs = [r[group][name] for r in rows]
        if not vecs:
            return [[] for _ in tab["labels"]]
        width = len(vecs[0])
        assert width == len(tab["labels"]), \
            f"channel {key!r}: {width} dims but {len(tab['labels'])} labels — the report would mislabel it"
        return [[_num(v[d]) for v in vecs] for d in range(width)]
