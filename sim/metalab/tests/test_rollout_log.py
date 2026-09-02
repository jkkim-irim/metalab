"""Tests for the record-path artifacts: the rollout log and the HTML report built from it
(``dashboard/rollout_log.PerEnvRolloutLog`` → ``dashboard/rollout_report.build_report``).

What is under test is the log's own logic — the tab set it derives from a contract's cards, the per-dim
transpose it writes, the success latch, and the strictness of the JSON a browser has to ``JSON.parse``.
The driver side (``EnvDriver.snapshot_describe`` / ``snapshot_rows``) is a stub here and is exercised for
real by the record smoke run (``metalab_eval.sh RECORD=1``), which is also what proves frame alignment.

Runs under pytest, or directly (``python3 sim/metalab/tests/test_rollout_log.py``); self-skips where torch
is not installed (the log takes the driver's ``dones``/``task_success`` tensors).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
import tempfile

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import torch
except ImportError:                                    # torch-less env → nothing here can run
    torch = None

from sim.metalab.dashboard.rollout_log import PerEnvRolloutLog  # noqa: E402
from sim.metalab.dashboard.rollout_report import build_report  # noqa: E402

_JOINTS = ["j0", "j1"]
_OBS_LABELS = ["a", "b", "c", "d"]


class _Backend:
    """Backend reads the state channels go through. The torque split is part of the REQUIRED SimBackend
    surface, so every spoke has it and the stub does too; ``contact_budget_t`` is engine-partial (newton) and
    is reached through ``EnvDriver.capabilities``, not by sniffing this object."""

    num_envs = 4

    def joint_torque_pd(self, names):
        return self._col(names, 0.25)

    def joint_torque_gravcomp(self, names):
        return self._col(names, 0.75)

    def contact_budget_t(self):
        # [nacon, ncollision, nefc, naconmax, njmax]
        return torch.tensor([7.0, 11.0, 23.0, 128.0, 256.0])

    def _col(self, names, v):
        return torch.full((4, len(names)), float(v))

    def joint_pos(self, names):
        return self._col(names, math.pi)               # π rad → must plot as 180 deg
    def joint_vel(self, names):
        return self._col(names, math.pi / 2)           # → 90 deg/s
    def joint_torque(self, names):
        return self._col(names, 1.5)


class _GravComp:
    """RobotSpec.gravcomp, as much of it as the log reads: which joints the robot holds against gravity."""

    def __init__(self, joints):
        self._joints = list(joints)

    def joints(self):
        return list(self._joints)


class _Spec:
    def __init__(self, gc_joints):
        self.robot = type("_Robot", (), {"gravcomp": None if gc_joints is None else _GravComp(gc_joints)})


class _Driver:
    """Stub of EnvDriver's snapshot API (see module docstring)."""

    def __init__(self, obs_width: int = len(_OBS_LABELS), obs_val=None,
                 describe_needs_step: bool = False, gc_joints=(_JOINTS[0],), capabilities=()):
        self.backend = _Backend()
        self.num_envs = self.backend.num_envs
        # engine-partial backend features, resolved once at load by api.backend.assert_backend
        self.capabilities = frozenset(capabilities)
        self.spec = _Spec(gc_joints)               # gravcomp covers ONE of the two action joints by default
        self.num_actions = len(_JOINTS)
        self.step = 0
        self.enabled = False
        self.stepped = False
        self._obs_width = obs_width
        self._obs_val = obs_val
        # mimics an obs term that is only evaluable after a step (curriculum_state on a curriculum that has
        # published nothing yet): describing at construction would blow up the sim-service at boot.
        self._describe_needs_step = describe_needs_step

    def snapshot_describe(self):
        assert self.stepped or not self._describe_needs_step, \
            "snapshot_describe called before any step ran"
        return {"task": "unit-task", "engine": "newton", "hz": 60, "dt": 1 / 60.0, "max_step": 100,
                "cards": [
                    {"group": "obs", "name": "joint_state", "labels": list(_OBS_LABELS),
                     "obs_groups": ["actor", "critic"]},
                    {"group": "obs", "name": "secret", "labels": ["s0"], "obs_groups": ["privileged"]},
                    {"group": "reward", "name": "reward", "labels": ["lift", "goal"]},
                    {"group": "action", "name": "action", "labels": list(_JOINTS)},
                    {"group": "eval", "name": "val/SR", "kind": "eval_sr", "labels": ["success", "attempts"]},
                ]}

    def snapshot_rows(self, idxs, extra=None):
        rows = {}
        for e in idxs:
            base = float(e) + self.step / 10.0
            state = {}
            for name, fn, params, scale in (extra or ()):   # the log's REAL obs terms + knobs + scale
                state[name] = [float(x) for x in (fn(self.backend, **params)[e] * scale)]
            obs = ([self._obs_val] * self._obs_width if self._obs_val is not None
                   else [base + i for i in range(self._obs_width)])
            rows[str(e)] = {"step": self.step, "max_step": 100,
                            "obs": {"joint_state": obs, "secret": [base]},
                            "reward": {"lift": base, "goal": -base},
                            "action": [base, base + 0.5], "state": state}
        return rows


def _run(log, driver, steps, *, done_at=(), success_at=()):
    """Drive ``steps`` policy steps through the log, flagging done/success on the given step indices."""
    for s in range(steps):
        driver.step = s
        driver.stepped = True                       # after_step always follows a completed env.step()
        d = torch.zeros(4, dtype=torch.bool)
        ts = torch.zeros(4, dtype=torch.bool)
        if s in done_at:
            d[:] = True
        if s in success_at:
            ts[:] = True
        log.after_step(d, ts)


def _finish(log) -> dict:
    return json.loads(Path(log.finish()).read_text())   # strict: NaN/Infinity would raise here


def _build(tmp, *, idxs=(0, 1), fps=60, **kw):
    drv = _Driver(**kw)
    return drv, PerEnvRolloutLog(drv, idxs, tmp, fps=fps)


def test_describe_is_deferred_until_a_step_has_run():
    """Terms like curriculum_state only become evaluable after a step. Resolving the channel set in the
    constructor killed the sim-service at BOOT for a contract that steps fine, taking the recording with it."""
    with tempfile.TemporaryDirectory() as tmp:
        drv, log = _build(tmp, describe_needs_step=True)   # constructing must not describe...
        _run(log, drv, 2)                                  # ...the first sample does
        assert len(_finish(log)["meta"]["tabs"]) == 9      # 4 cards + 5 joint-state, on the first sample


def test_no_steps_sampled_writes_nothing():
    """A session that died during boot must not leave an empty file that looks like a recording."""
    with tempfile.TemporaryDirectory() as tmp:
        _drv, log = _build(tmp)
        assert log.finish() == ""
        assert not list(Path(tmp).glob("*.json"))


def test_tabs_cover_the_cards_plus_the_joint_state_block():
    with tempfile.TemporaryDirectory() as tmp:
        _drv, log = _build(tmp)
        _run(log, _drv, 1)
        tabs = _finish(log)["meta"]["tabs"]
        keys = [t["key"] for t in tabs]
        assert keys[:4] == ["obs.joint_state", "obs.secret", "reward", "action"]
        assert "state.joint_pos" in keys and "state.joint_torque" in keys
        assert "val/SR" not in keys                    # a counter, not a per-step series
        by_key = {t["key"]: t for t in tabs}
        # Joint state is not a contract channel: it shares the CUSTOM section with the report's own synthetic
        # views instead of opening a heading of its own.
        assert by_key["state.joint_pos"]["section"] == "custom"
        assert by_key["state.joint_pos"]["labels"] == _JOINTS
        assert by_key["state.joint_pos"]["unit"] == "deg"
        # The obs SECTION is the privilege partition, not raw group membership: an asymmetric contract puts
        # every actor term in the critic group too, so "in the actor group" (= what the deployed policy sees)
        # is what the heading must say. The exact membership stays in the unit line.
        assert by_key["obs.joint_state"]["section"] == "actor obs"
        assert by_key["obs.joint_state"]["unit"] == "obs · actor + critic"
        assert by_key["obs.secret"]["section"] == "privileged obs"
        assert by_key["obs.secret"]["unit"] == "obs · privileged"


def test_the_torque_split_is_always_plotted():
    """Applied torque and both PRE-clamp components, unconditionally: they are part of the REQUIRED
    SimBackend surface, so a spoke that lacks one fails assert_backend at load rather than quietly logging
    three tabs on one engine and one on another."""
    with tempfile.TemporaryDirectory() as tmp:
        _drv, log = _build(tmp)
        _run(log, _drv, 1)
        keys = [t["key"] for t in _finish(log)["meta"]["tabs"]]
        for need in ("state.joint_torque", "state.joint_torque_pd", "state.joint_torque_gravcomp"):
            assert need in keys, keys


def test_contact_budget_tabs_follow_the_engine_capability():
    """The buffer-overflow panels are newton-only, so they are gated on the CAPABILITY the driver resolved —
    not on whether the read happens to exist on the backend object."""
    with tempfile.TemporaryDirectory() as tmp:
        _drv, log = _build(tmp)                                  # no capabilities → no panels
        _run(log, _drv, 1)
        keys = [t["key"] for t in _finish(log)["meta"]["tabs"]]
        assert "state.nacon" not in keys and "state.nefc" not in keys, keys
    with tempfile.TemporaryDirectory() as tmp:
        _drv, log = _build(tmp, capabilities=("contact_budget",))
        _run(log, _drv, 1)
        tabs = {t["key"]: t for t in _finish(log)["meta"]["tabs"]}
        assert tabs["state.nacon"]["labels"] == ["nacon", "ncollision", "naconmax"]
        assert tabs["state.nefc"]["labels"] == ["nefc", "njmax"]


def test_grav_torque_covers_the_gravcomp_joints_only():
    """Gravcomp is a robot fact and covers the arm only — a finger column there is a flat zero, and the tab
    goes away entirely on a contract that compensates nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        _drv, log = _build(tmp)                            # gravcomp = j0 of the two action joints
        _run(log, _drv, 1)
        by_key = {t["key"]: t for t in _finish(log)["meta"]["tabs"]}
        assert by_key["state.joint_torque_gravcomp"]["labels"] == [_JOINTS[0]]
        assert by_key["state.joint_torque_pd"]["labels"] == _JOINTS, "PD still spans the whole action set"
    with tempfile.TemporaryDirectory() as tmp:
        _drv, log = _build(tmp, gc_joints=None)            # no gravcomp contract
        _run(log, _drv, 1)
        keys = [t["key"] for t in _finish(log)["meta"]["tabs"]]
        assert "state.joint_torque_gravcomp" not in keys and "state.joint_torque_pd" in keys


def test_joint_state_channels_are_in_display_units():
    """π rad → 180 deg: the scale the log declares in its unit string is the one it applies."""
    with tempfile.TemporaryDirectory() as tmp:
        drv, log = _build(tmp)
        _run(log, drv, 2)
        s = _finish(log)["data"]["0"]["series"]
        assert s["state.joint_pos"][0] == [180.0, 180.0]
        assert s["state.joint_vel"][0] == [90.0, 90.0]
        assert s["state.joint_torque"][0] == [1.5, 1.5]      # N·m, unscaled
        assert s["state.joint_torque_pd"][0] == [0.25, 0.25]


def test_series_are_per_dim_over_time():
    """series[key][dim][step] — the shape the report plots directly."""
    with tempfile.TemporaryDirectory() as tmp:
        drv, log = _build(tmp)
        _run(log, drv, 5)
        out = _finish(log)
        env0 = out["data"]["0"]
        assert env0["step"] == [0, 1, 2, 3, 4]
        js = env0["series"]["obs.joint_state"]
        assert len(js) == len(_OBS_LABELS) and all(len(s) == 5 for s in js)
        assert js[0] == [0.0, 0.1, 0.2, 0.3, 0.4]      # dim 0 over time, env 0
        assert env0["series"]["reward"][0] == [0.0, 0.1, 0.2, 0.3, 0.4]     # 'lift', in label order
        assert env0["series"]["reward"][1] == [0.0, -0.1, -0.2, -0.3, -0.4]  # 'goal'
        assert env0["series"]["action"][1] == [0.5, 0.6, 0.7, 0.8, 0.9]
        # env 1's rows are its own (base = env id + step/10), not env 0's
        assert out["data"]["1"]["series"]["obs.joint_state"][0] == [1.0, 1.1, 1.2, 1.3, 1.4]


def test_success_latches_only_on_a_done_step():
    """The clip tag means "reached the goal at a termination", matching the recorder's own latch."""
    with tempfile.TemporaryDirectory() as tmp:
        drv, log = _build(tmp)
        _run(log, drv, 4, done_at=(), success_at=(1,))       # success without a done → no latch
        assert _finish(log)["meta"]["envs"][0]["success"] is False
    with tempfile.TemporaryDirectory() as tmp:
        drv, log = _build(tmp)
        _run(log, drv, 4, done_at=(2,), success_at=(2,))
        meta = _finish(log)["meta"]
        assert meta["envs"][0]["success"] is True
        assert _finish(log)["data"]["0"]["done"] == [0, 0, 1, 0]


def test_meta_carries_the_series_axis_and_env_tabs():
    """`fps` is the policy rate — the page's only way to turn a series index into a time (t = index / fps).
    The env list is the report's top-level tabs, in the order the log was given."""
    with tempfile.TemporaryDirectory() as tmp:
        drv, log = _build(tmp, idxs=(0, 2), fps=120)
        _run(log, drv, 3)
        meta = _finish(log)["meta"]
        assert meta["fps"] == 120
        assert [e["id"] for e in meta["envs"]] == [0, 2]
        assert meta["envs"][1]["steps"] == 3
        assert (meta["task"], meta["engine"], meta["hz"]) == ("unit-task", "newton", 60)


def test_non_finite_values_stay_strict_json():
    """A diverged step must become null (a gap), not NaN — JSON.parse rejects NaN and the page would die."""
    with tempfile.TemporaryDirectory() as tmp:
        drv, log = _build(tmp, obs_val=float("nan"))
        _run(log, drv, 2)
        out = _finish(log)                                   # would raise on NaN/Infinity
        assert out["data"]["0"]["series"]["obs.joint_state"][0] == [None, None]


def test_label_width_mismatch_fails_loud():
    """A contract whose card labels do not match its term width would silently mislabel every dim."""
    with tempfile.TemporaryDirectory() as tmp:
        drv, log = _build(tmp, obs_width=len(_OBS_LABELS) + 1)
        _run(log, drv, 1)
        try:
            log.finish()
        except AssertionError as e:
            assert "labels" in str(e)
        else:
            raise AssertionError("expected an AssertionError on a labels/width mismatch")


# --- the HTML report built from that log (dashboard/rollout_report.build_report) --------------------
# Only its Python half is checked here — the .rrd wiring, self-containment, fail-loud. The page's sync
# behaviour is browser behaviour and was verified against a real recording in Chrome (playhead <-> series
# index in both directions, env/channel tab switching, paused scrubbing).

def _recording(tmp, *, steps=4, idxs=(0, 1), rrds=("rollout.rrd",)):
    """A real recording dir: this log's own data.json beside the .rrd the sim-service writes."""
    drv, log = _build(tmp, idxs=idxs)
    _run(log, drv, steps)
    log.finish()
    for name in rrds:
        Path(tmp, name).write_bytes(b"\x00")
    return tmp


def _payload(html: str) -> dict:
    """The inlined series, back out of the page."""
    m = re.search(r"const R=(\{.*?\}), M=R\.meta", html, re.S)
    assert m, "page does not inline its payload as `const R={...}, M=R.meta`"
    return json.loads(m.group(1).replace("\\u003c", "<"))


def test_report_points_the_3d_pane_at_the_rrd_by_basename():
    """The 3D pane fetches the recording RELATIVE to the page, so the whole directory can be uploaded
    anywhere and the link still resolves. A bare basename is the only form that survives that."""
    with tempfile.TemporaryDirectory() as tmp:
        out = build_report(_recording(tmp))
        assert out == str(Path(tmp, "report.html"))
        rrd = _payload(Path(out).read_text())["meta"]["rrd"]
        assert rrd == "rollout.rrd"
        assert "/" not in rrd and "\\" not in rrd


def test_report_without_a_recording_still_plots():
    """A rollout log with no .rrd beside it is a real case (RRD=0, or a rebuild from data.json alone): the
    pane must degrade to a message instead of failing the report, since the plots are still the point."""
    with tempfile.TemporaryDirectory() as tmp:
        out = build_report(_recording(tmp, steps=3, rrds=()))
        p = _payload(Path(out).read_text())
        assert p["meta"]["rrd"] == ""
        assert len(p["data"]["0"]["step"]) == 3


def test_report_is_self_contained_apart_from_the_pinned_viewer():
    """One file wherever it is opened: the series are INLINED (no data fetch), and the only external host
    is the version-pinned rerun viewer the 3D pane loads."""
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(build_report(_recording(tmp, steps=3))).read_text()
        p = _payload(html)                                  # inlined, and strict JSON
        assert len(p["data"]["0"]["step"]) == 3
        assert "data.json" not in html                      # the series ride in the page, not a sibling fetch
        hosts = set(re.findall(r"https?://([^/\s\"']+)", html))
        assert hosts == {"app.rerun.io"}, hosts
        # the viewer is PINNED: an .rrd only stays readable across adjacent rerun minors
        assert re.search(r"app\.rerun\.io/version/\d+\.\d+\.\d+", html), "the viewer URL must pin a version"


def test_report_title_defaults_to_task_and_engine():
    with tempfile.TemporaryDirectory() as tmp:
        d = _recording(tmp)
        assert "<title>unit-task · newton · eval rollout</title>" in Path(build_report(d)).read_text()
        out = build_report(d, str(Path(tmp, "named.html")), "model_500 · my-run")
        assert "<title>model_500 · my-run</title>" in Path(out).read_text()


def test_report_fails_loud_without_the_rollout_log():
    """The series ARE the report — no data.json means there is nothing to build, so say so rather than
    emitting an empty page. (A missing .rrd is the opposite case: deliberately survivable, see above.)"""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            build_report(tmp)
        except AssertionError as e:
            assert "data.json" in str(e)
        else:
            raise AssertionError("expected an AssertionError when the rollout log is absent")


# --- the report's CUSTOM section (rollout_report._add_custom_section / _drop_state_channels) ------
# Synthesized at BUILD time from the series the log already wrote, so these go straight at build_report with a
# hand-written data.json — that is the real path for a rebuild of an existing recording.

def _custom_payload() -> dict:
    """A recording whose obs tabs are an overlay's two sources. The position card lists its joints in the
    OPPOSITE order, which is what catches picking by position instead of by label."""
    return {
        "meta": {"task": "t", "engine": "newton", "hz": 60, "dt": 1 / 60.0, "max_step": 9, "fps": 60,
                 "envs": [{"id": 0, "success": True, "steps": 2}],
                 "tabs": [
                     {"key": "obs.prev_action_targets", "title": "prev_action_targets", "unit": "obs · actor",
                      "labels": ["j0", "j1"], "section": "actor obs"},
                     {"key": "obs.arm_joint_pos_clean", "title": "arm_joint_pos_clean", "unit": "obs · critic",
                      "labels": ["j1", "j0"], "section": "critic obs"},
                     {"key": "state.joint_pos", "title": "Joint Position", "unit": "deg",
                      "labels": ["j0", "j1"], "section": "joint state"},
                 ]},
        "data": {"0": {"step": [1, 2], "done": [0, 1], "success": [0, 1], "series": {
            "obs.prev_action_targets": [[10.0, 11.0], [20.0, 21.0]],        # j0, j1
            "obs.arm_joint_pos_clean": [[200.0, 201.0], [100.0, 101.0]],    # j1, j0  <- reversed
            "state.joint_pos": [[1.0, 2.0], [3.0, 4.0]],
        }}},
    }


def _built(payload: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "data.json").write_text(json.dumps(payload))
        return _payload(Path(build_report(tmp)).read_text())


def test_report_builds_the_overlay_from_the_stored_series_by_label():
    """target vs pos rides on the obs series already in the log — no new sampling, and paired BY JOINT NAME."""
    p = _built(_custom_payload())
    ov = {t["key"]: t for t in p["meta"]["tabs"]}["custom.target_vs_pos_arm"]
    assert ov["section"] == "custom", "it lands in the CUSTOM section of the channel picker"
    assert ov["line"] is False and ov["marker"] is True, \
        "SCATTER only: a line through the samples draws motion that never happened"
    assert ov["labels"] == ["j0 tgt", "j0 pos", "j1 tgt", "j1 pos"], "joints in the TARGET's order"
    s = p["data"]["0"]["series"]["custom.target_vs_pos_arm"]
    assert s[0] == [10.0, 11.0], "j0 tgt"
    assert s[1] == [100.0, 101.0], "j0 pos = index 1 of the position card, not index 0"
    assert ov["rows"][0] == {"joint": "j0", "items": [[0, "j0 tgt"], [1, "j0 pos"]]}, \
        "the page colours a pair by joint off `rows`"


def test_report_files_a_legacy_joint_state_block_under_custom():
    """A recording made while joint state had a section of its own is re-filed, not dropped: those are backend
    measurements a rebuild cannot recreate, and two headings for the same kind of thing is the bug."""
    p = _built(_custom_payload())
    st = [t for t in p["meta"]["tabs"] if t["key"] == "state.joint_pos"]
    assert st and st[0]["section"] == "custom"
    assert p["data"]["0"]["series"]["state.joint_pos"] == [[1.0, 2.0], [3.0, 4.0]], "series kept as recorded"


def test_report_offers_val_sr_as_a_counter_not_a_series():
    p = _built(_custom_payload())
    sr = {t["key"]: t for t in p["meta"]["tabs"]}["eval.val/SR"]
    assert sr["kind"] == "eval_sr" and sr["section"] == "custom"
    assert "eval.val/SR" not in p["data"]["0"]["series"], \
        "the page counts it off done/success — a duplicated series could disagree with the env tabs' ✓/✗"


def _main() -> int:
    if torch is None:
        print("SKIP: torch not installed")
        return 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
