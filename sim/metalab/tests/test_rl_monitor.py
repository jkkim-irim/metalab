"""RL monitor adapter — env_driver snapshots re-shaped into the standalone dashboard schema.

Exercises ``dashboard/rl_monitor.py`` against the SHAPES env_driver actually produces
(``snapshot_describe()`` / ``snapshot_rows()``), so the fixtures below are copies of those contracts rather
than of the adapter's logic. Pure dict -> dict, so no torch / engine / GPU is needed.

Runs under pytest, or directly: ``python3 sim/metalab/tests/test_rl_monitor.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sim.metalab.dashboard import rl_monitor  # noqa: E402

# --- fixtures: exactly what env_driver.snapshot_describe() / snapshot_rows() return ------------------
DESCRIBE = {
    "task": "hammer_lift_student", "engine": "newton", "num_envs": 4,
    "hz": 100, "dt": 0.01, "episode_length_s": 10.0, "max_step": 600,
    "cards": [
        {"group": "obs", "name": "joint_pos", "labels": ["a", "b"], "obs_groups": ["actor", "critic"]},
        {"group": "obs", "name": "joint_vel", "labels": ["a", "b"], "obs_groups": ["critic"]},
        {"group": "reward", "name": "reward", "labels": ["prox", "contact"]},
        {"group": "action", "name": "action", "labels": ["j0", "j1", "j2"]},
        {"group": "eval", "name": "val/SR", "kind": "eval_sr", "labels": ["success", "attempts"]},
    ],
    "goal": {"pos": [0.6, -0.1, 1.0], "quat": [1.0, 0.0, 0.0, 0.0], "goal_dist_tol": 0.01},
}

ROWS = {
    "0": {"step": 7, "max_step": 600,
          "obs": {"joint_pos": [0.1, 0.2], "joint_vel": [1.0, 2.0]},
          "reward": {"prox": 0.5, "contact": 0.25}, "action": [1.0, 2.0, 3.0], "state": {}},
    "2": {"step": 3, "max_step": 600,
          "obs": {"joint_pos": [0.3, 0.4], "joint_vel": [3.0, 4.0]},
          "reward": {"prox": 0.1, "contact": 0.0}, "action": [4.0, 5.0, 6.0], "state": {}},
}


def test_channels_are_the_standalone_shape():
    """One channel per card, with the keys drive/monitor.describe emits."""
    chs = rl_monitor.channels(DESCRIBE)
    assert [c["key"] for c in chs] == ["obs.joint_pos", "obs.joint_vel", "reward.reward", "action.action",
                                       "eval.val/SR"]
    for c in chs:
        assert set(c) >= {"key", "title", "unit", "labels", "digits", "section"}, f"missing field: {c}"
        assert isinstance(c["digits"], int) and isinstance(c["unit"], str)
    assert chs[0]["title"] == "joint_pos" and chs[0]["labels"] == ["a", "b"]
    assert chs[0]["obs_groups"] == ["actor", "critic"]
    assert "obs_groups" not in chs[2], "only obs terms belong to obs groups"


def test_sections_order_the_tab_row():
    """actor -> critic -> reward -> action -> custom, whatever order the contract declared them in."""
    shuffled = {"cards": list(reversed(DESCRIBE["cards"]))}
    chs = rl_monitor.channels(shuffled)
    assert [c["section"] for c in chs] == ["actor obs", "critic obs", "reward", "action", "custom"]
    assert [c["key"] for c in chs][:4] == ["obs.joint_pos", "obs.joint_vel", "reward.reward", "action.action"]


def test_table_kinds_are_reachable_under_custom():
    """val/SR is a per-env counter TABLE, not a series. It still needs a combo entry to be reachable, and
    the page switches on `kind` rather than plotting it."""
    sr = next(c for c in rl_monitor.channels(DESCRIBE) if c["key"] == "eval.val/SR")
    assert sr["kind"] == "eval_sr", "the page keys its table rendering on this"
    assert sr["section"] == "custom"
    assert sr["labels"] == ["success", "attempts"]
    plots = [c for c in rl_monitor.channels(DESCRIBE) if not c.get("kind")]
    assert "eval.val/SR" not in [c["key"] for c in plots]


def test_a_term_in_both_groups_is_actor_visible():
    """`joint_pos` is in actor AND critic — it is an ACTOR input, so it must not land under privileged."""
    by_key = {c["key"]: c for c in rl_monitor.channels(DESCRIBE)}
    assert by_key["obs.joint_pos"]["section"] == "actor obs"
    assert by_key["obs.joint_vel"]["section"] == "critic obs", "critic-only == privileged"


def test_unit_and_digits_ride_the_card():
    chs = {c["key"]: c for c in rl_monitor.channels(
        {"cards": [{"group": "obs", "name": "t", "labels": ["a"], "unit": "N*m", "digits": 1,
                    "obs_groups": ["actor"]}]})}
    assert chs["obs.t"]["unit"] == "N*m" and chs["obs.t"]["digits"] == 1


# --- overlay view cards (rl_monitor.OVERLAYS) --------------------------------------------------------
OV_DESCRIBE = {
    "cards": [
        {"group": "obs", "name": "prev_action_targets", "labels": ["j0", "j1"], "obs_groups": ["actor"]},
        {"group": "obs", "name": "joint_pos", "labels": ["j1", "j0", "j2"], "obs_groups": ["actor"]},
    ],
}
OV_ROWS = {"0": {"step": 1, "max_step": 9, "obs": {"prev_action_targets": [10.0, 11.0],
                                                  "joint_pos": [1.1, 1.0, 1.2]}}}


def test_overlay_pairs_the_two_sources_per_joint():
    ov = rl_monitor.overlay_plan(OV_DESCRIBE)[0]
    assert ov["labels"] == ["j0 tgt", "j0 pos", "j1 tgt", "j1 pos"], \
        "joints follow the FIRST source's order (a target exists only for a commanded joint)"
    assert [r["joint"] for r in ov["rows"]] == ["j0", "j1"]
    assert ov["rows"][0]["items"] == [[0, "j0 tgt"], [1, "j0 pos"]], "one pane per joint holds its pair"
    assert ov["unit"] == "rad" and ov["marker"] is True


def test_overlay_values_come_from_the_right_source_index():
    """joint_pos lists the joints in a DIFFERENT order — the plan must index by name, not by position."""
    ov = rl_monitor.overlay_plan(OV_DESCRIBE)
    ch = rl_monitor.snapshot(OV_ROWS, overlays=ov)["envs"]["0"]["custom.target_vs_pos"]
    assert ch == [10.0, 1.0, 11.0, 1.1], f"got {ch}"    # j0: tgt 10.0 / pos 1.0 (index 1 of joint_pos)


def test_overlay_is_a_channel_but_its_value_plan_is_not_published():
    ch = [c for c in rl_monitor.channels(OV_DESCRIBE) if c["section"] == "custom"]
    assert len(ch) == 1 and ch[0]["key"] == "custom.target_vs_pos"
    assert "pick" not in ch[0], "the source-index plan is snapshot's business, not /describe's"
    assert "rows" in ch[0], "the page groups panes by joint off `rows`"


def test_overlay_drops_when_a_source_is_absent_or_disjoint():
    assert rl_monitor.overlay_plan({"cards": [OV_DESCRIBE["cards"][0]]}) == [], "missing source -> no channel"
    disjoint = {"cards": [{"group": "obs", "name": "prev_action_targets", "labels": ["a"]},
                          {"group": "obs", "name": "joint_pos", "labels": ["b"]}]}
    assert rl_monitor.overlay_plan(disjoint) == [], "no shared joint -> no channel"


def test_index_labelled_source_drops_the_overlay():
    """The bug this caught in review: `prev_action_targets` takes no joint-list knob, so the card fell back to
    "0","1",... while joint_pos carried names — no shared label, overlay silently gone, no `custom` section.
    The adapter is right to drop it; env_driver._ACTION_VECTOR_TERMS is what supplies the names."""
    indexed = {"cards": [
        {"group": "obs", "name": "prev_action_targets", "labels": ["0", "1"]},
        {"group": "obs", "name": "joint_pos", "labels": ["j0", "j1"]},
    ]}
    assert rl_monitor.overlay_plan(indexed) == []
    assert not [c for c in rl_monitor.channels(indexed) if c["section"] == "custom"]


def test_overlay_width_matches_its_labels():
    """Same invariant as every other channel: values plotted == labels declared."""
    ov = rl_monitor.overlay_plan(OV_DESCRIBE)
    ch = rl_monitor.snapshot(OV_ROWS, overlays=ov)["envs"]["0"]
    assert len(ch["custom.target_vs_pos"]) == len(ov[0]["labels"])


def test_snapshot_without_overlays_emits_none():
    assert "custom.target_vs_pos" not in rl_monitor.snapshot(OV_ROWS)["envs"]["0"]


def test_describe_hides_the_standalone_only_subtabs():
    d = rl_monitor.describe(DESCRIBE)
    assert d["groups"] == [] and d["controls"] == [], \
        "RL has no trajectory playback and no joint take-over — the page keys those sub-tabs on emptiness"
    assert d["engine"] == "newton" and d["task"] == "hammer_lift_student"
    assert d["num_envs"] == 4 and d["max_step"] == 600      # the env selector needs num_envs
    assert d["control_hz"] == 100


def test_snapshot_is_one_channel_map_per_env():
    s = rl_monitor.snapshot(ROWS, t=1.25)
    assert set(s["envs"]) == {"0", "2"}, "envs are keyed by GLOBAL env id, not by position"
    e0 = s["envs"]["0"]
    assert e0["obs.joint_pos"] == [0.1, 0.2]
    assert e0["action.action"] == [1.0, 2.0, 3.0]
    assert e0["reward.reward"] == [0.5, 0.25], "a reward dict flattens in term order"
    assert s["envs"]["2"]["obs.joint_vel"] == [3.0, 4.0]
    assert s["t"] == 1.25 and s["max_step"] == 600


def test_every_channel_is_present_for_every_env():
    """The page buffers all SERIES tabs at once, so a snapshot may not omit one for an env. Table kinds carry
    no per-env series (val/SR rides the payload whole), so they are excluded."""
    keys = {c["key"] for c in rl_monitor.channels(DESCRIBE) if not c.get("kind")}
    for env_id, ch in rl_monitor.snapshot(ROWS)["envs"].items():
        assert set(ch) == keys, f"env {env_id}: {keys ^ set(ch)} missing/extra"


def test_value_count_matches_the_declared_labels():
    """Same invariant drive/monitor.sample asserts: dims plotted == labels declared."""
    width = {c["key"]: len(c["labels"]) for c in rl_monitor.channels(DESCRIBE) if not c.get("kind")}
    for env_id, ch in rl_monitor.snapshot(ROWS)["envs"].items():
        for key, vals in ch.items():
            assert len(vals) == width[key], f"env {env_id} {key}: {len(vals)} values vs {width[key]} labels"


def test_reward_labels_give_the_flatten_order():
    assert rl_monitor.reward_labels(DESCRIBE) == ["prox", "contact"]
    assert rl_monitor.reward_labels({"cards": []}) == []


def test_empty_and_partial_rows_do_not_invent_channels():
    assert rl_monitor.snapshot({})["envs"] == {}
    bare = {"9": {"step": 0, "max_step": 600, "obs": {"joint_pos": [0.0, 0.0]}}}
    ch = rl_monitor.snapshot(bare)["envs"]["9"]
    assert set(ch) == {"obs.joint_pos"}, "no reward/action published -> no reward/action channel"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
