"""Unit tests for the pure-logic sim-eval helpers (``learning/eval/sim_eval_helpers.py``).

Torch-free / sim-free: exercises the REAL ``replan_cap`` + ``video_filename`` that
``learning/eval/policies/groot.py`` imports and calls, so they run locally (off the node) without the
GR00T / imageio / wandb stack. ``policies/groot.py`` itself can't be imported here (it pulls torch/gr00t
at module top), which is exactly why the pure logic lives in its own module.

Run:  python -m pytest learning/eval -q
"""
import pytest

from learning.eval.sim_eval_helpers import (
    aggregate_sim_metrics,
    parse_tasks,
    replan_cap,
    resolve_suite,
    safe_name,
    video_filename,
)


def _rollout_chunk_lengths(chunk_len: int, replan_steps: int, horizon: int) -> list[int]:
    """Mirror the groot runner's inner loop: for each predicted chunk we execute ``replan_cap`` steps, then
    re-observe / re-predict, until the horizon. Returns the per-chunk executed-step counts — this is
    the exact production semantics (re-predict cadence) the ``--replan-steps`` flag controls."""
    per_chunk, t = [], 0
    while t < horizon:
        executed = 0
        for _ in range(replan_cap(chunk_len, replan_steps)):
            if t >= horizon:
                break
            t += 1
            executed += 1
        per_chunk.append(executed)
    return per_chunk


def test_replan_cap_full_chunk_is_default():
    # replan_steps == 0 -> execute the whole chunk (original behaviour), never re-predicting mid-chunk.
    assert replan_cap(8, 0) == 8
    assert replan_cap(16, 0) == 16


def test_replan_cap_partial_chunk():
    # The reference LIBERO eval used 5: execute 5 of an 8-step chunk, then re-observe.
    assert replan_cap(8, 5) == 5
    assert replan_cap(16, 5) == 5


def test_replan_cap_never_exceeds_chunk_len():
    assert replan_cap(8, 20) == 8   # asking for more than the chunk has -> capped at the chunk length
    assert replan_cap(8, 8) == 8


def test_replan_cap_rejects_bad_args():
    with pytest.raises(AssertionError):
        replan_cap(0, 5)            # empty chunk
    with pytest.raises(AssertionError):
        replan_cap(8, -1)           # negative replan window


def test_replan_cadence_matches_flag():
    # chunk_len=8: replan_steps=5 executes 5 then re-predicts; replan_steps=0 executes the full 8.
    assert _rollout_chunk_lengths(8, 5, horizon=20) == [5, 5, 5, 5]
    assert _rollout_chunk_lengths(8, 0, horizon=16) == [8, 8]
    # a partial replan re-predicts more often over the same horizon than the full-chunk default.
    assert len(_rollout_chunk_lengths(8, 5, 40)) > len(_rollout_chunk_lengths(8, 0, 40))


def test_video_filename():
    assert video_filename(0, True) == "ep0_success.mp4"
    assert video_filename(3, False) == "ep3_fail.mp4"
    assert video_filename(12, success=True) == "ep12_success.mp4"


def test_resolve_suite_maps_family_and_task_kind():
    # 'mikasa' -> ManiSkill server + env-id tasks; any 'libero*' suite -> LIBERO server + int task-ids.
    assert resolve_suite("mikasa") == ("mikasa", "envid")
    assert resolve_suite("libero_90") == ("libero", "int")
    assert resolve_suite("libero_spatial") == ("libero", "int")


def test_resolve_suite_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_suite("rlbench")        # not a known family -> fail loud, don't guess a server


def test_parse_tasks_int_vs_envid():
    # LIBERO: integer task-ids, whitespace trimmed; single or set.
    assert parse_tasks("int", "11") == [11]
    assert parse_tasks("int", "0, 11 ,42") == [0, 11, 42]
    # MIKASA: env-id strings, kept verbatim.
    assert parse_tasks("envid", "ShellGameTouch-VLA-v0") == ["ShellGameTouch-VLA-v0"]
    assert parse_tasks("envid", "A-VLA-v0,B-VLA-v0") == ["A-VLA-v0", "B-VLA-v0"]


def test_parse_tasks_rejects_empty():
    with pytest.raises(ValueError):
        parse_tasks("int", "")
    with pytest.raises(ValueError):
        parse_tasks("int", " , ")       # only separators/whitespace -> empty


def test_parse_tasks_int_rejects_nonint():
    with pytest.raises(ValueError):
        parse_tasks("int", "0,foo")     # a LIBERO task-id must be an int


def test_safe_name_namespaces_video_dirs():
    # env_id -> filesystem-safe token so a multi-task set's ep{idx} videos never collide.
    assert safe_name("libero_90:11") == "libero_90_11"
    assert safe_name("ShellGameTouch-VLA-v0") == "ShellGameTouch-VLA-v0"


def test_aggregate_sim_metrics_micro_averages_over_episodes():
    # The in-training sim-eval hook aggregates the per-task run_task results into the sim_eval/ dict
    # logged to wandb. SR is the MICRO-average (successes/episodes over the whole set), so uneven
    # per-task episode counts are episode-weighted — NOT a mean of per-task SRs.
    results = [
        {"env_id": "libero_90:0", "sr": 0.25, "succ": 1, "total": 4},   # per-task SR 0.25
        {"env_id": "libero_90:11", "sr": 1.0, "succ": 6, "total": 6},   # per-task SR 1.0
    ]
    m = aggregate_sim_metrics(results)
    assert m["sim_eval/SR"] == (1 + 6) / (4 + 6)          # 0.7 micro-avg, not mean(0.25,1.0)=0.625
    assert m["sim_eval/n_success"] == 7
    assert m["sim_eval/n_episodes"] == 10
    assert m["sim_eval/n_tasks"] == 2
    # per-task SR keyed by env_id so a task set logs one each without collision
    assert m["sim_eval/libero_90:0/SR"] == 0.25
    assert m["sim_eval/libero_90:11/SR"] == 1.0


def test_aggregate_sim_metrics_fails_loud_on_nothing_evaluated():
    with pytest.raises(AssertionError):
        aggregate_sim_metrics([])                                       # no tasks at all
    with pytest.raises(AssertionError):
        aggregate_sim_metrics([{"env_id": "x", "sr": 0.0, "succ": 0, "total": 0}])  # zero episodes
