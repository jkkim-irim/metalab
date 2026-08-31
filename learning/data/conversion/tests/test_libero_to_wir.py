"""Unit tests for ``libero_to_wir.select_all_tasks`` — the full-suite (multi-task) registry builder.

Exercises the PRODUCTION function the ``--all-tasks`` conversion uses: keep every episode, map each to
its own NL task string, and assign distinct tasks contiguous indices in first-appearance order.
"""
import pytest

from learning.data.conversion.libero_to_wir import select_all_tasks


def _eps(pairs):
    """episodes.jsonl-shaped dicts from (episode_index, task_string) pairs."""
    return [{"episode_index": ei, "tasks": [t], "length": 1} for ei, t in pairs]


def test_all_tasks_builds_first_appearance_registry():
    episodes = _eps([(0, "A"), (1, "B"), (2, "A"), (3, "C")])
    selected, per_ep, ordered = select_all_tasks(episodes, max_episodes=0)
    assert selected == [0, 1, 2, 3]                    # every episode, in episode_index order
    assert ordered == ["A", "B", "C"]                  # distinct tasks, first-appearance order
    assert per_ep == {0: "A", 1: "B", 2: "A", 3: "C"}
    task_to_index = {t: i for i, t in enumerate(ordered)}
    assert [task_to_index[per_ep[e]] for e in selected] == [0, 1, 0, 2]


def test_all_tasks_cap_limits_episodes_and_registry():
    episodes = _eps([(0, "A"), (1, "B"), (2, "C")])
    selected, per_ep, ordered = select_all_tasks(episodes, max_episodes=2)
    assert selected == [0, 1]                          # first N by episode_index
    assert ordered == ["A", "B"]                       # capped-away task C absent from the registry
    assert set(per_ep) == {0, 1}


def test_all_tasks_requires_exactly_one_task_per_episode():
    with pytest.raises(ValueError, match="exactly 1"):
        select_all_tasks([{"episode_index": 0, "tasks": ["A", "B"]}], max_episodes=0)
    with pytest.raises(ValueError, match="exactly 1"):
        select_all_tasks([{"episode_index": 0, "tasks": []}], max_episodes=0)
