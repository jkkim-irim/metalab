"""Unit tests for ``mikasa_to_wir.list_task_names`` — the ``--all-tasks`` batch task enumerator.

Exercises the PRODUCTION function the full-suite conversion uses to discover the per-task datasets
under a parent directory: return every ``<task>/`` that is itself a LeRobot dataset (has
``meta/info.json``), sorted, and fail loud when there are none.
"""
import pytest

from learning.data.conversion.mikasa_to_wir import list_task_names


def _make_task(parent, name, *, with_dataset=True):
    """Create ``parent/<name>/`` — a LeRobot-shaped task dir (has meta/info.json) unless disabled."""
    task = parent / name
    if with_dataset:
        (task / "meta").mkdir(parents=True)
        (task / "meta" / "info.json").write_text("{}")
    else:
        task.mkdir(parents=True)  # a non-dataset subdir that must be ignored
    return task


def test_lists_only_dataset_subdirs_sorted(tmp_path):
    _make_task(tmp_path, "shell_game_touch_vla_v0")
    _make_task(tmp_path, "another_task_vla_v0")
    _make_task(tmp_path, "not_a_dataset", with_dataset=False)  # no meta/info.json -> ignored
    (tmp_path / "loose_file.txt").write_text("x")              # a stray file -> ignored

    assert list_task_names(str(tmp_path)) == ["another_task_vla_v0", "shell_game_touch_vla_v0"]


def test_fails_loud_when_no_task_datasets(tmp_path):
    _make_task(tmp_path, "not_a_dataset", with_dataset=False)
    with pytest.raises(FileNotFoundError, match="no MIKASA task datasets"):
        list_task_names(str(tmp_path))
