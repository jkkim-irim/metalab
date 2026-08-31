"""Unit tests for the S3 -> local dataset sync guard (learning/data/s3_sync.py).

Exercises the PRODUCTION ``ensure_local_dataset`` with an injected runner (no AWS): the COMPLETE
marker must make a completed sync skip, and must never be left behind by a partial one — a dataset
that was only half-copied has to re-sync rather than silently train on missing episodes.
"""
import pytest

from learning.data.s3_sync import MARKER, default_root, ensure_local_dataset, marker_matches

URI = "s3://bkt/datasets/ds_a"


class _Runner:
    """Records aws-cli invocations; optionally fails to simulate an interrupted sync."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, cmd, check=False):
        self.calls.append(cmd)
        if self.fail:
            raise RuntimeError("sync interrupted")
        return None


def test_syncs_then_writes_marker(tmp_path):
    r = _Runner()
    root = ensure_local_dataset(URI, str(tmp_path / "ds"), runner=r)
    assert len(r.calls) == 1
    cmd = r.calls[0]
    assert cmd[:3] == ["aws", "s3", "sync"] and cmd[3] == URI + "/" and cmd[4] == root
    assert marker_matches(root, URI)
    assert (tmp_path / "ds" / MARKER).read_text().splitlines()[0] == URI


def test_second_call_skips_when_marker_matches(tmp_path):
    r = _Runner()
    root = ensure_local_dataset(URI, str(tmp_path / "ds"), runner=r)
    ensure_local_dataset(URI, root, runner=r)          # same URI + COMPLETE present -> no re-sync
    assert len(r.calls) == 1


def test_partial_sync_leaves_no_marker_and_resyncs(tmp_path):
    root = str(tmp_path / "ds")
    failing = _Runner(fail=True)
    with pytest.raises(RuntimeError):
        ensure_local_dataset(URI, root, runner=failing)
    assert not marker_matches(root, URI)               # never mark an interrupted sync complete
    ok = _Runner()
    ensure_local_dataset(URI, root, runner=ok)         # the retry must actually re-sync
    assert len(ok.calls) == 1
    assert marker_matches(root, URI)


def test_different_uri_into_same_root_resyncs(tmp_path):
    root = str(tmp_path / "ds")
    r = _Runner()
    ensure_local_dataset(URI, root, runner=r)
    ensure_local_dataset("s3://bkt/datasets/ds_b", root, runner=r)   # stale cache for another dataset
    assert len(r.calls) == 2
    assert marker_matches(root, "s3://bkt/datasets/ds_b")
    assert not marker_matches(root, URI)


def test_rejects_non_s3_uri(tmp_path):
    with pytest.raises(ValueError):
        ensure_local_dataset("/local/path", str(tmp_path / "ds"), runner=_Runner())


def test_default_root_uses_cache_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLEX_DATASET_CACHE", str(tmp_path))
    assert default_root("s3://bkt/datasets/ds_a/") == str(tmp_path / "ds_a")
