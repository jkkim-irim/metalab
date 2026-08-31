"""S3 → local dataset sync, guarded by a ``COMPLETE`` marker.

Datasets are always sourced from S3 (``DatasetConfig.s3_uri``); ``make_dataset`` calls
``ensure_local_dataset`` to mirror the URI into a local cache before loading. A ``COMPLETE`` marker
is written only at the end of a full ``aws s3 sync`` and records which URI the cache holds — so a
re-run whose marker matches skips the sync, and a partial/interrupted sync (missing or stale marker)
re-syncs. Kept torch-free so this guard logic is unit-testable without the training stack.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

MARKER = "COMPLETE"


def default_root(s3_uri: str) -> str:
    """Local cache dir for an ``s3://`` dataset: ``$ALLEX_DATASET_CACHE/<basename>`` (cache root
    defaults to ``/opt/dlami/nvme/data``, the training-node convention)."""
    base = s3_uri.rstrip("/").rsplit("/", 1)[-1]
    if not base:
        raise ValueError(f"cannot derive a local dir from s3_uri {s3_uri!r}")
    cache = os.environ.get("ALLEX_DATASET_CACHE", "/opt/dlami/nvme/data")
    return str(Path(cache) / base)


def marker_matches(root: str | Path, s3_uri: str) -> bool:
    """True iff ``<root>/COMPLETE`` exists and records this exact ``s3_uri`` (first line)."""
    marker = Path(root) / MARKER
    return marker.is_file() and marker.read_text().splitlines()[0].strip() == s3_uri


def ensure_local_dataset(s3_uri: str, root: str | None = None, *, runner=subprocess.run) -> str:
    """Mirror ``s3_uri`` to a local ``root`` (derived from the basename if None); return the root.

    Idempotent + partial-safe: skip iff the ``COMPLETE`` marker already records this exact ``s3_uri``;
    otherwise invalidate the marker, run ``aws s3 sync``, and only then (re)write it — so a crash
    mid-sync leaves no ``COMPLETE`` and the next run re-syncs. ``runner`` is the subprocess runner
    (injectable for tests).
    """
    if not (isinstance(s3_uri, str) and s3_uri.startswith("s3://")):
        raise ValueError(f"s3_uri must be an s3:// URI, got {s3_uri!r}")
    root = root or default_root(s3_uri)
    root_p = Path(root)
    if marker_matches(root_p, s3_uri):
        return root
    root_p.mkdir(parents=True, exist_ok=True)
    (root_p / MARKER).unlink(missing_ok=True)   # no COMPLETE while a (re)sync is in flight
    runner(
        ["aws", "s3", "sync", s3_uri.rstrip("/") + "/", str(root_p), "--only-show-errors"],
        check=True,
    )
    (root_p / MARKER).write_text(
        f"{s3_uri}\nsynced_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )
    return root
