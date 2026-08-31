"""Contract asset path → a local file both spokes can open, for repo-relative paths and for ``s3://`` ones.

Object assets are heavy and deliberately not in git (the 78 YCB scans are 166 MB), so a contract may point
``asset.mjcf`` straight at the bucket instead of at a working copy:

    "s3://wirobotics-internal/shared/sim_assets/objects/ycb_002_master_chef_can/ycb_002_master_chef_can.xml"

What is fetched is the file's whole PARENT prefix, not the file: an object MJCF names its meshes and its
texture through ``meshdir``/``texturedir``, so the ``meshes/`` folder beside it has to land in the same
relative place or the compile dies on the first ``<mesh>``.

The sync runs on every resolve — one LIST call when nothing changed — so what runs is what the bucket
holds, and an unreachable bucket fails loudly instead of quietly running a stale copy. Needs the ``aws``
CLI on PATH with credentials for the bucket (neither sim env ships boto3).
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

_REPO = Path(__file__).resolve().parents[3]   # sim/metalab/contract/asset_path.py -> repo root
_CACHE = Path(os.environ.get("ALLEX_ASSET_CACHE", Path.home() / ".cache" / "metalab" / "sim_assets"))

_S3 = "s3://"


def resolve_asset(path: str) -> Path:
    """Contract asset path (repo-root-relative, or ``s3://bucket/key/file``) → absolute local file."""
    if path.startswith(_S3):
        return _fetch(path)
    p = (_REPO / path).resolve()
    assert p.is_file(), f"asset not found: {path}  (→ {p})"
    return p


def _fetch(uri: str) -> Path:
    """Sync the URI's parent prefix into the cache and return the local path of the named file."""
    prefix, _, filename = uri[len(_S3):].rpartition("/")
    assert prefix and filename, f"s3 asset must be bucket/key/file, got: {uri}"
    dest = _CACHE / prefix
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", f"{_S3}{prefix}/", str(dest), "--only-show-errors"], check=True)
    p = dest / filename
    assert p.is_file(), f"asset not found under {_S3}{prefix}/: {filename}"
    return p
