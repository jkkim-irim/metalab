"""Where a contract's object assets come from — the bucket, not the repo.

Object assets are heavy and deliberately not in git (the YCB scans alone are 166 MB), so every contract
under ``tasks/`` addresses them in S3 and ``contract.asset_path`` fetches on resolve. What it fetches is
the URI's whole PARENT prefix, not the file: an object MJCF names its meshes and texture through
``meshdir``/``texturedir``, so the ``meshes/`` folder beside it has to land in the same relative place or
the compile dies on the first ``<mesh>``.

``S3_OBJECTS`` is the ONE place that prefix is written and :func:`s3_mjcf` the one way to build a URI, so
moving the bucket is a one-line change. The robot is NOT addressed here — it loads from
``allex_description``, which is shipped separately.

This is a library, not a contract: ``_*.py`` is skipped by the task and recipe listings.
"""
from __future__ import annotations

S3_OBJECTS = "s3://wirobotics-internal/shared/sim_assets/objects"
"""Root prefix of the shared sim object assets. One asset = one folder named after it."""


def s3_mjcf(asset: str) -> str:
    """Asset folder name → the S3 URI of its MJCF (folder and file share the name)."""
    return f"{S3_OBJECTS}/{asset}/{asset}.xml"
