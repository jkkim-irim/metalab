"""Where a contract's object assets come from.

Objects live IN THE REPO under ``sim/metalab/assets/objects/<name>/<name>.xml``, so a clone is enough to
run every task here — no download step, no account. ``OBJECTS`` is the ONE place that root is written and
:func:`object_mjcf` the one way to build a path.

What gets addressed is a folder, not a lone file: an object MJCF names its meshes and texture through
``meshdir``/``texturedir``, so ``meshes/`` has to sit beside the ``.xml`` or the compile dies on the first
``<mesh>``. One asset = one folder named after it.

The robot is NOT addressed here — it loads from its own description tree.

This is a library, not a contract: ``_*.py`` is skipped by the task and recipe listings.
"""
from __future__ import annotations

OBJECTS = "sim/metalab/assets/objects"
"""Repo-relative root of the object assets."""


def object_mjcf(asset: str) -> str:
    """Asset folder name → the repo-relative path of its MJCF (folder and file share the name)."""
    return f"{OBJECTS}/{asset}/{asset}.xml"
