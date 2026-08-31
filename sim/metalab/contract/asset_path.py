"""Contract asset path → a local file both spokes can open.

Asset paths are repo-root-relative and everything a contract names ships in the repo
(``sim/metalab/assets/``), so resolving one is a path join: no network, no cache, no credentials. A path
that does not exist fails loudly rather than resolving to something else.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]   # sim/metalab/contract/asset_path.py -> repo root


def resolve_asset(path: str) -> Path:
    """Repo-root-relative contract asset path → absolute local file."""
    p = (_REPO / path).resolve()
    assert p.is_file(), f"asset not found: {path}  (→ {p})"
    return p
