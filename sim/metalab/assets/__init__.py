# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Package containing ALLEX robot and object asset configurations."""

import os

# Backend-neutral asset package (sim/metalab/assets, sim/isaaclab 밖). object USD 는 ``usd/objects/<name>/`` 아래,
# 비-USD 데이터(trajectory npz 등)는 ``data/`` 아래. Genesis 등 다른 백엔드도 파일 경로로 참조한다
# (import 아님). __version__ 은 static (Isaac Lab kit extension.toml 미사용).
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLEX_ASSETS_DATA_DIR = os.path.join(_PACKAGE_DIR, "data")
"""비-USD 데이터(trajectory npz 등) 루트 (``data/``)."""


def _resolve_allex_description_dir() -> str:
    """Robot URDF/MJCF/USD tree. ``allex_description`` is gitignored + shipped separately, so on a
    node set ``ALLEX_DESCRIPTION_DIR`` (authoritative). The relative fallback assumes the ``allex/``
    monorepo checkout: ``<allex>/sim/metalab/assets`` → ``<allex>/allex_description``."""
    override = os.environ.get("ALLEX_DESCRIPTION_DIR")
    if override:
        return os.path.abspath(override)
    return os.path.abspath(os.path.join(_PACKAGE_DIR, "..", "..", "allex_description"))


ALLEX_DESCRIPTION_DIR = _resolve_allex_description_dir()
"""Robot URDF/MJCF/USD tree (override via ``ALLEX_DESCRIPTION_DIR``; default: ``<allex>/allex_description``)."""
ALLEX_USD_DIR = os.path.join(_PACKAGE_DIR, "usd")
"""USD 에셋 루트 (``usd/``)."""
ALLEX_OBJECTS_DIR = os.path.join(ALLEX_USD_DIR, "objects")
"""Task object USD 루트 (``usd/objects/``)."""
ALLEX_HAMMER_DIR = os.path.join(ALLEX_OBJECTS_DIR, "hammer")
"""Hammer variant USDs (``usd/objects/hammer/``)."""
__version__ = "0.1.0"
