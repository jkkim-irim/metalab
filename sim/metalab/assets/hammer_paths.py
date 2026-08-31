# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve ALLEX asset paths (``assets`` SSOT re-exports for task code)."""

from __future__ import annotations

from pathlib import Path

from sim.metalab.assets import (
    ALLEX_ASSETS_DATA_DIR,
    ALLEX_DESCRIPTION_DIR,
    ALLEX_HAMMER_DIR,
    ALLEX_OBJECTS_DIR,
)

ASSETS_DIR: Path = Path(ALLEX_ASSETS_DATA_DIR)
OBJECTS_DIR: Path = Path(ALLEX_OBJECTS_DIR)
HAMMER_DIR: Path = Path(ALLEX_HAMMER_DIR)
DESCRIPTION_DIR: Path = Path(ALLEX_DESCRIPTION_DIR)
