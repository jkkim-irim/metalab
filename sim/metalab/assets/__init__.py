"""In-repo assets: robot MJCF (``robots/``), object MJCF (``objects/``), trajectory data (``data/``).

Everything here is addressed by PATH, not by import — contracts name a repo-relative path and
``sim.metalab.contract.asset_path`` resolves it. This module exists only to make the directory a
package so ``tools/`` is importable.
"""
