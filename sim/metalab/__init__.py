"""MetaLab — the engine-agnostic ALLEX sim stack (a peer of ``sim/isaaclab``, ``sim/libero``, ...).

Submodules:
- ``sim.metalab.contract``    — engine-agnostic env contract (hub): tasks + parts (object/robot/reward/obs)
- ``sim.metalab.api``     — engine-agnostic primitive API (state/frames/keypoints/shaping)
- ``sim.metalab.runtime`` — shared runtime (env_driver, telemetry); opt-in for training
- ``sim.metalab.backends.genesis`` / ``sim.metalab.backends.newton`` — engine spokes: each ships a parser
  (EnvSpec → engine scene), a backend (engine state/control → SimBackend), and a ``server.py``
  (contract → EnvDriver → RPC sim-service). Engine imports are isolated to the spokes; the hub
  (``envs``) and runtime never import an engine.

The RPC + CUDA-IPC process boundary the spokes serve over lives in ``sim.service`` (shared across sims).
"""
