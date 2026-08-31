"""Engine spokes (``sim.metalab.backends.genesis`` / ``sim.metalab.backends.newton``).

Each spoke ships a parser (EnvSpec → engine scene), a backend (engine state/control → SimBackend), and a
``server.py`` (contract → EnvDriver → RPC sim-service). Engine imports are isolated to the spokes; the hub
(``sim.metalab.contract``) and runtime never import an engine.
"""
