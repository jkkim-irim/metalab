"""Newton backend (spoke) — builds a Newton scene from an EnvSpec (standalone, no IsaacLab).

- :mod:`sim.metalab.backends.newton.parser` — EnvSpec → newton.Model (robot/object add_mjcf + per-world round-robin)
- :mod:`sim.metalab.backends.newton.backend`      — newton.Model/State/Control/Contacts → SimBackend (read+control+step+reset)
- :mod:`sim.metalab.backends.newton.server`       — build_env + EnvDriver(VecEnv) + RPC (mirrors genesis server)

Symmetric with the Genesis spoke (:mod:`sim.metalab.backends.genesis`): loads the same MJCF copy so only solver differences remain (parity).
"""
