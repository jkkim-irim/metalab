"""Shared runtime (engine-agnostic, opt-in for training).

- :mod:`sim.metalab.runtime.backend`    — SimBackend Protocol (engine implements: read+control+step+reset)
- :mod:`sim.metalab.runtime.env_driver` — EnvDriver: EnvSpec + SimBackend → VecEnv (duck-typed)

sim2sim/parity works with parser+state_adapter alone, without this runtime.
"""
