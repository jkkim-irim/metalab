"""Genesis backend (spoke) — builds a Genesis scene from an EnvSpec contract.

- :mod:`sim.metalab.backends.genesis.parser` — EnvSpec → gs.Scene (parsing bridge)
- :mod:`sim.metalab.backends.genesis.by_basename`  — proxy to look up entity joints/links by contract basename (robot MJCF)

(Legacy sim2sim adapters like ``newton_to_genesis`` run separately via sys.path bare-import.)
"""
