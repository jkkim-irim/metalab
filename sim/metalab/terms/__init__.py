"""MDP term libraries — the layer that turns primitives into what the task MEANS.

One subpackage per kind: :mod:`obs`, :mod:`reward`, :mod:`terminate`, :mod:`events`,
:mod:`curriculum`, :mod:`gate`. Each holds flat ``fn(env, **knobs)`` functions that a contract names and
hands its values to as knobs, so reading a term tells you what it measures without leaving the file.

A term composes two things and nothing else: backend PRIMITIVES (read through the driver's Ctx) and pure
math from :mod:`sim.metalab.api`. It never reaches for the contract itself, and the driver never does a
term's arithmetic for it — see the layer table in sim/metalab/CLAUDE.md.
"""
