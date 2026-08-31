"""Task-contract knobs as VALUES — the RECIPE a run was trained with, for its W&B run config.

Read off the **TaskSpec** (the loader's input), not the resolved EnvSpec: a ``Curr`` entry's knobs are
bound into the curriculum fn at load and are no longer readable afterwards, so the EnvSpec cannot answer
what the curriculum was tuned to. Sections mirror the contract blocks a run is actually tuned on
(PHYSICS / ACTION / REWARD / EVENTS / TERMINATE / GATE / CURRICULUM); the world (SCENE) and the obs sets
are structure, not tuning, and stay out.
"""
from __future__ import annotations

from typing import Any

from sim.metalab.contract.spec import Curr, Done, Event, Rew, TaskSpec, terms


def _plain(v: Any) -> Any:
    """W&B config takes JSON: term fns become their name, tuples become lists."""
    if callable(v):
        return getattr(v, "__name__", str(v))
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def recipe(ts: TaskSpec) -> dict:
    """{section: {term name: {knob: value}}} — flat enough that each knob is one W&B column."""
    def block(b, kind, head, knobs="params"):     # Curr carries its knobs as `kwargs`, the rest as `params`
        return {t.name: {**head(t), **{k: _plain(v) for k, v in getattr(t, knobs).items()}}
                for t in terms(b, kind)}

    return {
        "physics": ts.physics.model_dump(),
        # `joints` is the action DIM (structure), not a tuning knob — the scales/EMA are.
        "action": {**{g: c.model_dump(exclude={"joints"}) for g, c in ts.action.items()},
                   "delay": ts.action_delay.model_dump()},
        "reward": block(ts.reward, Rew, lambda t: {"weight": t.weight}),
        "events": block(ts.events, Event, lambda t: {"mode": t.mode}),
        "terminate": block(ts.terminate, Done, lambda t: {"truncation": t.truncation}),
        "gate": _plain(ts.gate.model_dump()) if ts.gate else {},
        "curriculum": block(ts.curriculum, Curr, lambda t: {"fn": t.fn.__name__}, knobs="kwargs"),
    }
