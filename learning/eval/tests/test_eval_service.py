"""Unit tests for the eval/policy seam: the ``EvalPolicy`` contract + the ``eval_service`` dispatcher.

Dependency-free (no torch / gr00t / RL stack): ``EvalPolicy`` is an abstract base with no heavy deps,
and the dispatcher is tested by stubbing ``importlib.import_module`` so no heavy policy module is ever
imported — exercising the REAL selection + argv-forwarding logic.

Run:  python -m pytest learning/eval -q
"""
import importlib
import sys

import pytest

from learning.eval import eval_service
from learning.eval.policies.eval_policy import EvalPolicy


def test_evalpolicy_is_abstract():
    with pytest.raises(TypeError):
        EvalPolicy()  # reset + act unimplemented


def test_incomplete_subclass_cannot_instantiate():
    class OnlyAct(EvalPolicy):
        def act(self, obs):
            return obs

    with pytest.raises(TypeError):
        OnlyAct()  # missing reset


def test_complete_subclass_works():
    class Fake(EvalPolicy):
        action_dim = 3

        def reset(self, done=None):
            self.reset_called = True

        def act(self, obs):
            return [obs] * self.action_dim

    p = Fake()
    p.reset()
    assert p.reset_called
    assert p.act("x") == ["x", "x", "x"]


class _StubModule:
    """Stand-in for a policy module: records the argv it was handed, returns a sentinel rc."""
    def __init__(self, rc):
        self.rc = rc
        self.seen = None

    def run(self, rest):
        self.seen = rest
        return self.rc


def _capture_import(imported, stub):
    """Return an ``import_module`` stand-in that records the requested name and returns ``stub``
    (a plain lambda can't: ``dict.setdefault`` returns the truthy name string, so ``... or stub``
    would hand the dispatcher a ``str`` instead of the module)."""
    def fake_import(name):
        imported["name"] = name
        return stub
    return fake_import


def test_dispatcher_selects_named_policy(monkeypatch):
    """`--policy groot` -> imports learning.eval.policies.groot and forwards the remaining argv."""
    stub = _StubModule(rc=0)
    imported = {}
    monkeypatch.setattr(importlib, "import_module", _capture_import(imported, stub))
    monkeypatch.setattr(sys, "argv",
                        ["eval_service", "--policy", "groot", "--suite", "libero_90", "--tasks", "11"])
    rc = eval_service.main()
    assert rc == 0
    assert imported["name"] == "learning.eval.policies.groot"
    assert stub.seen == ["--suite", "libero_90", "--tasks", "11"]  # --policy stripped, rest forwarded


def test_dispatcher_defaults_to_actor(monkeypatch):
    """No `--policy` -> actor, so rl_eval.sh's existing flags route to the actor path unchanged."""
    stub = _StubModule(rc=7)
    imported = {}
    monkeypatch.setattr(importlib, "import_module", _capture_import(imported, stub))
    monkeypatch.setattr(sys, "argv", ["eval_service", "--checkpoint", "m.pt", "--num_envs", "8"])
    rc = eval_service.main()
    assert rc == 7
    assert imported["name"] == "learning.eval.policies.actor"
    assert stub.seen == ["--checkpoint", "m.pt", "--num_envs", "8"]


def test_dispatcher_rejects_unknown_policy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval_service", "--policy", "nope"])
    with pytest.raises(SystemExit):  # argparse choices -> exit 2
        eval_service.main()
