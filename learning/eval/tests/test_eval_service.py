"""Unit tests for the ``eval_service`` dispatcher.

Dependency-free: ``importlib.import_module`` is stubbed so no heavy policy module is ever imported —
which exercises the REAL selection + argv-forwarding logic.

Run:  python -m pytest learning/eval -q
"""
import importlib
import sys

import pytest

from learning.eval import eval_service


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
    """`--policy actor` -> imports learning.eval.policies.actor and forwards the remaining argv."""
    stub = _StubModule(rc=0)
    imported = {}
    monkeypatch.setattr(importlib, "import_module", _capture_import(imported, stub))
    monkeypatch.setattr(sys, "argv",
                        ["eval_service", "--policy", "actor", "--num_envs", "8", "--episodes", "4"])
    rc = eval_service.main()
    assert rc == 0
    assert imported["name"] == "learning.eval.policies.actor"
    assert stub.seen == ["--num_envs", "8", "--episodes", "4"]  # --policy stripped, rest forwarded


def test_dispatcher_defaults_to_actor(monkeypatch):
    """No `--policy` -> actor, so metalab_eval.sh's flags route to the actor path unchanged."""
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
