"""Guard: the runtime import graph imports neither lerobot, draccus, nor HuggingFace ``datasets``.

Dropping these as runtime dependencies is the whole point of internalizing the ACT model / v3.0
dataset / config / utils. This is a pure AST scan of ``learning/`` (excluding ``tests/``), so it
needs no third-party packages and runs anywhere — including before the equivalence-test files (which
DO import lerobot, under tests/) are removed.

Run in the node env from the repo root:  python -m pytest learning -q
"""
import ast
import pathlib

FORBIDDEN = {"lerobot", "draccus", "datasets"}
LEARNING_DIR = pathlib.Path(__file__).resolve().parents[1]  # the learning/ package root


def _runtime_py_files():
    for py in sorted(LEARNING_DIR.rglob("*.py")):
        if "tests" in py.parts:        # skip test modules (the equivalence tests may import lerobot)
            continue
        yield py


def _top_level_imports(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def test_no_forbidden_runtime_imports():
    offenders: dict[str, list[str]] = {}
    for py in _runtime_py_files():
        bad = _top_level_imports(ast.parse(py.read_text())) & FORBIDDEN
        if bad:
            offenders[str(py.relative_to(LEARNING_DIR.parent))] = sorted(bad)
    assert not offenders, f"runtime modules import forbidden deps: {offenders}"
