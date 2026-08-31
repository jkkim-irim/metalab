#!/usr/bin/env bash
# Run the full pre-PR gate ON the node (ruff + the whole test suite):
#   1) ruff   — import order + lint on learning/
#   2) pytest — the whole learning/ suite (auto-discovers every test_*.py under it)
# Extra args pass through to pytest, e.g.:
#   bash learning/scripts/run_ci_tests.sh -k split
# Trigger from a laptop with test_aws.sh (scp the code over SSM, then run this).
#
# TRANSIENT — runs in the GR00T venv (torch 2.7.1 + diffusers + the vendored _hf), NOT the ACT venv.
# The GR00T model tests (learning/model/tests/test_groot_*.py) import diffusers + learning.model.groot,
# which the ACT venv (pyproject, torch 2.10) doesn't carry — so they never actually ran under the old
# ACT-venv gate. The GR00T env is a superset of the ACT test deps (ACT uses no lerobot), so the ACT tests
# pass here too. (transformers is no longer a GR00T dependency — the Qwen3-VL backbone/processor are
# vendored under learning/model/groot/_hf.)
# TODO: proper two-venv CI (ACT torch 2.10 + a separate GR00T torch 2.7.1 run) when the split matters;
# then revert this to `_setup_venv.sh` + `.[test,fk]` and add a dedicated GR00T pytest step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

# GR00T venv — same build as train_groot.sh (requirements-groot.txt is the single dep source) + pytest.
export HOME="${HOME:-/root}"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/opt/dlami/nvme/uv-cache}"
GROOT_VENV="${GROOT_VENV:-/opt/dlami/nvme/groot-venv}"
PY="$GROOT_VENV/bin/python"
command -v uv >/dev/null 2>&1 || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
ldconfig -p | grep -q 'libavutil\.so' || { echo "installing ffmpeg"; apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg; }
if ! "$PY" -c 'import torch, diffusers, pytest' 2>/dev/null; then
  echo "building GR00T venv at $GROOT_VENV from requirements-groot.txt (torch 2.7.1 line)"
  uv venv --python 3.10 "$GROOT_VENV"
  # unsafe-best-match: torch/vision/codec from the cu128 index, everything else from PyPI.
  VIRTUAL_ENV="$GROOT_VENV" uv pip install --index-strategy unsafe-best-match -r "$SCRIPT_DIR/requirements-groot.txt" pytest
fi
[ -x "$PY" ] || { echo "FATAL: GR00T venv missing at $GROOT_VENV." >&2; exit 1; }

echo "== ruff (import order + lint) =="
if command -v uvx >/dev/null 2>&1; then
	uvx ruff@0.15.17 check learning           # pinned, matches the pre-commit hook + CI
else
	uv pip install --python "$PY" -q ruff==0.15.17
	"$PY" -m ruff check learning
fi

echo "== pytest (whole suite, GR00T venv) =="
exec "$PY" -m pytest learning -q "$@"
