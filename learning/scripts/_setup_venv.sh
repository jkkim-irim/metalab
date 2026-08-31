#!/usr/bin/env bash
# Sourced helper (not executed) — ensure uv + the venv + pinned deps from pyproject.toml.
# Sets/exports $VENV and $PY. The caller must `cd` to the repo root first.
# Uses uv (installed if missing) so a bare DLAMI — which ships neither uv nor python3-venv
# (ensurepip) — still works; $VENV is created once, kept outside the deploy dir.
VENV="${VENV:-/opt/dlami/nvme/allex-venv}"

# ensure uv: fast, self-contained, and sidesteps the python3-venv/ensurepip gap on a bare DLAMI.
# SSM Run Command may not set HOME (and set -u would abort), so default it — it's also where the
# uv installer drops the uv/uvx binaries.
export HOME="${HOME:-/root}"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# torchcodec (video decode) loads system FFmpeg shared libs (libavutil…); the bare DLAMI lacks them
if ! ldconfig -p | grep -q 'libavutil\.so'; then
    echo "installing ffmpeg"
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "creating venv at $VENV"
    uv venv "$VENV"
fi
PY="$VENV/bin/python"
uv pip install --python "$PY" -e ".[fk]" -q   # pinned deps + FK extra (pytorch_kinematics/matplotlib/imageio-ffmpeg) — FK validation is on by default; no-op when satisfied, fails loudly on conflict
export VENV PY
