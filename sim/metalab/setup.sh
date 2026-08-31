#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh — one-command onboarding for MetaLab (local workstation OR AWS node).
#
# From a freshly-cloned `allex` repo this makes the machine ready to train/eval, per engine:
#   1) clone the PINNED simulator source (newton, genesis-world) as a SIBLING of this repo
#   2) `uv sync` the engine's uv env from its committed uv.lock  (no conda, no S3 snapshot)
# Both steps are done by learning/scripts/local/setup_env.sh; this script just preflights and
# calls it for each engine. Idempotent — re-run any time; already-current pieces are no-ops.
#
# Simulator versions are pinned in sim/metalab/sim_versions.env (single source of truth); the env
# owner bumps them + re-locks. Python deps come from each engine's uv.lock — reproducible from git.
#
# Usage:  sim/metalab/setup.sh          # sets up both engines (newton + genesis)
#
# Prerequisites (details in sim/metalab/README.md): a GPU + NVIDIA driver to actually run, ~30 GB disk, and
# a C toolchain for the source build (Ubuntu: `sudo apt install build-essential`). uv is installed
# automatically if missing (it also provides Python — no miniconda needed). `wandb login` to log runs.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # sim/metalab/
REPO="$(cd "$HERE/../.." && pwd)"                            # <repo>
SETUP_ENV="$REPO/learning/scripts/local/setup_env.sh"

log(){ printf '[setup] %s\n' "$*"; }
die(){ printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

case "${1:-}" in -h|--help) sed -n '2,20p' "$0"; exit 0 ;; esac

# ── preflight ─────────────────────────────────────────────────────────────────
command -v git >/dev/null 2>&1 || die "git not found — install it first (sudo apt install git)."
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || \
  die "C compiler not found — the sim sources build from source. Install: sudo apt install build-essential"

# uv: install automatically if missing (it also manages the Python versions the engines need).
if ! command -v uv >/dev/null 2>&1; then
  log "uv not found — installing (astral.sh)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed."
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH — add \$HOME/.local/bin to PATH and re-run."
  log "uv installed. Add \$HOME/.local/bin to your shell PATH to keep it."
fi

# wandb is optional for setup; training WITH wandb needs a login (rl_train.sh fails loud otherwise).
if ! grep -qs 'api.wandb.ai' "$HOME/.netrc" 2>/dev/null && [ -z "${WANDB_API_KEY:-}" ]; then
  log "NOTE: wandb not logged in — run 'wandb login' to log runs, or train with --no_wandb."
fi

# ── per engine: clone pinned source + uv sync (both handled by setup_env.sh) ────
for engine in newton genesis; do
  log "── $engine: clone pinned source + uv sync ──"
  bash "$SETUP_ENV" --sim "$engine"
done

log "done — this machine is ready. Next:"
log "  sim/metalab/launchpad.sh                                                    # web console (one-click train/eval)"
log "  learning/scripts/local/rl_train.sh --sim genesis --task hammer-lift-teacher    # or straight from the CLI"
