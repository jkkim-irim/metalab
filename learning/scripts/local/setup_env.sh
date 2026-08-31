#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_env.sh — make sure the engine's uv env exists and matches its committed uv.lock. So a
# freshly-cloned repo can train the moment you run metalab_train.sh: no manual conda/pip.
#
#   1) clone the PINNED simulator source (a sibling of the repo), or converge it to the pin
#   2) `uv sync` the engine's uv project (sim/_setup/<engine>) into its venv (METALAB_VENV_ROOT)
#
# Engine + all deps come from uv.lock (torch cu128, warp nightly from NVIDIA's index, usd, …) —
# reproducible from git. Versions pinned in sim/metalab/sim_versions.env.
#
# Usage:  setup_env.sh --sim genesis        # or --sim newton
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# Pinned upstream sim versions (single source of truth): sets *_GIT + *_REF.
PINS="$ROOT/sim/metalab/sim_versions.env"
[ -f "$PINS" ] && source "$PINS"

SIM=""; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sim) SIM="$2"; shift 2 ;;
    --sim=*) SIM="${1#*=}"; shift ;;
    --extra) EXTRA="$2"; shift 2 ;;        # optional uv extra to activate (e.g. rtx = newton OVRTX viewer)
    --extra=*) EXTRA="${1#*=}"; shift ;;
    *) shift ;;
  esac
done
[ -n "$SIM" ] || { echo "[setup_env] --sim genesis|newton 필요" >&2; exit 2; }
case "$SIM" in
  genesis) SRC_DIR=genesis-world; SRC_GIT="${GENESIS_WORLD_GIT:-}"; SRC_REF="${GENESIS_WORLD_REF:-}" ;;
  newton)  SRC_DIR=newton;        SRC_GIT="${NEWTON_GIT:-}";        SRC_REF="${NEWTON_REF:-}" ;;
  *) echo "[setup_env] --sim 는 genesis|newton (got '$SIM')" >&2; exit 2 ;;
esac

log(){ printf '[setup_env] %s\n' "$*"; }
command -v uv >/dev/null 2>&1 || {
  echo "[setup_env] uv 없음 — 설치 후 재실행: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "[setup_env] git 필요" >&2; exit 1; }

WS="$(dirname "$ROOT")"                       # workspace (repo parent) — sibling sims live here
SRC="$WS/$SRC_DIR"

# ── 1) pinned simulator source (editable), sibling of the repo: clone / converge to the pin ──────
[ -n "$SRC_REF" ] || { echo "[setup_env] $SIM ref 미지정 ($PINS 확인)" >&2; exit 1; }
if [ -d "$SRC/.git" ]; then
  CUR="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo '?')"
  if [ "$CUR" != "$SRC_REF" ]; then
    if [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ]; then
      log "WARN: $SRC_DIR 에 로컬 변경 있음 ($CUR, pin $SRC_REF) — 'git -C $SRC checkout $SRC_REF' 수동 (로컬 작업 보호차 유지)."
    else
      log "$SRC_DIR: $CUR → pinned $SRC_REF"
      git -C "$SRC" fetch --quiet origin || true
      git -C "$SRC" checkout --quiet "$SRC_REF" || { echo "[setup_env] $SRC_DIR checkout $SRC_REF 실패" >&2; exit 1; }
    fi
  fi
else
  [ -n "$SRC_GIT" ] || { echo "[setup_env] $SRC_DIR 소스 없음 + git URL 미지정" >&2; exit 1; }
  log "$SRC_DIR: cloning $SRC_GIT → $SRC"
  git clone "$SRC_GIT" "$SRC"
  git -C "$SRC" checkout "$SRC_REF" || { echo "[setup_env] $SRC_DIR checkout $SRC_REF 실패" >&2; exit 1; }
fi

# ── 2) uv sync the engine's project into its venv (reproducible from uv.lock) ─────────────────────
PROJ="$(engine_uv_project "$SIM")"
VENV="$(engine_venv "$SIM")"
[ -f "$PROJ/pyproject.toml" ] || { echo "[setup_env] uv 프로젝트 없음: $PROJ" >&2; exit 1; }
mkdir -p "$(dirname "$VENV")"
log "uv sync → $VENV  (project $PROJ${EXTRA:+, extra: $EXTRA}, ~수 GB 첫 실행)"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --project "$PROJ" ${EXTRA:+--extra "$EXTRA"}
log "완료: $SIM env 준비됨 ($VENV)."
