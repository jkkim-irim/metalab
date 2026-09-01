#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# standalone.sh — run ONE MetaLab task env with the GUI, no policy / no learning.
#
# Builds the contract env via the engine spoke's build_env and steps it forever with a zero action
# (robot holds its init pose) while the GL viewer renders and the physics plays out. For eyeballing a
# task's scene, object placement, contacts and initial pose — sim2real physics tuning, new-task setup
# checks — env only, with no policy and no training/eval artifacts.
#
# Runs on the DEFAULT (display) GPU so compute AND the GL viewer share ONE GPU — identical on a 1-GPU or
# 2-GPU box. No device selection on purpose: the on-screen GL window renders on the display GPU no matter
# what, so pinning compute to another GPU (CUDA_VISIBLE_DEVICES) only splits the run across two GPUs.
#
# Lives under sim/ (learning-independent). Reuses ONLY the shared engine-venv provisioning helpers
# (learning/scripts/local/{lib.sh,setup_env.sh}) — engine-venv utilities, not learning logic — to make
# sure the engine's uv venv exists and to activate it, then runs the engine-agnostic runner
# sim/metalab/runtime/standalone.py in that venv. Always a single env + GL viewer (hardcoded in the runner).
#
# Usage (from anywhere):
#   sim/standalone.sh --sim newton  --task hammer-lift-teacher
#   sim/standalone.sh --sim genesis --task hammer-lift-teacher
# ─────────────────────────────────────────────────────────────────────────────
LOG_TAG=standalone
source "$(dirname "${BASH_SOURCE[0]}")/../../learning/scripts/local/lib.sh"

TASK="${TASK:-}"; SIM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sim)      SIM="$2"; shift 2 ;;
    --sim=*)    SIM="${1#*=}"; shift ;;
    --task)     TASK="$2"; shift 2 ;;
    --task=*)   TASK="${1#*=}"; shift ;;
    -h|--help)  sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "[standalone] unknown arg '$1' (flags: --sim --task)" >&2; exit 1 ;;
  esac
done

# backend (REQUIRED, fail-loud — no default): genesis | newton (each in its own uv venv).
case "$SIM" in
  genesis|newton) ;;
  "") echo "[standalone] --sim is required: --sim genesis | --sim newton" >&2; exit 2 ;;
  *)  echo "[standalone] --sim must be genesis|newton (got '$SIM')" >&2; exit 2 ;;
esac

# task (REQUIRED, fail-loud — same as rl_train.sh): print the discoverable tasks, then exit.
if [ -z "$TASK" ]; then
  echo "[standalone] --task is required (no default). Available tasks:" >&2
  list_standalone_tasks | sed 's/^/  - /' >&2 || true
  echo "  e.g.  sim/metalab/standalone.sh --sim $SIM --task hammer-lift" >&2
  exit 2
fi

# ensure the engine's uv env exists & matches uv.lock (clone pinned source + `uv sync`; no-op if current),
# then activate it — the runner imports genesis/newton in this venv (in-process).
bash "$ROOT/learning/scripts/local/setup_env.sh" --sim "$SIM"
resolve_display || exit 2
_VENV="$(engine_venv "$SIM")"
source "$_VENV/bin/activate"
log "$SIM · $TASK — env only, GL viewer, default GPU, no policy (venv=$_VENV). Ctrl-C to stop."
cd "$ROOT"

python -m sim.metalab.runtime.standalone --engine "$SIM" --task "$TASK"
