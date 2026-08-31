#!/usr/bin/env bash
# verify_init_state.sh — run each MetaLab spoke sequentially headless & no-wandb for a task,
# and dump env-0 initial state **as actually read by the simulator** after reset to sim/metalab/parity/<task>/<sim>.json.
# (values read from the engine state buffer, not the input contract — for per-engine comparison.)
#
#   sim/metalab/verify_init_state.sh --task hammer_lift_teacher --recipe privileged
#
# Args are --task and --recipe (a task FAMILY needs both; a single-file contract takes no recipe). Auto-discovers sim/metalab/ standalone spokes (genesis, newton) and runs each in its own
# uv venv sequentially (same venvs as training — ~/.metalab/venvs/<engine>, built by
# learning/scripts/local/setup_env.sh). Venv root, num_envs, device, output root are env-overridable.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"                 # sim/metalab/
# uv engine venvs — same per-worktree root lib.sh computes, so this resolves the venv TRAINING uses
METALAB_VENV_ROOT="${METALAB_VENV_ROOT:-$HOME/.metalab/venvs/$(basename "$(cd "$HERE/../.." && pwd)")}"
OUT_DIR="${OUT_DIR:-$HERE/parity}"                   # sim/metalab/parity (task subfolder is gitignored)
NUM_ENVS="${NUM_ENVS:-1}"; DEVICE="${DEVICE:-cuda:0}"

TASK=""; RECIPE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --task)    TASK="$2"; shift 2;;
    --recipe)  RECIPE="$2"; shift 2;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) echo "unknown arg: '$1' (--task <name> [--recipe <name>])" >&2; exit 2;;
  esac
done
[ -n "$TASK" ] || { echo "--task <name> required" >&2; exit 2; }
TASK="${TASK//-/_}"

export PYTHONPATH="$(cd "$HERE/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"   # repo root (sim is not an installed package)
d="$OUT_DIR/$TASK${RECIPE:+_$RECIPE}"; mkdir -p "$d"

dump_one(){  # $1=sim (its uv venv = $METALAB_VENV_ROOT/<sim>)
  local sim="$1" venv="$METALAB_VENV_ROOT/$1"
  [ -f "$HERE/$sim/server.py" ] || { echo "### skip $sim (no $sim/server.py)"; return 0; }
  [ -x "$venv/bin/python" ]     || { echo "### skip $sim (no uv venv at $venv — run learning/scripts/local/setup_env.sh --sim $sim)"; return 0; }
  echo "### $sim  (venv=$venv, task=$TASK${RECIPE:+/$RECIPE}) → $d/$sim.json"
  "$venv/bin/python" -m sim.metalab.runtime.parity_rollout initdump --engine "$sim" --task "$TASK" \
    ${RECIPE:+--recipe "$RECIPE"} --num_envs "$NUM_ENVS" --device "$DEVICE" --out "$d/$sim.json"
}

rc=0
dump_one genesis || rc=$?
dump_one newton  || rc=$?
echo "### done (exit $rc) — init-state dump: $d/"
exit $rc
