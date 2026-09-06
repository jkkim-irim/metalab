#!/usr/bin/env bash
# parity.sh — record one engine's reads under the contract's COMMAND (headless, no viewer).
#
# Provisions/activates the engine venv (same helpers as standalone.sh) and runs
# sim/metalab/tools/parity_record.py. The trajectory (joints, amplitude, frequency, bodies, or the MDP action)
# is declared as COMMAND in the parity_test contract file, so both engines get bit-identical input.
#
# Usage (run once per engine, then diff the two files):
#   sim/metalab/parity.sh --sim newton  --task parity-joint-torque
#   sim/metalab/parity.sh --sim genesis --task parity-joint-torque
#   python -m sim.metalab.tools.parity_diff _logs/parity/<task>/<a>.npz _logs/parity/<task>/<b>.npz [--out diff.md]
LOG_TAG=parity
source "$(dirname "${BASH_SOURCE[0]}")/../../learning/scripts/local/lib.sh"

SIM=""; TASK=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --sim)      SIM="$2"; shift 2 ;;
    --sim=*)    SIM="${1#*=}"; shift ;;
    --task)     TASK="$2"; shift 2 ;;
    --task=*)   TASK="${1#*=}"; shift ;;
    -h|--help)  sed -n '2,10p' "$0"; exit 0 ;;
    *)          EXTRA+=("$1"); shift ;;
  esac
done

case "$SIM" in
  genesis|newton) ;;
  "") echo "[parity] --sim is required: --sim genesis | --sim newton" >&2; exit 2 ;;
  *)  echo "[parity] --sim must be genesis|newton (got '$SIM')" >&2; exit 2 ;;
esac
if [ -z "$TASK" ]; then
  echo "[parity] --task is required (a standalone contract name). Available:" >&2
  list_standalone_tasks | sed 's/^/  - /' >&2 || true
  exit 2
fi

bash "$ROOT/learning/scripts/local/setup_env.sh" --sim "$SIM"
_VENV="$(engine_venv "$SIM")"
source "$_VENV/bin/activate"
log "$SIM · $TASK — headless parity recording (venv=$_VENV)"
cd "$ROOT"

python -m sim.metalab.tools.parity_record --engine "$SIM" --task "$TASK" "${EXTRA[@]}"
