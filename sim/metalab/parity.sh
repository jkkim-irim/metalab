#!/usr/bin/env bash
# parity.sh — record one engine's StateAdapter reads under a sinusoidal joint command (headless, no viewer).
#
# Provisions/activates the engine venv (same helpers as standalone.sh) and runs
# sim/metalab/tools/parity_record.py. Extra flags pass through to the recorder.
#
# Usage (run each line once per engine, then diff the two files):
#   B=panda0_gripper,panda0_leftfinger,panda0_rightfinger
#   sim/metalab/parity.sh --sim newton --task parity-joint-torque --joints panda0_joint1,panda0_joint2,panda0_joint3,panda0_joint4,panda0_joint5,panda0_joint6,panda0_joint7 --amp-deg 5 --freq-hz 0.25 --bodies $B
#   sim/metalab/parity.sh --sim newton --task parity-contact      --joints panda0_joint2 --amp-deg 4 --freq-hz 0.5 --bodies $B
#   sim/metalab/parity.sh --sim newton --task parity-objects      --joints panda0_joint1 --amp-deg 0 --bodies $B
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
