#!/usr/bin/env bash
# parity.sh — record the SAME contract on genesis and newton, then diff + plot the pair (headless, no viewer).
#
# The trajectory (joints, amplitude, frequency, bodies, or the MDP action) is the COMMAND block of the
# parity_test contract, so both engines get bit-identical input. Each engine runs in its own uv venv
# (provisioned/activated with the same helpers as standalone.sh) via sim/metalab/tools/parity_record.py;
# the two .npz files are then compared with parity_diff (markdown) and parity_plot (png).
#
# Usage:
#   sim/metalab/parity.sh --task parity-joint-torque             # genesis + newton → diff .md + plot .png
#   sim/metalab/parity.sh --task parity-joint-torque --sim newton  # record one engine only, no comparison
# Outputs land in _logs/parity/<task>/: <engine>_<mode>_<stamp>.{npz,json}, genesis_vs_newton_<stamp>.md, genesis_vs_newton_<stamp>_NN.png (2 channels per page)
LOG_TAG=parity
source "$(dirname "${BASH_SOURCE[0]}")/../../learning/scripts/local/lib.sh"

SIM=""; TASK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --sim)      SIM="$2"; shift 2 ;;
    --sim=*)    SIM="${1#*=}"; shift ;;
    --task)     TASK="$2"; shift 2 ;;
    --task=*)   TASK="${1#*=}"; shift ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "[parity] unknown arg '$1' (flags: --task --sim)" >&2; exit 1 ;;
  esac
done

case "$SIM" in
  ""|genesis|newton) ;;
  *)  echo "[parity] --sim must be genesis|newton (got '$SIM')" >&2; exit 2 ;;
esac
if [ -z "$TASK" ]; then
  echo "[parity] --task is required (a parity_test contract name). Available:" >&2
  list_standalone_tasks | sed 's/^/  - /' >&2 || true
  exit 2
fi
cd "$ROOT"

# record one engine in its venv (subshell keeps the activation local); echoes the written .npz path.
record(){
  local engine="$1" venv out
  bash "$ROOT/learning/scripts/local/setup_env.sh" --sim "$engine" >&2
  venv="$(engine_venv "$engine")"
  log "$engine · $TASK — headless parity recording (venv=$venv)" >&2
  out="$(
    source "$venv/bin/activate"
    python -m sim.metalab.tools.parity_record --engine "$engine" --task "$TASK" | tee /dev/stderr
  )"
  sed -n 's/^\[parity\] wrote //p' <<<"$out"
}

if [ -n "$SIM" ]; then
  record "$SIM" >/dev/null
  exit 0
fi

A="$(record genesis)"
B="$(record newton)"
[ -n "$A" ] && [ -n "$B" ] || { echo "[parity] recorder did not report an output path (A='$A' B='$B')" >&2; exit 1; }

OUT="$(dirname "$A")/genesis_vs_newton_$(date +%Y%m%d_%H%M%S)"
source "$(engine_venv genesis)/bin/activate"
python -m sim.metalab.tools.parity_diff "$A" "$B" --out "$OUT.md"
python -m sim.metalab.tools.parity_plot "$A" "$B" --out "$OUT.png"
