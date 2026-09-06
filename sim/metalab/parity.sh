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
#   sim/metalab/parity.sh --task parity-contact --video           # + one side-by-side mp4 (newton needs a live X display)
# Outputs land in _logs/parity/<task>/: <engine>_<mode>_<stamp>.{npz,json}, genesis_vs_newton_<stamp>.md,
# genesis_vs_newton_<stamp>_NN.png (2 channels per page), video/genesis_vs_newton_<stamp>.mp4 (left genesis,
# right newton, each cropped to the middle 640 px around the robot; the per-engine recordings are deleted).
LOG_TAG=parity
source "$(dirname "${BASH_SOURCE[0]}")/../../learning/scripts/local/lib.sh"

SIM=""; TASK=""; VIDEO=()
while [ $# -gt 0 ]; do
  case "$1" in
    --sim)      SIM="$2"; shift 2 ;;
    --sim=*)    SIM="${1#*=}"; shift ;;
    --task)     TASK="$2"; shift 2 ;;
    --task=*)   TASK="${1#*=}"; shift ;;
    --video)    VIDEO=(--video); shift ;;
    -h|--help)  sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "[parity] unknown arg '$1' (flags: --task --sim --video)" >&2; exit 1 ;;
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
[ ${#VIDEO[@]} -eq 0 ] || resolve_display || exit 2

record(){
  local engine="$1" venv out
  bash "$ROOT/learning/scripts/local/setup_env.sh" --sim "$engine" >&2
  venv="$(engine_venv "$engine")"
  log "$engine · $TASK — headless parity recording (venv=$venv)" >&2
  out="$(
    source "$venv/bin/activate"
    python -m sim.metalab.tools.parity_record --engine "$engine" --task "$TASK" "${VIDEO[@]}" | tee /dev/stderr
  )"
  sed -n 's/^\[parity\] wrote \(.*\.npz\)$/\1/p' <<<"$out"
  sed -n 's/^\[parity\] wrote \(.*\.mp4\)$/\1/p' <<<"$out"
}

side_by_side(){
  local a="$1" b="$2" out="$3"
  ffmpeg -v error -y -i "$a" -i "$b" -filter_complex \
    "[0:v]crop=640:720:320:0,drawtext=text='genesis':x=16:y=16:fontsize=36:fontcolor=white:font=DejaVuSans[a];\
     [1:v]crop=640:720:320:0,drawtext=text='newton':x=16:y=16:fontsize=36:fontcolor=white:font=DejaVuSans[b];\
     [a][b]hstack=inputs=2" -c:v libx264 -crf 18 -pix_fmt yuv420p "$out"
  rm -f "$a" "$b"
  echo "[parity] wrote $out"
}

if [ -n "$SIM" ]; then
  record "$SIM" >/dev/null
  exit 0
fi

mapfile -t GA < <(record genesis)
mapfile -t NB < <(record newton)
A="${GA[0]:-}"; B="${NB[0]:-}"
[ -n "$A" ] && [ -n "$B" ] || { echo "[parity] recorder did not report an output path (A='$A' B='$B')" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$(dirname "$A")/genesis_vs_newton_$STAMP"
source "$(engine_venv genesis)/bin/activate"
python -m sim.metalab.tools.parity_diff "$A" "$B" --out "$OUT.md"
python -m sim.metalab.tools.parity_plot "$A" "$B" --out "$OUT.png"
if [ ${#VIDEO[@]} -gt 0 ]; then
  [ -n "${GA[1]:-}" ] && [ -n "${NB[1]:-}" ] || { echo "[parity] --video set but no mp4 reported (${GA[*]} / ${NB[*]})" >&2; exit 1; }
  side_by_side "${GA[1]}" "${NB[1]}" "$(dirname "$A")/video/genesis_vs_newton_$STAMP.mp4"
fi
