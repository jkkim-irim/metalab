#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# metalab_eval.sh (local) — eval a LOCAL checkpoint over the RPC sim-service (RPC-only, team rule).
# Reports SR + a per-episode S/F table to a local json.
#
# DEFAULT (no --viz): headless — one full episode window, recorded by the SERVER as a rerun `.rrd` plus the
# synced `data.json` + `report.html` (the report's 3D pane replays that .rrd with a free camera). RECORD_ENVS
# envs (default 3) get a series + a report tab; num_envs is raised to RECORD_ENVS if smaller. Nothing is
# rendered to pixels here — no GL context, no RT cores, no display. Kept LOCAL under the eval out-dir.
# --viz instead opens the backend's GUI for a live watch (no recording).
#
# Backend: --sim {genesis|newton} is REQUIRED. The checkpoint's task is inferred from its run-dir name;
# override --task.
#
# Checkpoint: --checkpoint <path> (or CHECKPOINT=<path>). If omitted, the newest LOCAL model_*.pt for
# --task under $RL_LOG_ROOT (metalab_train.sh keeps local copies).
#
# Auto-export (play.py-style): every run bakes <ckpt_dir>/exported/{policy.pt, policy.onnx} (raw obs in
# -> action out; obs_normalizer + deterministic output folded in) for sim2sim / sim2real. EXPORT=0 skips.
#
# Args: --sim / --viz / --task / --num_envs / --checkpoint are CLI flags; other knobs are env vars
# (EPISODES, STEPS, SEED, EXPORT, GPU, EXPNAME). --viz is a boolean (backend inferred from --sim).
# Usage (from anywhere):
#   learning/scripts/local/metalab_eval.sh --sim genesis --checkpoint _logs/rl/.../model_499.pt   # DEFAULT: record → .rrd + report
#   learning/scripts/local/metalab_eval.sh --sim genesis --viz --num_envs 1                       # live GUI watch (∞, Ctrl-C)
#   RECORD=0 EPISODES=64 learning/scripts/local/metalab_eval.sh --sim newton --num_envs 16        # headless, 64-ep SR (no recording)
#   GPU=1 learning/scripts/local/metalab_eval.sh --sim genesis --checkpoint .../model_499.pt      # pin to physical GPU 1
# ──────────────────────────────────────────────────────────────────────────────
LOG_TAG=eval
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TASK="${TASK:-}"          # inferred from the checkpoint run-dir if unset; --task overrides
RECIPE="${RECIPE:-}"      # REQUIRED when --task is a family folder; empty for a single-file contract
# SEED: 미지정 시 42 고정(재현 — 로컬 경로는 랜덤 시드 없이 42 통일). 다른 상황을 뽑으려면 SEED=<n>.
NUM_ENVS="${NUM_ENVS:-1}"; EPISODES="${EPISODES:--1}"; STEPS="${STEPS:-400}"; SEED="${SEED:-42}"
EXPORT="${EXPORT:-1}"     # 1 (DEFAULT) bake exported/{policy.pt,policy.onnx} after load · 0 = skip
# EPISODES: -1 (DEFAULT) infinite watch (Ctrl-C) · >0 stop after N episodes · 0 fixed STEPS.
# 기본 동작(--viz 없을 때): headless 로 1 에피소드를 돌리고 서버가 rerun .rrd + data.json + report.html 을
# 로컬 out-dir 에 씀. RECORD_ENVS 개 env 가 리포트 탭/시리즈를 받는다.
# --viz 면 대신 GUI 라이브 watch(녹화 off).
RECORD="${RECORD:-1}"; RECORD_ENVS="${RECORD_ENVS:-3}"

# ── args: --sim picks the backend (REQUIRED), --viz opens its viewer (boolean), --task/--num_envs/
# --checkpoint override the env-var defaults. Everything else is an env var (see header).
SIM=""; VIZ_ON=0
while [ $# -gt 0 ]; do
  case "$1" in
    --sim)          SIM="$2"; shift 2 ;;
    --sim=*)        SIM="${1#*=}"; shift ;;
    --viz)          VIZ_ON=1; shift ;;                    # boolean: open the --sim backend's viewer
    --task)         TASK="$2"; shift 2 ;;
    --task=*)       TASK="${1#*=}"; shift ;;
    --recipe)       RECIPE="$2"; shift 2 ;;
    --recipe=*)     RECIPE="${1#*=}"; shift ;;
    --num_envs)     NUM_ENVS="$2"; shift 2 ;;
    --num_envs=*)   NUM_ENVS="${1#*=}"; shift ;;
    --checkpoint)   CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    -h|--help)      sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "[eval] unknown arg '$1' (flags: --sim --viz --task --recipe --num_envs --checkpoint; rest are env vars)"; exit 1 ;;
  esac
done

# backend (REQUIRED, fail-loud): standalone in-process (genesis·newton), each in its own uv env.
case "$SIM" in
  genesis) export SIMULATOR=genesis SIM_ENGINE=genesis ;;
  newton)  export SIMULATOR=newton SIM_ENGINE=newton ;;
  "")      echo "[eval] --sim is required: --sim genesis | --sim newton" >&2; exit 2 ;;
  *)       echo "[eval] --sim must be genesis|newton (got '$SIM')" >&2; exit 2 ;;
esac
# eval goes through the RPC sim-service (RPC-only).
# --viz → GUI 라이브 watch(녹화 off; 뷰어+오프스크린 이중 GL 컨텍스트 회피). 미지정 → headless 녹화(로컬, 기본).
if [ "$VIZ_ON" = 1 ]; then VIZ=gl; RECORD=0; else VIZ=none; fi
# Headless GL for the offscreen recorder: newton's ViewerGL uses pyglet, which opens an X display at import
# (even with headless=True) and crashes on a display-less machine (NoSuchDisplayException). PYGLET_HEADLESS=1
# routes pyglet to EGL (hardware GL on the GPU, no X). Only when there's no DISPLAY → local --viz (needs X) untouched.
[ -z "${DISPLAY:-}" ] && export PYGLET_HEADLESS=1
# record: 각 env 를 개별 클립으로 녹화하므로 num_envs 는 최소 RECORD_ENVS 필요 — 부족하면 상향(BASENAME 반영 전).
[ "$RECORD" = 1 ] && [ "$NUM_ENVS" -lt "$RECORD_ENVS" ] && NUM_ENVS="$RECORD_ENVS"

# task (REQUIRED, fail-loud — metalab_train.sh 와 동일; 추론하지 않음): 미지정 시 가능한 task 를 출력하고 종료.
if [ -z "$TASK" ]; then
  echo "[eval] --task is required (no default). Available tasks:" >&2
  list_tasks | sed 's/^/  - /' >&2 || true
  echo "  e.g.  learning/scripts/local/metalab_eval.sh --sim $SIM --task hammer-lift-teacher --recipe privileged" >&2
  exit 2
fi
require_task_recipe eval "$TASK" "$RECIPE"

# Pin the eval to a physical GPU (CUDA_VISIBLE_DEVICES → that GPU is cuda:0 in-process). Evaluating
# DURING a training run (which holds GPU 0)? set GPU=1 to use a free GPU.
GPU="${GPU:-}"
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"

# --- resolve the checkpoint (LOCAL only). --checkpoint/CHECKPOINT wins; else newest local model_*.pt
#     for --task under $RL_LOG_ROOT (needs --task). ---
CKPT="${CHECKPOINT:-}"
if [ -z "$CKPT" ]; then
  CKPT="$(ls -t "$RL_LOG_ROOT"/*/*"$TASK"*/model_*.pt 2>/dev/null | head -1 || true)"
  [ -n "$CKPT" ] || { echo "[eval] $RL_LOG_ROOT 아래 task '$TASK' 체크포인트 없음 — 먼저 학습하거나 --checkpoint <path>" >&2; exit 1; }
  log "latest local checkpoint: $CKPT"
fi
[ -f "$CKPT" ] || { echo "[eval] checkpoint not found: $CKPT" >&2; exit 1; }

# --- provenance (from the local checkpoint path: <run_name>/model_<it>.pt) ---
RUN_NAME="$(basename "$(dirname "$CKPT")")"
EXPNAME="${EXPNAME:-$TASK}"
EVAL_DATE="$(date -u +%Y%m%d)"
EVAL_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
git -C "$ROOT" diff --quiet HEAD 2>/dev/null || EVAL_SHA="${EVAL_SHA}-dirty"
CKPT_ITER="$(basename "$CKPT")"; CKPT_ITER="${CKPT_ITER#model_}"; CKPT_ITER="${CKPT_ITER%.pt}"
TRAIN_TAG="${TRAIN_TAG:-${RUN_NAME}-iter${CKPT_ITER}}"
BASENAME="seed${SEED}_n${NUM_ENVS}"
ITER_TAG="$(printf 'iter%06d' "$CKPT_ITER" 2>/dev/null || echo "iter${CKPT_ITER}")"

# --- run the eval locally, in-process, in the engine's uv env ---
# ensure the engine's uv env exists & matches uv.lock (clone pinned source + `uv sync`; no-op if current)
bash "$ROOT/learning/scripts/local/setup_env.sh" --sim "$SIMULATOR"
_VENV="$(engine_venv "$SIMULATOR")"
source "$_VENV/bin/activate"
OUT_DIR="${OUT_DIR:-$ROOT/_logs/eval/$EVAL_DATE/$RUN_NAME/$ITER_TAG}"
mkdir -p "$OUT_DIR"
META_OUT="$OUT_DIR/$BASENAME.json"

[ "$EXPORT" = "1" ] && EXPORT_FLAG="--export" || EXPORT_FLAG="--no-export"
# 녹화 모드(기본): 한 에피소드 윈도우를 돌리고 서버가 rerun .rrd 를 쓴다. data.json + report.html 이 그 옆에
# 생성되고(리포트의 3D 패널이 그 .rrd 를 재생), 결과물은 로컬 $VIDEO_DIR 에 보관. RRD=0 이면 녹화 없이 SR 만.
VIDEO_DIR="$OUT_DIR/video"
RECORD_ARGS=()
[ "$RECORD" = 1 ] && RECORD_ARGS=(--record --record_envs "$RECORD_ENVS")
[ "$RECORD" = 1 ] && [ "${RRD:-1}" = 1 ] && RECORD_ARGS+=(--rrd "$VIDEO_DIR/rollout.rrd")
log "eval over RPC sim-service (sim=$SIMULATOR env=$_VENV task=$TASK${RECIPE:+/$RECIPE} num_envs=$NUM_ENVS seed=$SEED viz=$VIZ export=$EXPORT record=$RECORD)"
[ "$RECORD" = 1 ] && log "record (local): ${RECORD_ENVS} env → $VIDEO_DIR/{rollout.rrd,data.json,report.html}"
cd "$ROOT"
# SIM selects the sim package for the service spawn (learning/rl/service.py → sim/$SIM/launch.py);
# SIM_ENGINE (exported above) is MetaLab's own engine knob read by sim/metalab/launch.py.
export SIM=metalab
python -m learning.eval.eval_service --policy actor --experiment_pkg dexblind --curriculum_end \
  --task "$TASK" ${RECIPE:+--recipe "$RECIPE"} --checkpoint "$CKPT" \
  --num_envs "$NUM_ENVS" --episodes "$EPISODES" --steps "$STEPS" --seed "$SEED" \
  --viz "$VIZ" --meta_out "$META_OUT" $EXPORT_FLAG "${RECORD_ARGS[@]}" \
  --experiment "$EXPNAME" --eval_sha "$EVAL_SHA" --train "$TRAIN_TAG" --eval_date "$EVAL_DATE"

# The recording + meta are kept local — point the user at them.
if [ "$RECORD" = 1 ]; then
  if [ -f "$VIDEO_DIR/report.html" ]; then
    log "recorded → $VIDEO_DIR/report.html (local; meta $META_OUT)"
  else
    log "WARN: no report produced — see the eval output above"
  fi
fi

log "DONE meta=$META_OUT"
