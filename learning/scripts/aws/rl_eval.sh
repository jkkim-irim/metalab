#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# rl_eval.sh — evaluate a checkpoint over the sim-service client boundary. Given a launched node, it
# calls setup_sim_node.sh to ensure the isaaclab env is set up (idempotent — a no-op if already provisioned),
# then deploys sim/isaaclab + learning over SSM-SSH (transport in lib.sh) and runs the eval. Fully
# reproducible from the S3 env snapshot alone: the eval runs in the isaaclab conda env (a superset with
# torch / tensordict — no separate venv), no project code on S3. It spawns server.py, drives it via
# SimServiceVecEnv over RPC, rebuilds the actor from the checkpoint, and runs a fixed EPISODES=64 sample
# (NUM_ENVS=64 → one wave), reporting SR = successes / episodes (the headline metric). By default it also
# records ONE MP4 PER ENV (each env in isolation at the close 3/4 angle, its own file — a tiled grid made
# each too small to watch); VIDEO=0 skips them.
# Each run gets a dir on S3 (internal, auth-gated), keyed by date + experiment + TRAIN run + EVAL sha:
#   eval_videos/<date>/<exp>_train-<train-run>_eval-<eval_sha>/
#       seed<S>_n<N>.json                         # SR + per-episode success/fail table + config + provenance
#       seed<S>_n<N>/env<i>_{success,fail}.mp4    # one video per env (its first episode)
#
# Usage (targets a provisioned node):
#   NODE=i-0abc... learning/scripts/aws/rl_eval.sh                                 # pull ref ckpt, eval 64 eps -> S3
#   NODE=i-0abc... CHECKPOINT=/home/ubuntu/.../model_X.pt learning/scripts/aws/rl_eval.sh   # eval a node-local ckpt
#   NODE=i-0abc... EPISODES=128 learning/scripts/aws/rl_eval.sh                             # larger sample
#   NODE=i-0abc... VIDEO=0 learning/scripts/aws/rl_eval.sh                                  # SR + S/F table only
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
LOG_TAG=eval
AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$AWS_DIR/lib.sh"

NODE="${NODE:?set NODE=i-<instance-id> (a provisioned GPU node)}"
DEST="ubuntu@${NODE}"
CONDA="/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
SSH="$(ensure_ssm_ssh_wrapper)"   # self-contained SSM-SSH wrapper (rsync/ssh ride it)

SIM_DIR="$(cd "$AWS_DIR/../../../sim/isaaclab" && pwd)"   # <repo>/sim/isaaclab
LEARNING_DIR="$(cd "$AWS_DIR/../.." && pwd)"          # <repo>/learning
SIM_REMOTE="/home/ubuntu/sim/isaaclab"
LEARNING_REMOTE="/home/ubuntu/learning_repo/learning"
ENV_NAME="${SIM_CONDA_ENV:-isaaclab}"

# Reference checkpoint = run hdtb23tj iter 400, at its canonical run-name path (traceable to the run that
# produced it — same convention rl_train.sh publishes to: ckpts/<run_name>/model_<iter>.pt).
REF_CKPT_S3="${REF_CKPT_S3:-s3://wirobotics-internal/chrisryu/sim_rl/ckpts/2026-07-03-05-24_hammer-lift_1000it_1024envs-nogit/model_400.pt}"
VIDEO_S3="${VIDEO_S3:-s3://wirobotics-internal/chrisryu/sim_rl/eval_videos}"  # eval MP4s land here (internal bucket)
# Eval a fixed, comprehensible sample: EPISODES=64. With NUM_ENVS=64 that's one wave (each env one
# episode) → SR over 64 AND all 64 envs in the grid video. SR = successes / episodes; STEPS is only a
# safety cap (the run stops when EPISODES complete).
NUM_ENVS="${NUM_ENVS:-64}"; EPISODES="${EPISODES:-64}"; STEPS="${STEPS:-400}"; SEED="${SEED:-42}"
# Record + upload the rollout video by default ("a run without a video didn't happen"); set VIDEO=0 to skip.
if [ "${VIDEO:-1}" = "0" ]; then VIDEO=""; else VIDEO="1"; fi

# Provenance for the S3 layout — a run is traceable to WHEN it ran, which TRAIN run produced the policy,
# and which EVAL code ran it (see the run-dir naming below).
EXPNAME="${EXPNAME:-hammer-lift}"
EVAL_DATE="$(date -u +%Y%m%d)"
EVAL_SHA="$(git -C "$AWS_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
git -C "$AWS_DIR" diff --quiet HEAD 2>/dev/null || EVAL_SHA="${EVAL_SHA}-dirty"
# Train provenance: auto-parsed from a wandb run dir in CHECKPOINT (run-<YYYYMMDD>_<hhmmss>-<runid>/…/model_<it>.pt);
# for the S3 ref (no wandb path) it falls back to TRAIN_TAG — the ref's provenance; update it when promoting a new ref.
TRAIN_TAG="${TRAIN_TAG:-20260703-hdtb23tj-iter400}"
if [[ "${CHECKPOINT:-}" =~ run-([0-9]{8})_[0-9]+-([A-Za-z0-9]+) ]]; then
  TRAIN_TAG="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
  [[ "${CHECKPOINT:-}" =~ model_([0-9]+)\.pt ]] && TRAIN_TAG="${TRAIN_TAG}-iter${BASH_REMATCH[1]}"
fi
# One dir per eval run, keyed by date + experiment + TRAIN run + EVAL sha — so the dir name alone says
# which policy (training run) was evaluated by which eval code. TRAIN_TAG carries the train run name (+date+iter).
RUN_DIR="$VIDEO_S3/$EVAL_DATE/${EXPNAME}_train-${TRAIN_TAG}_eval-${EVAL_SHA}"
BASENAME="seed${SEED}_n${NUM_ENVS}"

# Ensure the node's isaaclab env is set up (idempotent — restores the snapshot only if missing).
log "ensure node $NODE is set up (setup_sim_node.sh provision; no-op if already provisioned)"
NODE_ID="$NODE" bash "$AWS_DIR/setup_sim_node.sh" provision

deploy_sim_and_learning

# CKPT = node-local path to load; CKPT_S3 = its canonical S3 source (recorded in the meta, not the local
# path). Default evals the reference ckpt (source = REF_CKPT_S3); for a node-local CHECKPOINT, set
# CHECKPOINT_S3 to record its source (else the meta falls back to the local path).
if [ -n "${CHECKPOINT:-}" ]; then
  CKPT="$CHECKPOINT"; CKPT_S3="${CHECKPOINT_S3:-}"; log "eval node-local checkpoint $CKPT"
else
  CKPT=/home/ubuntu/ref_ckpt.pt; CKPT_S3="$REF_CKPT_S3"
  log "pull reference checkpoint $REF_CKPT_S3 -> $CKPT"
  "$SSH" "$DEST" "aws s3 cp $REF_CKPT_S3 $CKPT"
fi

OUT_REMOTE=/home/ubuntu/sim_eval_out
META_REMOTE="$OUT_REMOTE/meta.json"
"$SSH" "$DEST" "rm -rf $OUT_REMOTE && mkdir -p $OUT_REMOTE"   # fresh dir each run (per-episode clips + meta)
VIDEO_ARGS=""; [ -n "${VIDEO:-}" ] && VIDEO_ARGS="--video"

log "eval over the service-client (env=$ENV_NAME num_envs=$NUM_ENVS episodes=$EPISODES seed=$SEED${VIDEO:+ +video})"
EVAL_LOG="$(mktemp)"
"$SSH" "$DEST" "source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") && \
  SIM_ROOT=/home/ubuntu python -m learning.eval.eval_service --checkpoint $CKPT ${CKPT_S3:+--checkpoint_s3 $CKPT_S3} \
    --num_envs $NUM_ENVS --episodes $EPISODES --steps $STEPS --seed $SEED --video_dir $OUT_REMOTE \
    --meta_out $META_REMOTE --experiment $EXPNAME --eval_sha $EVAL_SHA --train $TRAIN_TAG \
    --eval_date $EVAL_DATE $VIDEO_ARGS" 2>&1 | tee "$EVAL_LOG"
SR_LINE="$(grep 'EVAL_OVER_SERVICE_OK' "$EVAL_LOG" | tail -1 || true)"
OUT_DIR="${OUT_DIR:-$AWS_DIR/out}"; mkdir -p "$OUT_DIR"

# --- meta json (SR + per-episode success/fail table + train/eval provenance), authored by eval_service ---
if "$SSH" "$DEST" "test -f $META_REMOTE"; then
  rsync -a -e "$SSH" "$DEST:$META_REMOTE" "$OUT_DIR/$BASENAME.json" 2>/dev/null || \
    "$SSH" "$DEST" "cat $META_REMOTE" > "$OUT_DIR/$BASENAME.json"
  log "upload metadata -> $RUN_DIR/$BASENAME.json"
  aws s3 cp "$OUT_DIR/$BASENAME.json" "$RUN_DIR/$BASENAME.json"
else
  log "WARN: no meta json produced (see the eval log above)"
fi

# --- per-env videos (env<i>_{success,fail}.mp4, one per env): publish all to a subdir by the meta json ---
NVID=0
if [ -n "${VIDEO:-}" ]; then
  NVID="$("$SSH" "$DEST" "ls $OUT_REMOTE/env*.mp4 2>/dev/null | wc -l" | tr -dc '0-9' || echo 0)"
  if [ "${NVID:-0}" -gt 0 ]; then
    log "upload $NVID per-env videos -> $RUN_DIR/$BASENAME/"
    # one recursive cp straight from the node (avoids ssh-in-a-loop eating stdin)
    "$SSH" "$DEST" "aws s3 cp $OUT_REMOTE $RUN_DIR/$BASENAME --recursive --exclude '*' --include 'env*.mp4'" < /dev/null >/dev/null
    rsync -a -e "$SSH" --include='env*.mp4' --exclude='*' "$DEST:$OUT_REMOTE/" "$OUT_DIR/$BASENAME/" 2>/dev/null || true
  else
    log "WARN: no per-env videos produced — see the eval log above"
  fi
fi
# one-line summary: SR (headline metric) + the run dir it published to
[ -n "${SR_LINE:-}" ] && log "SUMMARY ${SR_LINE#EVAL_OVER_SERVICE_OK } run_dir=$RUN_DIR${VIDEO:+ videos=$NVID}"
