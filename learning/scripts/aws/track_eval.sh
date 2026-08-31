#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# track_eval.sh — DexDeepMimic: eval a TRACKING checkpoint in the tracking env, render one MP4 per env,
# upload a sample to W&B (visual validation in the wandb UI) and publish all per-env videos to S3. Reuses
# the reference dataset already on the node (builds it from TRAJ_S3 if absent). Given a provisioned node.
#
# Usage:
#   NODE=i-0abc... CKPT_S3=s3://.../ckpts/<run>/model_N.pt TRAJ_S3=s3://.../trajectories/<date>/<run>/ \
#       learning/scripts/aws/track_eval.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
LOG_TAG=trackeval
AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$AWS_DIR/lib.sh"

NODE="${NODE:?set NODE=i-<instance-id> (a provisioned GPU node)}"
DEST="ubuntu@${NODE}"
CONDA="/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
SSH="$(ensure_ssm_ssh_wrapper)"

SIM_DIR="$(cd "$AWS_DIR/../../../sim/isaaclab" && pwd)"
LEARNING_DIR="$(cd "$AWS_DIR/../.." && pwd)"
SIM_REMOTE="/home/ubuntu/sim/isaaclab"
LEARNING_REMOTE="/home/ubuntu/learning_repo/learning"
ENV_NAME="${SIM_CONDA_ENV:-isaaclab}"

CKPT_S3="${CKPT_S3:?set CKPT_S3=s3://.../model_N.pt (a TRACKING checkpoint)}"
TRAJ_S3="${TRAJ_S3:?set TRAJ_S3=s3://.../trajectories/<date>/<run>/ (reference dataset source)}"
REF_REMOTE=/home/ubuntu/sim_references
TRAJ_REMOTE=/home/ubuntu/sim_trajectories
OUT_REMOTE="/home/ubuntu/sim_eval_out_$(date -u +%Y%m%d%H%M%S)_$$"   # unique per invocation — video dirs are never reused
VIDEO_S3="${VIDEO_S3:-s3://wirobotics-internal/chrisryu/sim_rl/track_eval_videos}"
NUM_ENVS="${NUM_ENVS:-16}"; EPISODES="${EPISODES:-16}"; SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-chrisryu-simrl}"; WANDB_VIDEO_N="${WANDB_VIDEO_N:-8}"
TRAIN_TAG="${TRAIN_TAG:-wbt-track}"
# Experiment + env variant. Default = DexDeepMimic tracking. WBT (full-state tracking):
#   EXPERIMENT=tracking VARIANT_FLAG=--wbt REFERENCE_MODULE=learning.rl.tracking.reference
EXPERIMENT="${EXPERIMENT:-tracking}"; VARIANT_FLAG="${VARIANT_FLAG:---wbt}"
REFERENCE_MODULE="${REFERENCE_MODULE:-learning.rl.tracking.reference}"
REFERENCE_ARGS="${REFERENCE_ARGS:-}"   # extra reference-build flags — MUST match the training build (e.g. "--min_lift_cm 8")
EVAL_DATE="$(date -u +%Y%m%d)"
EVAL_SHA="$(git -C "$AWS_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
git -C "$AWS_DIR" diff --quiet HEAD 2>/dev/null || EVAL_SHA="${EVAL_SHA}-dirty"
RUN_DIR="$VIDEO_S3/$EVAL_DATE/track_${TRAIN_TAG}_eval-${EVAL_SHA}"

log "ensure node $NODE is set up (setup_sim_node.sh provision; no-op if provisioned)"
NODE_ID="$NODE" bash "$AWS_DIR/setup_sim_node.sh" provision

deploy_sim_and_learning

log "ensure reference dataset on node ($REF_REMOTE); build from $TRAJ_S3 if absent"
"$SSH" "$DEST" "ls $REF_REMOTE/ref_0000.npz >/dev/null 2>&1 || { rm -rf $TRAJ_REMOTE $REF_REMOTE && mkdir -p $TRAJ_REMOTE && \
  aws s3 cp $TRAJ_S3 $TRAJ_REMOTE --recursive --exclude '*' --include 'traj_*.npz' && \
  source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") && \
  python -m $REFERENCE_MODULE --traj_dir $TRAJ_REMOTE --out_dir $REF_REMOTE $REFERENCE_ARGS; }"

CKPT=/home/ubuntu/track_ckpt.pt
log "pull tracking checkpoint $CKPT_S3 -> $CKPT"
"$SSH" "$DEST" "aws s3 cp $CKPT_S3 $CKPT"
"$SSH" "$DEST" "rm -rf $OUT_REMOTE && mkdir -p $OUT_REMOTE"

log "eval tracking policy in the tracking env (num_envs=$NUM_ENVS episodes=$EPISODES) + per-env video + wandb"
"$SSH" "$DEST" "source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") && \
  SIM_ROOT=/home/ubuntu python -m learning.eval.eval_service --checkpoint $CKPT --checkpoint_s3 $CKPT_S3 $VARIANT_FLAG --reference_dir $REF_REMOTE \
    --reference_s3 $TRAJ_S3 \
    --num_envs $NUM_ENVS --episodes $EPISODES --seed $SEED --video --video_dir $OUT_REMOTE \
    --wandb --wandb_project $WANDB_PROJECT --wandb_run track_${TRAIN_TAG}_${EVAL_SHA} --wandb_video_n $WANDB_VIDEO_N \
    --experiment $EXPERIMENT --train $TRAIN_TAG --eval_sha $EVAL_SHA --eval_date $EVAL_DATE" 2>&1

NVID="$("$SSH" "$DEST" "ls $OUT_REMOTE/env*.mp4 2>/dev/null | wc -l" | tr -dc '0-9' || echo 0)"
if [ "${NVID:-0}" -gt 0 ]; then
  log "publish $NVID per-env videos -> $RUN_DIR/"
  "$SSH" "$DEST" "aws s3 cp $OUT_REMOTE $RUN_DIR --recursive --exclude '*' --include 'env*.mp4'" < /dev/null >/dev/null
fi
log "SUMMARY tracking eval done: wandb project=$WANDB_PROJECT videos=$NVID s3=$RUN_DIR"
