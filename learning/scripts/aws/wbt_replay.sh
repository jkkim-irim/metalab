#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# wbt_replay.sh — render the STORED WBT reference trajectories by OPEN-LOOP REPLAY of their recorded
# actions: RSI each env to a reference's captured setup, then drive the sim with that reference's recorded
# actions (server-side, --replay), recording one MP4 per env. These are the exact reference motions the
# WBT tracking policy is trained to reproduce — all successful hammer lifts (the collector keeps only
# successes). No policy/checkpoint involved. Given a provisioned node; publishes per-env MP4s to S3.
#
# Usage:
#   NODE=i-0abc... TRAJ_S3=s3://.../trajectories/<date>/<wbt-run>/ learning/scripts/aws/wbt_replay.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
LOG_TAG=wbtreplay
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

# WBT trajectory dataset (traj_*.npz carry action + setup + tracking_state) — the references to replay.
TRAJ_S3="${TRAJ_S3:?set TRAJ_S3=s3://.../trajectories/<date>/<wbt-run>/ (WBT traj_*.npz with action+setup)}"
TRAJ_REMOTE=/home/ubuntu/sim_trajectories
REF_REMOTE=/home/ubuntu/sim_references_replay
OUT_REMOTE="/home/ubuntu/sim_replay_out_$(date -u +%Y%m%d%H%M%S)_$$"   # unique per invocation — video dirs are never reused
VIDEO_S3="${VIDEO_S3:-s3://wirobotics-internal/chrisryu/sim_rl/track_eval_videos}"
NUM_ENVS="${NUM_ENVS:-16}"; EPISODES="${EPISODES:-16}"; SEED="${SEED:-42}"
TAG="${TAG:-wbt-references}"
DATE="$(date -u +%Y%m%d)"
SHA="$(git -C "$AWS_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
git -C "$AWS_DIR" diff --quiet HEAD 2>/dev/null || SHA="${SHA}-dirty"
RUN_DIR="$VIDEO_S3/$DATE/replay_${TAG}_${SHA}"

log "ensure node $NODE is set up (setup_sim_node.sh provision; no-op if provisioned)"
NODE_ID="$NODE" bash "$AWS_DIR/setup_sim_node.sh" provision

deploy_sim_and_learning

log "pull $TRAJ_S3 -> build references WITH recorded actions -> $REF_REMOTE"
"$SSH" "$DEST" "rm -rf $TRAJ_REMOTE $REF_REMOTE && mkdir -p $TRAJ_REMOTE && \
  aws s3 cp $TRAJ_S3 $TRAJ_REMOTE --recursive --exclude '*' --include 'traj_*.npz' && \
  source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") && \
  python -m learning.rl.tracking.reference --traj_dir $TRAJ_REMOTE --out_dir $REF_REMOTE ${REFERENCE_ARGS:-}"
"$SSH" "$DEST" "rm -rf $OUT_REMOTE && mkdir -p $OUT_REMOTE"

log "replay references open-loop (recorded actions) in the WBT env + record one MP4 per env (n=$NUM_ENVS)"
"$SSH" "$DEST" "source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") && \
  SIM_ROOT=/home/ubuntu python -m learning.eval.eval_service --replay --wbt --reference_dir $REF_REMOTE \
    --num_envs $NUM_ENVS --episodes $EPISODES --seed $SEED --video --video_dir $OUT_REMOTE \
    --experiment tracking" 2>&1

NVID="$("$SSH" "$DEST" "ls $OUT_REMOTE/env*.mp4 2>/dev/null | wc -l" | tr -dc '0-9' || echo 0)"
if [ "${NVID:-0}" -gt 0 ]; then
  log "publish $NVID reference-replay videos -> $RUN_DIR/"
  "$SSH" "$DEST" "aws s3 cp $OUT_REMOTE $RUN_DIR --recursive --exclude '*' --include 'env*.mp4'" < /dev/null >/dev/null
fi
log "SUMMARY reference replay done: videos=$NVID s3=$RUN_DIR"
