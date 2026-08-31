#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# track_train.sh — DexDeepMimic P1: train the motion-tracking policy over the sim service. Given a
# launched node, self-provisions (setup_sim_node.sh, idempotent), deploys sim/isaaclab + learning, pulls
# the reference trajectory dataset from S3 (P0 collection WITH env setup), builds ref_*.npz on the node
# (learning.rl.tracking.reference), then trains the tracking policy through the sim's WBT env
# variant (learning.train --trainer rl --experiment tracking --wbt --reference_dir …). The
# tracking env RSIs each env to its assigned reference's recorded setup, so the reference is reachable.
#
# Usage (targets a provisioned node):
#   NODE=i-0abc... TRAJ_S3=s3://.../trajectories/<date>/<run>/ learning/scripts/aws/track_train.sh
#   NODE=i-0abc... TRAJ_S3=... MAX_ITERS=1000 NUM_ENVS=1024 learning/scripts/aws/track_train.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
LOG_TAG=track
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

# The reference trajectory dataset to track — a P0 collection that INCLUDES the per-episode env setup
# (traj_*.npz with a "setup" array); see learning/scripts/aws/collect_trajectories.sh.
TRAJ_S3="${TRAJ_S3:?set TRAJ_S3=s3://.../trajectories/<date>/<run>/ (re-collected traj_*.npz WITH setup)}"
TRAJ_REMOTE=/home/ubuntu/sim_trajectories
REF_REMOTE=/home/ubuntu/sim_references
CKPT_S3_ROOT="${CKPT_S3_ROOT:-s3://wirobotics-internal/chrisryu/sim_rl/ckpts}"

NUM_ENVS="${NUM_ENVS:-1024}"; MAX_ITERS="${MAX_ITERS:-5000}"; SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-chrisryu-simrl}"; LOGGER="${LOGGER:-wandb}"
# Experiment + env variant. Default = DexDeepMimic tracking. WBT (full-state tracking):
#   EXPERIMENT=tracking VARIANT_FLAG=--wbt REFERENCE_MODULE=learning.rl.tracking.reference
EXPERIMENT="${EXPERIMENT:-tracking}"; VARIANT_FLAG="${VARIANT_FLAG:---wbt}"
REFERENCE_MODULE="${REFERENCE_MODULE:-learning.rl.tracking.reference}"
REFERENCE_ARGS="${REFERENCE_ARGS:---trim_head 2 --min_lift_cm 8}"   # DEFAULT = the canonical gate-feasible build (trim + min-lift filter)

log "ensure node $NODE is set up (setup_sim_node.sh provision; no-op if already provisioned)"
NODE_ID="$NODE" bash "$AWS_DIR/setup_sim_node.sh" provision

deploy_sim_and_learning

if [ -n "${SKIP_REF_BUILD:-}" ]; then
  # Multi-run packing: a second run on the SAME reference set must not rm -rf the refs dir a
  # sibling's probes are reading mid-flight.
  log "SKIP_REF_BUILD set — using the reference set already at $REF_REMOTE"
else
log "pull reference trajectories $TRAJ_S3 -> $TRAJ_REMOTE, build ref_*.npz -> $REF_REMOTE"
"$SSH" "$DEST" "rm -rf $TRAJ_REMOTE $REF_REMOTE && mkdir -p $TRAJ_REMOTE && \
  aws s3 cp $TRAJ_S3 $TRAJ_REMOTE --recursive --exclude '*' --include 'traj_*.npz' && \
  source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") && \
  python -m $REFERENCE_MODULE --traj_dir $TRAJ_REMOTE --out_dir $REF_REMOTE $REFERENCE_ARGS"
fi

if [ -n "${RESUME_CKPT_S3:-}" ]; then
  log "pull resume checkpoint $RESUME_CKPT_S3 -> /home/ubuntu/resume_ckpt.pt"
  "$SSH" "$DEST" "aws s3 cp $RESUME_CKPT_S3 /home/ubuntu/resume_ckpt.pt"
fi
log "train tracking policy in $ENV_NAME: num_envs=$NUM_ENVS max_iterations=$MAX_ITERS seed=$SEED logger=$LOGGER"
TRAIN_LOG="$(mktemp)"
# The trainer runs under nohup with output to a NODE-side log; the SSH session only tails it.
# An SSM session drop then kills the tail, not the training (two launcher sessions died mid-run
# on 2026-07-09 — the first training survived SIGHUP by luck, the second did not).
"$SSH" "$DEST" "source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") ; \
  TL=/home/ubuntu/train_node_\$\$.log ; \
  ${WBT_ENTROPY_COEF:+WBT_ENTROPY_COEF=$WBT_ENTROPY_COEF }${WBT_RESUME_STD:+WBT_RESUME_STD=$WBT_RESUME_STD }${WBT_LR:+WBT_LR=$WBT_LR }\
  ${WBT_ENTROPY_SCHED:+WBT_ENTROPY_SCHED=$WBT_ENTROPY_SCHED }${WBT_STD_MAX:+WBT_STD_MAX=$WBT_STD_MAX }${WBT_PHASE_GUARD:+WBT_PHASE_GUARD=$WBT_PHASE_GUARD }\
  ${WBT_HOLD_PARTIAL_WEIGHT:+WBT_HOLD_PARTIAL_WEIGHT=$WBT_HOLD_PARTIAL_WEIGHT }${WBT_HOLD_PARTIAL_POWER:+WBT_HOLD_PARTIAL_POWER=$WBT_HOLD_PARTIAL_POWER }${WBT_CONTACT_WEIGHT:+WBT_CONTACT_WEIGHT=$WBT_CONTACT_WEIGHT }${WBT_HOLD_WEIGHT:+WBT_HOLD_WEIGHT=$WBT_HOLD_WEIGHT }${WBT_JOINT_WEIGHT:+WBT_JOINT_WEIGHT=$WBT_JOINT_WEIGHT }${WBT_JOINT_VEL_WEIGHT:+WBT_JOINT_VEL_WEIGHT=$WBT_JOINT_VEL_WEIGHT }${WBT_RSI_NOISE:+WBT_RSI_NOISE=$WBT_RSI_NOISE }${WBT_RSI_NOISE_MID:+WBT_RSI_NOISE_MID=$WBT_RSI_NOISE_MID }${WBT_REWARD_SET:+WBT_REWARD_SET=$WBT_REWARD_SET }\
  ${WBT_ACTOR_FREEZE:+WBT_ACTOR_FREEZE=$WBT_ACTOR_FREEZE }${WBT_TEACHER_PRIV:+WBT_TEACHER_PRIV=$WBT_TEACHER_PRIV }\
  ${WANDB_RUN_ID:+WANDB_RUN_ID=$WANDB_RUN_ID }${WANDB_RESUME:+WANDB_RESUME=$WANDB_RESUME }${WBT_RUN_TAG:+WBT_RUN_TAG=$WBT_RUN_TAG }${WBT_NOTES:+WBT_NOTES=$(printf '%q' "$WBT_NOTES") }nohup setsid bash -c \"SIM_ROOT=/home/ubuntu python -m learning.train --trainer rl --experiment $EXPERIMENT $VARIANT_FLAG --reference_dir $REF_REMOTE \
    --reference_s3 $TRAJ_S3 \
    --num_envs $NUM_ENVS --max_iterations $MAX_ITERS --seed $SEED --logger $LOGGER \
    --wandb_project $WANDB_PROJECT \
    ${RESUME_CKPT_S3:+--resume_ckpt /home/ubuntu/resume_ckpt.pt} \
    ${SAVE_INTERVAL:+--save_interval $SAVE_INTERVAL} \
    --val_video_every ${VAL_VIDEO_EVERY:-100} --val_eval_envs ${VAL_EVAL_ENVS:-64} --val_video_envs ${VAL_VIDEO_ENVS:-4} --val_video_n ${VAL_VIDEO_N:-4} \
    > \$TL 2>&1; echo TRAINER_EXIT rc=\\\$? at \\\$(date -u +%H:%M:%S) >> \$TL\" & TPID=\$! ; echo NODE_TRAIN_PID \$TPID LOG \$TL ; tail --pid=\$TPID -n +1 -f \$TL" 2>&1 | tee "$TRAIN_LOG"
    # (';' before TL and after the '&' matter: '&' backgrounds the whole preceding AND-OR list, so
    #  with '&&' chaining, TL/TPID were set only inside the backgrounded subshell and the foreground
    #  tail ran file-less on stdin — training ran fine but the launcher saw nothing)

# Publish the run's checkpoints to S3 under the run name (same convention as rl_train.sh).
LOG_DIR="$(grep -oE '\[rl-trainer\] log_dir=\S+' "$TRAIN_LOG" | tail -1 | sed 's/.*log_dir=//')"
if [ -n "$LOG_DIR" ]; then
  RUN_NAME="$(basename "$LOG_DIR")"
  log "publish checkpoints -> $CKPT_S3_ROOT/$RUN_NAME/ (run $RUN_NAME)"
  "$SSH" "$DEST" "aws s3 cp $LOG_DIR $CKPT_S3_ROOT/$RUN_NAME --recursive --exclude '*' --include 'model_*.pt'" < /dev/null \
    && log "TRACK_CKPTS_S3=$CKPT_S3_ROOT/$RUN_NAME/"
else
  log "WARN: no [rl-trainer] log_dir in the train log — checkpoints not published"
fi
