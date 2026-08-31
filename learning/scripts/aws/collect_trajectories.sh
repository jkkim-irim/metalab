#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# collect_trajectories.sh — DexDeepMimic P0: roll out a trained (blind) checkpoint over the sim-service
# client and collect SUCCESSFUL episodes as reference motions to track. Given a launched node, it calls
# setup_sim_node.sh to ensure the isaaclab env is set up (idempotent), deploys sim/isaaclab + learning
# over SSM-SSH, pulls the checkpoint, and runs learning.rl.tracking.collect_trajectories until
# >= MIN_TRAJ successes are collected. Each run publishes to S3 (internal, auth-gated), keyed by date +
# experiment + TRAIN run + COLLECT sha:
#   trajectories/<date>/<exp>_train-<train-run>_collect-<sha>/
#       traj_<i>.npz    # [T, dim] per obs group (privileged = object pose + joints + contacts) + action + reward
#       meta.json       # count + traj-length stats + obs layout + provenance
#
# Usage (targets a provisioned node):
#   NODE=i-0abc... learning/scripts/aws/collect_trajectories.sh                        # ref ckpt, >=100 trajs -> S3
#   NODE=i-0abc... MIN_TRAJ=200 learning/scripts/aws/collect_trajectories.sh           # collect more
#   NODE=i-0abc... CHECKPOINT=/home/ubuntu/.../model_X.pt CHECKPOINT_S3=s3://... learning/scripts/aws/collect_trajectories.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
LOG_TAG=collect
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

# Default: collect from the full-training reference checkpoint (run j2u675w7, model_999) at its canonical
# run-name S3 path — the policy whose successes are full lifts (eval SR 0.75, return ~12.2).
REF_CKPT_S3="${REF_CKPT_S3:-s3://wirobotics-internal/chrisryu/sim_rl/ckpts/2026-07-06-02-57_hammer-lift_1000it_1024envs-nogit/model_999.pt}"
TRAJ_S3="${TRAJ_S3:-s3://wirobotics-internal/chrisryu/sim_rl/trajectories}"  # dataset lands here (internal bucket)
NUM_ENVS="${NUM_ENVS:-64}"; MIN_TRAJ="${MIN_TRAJ:-100}"; MAX_STEPS="${MAX_STEPS:-6000}"; SEED="${SEED:-42}"
# Collector module. WBT (full-state tracking): set
#   COLLECTOR=learning.rl.tracking.collect_trajectories  (rolls out the same lift ckpt in the --wbt collect env).
COLLECTOR="${COLLECTOR:-learning.rl.tracking.collect_trajectories}"

# Provenance for the S3 layout — traceable to WHEN, which TRAIN run produced the policy, and which
# COLLECT code ran it.
EXPNAME="${EXPNAME:-hammer-lift}"
COLLECT_DATE="$(date -u +%Y%m%d)"
COLLECT_SHA="$(git -C "$AWS_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
git -C "$AWS_DIR" diff --quiet HEAD 2>/dev/null || COLLECT_SHA="${COLLECT_SHA}-dirty"
TRAIN_TAG="${TRAIN_TAG:-20260706-j2u675w7-iter999}"
# RUN_DIR gains _n<count> at publish time (the count exists only after collection) — the dataset
# size is then visible in the S3 path itself.
RUN_DIR_BASE="$TRAJ_S3/$COLLECT_DATE/${EXPNAME}_train-${TRAIN_TAG}_collect-${COLLECT_SHA}"

# Ensure the node's isaaclab env is set up (idempotent — restores the snapshot only if missing).
log "ensure node $NODE is set up (setup_sim_node.sh provision; no-op if already provisioned)"
NODE_ID="$NODE" bash "$AWS_DIR/setup_sim_node.sh" provision

deploy_sim_and_learning

# CKPT = node-local path to load; CKPT_S3 = its canonical S3 source (recorded in meta). Default pulls
# the reference ckpt; for a node-local CHECKPOINT, set CHECKPOINT_S3 to record its source.
if [ -n "${CHECKPOINT:-}" ]; then
  CKPT="$CHECKPOINT"; CKPT_S3="${CHECKPOINT_S3:-}"; log "collect from node-local checkpoint $CKPT"
else
  CKPT=/home/ubuntu/ref_ckpt.pt; CKPT_S3="$REF_CKPT_S3"
  log "pull reference checkpoint $REF_CKPT_S3 -> $CKPT"
  "$SSH" "$DEST" "aws s3 cp $REF_CKPT_S3 $CKPT"
fi

OUT_REMOTE=/home/ubuntu/sim_trajectories
"$SSH" "$DEST" "rm -rf $OUT_REMOTE && mkdir -p $OUT_REMOTE"   # fresh dir each run

log "collect over the service-client (env=$ENV_NAME num_envs=$NUM_ENVS min_trajectories=$MIN_TRAJ seed=$SEED)"
COLLECT_LOG="$(mktemp)"
"$SSH" "$DEST" "source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") && \
  ${WBT_COLLECT_HOLD_STEPS:+WBT_COLLECT_HOLD_STEPS=$WBT_COLLECT_HOLD_STEPS }${WBT_COLLECT_CLAIM_ONLY:+WBT_COLLECT_CLAIM_ONLY=$WBT_COLLECT_CLAIM_ONLY }SIM_ROOT=/home/ubuntu python -m $COLLECTOR --checkpoint $CKPT ${CKPT_S3:+--checkpoint_s3 $CKPT_S3} \
    --num_envs $NUM_ENVS --min_trajectories $MIN_TRAJ --max_steps $MAX_STEPS --seed $SEED \
    --out_dir $OUT_REMOTE --experiment $EXPNAME --eval_sha $COLLECT_SHA --train $TRAIN_TAG \
    --eval_date $COLLECT_DATE" 2>&1 | tee "$COLLECT_LOG"
OK_LINE="$(grep 'COLLECT_OK' "$COLLECT_LOG" | tail -1 || true)"

# publish the dataset (traj_*.npz + meta.json) to S3, one recursive cp straight from the node
NTRAJ="$("$SSH" "$DEST" "ls $OUT_REMOTE/traj_*.npz 2>/dev/null | wc -l" | tr -dc '0-9' || echo 0)"
RUN_DIR="${RUN_DIR_BASE}_n${NTRAJ}"
if [ "${NTRAJ:-0}" -gt 0 ]; then
  log "upload $NTRAJ trajectories + meta -> $RUN_DIR/"
  "$SSH" "$DEST" "aws s3 cp $OUT_REMOTE $RUN_DIR --recursive --exclude '*' --include 'traj_*.npz' --include 'meta.json'" < /dev/null >/dev/null
  OUT_DIR="${OUT_DIR:-$AWS_DIR/out}"; mkdir -p "$OUT_DIR"
  rsync -a -e "$SSH" "$DEST:$OUT_REMOTE/meta.json" "$OUT_DIR/collect_meta.json" 2>/dev/null || \
    "$SSH" "$DEST" "cat $OUT_REMOTE/meta.json" > "$OUT_DIR/collect_meta.json"
else
  log "WARN: no trajectories produced — see the collect log above"
fi
[ -n "${OK_LINE:-}" ] && log "SUMMARY ${OK_LINE#COLLECT_OK } run_dir=$RUN_DIR trajectories=$NTRAJ"
