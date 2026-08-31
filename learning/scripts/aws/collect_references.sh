#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# collect_references.sh — the reference-DATASET pipeline, one command, fail-loudly:
#   1. collect N_TRAJ gate-passing episodes of SOURCE_CKPT_S3 over the sim service in
#      CLAIM-ONLY mode (WBT_COLLECT_CLAIM_ONLY=1: the gate latches but does not terminate, so
#      episodes run past the claim and the recording keeps the FULL hold window — a
#      success-terminating episode clips it: the 20260709 n2000 set shipped 1825/2000 refs one
#      frame short of the gate);
#   2. build ref_*.npz on the node (--trim_head 2 --min_lift_cm MIN_LIFT_CM);
#   3. VALIDATE the built artifact (learning.rl.tracking.reference --validate): setup + variant
#      present, lift >= MIN_LIFT_CM, hold window crosses the gate — any violation fails the
#      pipeline before anything trains on it;
#   4. the trajectory set is published to S3 by the collection driver (step 1) — the final
#      DATASET_S3 line is the provenance pointer for launches and the PR.
#
# Usage (targets a provisioned node; collection needs the GPU):
#   NODE=i-… SOURCE_CKPT_S3=s3://…/model_1600.pt N_TRAJ=2000 TRAIN_TAG=<source-run-tag> \
#     learning/scripts/aws/collect_references.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
LOG_TAG=collect-refs
AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$AWS_DIR/lib.sh"

NODE="${NODE:?set NODE=i-<instance-id> (a provisioned GPU node)}"
SOURCE_CKPT_S3="${SOURCE_CKPT_S3:?set SOURCE_CKPT_S3=s3://.../model_N.pt (the policy whose successes become references)}"
N_TRAJ="${N_TRAJ:-2000}"
MIN_LIFT_CM="${MIN_LIFT_CM:-8}"
TRAIN_TAG="${TRAIN_TAG:?set TRAIN_TAG=<source-run tag> (keys the S3 dataset path)}"
DEST="ubuntu@${NODE}"
CONDA="/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
SSH="$(ensure_ssm_ssh_wrapper)"
LEARNING_REMOTE="/home/ubuntu/learning_repo/learning"
COLLECT_LOG="$(mktemp)"

# Relay the source checkpoint through THIS host: node instance roles often cannot read another
# owner's S3 prefix (403 on jkkim's ckpts, measured), while the invoking host typically can. The
# node never needs read access to the source bucket; provenance still records the S3 path.
log "step 0/3: relay $SOURCE_CKPT_S3 -> node (invoking-host creds; node roles may lack cross-owner read)"
RELAY_LOCAL="$(mktemp --suffix=.pt)"
aws s3 cp "$SOURCE_CKPT_S3" "$RELAY_LOCAL" --only-show-errors
scp -q -o ConnectTimeout=30 "$RELAY_LOCAL" "$DEST:/home/ubuntu/ref_ckpt.pt"
rm -f "$RELAY_LOCAL"

log "step 1/3: collect $N_TRAJ claim-only successes of $SOURCE_CKPT_S3 (full hold windows recorded)"
NODE="$NODE" CHECKPOINT=/home/ubuntu/ref_ckpt.pt CHECKPOINT_S3="$SOURCE_CKPT_S3" MIN_TRAJ="$N_TRAJ" \
  NUM_ENVS="${NUM_ENVS:-1024}" MAX_STEPS="${MAX_STEPS:-20000}" TRAIN_TAG="$TRAIN_TAG" \
  COLLECTOR=learning.rl.tracking.collect_trajectories WBT_COLLECT_CLAIM_ONLY=1 \
  bash "$AWS_DIR/collect_trajectories.sh" 2>&1 | tee "$COLLECT_LOG"

DATASET_S3="$(grep -oE 's3://[^ ]*/trajectories/[^ ]+' "$COLLECT_LOG" | tail -1)"
[ -n "$DATASET_S3" ] || { log "FATAL: no published dataset path in the collection log"; exit 1; }

log "step 2/3 + 3/3: build refs on the node (trim 2, min-lift ${MIN_LIFT_CM}cm) + VALIDATE the artifact"
"$SSH" "$DEST" "source $CONDA && conda activate ${SIM_CONDA_ENV:-isaaclab} && cd \$(dirname $LEARNING_REMOTE) && \
  python -m learning.rl.tracking.reference --traj_dir /home/ubuntu/sim_trajectories --out_dir /home/ubuntu/sim_references \
    --trim_head 2 --min_lift_cm $MIN_LIFT_CM --max_post_claim ${MAX_POST_CLAIM:-5} --validate"

log "DATASET_S3=$DATASET_S3"
log "reference set built + validated at /home/ubuntu/sim_references on $NODE — train with TRAJ_S3=$DATASET_S3"
