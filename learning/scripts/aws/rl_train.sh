#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# rl_train.sh — deploy the Newton RL sim service + trainer to a provisioned GPU node and train.
# Sources lib.sh for the SSM-SSH transport + rsync deploy (deploy_sim_and_learning), shared with
# setup_sim_node.sh / rl_eval.sh.
#
# TWO code deploys on the node, but ONE conda env: the snapshot's `isaaclab` env is a superset (it
# already has torch / tensordict / wandb), so the Isaac-free trainer runs in it too — nothing else to
# build, so a node is reproducible from the S3 env snapshot alone.
#   sim/isaaclab/  -> <node>:/home/ubuntu/sim/isaaclab            (server.py — the Isaac Lab env)
#   learning/      -> <node>:/home/ubuntu/learning_repo/learning  (the vendored PPO trainer + client)
# `python -m learning.train --trainer rl` spawns server.py over the boundary (learning/rl/service.py);
# the server builds the env from the sim-owned task knobs (sim/isaaclab/envs/hammer_lift/knobs.py) —
# the trainer ships only run args.
#
# Self-provisions the node (idempotent): calls setup_sim_node.sh to ensure the isaaclab env + snapshot are
# present, so there is no separate provision step — same as rl_eval.sh.
# Idempotent: rsync deltas + an editable reinstall each run so the install always matches the code.
#
# Trained checkpoints are published to S3 keyed by RUN NAME — $CKPT_S3_ROOT/<run_name>/model_*.pt — so a
# run's models are traceable and evalable (rl_eval.sh with CHECKPOINT=<local> CHECKPOINT_S3=<that path>).
#
# Usage:
#   NODE=i-0abc... learning/scripts/aws/rl_train.sh
#   NODE=i-0abc... NUM_ENVS=1024 MAX_ITERS=250 SEED=42 WANDB_PROJECT=chrisryu-simrl learning/scripts/aws/rl_train.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
LOG_TAG=train
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
# ONE env: the snapshot's isaaclab env is a superset, so BOTH server.py and the Isaac-free trainer
# run in it — no separate trainer venv to build (a fresh node stays reproducible from the snapshot).
ENV_NAME="${SIM_CONDA_ENV:-isaaclab}"

# Training args (overridable).
NUM_ENVS="${NUM_ENVS:-1024}"; MAX_ITERS="${MAX_ITERS:-5000}"; SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-chrisryu-simrl}"; LOGGER="${LOGGER:-wandb}"
# Trained checkpoints are published to S3 keyed by RUN NAME, so a run's models are traceable + evalable:
#   $CKPT_S3_ROOT/<run_name>/model_<iter>.pt   (eval with CHECKPOINT_S3=<that path>).
# Default is per-user under the caller's own S3 prefix ($AWS_USER, exported by the devbox config.env and
# used for Owner tags). Required: with no AWS_USER and no explicit CKPT_S3_ROOT the trainer would have
# nowhere durable to put checkpoints, so fail loudly here rather than train and lose them.
: "${AWS_USER:?set AWS_USER=<your-username> (devbox config.env exports it) or set CKPT_S3_ROOT=s3://... explicitly}"
CKPT_S3_ROOT="${CKPT_S3_ROOT:-s3://wirobotics-internal/${AWS_USER}/sim_rl/ckpts}"

# Ensure the node's isaaclab env is set up (idempotent — restores the snapshot only if missing), same as
# rl_eval.sh. No separate provision step needed.
log "ensure node $NODE is set up (setup_sim_node.sh provision; no-op if already provisioned)"
NODE_ID="$NODE" bash "$AWS_DIR/setup_sim_node.sh" provision

deploy_sim_and_learning

log "train in the $ENV_NAME env: num_envs=$NUM_ENVS max_iterations=$MAX_ITERS seed=$SEED logger=$LOGGER"
TRAIN_LOG="$(mktemp)"
# CKPT_S3_ROOT is exported into the trainer env so it uploads each model_*.pt to
# $CKPT_S3_ROOT/<run_name>/ the moment the checkpoint is written (OnPolicyRunner._upload_ckpt_s3) —
# durable immediately, not only via the end-of-run backstop below.
"$SSH" "$DEST" "source $CONDA && conda activate $ENV_NAME && cd $(dirname "$LEARNING_REMOTE") && \
  SIM_ROOT=/home/ubuntu CKPT_S3_ROOT='$CKPT_S3_ROOT' python -m learning.train --trainer rl --num_envs $NUM_ENVS --max_iterations $MAX_ITERS --seed $SEED \
  --logger $LOGGER --wandb_project $WANDB_PROJECT" 2>&1 | tee "$TRAIN_LOG"

# Backstop: re-publish the run's checkpoints to S3 under the run name in case any per-checkpoint upload
# failed (the trainer prints its log_dir; run_name = its basename). Models land at
# $CKPT_S3_ROOT/<run_name>/model_*.pt — traceable to the run, evalable via CHECKPOINT_S3.
# (< /dev/null: keep the node ssh from eating this script's stdin.)
LOG_DIR="$(grep -oE '\[rl-trainer\] log_dir=\S+' "$TRAIN_LOG" | tail -1 | sed 's/.*log_dir=//')"
if [ -n "$LOG_DIR" ]; then
  RUN_NAME="$(basename "$LOG_DIR")"
  log "publish checkpoints -> $CKPT_S3_ROOT/$RUN_NAME/ (run $RUN_NAME)"
  "$SSH" "$DEST" "aws s3 cp $LOG_DIR $CKPT_S3_ROOT/$RUN_NAME --recursive --exclude '*' --include 'model_*.pt'" < /dev/null \
    && log "TRAIN_CKPTS_S3=$CKPT_S3_ROOT/$RUN_NAME/"
else
  log "WARN: no [rl-trainer] log_dir in the train log — checkpoints not published to S3"
fi
