#!/usr/bin/env bash
# CLIENT-SIDE launcher (runs on your laptop) — deploy the trainer to a remote GPU node and start
# training there. Pairs with the node-side recipe:
#
#   learning/scripts/train.sh           runs ON the node  (venv -> sync dataset -> train)
#   learning/scripts/aws/train_aws.sh   runs HERE         (scp code over SSM -> trigger it)
#
# The code (committed HEAD of learning/ + pyproject.toml, ~25 KB) and your local ~/.netrc (W&B
# auth) are pushed via scp over the SSM SSH proxy (AWS-StartSSHSession) — no S3. Then an SSM Run
# Command (root) extracts the code, installs the W&B credential at /root/.netrc, and launches
# train.sh detached.
#
# Usage (AWS_PROFILE, INSTANCE_ID, SSH_KEY are required — no defaults):
#   AWS_PROFILE=<profile> INSTANCE_ID=i-... SSH_KEY=~/.ssh/<key> \
#     bash learning/scripts/aws/train_aws.sh
# Env (optional): AWS_REGION (us-east-1), REMOTE_DIR, REMOTE_LOG, EXTRA_FLAGS (to the node script),
#   CUDA_VISIBLE_DEVICES (pin GPU(s), e.g. =1 to avoid a busy GPU 0), DATASET_S3,
#   POLICY (act|act_s1|groot; default act — act_s1 runs train_act_s1.sh, groot runs train_groot.sh),
#   HF_TOKEN_FILE (push a local token
#   for GR00T's gated nvidia/Cosmos-Reason2-2B backbone to the node). Any of these run-config vars,
#   if set locally, are forwarded to the node recipe (see NODE_ENV_VARS): STEPS, BATCH_SIZE, GPUS,
#   ZERO_STAGE, LR, TUNE_LLM, TUNE_VISUAL, VAL_EVERY, VAL_RATIO, VAL_MAX_BATCHES,
#   VAL_EVAL_MAX_BATCHES, IMAGE_AUG, OUTPUT_DIR.
set -euo pipefail

: "${INSTANCE_ID:?set INSTANCE_ID=i-... (the target GPU node)}"
export AWS_PROFILE="${AWS_PROFILE:?set AWS_PROFILE=<profile> (be explicit about the account)}"
: "${SSH_KEY:?set SSH_KEY=/path/to/your/ssh/private/key (for scp over the SSM proxy)}"
# AWS_PROFILE must belong to the gpu-launchers group:
WHO=$(aws sts get-caller-identity --query Arn --output text)
aws iam list-groups-for-user --user-name "${WHO##*/}" \
  --query 'Groups[].GroupName' --output text | grep -qw gpu-launchers \
  || { echo "REFUSING: $WHO is not in the gpu-launchers group." >&2; exit 1; }
export AWS_REGION="${AWS_REGION:-us-east-1}"
REMOTE_DIR="${REMOTE_DIR:-/opt/dlami/nvme/allex}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# POLICY selects the node-side recipe: act (default) -> train.sh; groot -> train_groot.sh.
# GR00T extra node prereqs (one-time): the Isaac-GR00T uv venv + an HF token for the gated
# nvidia/Cosmos-Reason2-2B backbone (see train_groot.sh header). Provide HF_TOKEN_FILE to push a
# local token to the node's HF_HOME/token; otherwise the token must already be on the node.
POLICY="${POLICY:-act}"
case "$POLICY" in
  act)    NODE_SCRIPT="learning/scripts/train.sh";        DEF_LOG="/opt/dlami/nvme/allex_train.log" ;;
  act_s1) NODE_SCRIPT="learning/scripts/train_act_s1.sh"; DEF_LOG="/opt/dlami/nvme/act_s1_train.log" ;;
  groot)  NODE_SCRIPT="learning/scripts/train_groot.sh";  DEF_LOG="/opt/dlami/nvme/groot_train.log" ;;
  *)      echo "REFUSING: unknown POLICY=$POLICY (want act|act_s1|groot)." >&2; exit 1 ;;
esac
LOG="${REMOTE_LOG:-$DEF_LOG}"
HF_HOME_NODE="${HF_HOME_NODE:-/opt/dlami/nvme/hf_cache}"
# Forward an allowlist of node-recipe env vars (train.sh / train_groot.sh read these from the
# environment) to the remote launch, single-quoting each value. Only vars set locally are forwarded.
#   DATASET_S3            : target a dataset other than the recipe default
#   CUDA_VISIBLE_DEVICES  : pin the run to specific GPU(s) (e.g. =1 to dodge a busy GPU 0)
#   STEPS BATCH_SIZE GPUS : total optimizer steps / per-GPU batch / num processes
#   ZERO_STAGE LR         : DeepSpeed ZeRO stage (groot full-VLM finetune) / optimizer LR
#   TUNE_LLM TUNE_VISUAL  : unfreeze the GR00T VLM backbone submodules
#   VAL_EVERY VAL_RATIO VAL_MAX_BATCHES VAL_EVAL_MAX_BATCHES : validation cadence / sizing
#   IMAGE_AUG             : off|cpu|gpu image-augmentation selector
#   OUTPUT_DIR            : checkpoint/output dir on the node
#   SIM_EVAL_*            : in-training closed-loop sim eval (groot) — cadence/suite/tasks/episodes/
#                          the SIM venv python + knobs; off unless SIM_EVAL_EVERY>0 (see train_groot.sh)
NODE_ENV_VARS="DATASET_S3 CUDA_VISIBLE_DEVICES STEPS BATCH_SIZE GPUS ZERO_STAGE LR \
  TUNE_LLM TUNE_VISUAL VAL_EVERY VAL_RATIO VAL_MAX_BATCHES VAL_EVAL_MAX_BATCHES IMAGE_AUG OUTPUT_DIR \
  WANDB_PROJECT FK_VIDEO_EVERY URDF_PATH \
  SIM_EVAL_EVERY SIM_EVAL_SUITE SIM_EVAL_TASKS SIM_EVAL_EPISODES SIM_EVAL_SIM_PYTHON \
  SIM_EVAL_RESOLUTION SIM_EVAL_MAX_EPISODE_STEPS SIM_EVAL_REPLAN_STEPS SIM_EVAL_BASE_SEED \
  SIM_EVAL_INFERENCE_TIMESTEPS SIM_EVAL_VIDEO_EPISODES SIM_EVAL_VIDEO_FPS \
  MUJOCO_GL"
ENV_FWD=""
for v in $NODE_ENV_VARS; do
  if [ -n "${!v:-}" ]; then ENV_FWD+="export $v='${!v}'; "; fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"

# 1) scoped archive of committed HEAD: the trainer + package metadata. Deliberately NOT the whole
#    repo — that pulls ~300 MB of tracked .usd sim assets ACT never uses. (The S1 recipe's pinned FK
#    URDF is vendored under learning/assets/ and rides along in `learning`; act/groot leave FK opt-in.)
# Per-invocation client-side temp paths (keyed on the target instance) so concurrent deploys to
# different nodes don't clobber each other's archive / SSM params — a shared /tmp path caused a
# config mix-up when two deploys ran at once.
LOCAL_TGZ="/tmp/allex_code_${INSTANCE_ID}.tgz"
LOCAL_PARAMS="/tmp/allex_ssm_params_${INSTANCE_ID}.json"
# GR00T in-training sim-eval also needs the sim SERVER + wire protocol on the node: sim/<suite>/
# (server + env proxy) and sim/service/transport.py (the unified RPC transport, imported by both the
# server and the client — the MetaLab restructure extracted it from sim/isaaclab/ into sim/service/).
# Add sim/libero + sim/maniskill + transport.py ONLY — NOT all of sim/isaaclab (418 MB of .usd hammer
# assets the sim eval never touches). Add other sim backends' sim/<x>/ dirs here as they land.
git -C "$REPO_ROOT" archive --format=tar.gz -o "$LOCAL_TGZ" HEAD \
  learning sim/libero sim/maniskill sim/service/transport.py pyproject.toml

# 2) push the code + the W&B credential (local ~/.netrc) over the SSM SSH proxy (no S3)
PXY='aws ssm start-session --target %h --document-name AWS-StartSSHSession'
PXY="$PXY --parameters portNumber=%p"
echo "scp $SHA + ~/.netrc -> ubuntu@$INSTANCE_ID (over SSM, POLICY=$POLICY)"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o "ProxyCommand=$PXY" \
    "$LOCAL_TGZ" "ubuntu@$INSTANCE_ID:/tmp/allex_code.tgz"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o "ProxyCommand=$PXY" \
    "$HOME/.netrc" "ubuntu@$INSTANCE_ID:/home/ubuntu/.netrc"
# Optional: provision the gated-backbone HF token for GR00T (scp the file content; not an SSM
# command param, so it never lands in SSM command history). Otherwise the token must be on the node.
TOKEN_CP=""
if [ "$POLICY" = "groot" ] && [ -n "${HF_TOKEN_FILE:-}" ]; then
  scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o "ProxyCommand=$PXY" \
      "$HF_TOKEN_FILE" "ubuntu@$INSTANCE_ID:/home/ubuntu/.hftoken"
  TOKEN_CP="mkdir -p $HF_HOME_NODE; cp /home/ubuntu/.hftoken $HF_HOME_NODE/token; chmod 600 $HF_HOME_NODE/token; rm -f /home/ubuntu/.hftoken; "
fi

# 3) extract + launch via SSM Run Command (root). /bin/sh on the node: no double-quotes / parens.
REMOTE="set -e; rm -rf $REMOTE_DIR; mkdir -p $REMOTE_DIR; "
REMOTE+="tar xzf /tmp/allex_code.tgz -C $REMOTE_DIR; "
REMOTE+="cp /home/ubuntu/.netrc /root/.netrc; chmod 600 /root/.netrc; "
REMOTE+="${TOKEN_CP}cd $REMOTE_DIR; ${ENV_FWD}export ALLEX_SHA='$SHA'; "
REMOTE+="export EXTRA_FLAGS='$EXTRA_FLAGS'; "
REMOTE+="setsid bash $NODE_SCRIPT > $LOG 2>&1 < /dev/null & sleep 4; "
REMOTE+="echo LAUNCHED $SHA $POLICY; pgrep -af learning.train | head -1; tail -n 8 $LOG 2>/dev/null"
printf '{"commands":["%s"]}\n' "$REMOTE" > "$LOCAL_PARAMS"
CMD_ID="$(aws ssm send-command --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --parameters "file://$LOCAL_PARAMS" \
  --query 'Command.CommandId' --output text)"

echo "Launched on $INSTANCE_ID (SSM $CMD_ID). Node log: $LOG"
