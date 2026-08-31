#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# lib.sh — shared config + helpers for the sim/isaaclab (Newton sim-service) AWS deploy toolkit.
#
# Source this from the other deploy scripts:  source "$(dirname "$0")/lib.sh"
# Every value below is overridable from the environment, e.g.:
#   AWS_PROFILE=my-gpu ROOT_GB=400 ./setup_sim_node.sh up
#
# Transport model: nodes have NO inbound ports. We reach them over SSH tunnelled
# through AWS SSM (Session Manager), which is also what rsync/scp ride on. This
# needs the `session-manager-plugin` installed locally (brew install --cask
# session-manager-plugin) and an AWS profile whose IAM allows ec2 run/stop,
# ssm:StartSession and ssm:Describe* on ManagedBy=gpu-launcher instances.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── AWS / instance config ─────────────────────────────────────────────────────
: "${AWS_PROFILE:=}"                 # your AWS profile (gpu-launcher + SSM perms); empty = default cred chain
: "${AWS_REGION:=us-east-1}"; export AWS_REGION
# L40S (g6e) / A10G (g5): have the RT cores Newton's headless renderer needs AND
# enough compute for 1024-env PPO. A100/H100/V100 CANNOT render Newton (no RT cores).
: "${INSTANCE_TYPES:=g6e.4xlarge g6e.2xlarge g5.4xlarge g5.2xlarge}"
: "${ROOT_GB:=300}"                  # gp3 root EBS; env snapshot (~12GB) + assets + checkpoints
: "${NODE_NAME:=${USER:-$(id -un)}-allex-newton}"  # username-prefixed so teammates don't collide on the Name tag
: "${OWNER:=${USER:-$(id -un)}}"     # tagged Owner=... on launch (who to ask before reusing/terminating)
: "${USAGE_PURPOSE:=training}"       # written into the node's /home/ubuntu/USAGE.lock
: "${IAM_INSTANCE_PROFILE:=project-x-ssm-profile}"  # MUST grant AmazonSSMManagedInstanceCore
: "${SECURITY_GROUP:=}"              # SG id; empty = account default VPC SG (SSM needs only egress 443)
: "${SUBNET:=}"                      # empty = let EC2 choose an AZ that has capacity
: "${LAUNCH_ATTEMPTS:=8}"            # capacity-retry rounds
: "${RETRY_SLEEP:=30}"

# ── SSH-over-SSM transport (rsync/scp ride this) ──────────────────────────────
: "${SSH_USER:=ubuntu}"
: "${SSH_PUBKEY:=$HOME/.ssh/id_ed25519.pub}"  # YOUR public key — injected into the node via user-data
: "${SSH_KEY:=${SSH_PUBKEY%.pub}}"            # the matching private key

# ── Remote (node) layout ──────────────────────────────────────────────────────
# The Newton stack is KITLESS (custom IsaacLab fork + newton + mujoco-warp; NO Isaac Sim) and is not
# cleanly pip-installable, so provisioning RESTORES a self-contained S3 env snapshot (conda env +
# IsaacLab fork) — see remote/provision.sh. The snapshot is a standalone artifact; nothing here references
# any specific/running box. The repo's sim/isaaclab/ project is rsync'd into the fork's projects/allex_rl/
# (the path the snapshot env editable-installs from); allex_dense.py resolves USD via allex_assets data dir.
: "${ISAACLAB_DIR:=/home/ubuntu/IsaacLab}"
: "${CONDA_ENV:=isaaclab}"
: "${ENV_BIN:=/home/ubuntu/miniconda3/envs/$CONDA_ENV/bin}"
# Rebuilt snapshot (2026-07-01) after the old chrisryu/allex-eval/envsnap was deleted. Holds
# env_isaaclab.tar (conda env: kitless IsaacLab release/3.0.0-beta2 + newton + mujoco-warp, with
# the allex_* editable installs) and isaaclab.tar (the fork + allex_description baked in; allex_rl
# code is rsync'd on top). Rebuild with sim/isaaclab/scripts/snapshot_env.sh. `ensure_snapshot` fails fast if missing.
: "${ENV_SNAPSHOT_S3:=s3://wirobotics-internal/chrisryu/sim_rl/envsnap}"  # node IAM role must read it
# Training deploy is learning/scripts/aws/rl_train.sh (rsync sim/isaaclab -> /home/ubuntu/sim/isaaclab + learning ->
# /home/ubuntu/learning_repo, over SSM-SSH). PROJ_DIR below is the sim service dir on the node (where
# server.py runs); EXPDIR is where the trainer (rltrainer venv) writes runs/checkpoints.
: "${REMOTE_BASE:=/home/ubuntu/sim}"          # parent of the deployed sim service
: "${PROJ_DIR:=$REMOTE_BASE/isaaclab}"        # sim service dir on the node (matches service.py SIM_SERVER_SCRIPT)
: "${EXPDIR:=/home/ubuntu/learning_repo/logs/rl/dexblind_newton_allex}"   # trainer runs (rltrainer venv)

# ── Local paths ───────────────────────────────────────────────────────────────
_AWS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # learning/scripts/aws
: "${REPO_ROOT:=$(cd "$_AWS_DIR/../../../sim/isaaclab" && pwd)}"   # learning/scripts/aws -> sim/isaaclab
: "${STATE_FILE:=$HOME/.allex-node}"          # remembers the launched instance id
: "${SSH_WRAPPER:=$HOME/.cache/allex-deploy/ssh-ssm.sh}"

log(){ echo "[${LOG_TAG:-allex-deploy} $(date -u +%H:%M:%S)] $*"; }
die(){ echo "[allex-deploy ERROR] $*" >&2; exit 1; }

need(){ command -v "$1" >/dev/null 2>&1 || die "missing prerequisite: $1${2:+ ($2)}"; }
preflight(){
  need aws "https://docs.aws.amazon.com/cli/"
  need session-manager-plugin "brew install --cask session-manager-plugin"
  need ssh; need rsync
  [ -f "$SSH_KEY" ] || die "SSH private key not found: $SSH_KEY (set SSH_PUBKEY to your key.pub)"
}

aws_(){ aws ${AWS_PROFILE:+--profile "$AWS_PROFILE"} --region "$AWS_REGION" "$@"; }

# Resolve the target instance id: $NODE_ID, else the state file, else the single
# running node tagged Name=$NODE_NAME ManagedBy=gpu-launcher.
node_id(){
  if [ -n "${NODE_ID:-}" ]; then printf '%s\n' "$NODE_ID"; return; fi
  if [ -s "$STATE_FILE" ]; then cat "$STATE_FILE"; return; fi
  local ids
  ids=$(aws_ ec2 describe-instances \
    --filters "Name=tag:Name,Values=$NODE_NAME" "Name=tag:ManagedBy,Values=gpu-launcher" \
              "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null)
  if [ "$(printf '%s\n' $ids | grep -c .)" != 1 ]; then
    # The default Name tag is "$USER-allex-newton" (lib.sh), which won't match nodes named
    # otherwise (e.g. chrisryu-gpu-l40s). List the gpu-launcher fleet so the caller can pin NODE_ID.
    echo "[allex-deploy] gpu-launcher nodes (set NODE_ID=i-... to target one):" >&2
    aws_ ec2 describe-instances --filters "Name=tag:ManagedBy,Values=gpu-launcher" \
        "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,Tags[?Key==`Name`]|[0].Value]' \
        --output text 2>/dev/null | sed 's/^/    /' >&2
    die "could not resolve a unique node by Name=$NODE_NAME (found: ${ids:-none}). Set NODE_ID=i-... (see above) or run setup_sim_node.sh launch."
  fi
  printf '%s\n' "$ids"
}

# Fail fast if the env snapshot the provisioner restores from is missing. Without this,
# `setup_sim_node.sh provision` only discovers it minutes in (after apt + miniconda), via a generic
# `aws s3 cp` failure inside remote/provision.sh.
ensure_snapshot(){
  local miss=0 f
  for f in env_isaaclab.tar isaaclab.tar; do
    aws_ s3 ls "$ENV_SNAPSHOT_S3/$f" >/dev/null 2>&1 || { echo "[allex-deploy] MISSING: $ENV_SNAPSHOT_S3/$f" >&2; miss=1; }
  done
  [ "$miss" = 0 ] || die "env snapshot not found under ENV_SNAPSHOT_S3=$ENV_SNAPSHOT_S3 (need env_isaaclab.tar + isaaclab.tar; see the README.md alongside them). Repoint ENV_SNAPSHOT_S3 or rebuild from a known-good node: sim/isaaclab/scripts/snapshot_env.sh"
}

# Fail fast if the node root lacks room for the ~12GB snapshot download + extraction,
# rather than dying mid-tar on an undersized box (seen: a 75GB root with 12GB free).
ensure_node_disk(){  # $1 = required GiB (default 30)
  local need="${1:-30}" avail
  avail=$(ssh_node "df -BG --output=avail / | tail -1 | tr -dc 0-9" 2>/dev/null)
  [ -n "$avail" ] || { echo "[allex-deploy] WARN: could not read node disk free; skipping disk preflight" >&2; return 0; }
  [ "$avail" -ge "$need" ] || die "node root has ${avail}GiB free, need >=${need}GiB for the snapshot restore. Expand the EBS root + grow the fs, or launch a larger node (ROOT_GB)."
}

# Write a self-contained ssh wrapper that tunnels through SSM (so rsync -e and
# scp can use it without ProxyCommand quoting headaches).
_ensure_wrapper(){
  mkdir -p "$(dirname "$SSH_WRAPPER")"
  cat > "$SSH_WRAPPER" <<EOF
#!/usr/bin/env bash
exec ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 \\
  -o ProxyCommand="aws ${AWS_PROFILE:+--profile $AWS_PROFILE }--region $AWS_REGION ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p" \\
  -i "$SSH_KEY" "\$@"
EOF
  chmod +x "$SSH_WRAPPER"
}

ssh_node(){ _ensure_wrapper; "$SSH_WRAPPER" "$SSH_USER@$(node_id)" "$@"; }
# rsync_node <rsync-args...>  — use $(node_dest)/path as the remote spec
rsync_node(){ _ensure_wrapper; rsync -az -e "$SSH_WRAPPER" "$@"; }
node_dest(){ printf '%s@%s' "$SSH_USER" "$(node_id)"; }

# Push the local wandb credential to the node so `--logger wandb` logs online without an interactive
# `wandb login`. The key rides the SSM tunnel straight into the node's ~/.netrc (never printed here),
# mirroring how do_launch injects the SSH pubkey. No-op (with a note) if the local box has no wandb cred.
seed_wandb_creds(){
  local nrc="${NETRC:-$HOME/.netrc}"
  if [ -f "$nrc" ] && grep -q '^machine api.wandb.ai' "$nrc"; then
    awk '/^machine /{p=($2=="api.wandb.ai")} p' "$nrc" \
      | ssh_node 'umask 077; touch ~/.netrc; grep -q api.wandb.ai ~/.netrc || cat >> ~/.netrc; chmod 600 ~/.netrc' \
      && log "wandb cred seeded to node ~/.netrc (--logger wandb logs online)."
  else
    log "no local wandb cred (${nrc} has no 'machine api.wandb.ai') — set LOGGER=tensorboard or 'wandb login' on the node."
  fi
}

wait_ssm_online(){  # $1 = instance id
  local id="$1" i
  for i in $(seq 1 60); do
    [ "$(aws_ ssm describe-instance-information --filters Key=InstanceIds,Values="$id" \
         --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)" = Online ] && return 0
    sleep 10
  done
  return 1
}

wait_ssh_ready(){  # block until we can actually ssh in (key landed via user-data)
  local i
  for i in $(seq 1 30); do
    ssh_node true 2>/dev/null && return 0
    sleep 10
  done
  return 1
}

# ── sim-service rsync deploy (rl_eval.sh / rl_train.sh) ───────────────────────
# These build on the SSH-over-SSM wrapper above. The RL deploy scripts source this lib, set
# SSH DEST CONDA SIM_DIR LEARNING_DIR SIM_REMOTE LEARNING_REMOTE ENV_NAME (+ LOG_TAG), then call
# deploy_sim_and_learning. `ensure_ssm_ssh_wrapper` returns the wrapper's path so they can use it
# directly as `"$SSH" host cmd` and `rsync -e "$SSH"` — the same wrapper ssh_node/rsync_node ride.
ensure_ssm_ssh_wrapper(){ _ensure_wrapper; printf '%s\n' "$SSH_WRAPPER"; }

rsync_delete(){  # $1=local dir  $2=remote dir — purge __pycache__ FIRST so --delete can drop stale dirs
  "$SSH" "$DEST" "mkdir -p $2 && find $2 -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true"
  rsync -a --delete -e "$SSH" --exclude __pycache__ --exclude '*.pyc' \
    --exclude 'logs/' --exclude 'outputs/' --exclude 'wandb/' "$1/" "$DEST:$2/"
}

# rsync sim/isaaclab + learning to a provisioned node and editable-reinstall the sim pkgs in ENV_NAME.
deploy_sim_and_learning(){
  log "deploy sim/isaaclab -> $DEST:$SIM_REMOTE (task assets ~600MB; first sync slow, later are deltas)"
  rsync_delete "$SIM_DIR" "$SIM_REMOTE"
  # The RPC wire protocol is unified in sim/service (shared by every sim eval service); ship it next to
  # sim/isaaclab so the server's `from transport import` (sys.path -> ../service) resolves on the node.
  log "deploy sim/service -> $DEST:$(dirname "$SIM_REMOTE")/service (unified transport)"
  rsync_delete "$(dirname "$SIM_DIR")/service" "$(dirname "$SIM_REMOTE")/service"
  log "editable-reinstall the sim packages (envs/robot/assets) in the $ENV_NAME env"
  # Uninstall first so a stale editable install baked into the env snapshot (an older layout, or a
  # prior hammer-lift-sim) can't shadow the freshly-rsynced code — then reinstall against it.
  "$SSH" "$DEST" "source $CONDA && conda activate $ENV_NAME && \
    pip uninstall -y hammer-lift-sim allex_rl_dexblind allex_assets allex_rl_config >/dev/null 2>&1 || true; \
    cd $SIM_REMOTE && pip install -e . --config-settings editable_mode=compat -q"
  log "deploy learning/ -> $DEST:$LEARNING_REMOTE"
  rsync_delete "$LEARNING_DIR" "$LEARNING_REMOTE"
  # Stamp the deployed code's provenance: the node-side tree is an rsync, not a git repo, so
  # short_git_sha() there resolved to the 'nogit' sentinel and EVERY run name shipped without the
  # one field that pins which code trained it. build_run_name reads this stamp when git is absent;
  # '-dirty' marks a worktree with uncommitted changes at deploy time.
  local _sha _dirty
  _sha=$(git -C "$LEARNING_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
  _dirty=$(git -C "$LEARNING_DIR" status --porcelain 2>/dev/null | head -1)
  [ -n "$_dirty" ] && _sha="${_sha}-dirty"
  "$SSH" "$DEST" "echo '$_sha' > $(dirname "$LEARNING_REMOTE")/GIT_SHA"
  log "deployed code stamped GIT_SHA=$_sha"
}
