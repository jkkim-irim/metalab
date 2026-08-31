#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_sim_node.sh — LOCAL CLI for the sim/isaaclab (Newton sim-service) GPU node's whole lifecycle (Newton-only).
# Runs on YOUR machine; talks to AWS and drives the node over SSH-over-SSM. The
# node-side payloads it ships live in remote/ (those run ON the node, not here).
#
#   setup_sim_node.sh up          launch + provision a ready-to-train node (one shot)
#   setup_sim_node.sh launch      create the instance only (your ssh pubkey injected via user-data)
#   setup_sim_node.sh provision   restore the env snapshot on the node + rsync repo code in
#   setup_sim_node.sh sync        re-rsync the sim/isaaclab/ project to the node (latest code)
#   setup_sim_node.sh status      ec2 state + SSM ping + latest run/checkpoint
#   setup_sim_node.sh ssh [cmd]   shell in / run a command over SSH-over-SSM
#   setup_sim_node.sh logs        tail the live train.log
#   setup_sim_node.sh stop        pause (keeps EBS + checkpoints; cheap; resume with `start`)
#   setup_sim_node.sh start       resume a stopped node (+ wait for SSM)
#   setup_sim_node.sh down        terminate / DESTROY the node (checkpoints lost unless fetched)
#
# Target a specific box with NODE_ID=i-... ; otherwise uses the state file (~/.allex-node).
# Prereqs (see README.md): aws cli, session-manager-plugin, an ssh keypair, and an AWS
# profile allowed to run/stop instances + ssm:StartSession on ManagedBy=gpu-launcher nodes.
# *** BILLABLE *** (~$1.5–3/hr for g6e/g5) until you `setup_sim_node.sh stop` (or `down`).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/lib.sh"

resolve_ami(){
  if [ -n "${AMI:-}" ]; then echo "$AMI"; return; fi
  aws_ ec2 describe-images --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
              "Name=state,Values=available" \
    --query 'reverse(sort_by(Images,&CreationDate))[0].ImageId' --output text
}

do_launch(){
  preflight
  [ -f "$SSH_PUBKEY" ] || die "public key not found: $SSH_PUBKEY (generate one: ssh-keygen -t ed25519)"
  local ami; ami=$(resolve_ami); [ -n "$ami" ] && [ "$ami" != None ] || die "could not resolve DL OSS AMI"
  log "AMI=$ami  types=[$INSTANCE_TYPES]  region=$AWS_REGION  profile=${AWS_PROFILE:-<default>}"

  # user-data: append the caller's pubkey to ubuntu's authorized_keys (per-user, no shared key)
  local ud; ud="$(mktemp)"; trap 'rm -f "$ud"' RETURN
  {
    echo '#!/bin/bash'
    echo 'install -d -m700 -o ubuntu -g ubuntu /home/ubuntu/.ssh'
    printf 'echo %q >> /home/ubuntu/.ssh/authorized_keys\n' "$(cat "$SSH_PUBKEY")"
    echo 'chown ubuntu:ubuntu /home/ubuntu/.ssh/authorized_keys; chmod 600 /home/ubuntu/.ssh/authorized_keys'
  } > "$ud"

  local subnet_arg="" sg_arg=""
  [ -n "$SUBNET" ] && subnet_arg="--subnet-id $SUBNET"
  [ -n "$SECURITY_GROUP" ] && sg_arg="--security-group-ids $SECURITY_GROUP"

  local iid="" itype="" attempt t out
  for attempt in $(seq 1 "$LAUNCH_ATTEMPTS"); do
    for t in $INSTANCE_TYPES; do
      out=$(aws_ ec2 run-instances --image-id "$ami" --instance-type "$t" --count 1 \
        $subnet_arg $sg_arg \
        --iam-instance-profile Name="$IAM_INSTANCE_PROFILE" \
        --user-data "file://$ud" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$ROOT_GB,\"VolumeType\":\"gp3\"}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NODE_NAME},{Key=Owner,Value=$OWNER},{Key=ManagedBy,Value=gpu-launcher}]" \
        --query 'Instances[0].InstanceId' --output text 2>&1) && [[ "$out" == i-* ]] && { iid="$out"; itype="$t"; break 2; }
      log "  round $attempt/$LAUNCH_ATTEMPTS $t: $(echo "$out" | grep -oE 'InsufficientInstanceCapacity|VcpuLimitExceeded|[A-Za-z]+Error' | head -1 || echo "$out" | tail -c 120)"
    done
    [ "$attempt" -lt "$LAUNCH_ATTEMPTS" ] && sleep "$RETRY_SLEEP"
  done
  [ -n "$iid" ] || die "no capacity for [$INSTANCE_TYPES] after $LAUNCH_ATTEMPTS rounds"
  printf '%s' "$iid" > "$STATE_FILE"
  log "LAUNCHED $itype: $iid  (state file: $STATE_FILE)"

  log "Waiting for status checks + SSM Online ..."
  aws_ ec2 wait instance-status-ok --instance-ids "$iid"
  wait_ssm_online "$iid" || die "SSM did not come Online"
  log "Waiting for SSH (your key landing via user-data) ..."
  NODE_ID="$iid" wait_ssh_ready || die "SSH not reachable (check SSH_PUBKEY / session-manager-plugin)"
  log "Node $iid reachable over SSH-over-SSM."
}

do_sync(){
  preflight
  # Warn (don't block) if the node carries someone else's active USAGE.lock — avoid clobbering a live run.
  _lk=$(ssh_node "cat /home/ubuntu/USAGE.lock 2>/dev/null" 2>/dev/null || true)
  if [ -n "$_lk" ] && ! printf '%s' "$_lk" | grep -q "owner=$OWNER"; then
    echo "[allex-deploy] WARNING: node has an active USAGE.lock (not owner=$OWNER) — another run may be using it:" >&2
    printf '    %s\n' "$_lk" >&2
  fi
  # Quick sim-code refresh to the service dir (compat editable install picks up changed .py without a
  # reinstall). For a full deploy + train (sim + learning, both envs) use learning/scripts/aws/rl_train.sh.
  log "rsync sim/isaaclab -> $(node_id):$PROJ_DIR (task assets ~600MB; first sync slow, later are deltas)"
  ssh_node "mkdir -p $PROJ_DIR"
  rsync_node --delete --exclude '__pycache__' --exclude '*.pyc' --exclude 'logs/' --exclude 'outputs/' --exclude 'wandb/' \
    "$REPO_ROOT/" "$(node_dest):$PROJ_DIR/"
  log "rsync complete."
}

do_provision(){
  # Idempotent: if the node already has the '$CONDA_ENV' conda env, the snapshot is already restored —
  # skip the ~10-min restore. rl_eval.sh calls this before each eval, so it must be a fast no-op then.
  if ssh_node "test -d $ENV_BIN" >/dev/null 2>&1; then
    log "node already has the '$CONDA_ENV' env — skipping snapshot restore (idempotent)."
    seed_wandb_creds
    ssh_node "echo \"owner=$OWNER name=$NODE_NAME purpose=$USAGE_PURPOSE started=\$(date -u +%FT%TZ)\" > /home/ubuntu/USAGE.lock" >/dev/null 2>&1 || true
    return 0
  fi
  # Preflight: snapshot present in S3 + node has disk for it. Fail fast (seconds) instead of
  # discovering the problem ~3 min in (missing snapshot) or mid-tar (full disk).
  ensure_snapshot
  ensure_node_disk 30
  # Restore the snapshot FIRST (it recreates $ISAACLAB_DIR incl projects/), THEN rsync repo code on top.
  rsync_node "$HERE/remote/provision.sh" "$(node_dest):~/provision.sh"
  log "Restoring env snapshot from $ENV_SNAPSHOT_S3 (kitless Newton; ~10GB) — polling ~/allex_setup.log ..."
  ssh_node "rm -f ~/allex_setup.log; ENV_SNAPSHOT_S3=$ENV_SNAPSHOT_S3 CONDA_ENV=$CONDA_ENV ISAACLAB_DIR=$ISAACLAB_DIR nohup bash ~/provision.sh > ~/allex_setup.log 2>&1 & echo SETUP_DISPATCHED" | grep -q SETUP_DISPATCHED || die "provision dispatch failed"
  local t=0 line
  while [ "$t" -lt 2400 ]; do   # up to 40 min
    sleep 30; t=$((t+30))
    line=$(ssh_node "tail -n 1 ~/allex_setup.log; grep -qa SETUP_DONE ~/allex_setup.log && echo HIT_DONE; grep -qaE 'FAILED|Traceback' ~/allex_setup.log && echo HIT_FAIL" 2>/dev/null || true)
    log "  t=$((t/60))m  $(printf '%s' "$line" | grep -aE '=== |HIT_' | tail -1)"
    printf '%s' "$line" | grep -qa HIT_DONE && break
    printf '%s' "$line" | grep -qa HIT_FAIL && { echo "--- provision log tail ---"; ssh_node "tail -n 40 ~/allex_setup.log"; die "node provision failed"; }
    [ "$t" -ge 2400 ] && { echo "--- provision log tail ---"; ssh_node "tail -n 40 ~/allex_setup.log"; die "node provision timed out"; }
  done
  # Code is NOT baked into the snapshot or synced here — deploy the sim + trainer and train with
  # learning/scripts/aws/rl_train.sh (rsync over SSM-SSH + editable reinstall). This restore is the env baseline only.
  # Seed the wandb cred so --logger wandb logs online out of the box (no interactive login on the node).
  seed_wandb_creds
  # Mark the node in-use so agents/teammates don't clobber it (see wir CLAUDE.md).
  ssh_node "echo \"owner=$OWNER name=$NODE_NAME purpose=$USAGE_PURPOSE started=\$(date -u +%FT%TZ)\" > /home/ubuntu/USAGE.lock" \
    && log "USAGE.lock written (owner=$OWNER purpose=$USAGE_PURPOSE)."
  log "Provision complete — env baseline ready. Deploy code + train with: NODE=$(node_id) learning/scripts/aws/rl_train.sh"
}

cmd_status(){
  local id st ping
  id=$(node_id)
  st=$(aws_ ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  ping=$(aws_ ssm describe-instance-information --filters Key=InstanceIds,Values="$id" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)
  echo "NODE=$id  state=${st:-?}  ssm=${ping:-None}"
  [ "$ping" = Online ] && ssh_node "echo latest_run=\$(ls -1dt $EXPDIR/*/ 2>/dev/null | head -1 | xargs -n1 basename); echo ckpts=\$(ls -1 $EXPDIR/*/model_*.pt 2>/dev/null | wc -l)" 2>/dev/null || true
}

cmd_down(){
  local id; id=$(node_id)
  read -r -p "Terminate (DESTROY) $id? checkpoints not fetched will be LOST. [y/N] " a
  [ "$a" = y ] || { echo aborted; exit 1; }
  aws_ ec2 terminate-instances --instance-ids "$id" --query 'TerminatingInstances[0].CurrentState.Name' --output text
  [ -f "$STATE_FILE" ] && rm -f "$STATE_FILE"
}

case "${1:-status}" in
  up)        preflight; ensure_snapshot   # verify the snapshot exists BEFORE paying for a node
             do_launch && do_provision
             log "READY: node $(node_id). Train:  python -m learning.train --trainer rl  |  eval:  learning/scripts/aws/rl_eval.sh" ;;
  launch)    do_launch ;;
  provision) do_provision ;;
  sync)      do_sync ;;
  status)    cmd_status ;;
  ssh)       shift; ssh_node "$@" ;;
  logs)      ssh_node "tail -n ${N:-40} -f ~/train.log" ;;
  stop)      aws_ ec2 stop-instances  --instance-ids "$(node_id)" --query 'StoppingInstances[0].CurrentState.Name' --output text ;;
  start)     id=$(node_id); aws_ ec2 start-instances --instance-ids "$id" --query 'StartingInstances[0].CurrentState.Name' --output text
             aws_ ec2 wait instance-running --instance-ids "$id"; wait_ssm_online "$id" && echo "SSM Online" ;;
  down)      cmd_down ;;
  *) echo "usage: setup_sim_node.sh {up|launch|provision|sync|status|ssh|logs|stop|start|down}"; exit 1 ;;
esac
