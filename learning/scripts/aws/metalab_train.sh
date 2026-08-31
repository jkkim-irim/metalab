#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# metalab_train.sh — ONE-COMMAND AWS training for the standalone MetaLab engines (genesis/newton) over the
# RPC sim-service (team rule — trainer + engine server as two processes on the node, localhost socket).
# This is the engine-agnostic MetaLab spoke running on a GPU node exactly like local/metalab_train.sh runs on
# your workstation (it launches that same script on the node) — just reached over SSM, code+env from S3.
#
# KEYLESS transport: the node is reached over AWS SSM (no SSH keys, no inbound ports); code is staged via
# an S3 tarball and the engine env is built ON the node with uv (learning/scripts/local/setup_env.sh:
# clone the pinned sim source + `uv sync` from the committed uv.lock — no conda). Nothing depends on your
# workstation once the node is running.
#
# Flow:  take the RUNNING node given by --node (never creates one) → install uv → deploy the
#        repo (S3 tar) → seed wandb → run local/metalab_train.sh on the node (uv sync + train) → wandb URL + tail.
#        Checkpoints mirror to S3 (jkkim/sim_rl/ckpts, --s3-sync) so they survive node termination.
#
# Run name is built HERE (learning/scripts/metalab_run_name.sh) and exported to the node, so it carries the
# real repo SHA + `aws` tag: {yymmdd-HHMM}_{envs}_{engine}_{recipe}_{label}_{sha}_aws (format: `_run_name` below,
# this script's own copy). --run-label <x> renames the label segment; RUN_NAME=<x> replaces the whole name.
#
# PREREQS (one-time):
#   • the node ALREADY carries your per-user node role `node-<your-IAM-username>` (SSM + read metalab/ +
#     read/write your own S3 prefix). This script does not create nodes, so it cannot attach a role either —
#     the role is chosen at CREATION time by the gpu-launcher kit (`SSM_INSTANCE_PROFILE=node-<user>`).
#     The shared legacy roles `metalab-node-role` / `project-x-ssm-role` are being retired (each hardcodes
#     one person's S3 prefix = cross-user exposure) and are dropping out of the gpu-launchers PassRole
#     allowlist — do NOT create new nodes with them. A node still running one keeps working until it is
#     replaced; this script only reads whatever role the node already has.
#   • your AWS_PROFILE is in the gpu-launchers group, and the node is tagged Owner=<your IAM username>
#     (the policy pins node control to your own Owner tag, so a node tagged to someone else is not yours
#     to run training on).
#
# Usage (--node is REQUIRED — this script never creates an instance; see "node lifecycle" below):
#   AWS_PROFILE=default bash learning/scripts/aws/metalab_train.sh --node <id> --sim genesis --task hammer-lift-teacher
#   … --node <id> --num-envs 8192 --record --max-iterations 5000  # real run (detached; node kept)
#   … --node <id> --num-envs 512 --max-iterations 3 --smoke       # short test (waits for completion)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# AWS_PROFILE defaults to the 'default' profile (so the Hub, which doesn't set it, works); override to
# pick a specific gpu-launchers profile. Exported so every aws CLI call below inherits it.
export AWS_PROFILE="${AWS_PROFILE:-default}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# local/lib.sh for the task·recipe discovery helpers only — this script otherwise shares nothing with the
# local path (it never activates an engine venv; it drives a node over SSM).
source "$ROOT/learning/scripts/local/lib.sh"
# Repo-tarball staging prefix — the CALLER'S OWN, not a shared one. The account grants each IAM user
# s3:PutObject only under s3://wirobotics-internal/<user>/ (the shared metalab/ prefix is read-only
# for users), so the previous shared default AccessDenied'd every launch at the deploy step below.
# AWS_USER is the same variable the gpu-launchers Owner tag uses (the devbox config.env exports it);
# with it unset we resolve the caller's IAM identity, so a plain `aws login` needs no extra setup.
# The node reads the tarball back with its own node-<user> instance role, which covers <user>/.
AWS_USER="${AWS_USER:-$(aws sts get-caller-identity --query Arn --output text 2>/dev/null | sed 's#.*[:/]##' || true)}"
: "${AWS_USER:?cannot resolve your S3 prefix — set AWS_USER=<your-username>, or METALAB_DEPLOY_S3=s3://… explicitly}"
S3_DEPLOY="${METALAB_DEPLOY_S3:-s3://wirobotics-internal/${AWS_USER}/_deploy}"

SIM=genesis; TASK=""; RECIPE=""; NUM_ENVS=4096; MAX_ITERS=""
RECORD=0; SMOKE=0; NODE_ID_IN=""; RUN_LABEL="${RUN_LABEL:-}"
while [ $# -gt 0 ]; do case "$1" in
  --sim)             SIM="$2"; shift 2 ;;
  --task)            TASK="$2"; shift 2 ;;
  --recipe)          RECIPE="$2"; shift 2 ;;
  --num-envs)        NUM_ENVS="$2"; shift 2 ;;
  --max-iterations)  MAX_ITERS="$2"; shift 2 ;;
  --run-label)       RUN_LABEL="$2"; shift 2 ;;            # run-name label segment (default: the task)
  --record)          RECORD=1; shift ;;
  --smoke)           SMOKE=1; shift ;;                     # wait for the run to finish (short tests)
  --node)            NODE_ID_IN="$2"; shift 2 ;;           # REQUIRED: the running node to train on
  -h|--help)         sed -n '2,40p' "$0"; exit 0 ;;
  *) echo "[metalab-aws] unknown arg: $1" >&2; exit 2 ;;
esac; done
[ -n "$TASK" ] || { echo "[metalab-aws] --task required (e.g. --task hammer-lift-teacher)" >&2; exit 2; }
# The node re-checks against ITS deployed tree; check here too so a typo fails before a node is touched.
require_task_recipe metalab-aws "$TASK" "$RECIPE"
REC_FLAG=""; [ -n "$RECIPE" ] && REC_FLAG="--recipe $RECIPE"
# Node lifecycle is the USER's call, never this script's — same rule the not-running check below applies,
# extended to creation. Launching an instance is the single most expensive thing this script could do and
# it used to happen implicitly whenever --node was omitted, so a mistyped flag silently billed a new GPU.
# Nodes are created with the gpu-launchers kit (see learning/CLAUDE.md); this script only ever RUNS on one.
if [ -z "$NODE_ID_IN" ]; then
  echo "[metalab-aws] --node <instance-id> 가 필요합니다 — 이 스크립트는 인스턴스를 생성하지 않습니다." >&2
  echo "[metalab-aws] Launchpad 를 쓰신다면 AWS 학습 카드의 인스턴스 선택 콤보에서 노드를 고르세요." >&2
  echo "[metalab-aws] 현재 사용 가능한 노드:" >&2
  aws ec2 describe-instances \
    --filters 'Name=instance-state-name,Values=running' \
    --query 'Reservations[].Instances[?starts_with(InstanceType,`g`)||starts_with(InstanceType,`p`)].[InstanceId,InstanceType,Tags[?Key==`Name`].Value|[0]]' \
    --output text >&2 2>/dev/null || echo "[metalab-aws]   (조회 실패 — aws 로그인 상태를 확인하세요)" >&2
  echo "[metalab-aws] 노드가 없으면 gpu-launchers 로 먼저 만드세요 (learning/CLAUDE.md):" >&2
  echo "[metalab-aws]   sandbox/chrisryu0/gpu-launchers/scripts/launch_l40s_node.sh   # g5/A10G" >&2
  exit 2
fi

log(){ printf '[metalab-aws %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ── keyless SSM command runner (script on stdin; prints stdout; returns non-zero on failure) ──────────
ssm_run(){  # $1 = instance-id ; SSM_TIMEOUT env (default 900)
  local iid="$1" tmo="${SSM_TIMEOUT:-900}" sf pf cid st end err
  sf=$(mktemp); pf=$(mktemp); cat > "$sf"
  python3 -c "import json,sys;json.dump({'commands':[open(sys.argv[1]).read()]},open(sys.argv[2],'w'))" "$sf" "$pf"
  cid=$(aws ssm send-command --instance-ids "$iid" --document-name AWS-RunShellScript \
        --timeout-seconds "$tmo" --parameters file://"$pf" --query 'Command.CommandId' --output text)
  end=$((SECONDS+tmo+120))
  while :; do
    st=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query Status --output text 2>/dev/null || echo Pending)
    case "$st" in Success|Failed|Cancelled|TimedOut) break ;; esac
    [ $SECONDS -ge $end ] && { st=LocalTimeout; break; }
    sleep 4
  done
  aws ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query StandardOutputContent --output text 2>/dev/null
  err=$(aws ssm get-command-invocation --command-id "$cid" --instance-id "$iid" --query StandardErrorContent --output text 2>/dev/null || true)
  [ -n "$err" ] && [ "$err" != None ] && printf '%s\n' "$err" >&2
  rm -f "$sf" "$pf"; [ "$st" = Success ]
}
# run the given remote script AS ubuntu with HOME=/home/ubuntu (SSM runs as root by default). SSM's
# AWS-RunShellScript executes the remote command under /bin/sh (dash on Ubuntu), which lacks bash's
# $'…' ANSI-C quoting — so `printf %q` (which emits $'…') breaks there. Base64-encode the script instead
# (b64 is metachar-free → safe under any /bin/sh) and decode|bash it on the node.
as_ubuntu(){ printf 'printf %%s %s | base64 -d | sudo -u ubuntu -H bash\n' "$(printf '%s' "$1" | base64 -w0)"; }

# ── 1) check the node the caller picked ───────────────────────────────────────────────────────────────
# Node lifecycle (create / start = billing) is kept SEPARATE from launching training — fail loud on
# anything but a running node instead of silently creating or starting one (a GPU cold-start bills money +
# adds a multi-minute blocking wait, and a fresh instance bills until someone notices it).
IID="$NODE_ID_IN"
_st=$(aws ec2 describe-instances --instance-ids "$IID" \
      --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo unknown)
log "using node $IID (state=$_st)"
case "$_st" in
  running) : ;;
  stopped|stopping|pending)
    echo "[metalab-aws] 노드 $IID 가 '$_st' 상태입니다 — 학습 전에 인스턴스를 먼저 시작하세요:" >&2
    echo "[metalab-aws]   aws ec2 start-instances --instance-ids $IID" >&2
    echo "[metalab-aws]   running 이 되면 다시 Launch 하세요." >&2
    exit 1 ;;
  *)
    echo "[metalab-aws] 노드 $IID 상태=$_st — 사용 불가. running 인 노드를 고르세요." >&2
    exit 1 ;;
esac

# wait for SSM to register the node (it is already running, but the agent can lag a restart)
log "waiting for SSM online…"
for i in $(seq 1 30); do
  [ "$(aws ssm describe-instance-information --filters Key=InstanceIds,Values="$IID" \
       --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)" = Online ] && break
  sleep 6
done

# ── 2) provision base: uv (git/aws/rsync ship with the DLAMI; uv provides Python + builds the engine env) ─
# No conda — local/setup_env.sh on the node uses uv (uv sync from the committed uv.lock), same as local.
log "provision base (uv)…"
SSM_TIMEOUT=600 ssm_run "$IID" <<REMOTE
$(as_ubuntu '
set -e
command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" --version 2>/dev/null || uv --version
')
REMOTE

# ── 3) deploy the repo via S3 tarball (keyless; the node's own node-<user> role reads metalab/) ───────
SHA=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)
log "deploy repo (sha $SHA) → $S3_DEPLOY → node…"
TARBALL=$(mktemp --suffix=.tgz)
# .venv is excluded (path-independent, so a nested one is caught too): a venv is NOT relocatable — its
# bin/python symlinks into the LAUNCHER's $HOME and is a dead link on the node, which builds its own
# engine env under ~/.metalab/venvs (setup_env.sh, from uv.lock). uv hides it from git with its own
# .gitignore, so it never shows up in `git status` — but tar does not read .gitignore. Measured: it was
# 4.25G of the 4.48G tarball (19.6x), ~6 min of every launch, for bytes the node cannot execute.
tar czf "$TARBALL" -C "$ROOT" --exclude=./.git --exclude=./logs --exclude=./wandb --exclude=./outputs \
  --exclude=__pycache__ --exclude='*.pyc' --exclude=.venv --exclude=./ros2_workspace \
  --exclude=./allex_groot_sim --exclude=./allex_groot --exclude=./allex_vla_pipeline .
aws s3 cp "$TARBALL" "$S3_DEPLOY/allex-repo.tgz" --only-show-errors
rm -f "$TARBALL"
SSM_TIMEOUT=300 ssm_run "$IID" <<REMOTE
$(as_ubuntu "
set -e
mkdir -p /home/ubuntu/allex
aws s3 cp $S3_DEPLOY/allex-repo.tgz /tmp/allex-repo.tgz --only-show-errors
tar xzf /tmp/allex-repo.tgz -C /home/ubuntu/allex && rm -f /tmp/allex-repo.tgz
echo deployed
")
REMOTE

# ── 4) seed wandb credential from your local ~/.netrc (key rides local→SSM only; not printed) ─────────
WKEY=$(awk '/^machine api\.wandb\.ai/{f=1} f&&$1=="password"{print $2; exit}' "$HOME/.netrc" 2>/dev/null || true)
if [ -n "${WKEY:-}" ]; then
  log "seeding wandb cred to node…"
  SSM_TIMEOUT=60 ssm_run "$IID" <<REMOTE >/dev/null
$(as_ubuntu "umask 077; touch /home/ubuntu/.netrc; grep -q api.wandb.ai /home/ubuntu/.netrc || echo 'machine api.wandb.ai login user password $WKEY' >> /home/ubuntu/.netrc; chmod 600 /home/ubuntu/.netrc")
REMOTE
else
  log "WARN: no wandb cred in ~/.netrc — the run logs offline (set WANDB or 'wandb login' locally)."
fi

# ── 5) launch training on the node (detached), tagged as an aws run ───────────────────────────────────
REC=""; [ "$RECORD" = 1 ] && REC="--record"
ITERS=""; [ -n "$MAX_ITERS" ] && ITERS="--max_iterations $MAX_ITERS"
# Forward the algorithm/physics selection (SAPG, gravcomp) from THIS process's env (set by the Hub /
# caller) to the node: SSM starts a fresh shell there, so only what we bake into the remote `export`
# reaches experiment.py / the sim backend.
ALGO_ENV=""
for _v in ALGO SAPG_BLOCKS SAPG_MODE SAPG_OFF_POLICY_RATIO SAPG_EMBED_DIM SAPG_IR_TYPE SAPG_IR_COEF_SCALE \
          METALAB_GRAVCOMP METALAB_MOTOR_COUPLING; do
  eval "_val=\${$_v:-}"
  [ -n "$_val" ] && ALGO_ENV="$ALGO_ENV $_v=$_val"
done
# ── run name — THIS SCRIPT'S OWN FORMAT (AWS runs) ──────────────────────────────────────────────────────
# EDIT `_run_name` BELOW AND NOTHING ELSE. Whatever it prints becomes, all at once:
#   • the wandb run name        (WandbSummaryWriter: os.path.split(log_dir)[-1])
#   • the S3 checkpoint folder  ($CKPT_S3_ROOT/<name>/model_*.pt)
#   • the local log dir         ($RL_LOG_ROOT/<experiment_name>/<name>/)
# The trainer takes $RUN_NAME verbatim and falls back to learning/rl/utils/run_naming.py only when it is
# unset, so this owns MetaLab naming while that shared default stays untouched for the isaaclab flow.
#
#   {yymmdd-HHMM}_{N}envs_{algo}_{engine}[_{recipe}][_{label}]_{sha}_aws
#   260728-2108_16384envs_ppo_newton_privileged_f7e77ac_aws   (no label given -> that segment is absent)
#   260728-2108_16384envs_sapg_newton_only-ycb_gain-sweep3_f7e77ac_aws
# recipe   = --recipe; OMITTED for a single-file contract, which has no recipe axis. The TASK stays out
#            of the name (it always has) — it is on the W&B run config as train_cfg.task.
# label    = --run-label / RUN_LABEL; OMITTED when not given (the task is not substituted in)
# algo     = $ALGO (what the Launchpad's algorithm selector exports), default ppo — resolved the SAME way
#            the experiment does, and it is the same variable this script forwards to the node in
#            $ALGO_ENV, so the name and the node's actual algorithm cannot disagree. ALWAYS present.
# sha      = this worktree's HEAD, read BEFORE deploy (the node's copy has no .git, which is
#            what used to make every AWS run read `-nogit`)
# built    = HERE, on your machine, then forwarded to the node through the remote `export`
#            below; local/metalab_train.sh honours a preset RUN_NAME instead of recomputing, so
#            ONE name covers the wandb run, the S3 folder and the node's log dir. That script
#            keeps its own copy of the format for workstation runs — deliberately independent.
# Override the WHOLE name: RUN_NAME=<name> …   (wins over everything below)
_run_name() {
  # segments joined by "_"; the label is OMITTED when --run-label / RUN_LABEL is not given (no filler word)
  local seg=() IFS algo="${ALGO:-ppo}"
  seg+=("$(date +%y%m%d-%H%M)" "${NUM_ENVS}envs" "${algo,,}" "$SIM")   # algo lowercased like experiment.py
  [ -n "$RECIPE" ] && seg+=("$RECIPE")                 # omitted for a single-file contract (no recipe axis)
  [ -n "$RUN_LABEL" ] && seg+=("$RUN_LABEL")
  seg+=("$SHA" aws)
  IFS=_; echo "${seg[*]}"
}
RUN_NAME="${RUN_NAME:-$(_run_name)}"
log "run name = $RUN_NAME  (wandb run · S3 ckpt folder · node log dir)"
log "starting: --sim $SIM --task $TASK $REC_FLAG --num_envs $NUM_ENVS $REC $ITERS$ALGO_ENV (location=aws, s3-sync on)"
SSM_TIMEOUT=120 ssm_run "$IID" <<REMOTE
$(as_ubuntu "
cd /home/ubuntu/allex
export METALAB_LOCATION=aws METALAB_GIT_SHA=$SHA RUN_NAME=$RUN_NAME$ALGO_ENV
setsid bash learning/scripts/local/metalab_train.sh --sim $SIM --task $TASK $REC_FLAG --num_envs $NUM_ENVS $REC $ITERS --s3-sync \
  > /home/ubuntu/train.log 2>&1 < /dev/null &
echo started
")
REMOTE

# ── 6) wait for first signal, report wandb URL ────────────────────────────────────────────────────────
log "waiting for first iteration / wandb URL…"
SSM_TIMEOUT=780 ssm_run "$IID" <<REMOTE
$(as_ubuntu '
for i in $(seq 1 70); do
  ps -eo cmd | grep -q "[l]earning.train" || { tr "\r" "\n" < /home/ubuntu/train.log | grep -qiE "Traceback|error:|out of memory" && { echo CRASH; break; }; }
  tr "\r" "\n" < /home/ubuntu/train.log | grep -qiE "Learning iteration [2-9]|out of memory|Traceback|error:" && break
  sleep 10
done
echo "=== run name ==="; tr "\r" "\n" < /home/ubuntu/train.log | grep -aE "Run name:" | tail -1
echo "=== wandb ==="; grep -aoE "https://wandb.ai/[^ ]+/runs/[^ ]+" /home/ubuntu/train.log | head -1
echo "=== last iter ==="; tr "\r" "\n" < /home/ubuntu/train.log | grep -aE "Learning iteration|Steps per second|Mean reward|out of memory|Traceback" | tail -6
echo "=== gpu ==="; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
')
REMOTE

if [ "$SMOKE" = 1 ]; then
  log "smoke: waiting for the run to finish (the node is left running — it is yours)…"
  SSM_TIMEOUT=1800 ssm_run "$IID" <<REMOTE || true
$(as_ubuntu 'for i in $(seq 1 178); do ps -eo cmd | grep -q "[l]earning.train" || { echo DONE; break; }; sleep 10; done
tr "\r" "\n" < /home/ubuntu/train.log | grep -aE "Learning iteration|TRAIN_SERVICE_OK|Traceback|error:" | tail -8')
REMOTE
else
  log "DONE. Training runs detached on $IID. Tail:  aws ssm start-session --target $IID  → tail -f /home/ubuntu/train.log"
  log "Stop billing when finished:  aws ec2 terminate-instances --instance-ids $IID"
fi
