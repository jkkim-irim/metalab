#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# metalab_train.sh (local) — MetaLab RL training on the LOCAL GPU via the RPC sim-service, with automatic
# checkpoint publishing to S3 (same convention as aws/metalab_train.sh).
#
# Workstation peer of the AWS launcher: same trainer (learning.train --trainer rl), same engines
# (genesis/newton) over the RPC sim-service — this runs on the workstation, not an AWS node.
# Task knobs (rewards, curriculum, action scales, DR) live inline in the sim task contract
# (sim/metalab/contract/tasks/<task>.py); the RL config (PPO, networks) in
# learning/rl/<experiment>/<task>/experiment.py — edit those, not this script. Engine selection reaches the trainer as SIM_ENGINE (learning/rl/service.py spawns
# `python -m sim.metalab.backends.<engine>.server` in the SAME venv).
#
# S3 checkpoint sync (S3_SYNC): while training, this periodically MIRRORS the run's model_*.pt to
#   $CKPT_S3_ROOT/<run_name>/model_*.pt   (default s3://wirobotics-internal/jkkim/sim_rl/ckpts)
# — the same layout aws/metalab_train.sh publishes to, so a run (local or node) is traceable from
# anywhere. This mirrors DURING the run (every SYNC_INTERVAL s) + a final sweep. S3 is THE checkpoint store
# and local retention is identical on a workstation and on a node: after a successful upload the local
# model_*.pt are pruned to the NEWEST one (KEEP_LOCAL_CKPTS=1 disables it). Re-evaluating the latest
# checkpoint therefore needs no S3 pull; an older one is fetched from S3. See _publish.
# ON by default, so a local run publishes to the SAME S3 path an AWS run does (no "where did it go?"
# difference between the two). --no-s3-sync (or S3_SYNC=0) turns it off. A workstation has no instance role,
# so this needs local creds (`aws login`): if they are MISSING and S3 was merely the default, the run warns
# loudly and continues local-only (a workstation keeps its checkpoints anyway); if you asked for S3
# EXPLICITLY (--s3-sync / S3_SYNC=1), missing creds are a hard error instead.
#
# Training-time eval videos (OFF by default; pass --record or RECORD=1): folded in so NO separate terminal
# is needed — on each checkpoint the trainer records a short eval rollout (rerun .rrd + synced HTML report
# over a short-lived RPC eval), publishes it beside that checkpoint and posts its LINK to the SAME W&B run
# under val/report (no sibling run). BLOCKING — the loop pauses per checkpoint until the link is logged.
# a background thread so training never pauses. Shares the training GPU. Needs wandb creds (`wandb login`).
#
# Args: every run knob is a --flag; the matching UPPERCASE env var still works as an alternative (so
# `TASK=hammer-lift-teacher metalab_train.sh --sim genesis` == `metalab_train.sh --sim genesis --task hammer-lift-teacher`).
# --sim {genesis|newton} AND --task <t> are REQUIRED — no defaults (fail-loud; missing --task lists the
# tasks for that backend). --sim runs headless; add --viz to open THAT backend's GUI on this machine.
# 로컬 학습도 RPC sim-service 경유(팀 규칙): 트레이너(client)+엔진 server 를 같은 uv env 에서 2 프로세스로. AWS 는 aws/metalab_train.sh (이 스크립트를 노드에서 실행).
# --device cuda:N pins the WHOLE run to physical GPU N (CUDA_VISIBLE_DEVICES) so two backends can train on
# two GPUs at once (genesis/warp otherwise ignore the index and land on GPU 0). Other unrecognized flags
# pass straight through to the trainer (--num_envs, --max_iterations, --seed).
# Script flags: --sim --viz --task --no_wandb --s3-sync/--no-s3-sync --sync-interval --ckpt-s3-root --record
#               --record-envs --record-steps
# --viz opens the backend GUI AND a live web dashboard (env-0..3 obs/reward/termination) that auto-opens in a
# browser window. --no_wandb disables all wandb logging (WANDB_MODE=disabled).
# Usage (from anywhere):
#   learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift-teacher                        # Genesis, RPC sim-service
#   learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift-teacher --num_envs 4 --viz gl     # Genesis, live GUI, 4 envs
#   learning/scripts/local/metalab_train.sh --sim newton --task perceptive-dexdeepmimic --num_envs 4 --viz rtx  # Newton, OVRTX viewer
#   learning/scripts/local/metalab_train.sh --sim newton --task perceptive-dexdeepmimic --num_envs 4096 --max_iterations 30000
#   learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift-teacher --record            # + per-checkpoint video/report (S3 mirror is already on)
#   learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift-teacher --no-s3-sync           # keep this run off S3 (local only)
#   learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift-teacher --record  # + per-checkpoint report link → W&B (val/report)
# Two backends on two GPUs at once (separate terminals):
#   learning/scripts/local/metalab_train.sh --sim newton  --task hammer-lift-teacher --device cuda:0   # newton  → physical GPU 0
#   learning/scripts/local/metalab_train.sh --sim genesis --task hammer-lift-teacher --device cuda:1   # genesis → physical GPU 1
# Long runs:
#   nohup learning/scripts/local/metalab_train.sh --sim newton --task perceptive-dexdeepmimic --num_envs 4096 > train.log 2>&1 &
# ──────────────────────────────────────────────────────────────────────────────
LOG_TAG=train
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TASK="${TASK:-}"          # REQUIRED (no default) — fail-loud after parse; pass --task <t> or TASK=<t>
RECIPE="${RECIPE:-}"      # REQUIRED when --task is a family folder; empty for a single-file contract
CKPT_S3_ROOT="${CKPT_S3_ROOT:-s3://wirobotics-internal/jkkim/sim_rl/ckpts}"
# Was S3 asked for EXPLICITLY (env or flag), or is it just the default? Only an explicit ask makes
# missing creds fatal — see the creds check below.
_s3_explicit=0; [ -n "${S3_SYNC+x}" ] && _s3_explicit=1
S3_SYNC="${S3_SYNC:-1}"             # boolean: 1=on (DEFAULT — same S3 path as an AWS run); --no-s3-sync disables
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"   # [s] between mid-run checkpoint syncs

# Training-time eval recording (OFF by default; --record / RECORD=1 enables): on each checkpoint the trainer
# records a short eval rollout (rerun .rrd + synced report, over a short-lived in-process RPC eval), files it
# beside that checkpoint and logs its link to the training W&B run's val/report panel (per-step media, so
# the panel updates live — a val/reports TABLE was tried and reverted: the table panel is cache-served).
# BLOCKING: the loop pauses per checkpoint until the report link is logged.
RECORD="${RECORD:-0}"
RECORD_ENVS="${RECORD_ENVS:-4}"       # envs given a series + a report tab. 4 = one
                                      # per object variant: the sim assigns env i variant i % N.
RECORD_STEPS="${RECORD_STEPS:-600}"   # policy steps per recording (0 = full episode; capped at one episode)
NO_WANDB="${NO_WANDB:-0}"             # boolean: --no_wandb / NO_WANDB=1 → WANDB_MODE=disabled (no wandb logging)
EXPERIMENT="${EXPERIMENT:-dexblind}"  # learning.rl.<experiment>.<task>.experiment package (trainer --experiment)

# ── args: every run knob is a --flag; the matching UPPERCASE env var (above) is its default, so the env
# form still works (`TASK=hammer-lift-teacher metalab_train.sh` == `metalab_train.sh --task hammer-lift-teacher`). --sim picks the
# backend (headless by default); add --viz to open THAT backend's GUI on this machine. Anything else falls
# through to the trainer verbatim (--num_envs, --max_iterations, --device, --seed, …).
SIM=""; VIZ=none; DEVICE=""; RUN_LABEL="${RUN_LABEL:-}"; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --sim)             SIM="$2"; shift 2 ;;
    --sim=*)           SIM="${1#*=}"; shift ;;
    --viz)             VIZ="${2:-}"                        # viewer: none | gl | rtx (value REQUIRED — fail-loud)
                       case "$VIZ" in
                         none|gl|rtx) shift 2 ;;
                         *) echo "[train] --viz requires one of: none | gl | rtx (got '${VIZ:-<missing>}')" >&2; exit 2 ;;
                       esac ;;
    --viz=*)           VIZ="${1#*=}"
                       case "$VIZ" in none|gl|rtx) ;; *) echo "[train] --viz requires one of: none | gl | rtx (got '$VIZ')" >&2; exit 2 ;; esac
                       shift ;;
    --task)            TASK="$2"; shift 2 ;;
    --task=*)          TASK="${1#*=}"; shift ;;
    --recipe)          RECIPE="$2"; shift 2 ;;
    --recipe=*)        RECIPE="${1#*=}"; shift ;;
    --device)          DEVICE="$2"; shift 2 ;;            # GPU 핀: cuda:N → 프로세스 전체를 물리 GPU N 에 고정
    --device=*)        DEVICE="${1#*=}"; shift ;;
    --run_label)       RUN_LABEL="$2"; shift 2 ;;          # run-name label segment (default: the task)
    --run_label=*)     RUN_LABEL="${1#*=}"; shift ;;
    --s3-sync)         S3_SYNC=1; _s3_explicit=1; shift ;;   # boolean: mirror checkpoints to S3 (default ON)
    --no-s3-sync)      S3_SYNC=0; _s3_explicit=1; shift ;;   # keep this run off S3 (local checkpoints only)
    --sync-interval)   SYNC_INTERVAL="$2"; shift 2 ;;
    --sync-interval=*) SYNC_INTERVAL="${1#*=}"; shift ;;
    --ckpt-s3-root)    CKPT_S3_ROOT="$2"; shift 2 ;;
    --ckpt-s3-root=*)  CKPT_S3_ROOT="${1#*=}"; shift ;;
    --record)          RECORD=1; shift ;;                  # boolean: per-checkpoint eval videos → W&B val/ (default off)
    --record-envs)     RECORD_ENVS="$2"; shift 2 ;;
    --record-envs=*)   RECORD_ENVS="${1#*=}"; shift ;;
    --record-steps)    RECORD_STEPS="$2"; shift 2 ;;
    --record-steps=*)  RECORD_STEPS="${1#*=}"; shift ;;
    --experiment)      EXPERIMENT="$2"; shift 2 ;;
    --experiment=*)    EXPERIMENT="${1#*=}"; shift ;;
    --no_wandb|--no-wandb) NO_WANDB=1; shift ;;            # boolean: disable wandb logging (WANDB_MODE=disabled)
    -h|--help)         sed -n '2,44p' "$0"; exit 0 ;;
    *)                 PASS+=("$1"); shift ;;             # trainer passthrough (--num_envs, --max_iterations, --device, --seed, …)
  esac
done

# backend (REQUIRED, fail-loud — no silent default): --sim wins; else honor a SIMULATOR env; else error.
# --viz {none|gl|rtx} opens a viewer on that backend (gl=OpenGL window, rtx=Newton OVRTX); omitted/none → headless.
if [ -n "$SIM" ]; then
  case "$SIM" in
    genesis)  export SIMULATOR=genesis ;;
    newton)   export SIMULATOR=newton ;;              # standalone Newton spoke (in-process)
    *) echo "[train] --sim must be 'genesis' or 'newton' (got '$SIM')" >&2; exit 2 ;;
  esac
elif [ -n "${SIMULATOR:-}" ]; then
  case "$SIMULATOR" in
    genesis|newton) export SIMULATOR ;;
    *) echo "[train] SIMULATOR must be genesis|newton (got '$SIMULATOR')" >&2; exit 2 ;;
  esac
else
  echo "[train] --sim is required (no default): --sim genesis | --sim newton" >&2
  exit 2
fi
[ "$SIMULATOR" = genesis ] && _VIZ_NAME=genesis || _VIZ_NAME=newton   # user-facing backend label (= --sim value)
[ "$VIZ" != none ] && PASS+=(--viz "$VIZ")                            # forward the chosen viewer (gl|rtx) to the trainer

# GPU 선택: --device cuda:N 은 프로세스 전체를 물리 GPU N 에 고정한다(CUDA_VISIBLE_DEVICES). genesis 의
# gs.init(gs.gpu)·warp·torch 가 모두 이 마스크를 따르므로 진짜 다중-GPU 격리가 된다 — 마스크 없이
# --device cuda:N 만 넘기면 gs.gpu 가 인덱스를 무시하고 GPU 0 에 올라간다. 마스크 뒤엔 보이는 GPU 가
# 1장뿐이므로 in-process 디바이스는 cuda:0 으로 트레이너에 넘긴다. 미지정 시 기존대로 GPU 0.
#   예) 두 GPU 동시 실행:  --sim newton ... --device cuda:0   //   --sim genesis ... --device cuda:1
if [ -n "$DEVICE" ]; then
  case "$DEVICE" in
    cpu)         PASS+=(--device cpu) ;;
    cuda|cuda:*) _gpu="${DEVICE#cuda}"; _gpu="${_gpu#:}"; _gpu="${_gpu:-0}"
                 export CUDA_VISIBLE_DEVICES="$_gpu"; PASS+=(--device cuda:0)
                 log "GPU 고정: 물리 GPU $_gpu (CUDA_VISIBLE_DEVICES=$_gpu) → cuda:0 (RPC: trainer+server share it)" ;;
    [0-9]*)      export CUDA_VISIBLE_DEVICES="$DEVICE"; PASS+=(--device cuda:0)
                 log "GPU 고정: 물리 GPU $DEVICE (CUDA_VISIBLE_DEVICES=$DEVICE) → cuda:0 (RPC: trainer+server share it)" ;;
    *)           PASS+=(--device "$DEVICE") ;;            # 알 수 없는 형식 → 트레이너에 그대로 전달
  esac
fi


# task · recipe (BOTH required for a task family, fail-loud): print what is discoverable, then exit.
if [ -z "$TASK" ]; then
  echo "[train] --task is required (no default). Available tasks for --sim ${_VIZ_NAME}:" >&2
  list_tasks | sed 's/^/  - /' >&2 || true
  echo "  e.g.  learning/scripts/local/metalab_train.sh --sim ${_VIZ_NAME} --task <one-of-the-above> --recipe <one-of-its-recipes>" >&2
  exit 2
fi
require_task_recipe train "$TASK" "$RECIPE"

# Publish checkpoints to S3? ON by default (same path as an AWS run); --no-s3-sync / S3_SYNC=0 opts out.
_do_sync=0
if [ "$S3_SYNC" = 1 ] && ! aws sts get-caller-identity >/dev/null 2>&1; then
  # Explicit ask → hard error. Default → warn LOUDLY and fall back: a workstation keeps its checkpoints
  # locally either way, so refusing to start a training run over an expired `aws login` costs more than
  # it protects. (On an AWS node the instance role always answers, and aws/metalab_train.sh asks
  # explicitly anyway, so the node path stays strict.)
  if [ "$_s3_explicit" = 1 ]; then
    echo "[train] --s3-sync given but no AWS creds — configure a profile (AWS_PROFILE / aws login) or drop --s3-sync" >&2
    exit 1
  fi
  echo "[train] ############################################################" >&2
  echo "[train] WARNING: no AWS creds → S3 publishing is OFF for this run." >&2
  echo "[train]          Checkpoints + reports stay LOCAL ONLY (under \$RL_LOG_ROOT, default ./logs/rl)." >&2
  echo "[train]          Run \`aws login\` and relaunch to publish to $CKPT_S3_ROOT/<run_name>/." >&2
  echo "[train] ############################################################" >&2
  S3_SYNC=0
fi
if [ "$S3_SYNC" = 1 ]; then
  _do_sync=1
  # The trainer uploads each checkpoint the moment it is written (OnPolicyRunner._upload_ckpt_s3) — it
  # reads CKPT_S3_ROOT from the env, so export it. The mirror below is then just the periodic backstop.
  export CKPT_S3_ROOT
else
  # No S3 for this run. rl_trainer requires either CKPT_S3_ROOT or --local, so say so explicitly rather
  # than inheriting a stray CKPT_S3_ROOT from the caller's shell and uploading behind the user's back.
  PASS+=(--local)
fi

# AWS node: put the training log/checkpoint root on the big local NVMe instance store (/opt/dlami/nvme,
# ~1.7T, DLAMI-mounted) instead of the small root EBS (~75G) — otherwise checkpoints fill / and break
# deploy + training. It is ephemeral (lost on stop/terminate), but --s3-sync mirrors checkpoints to S3
# (durable) and _publish prunes the local copies anyway. Only when RL_LOG_ROOT is unset AND the mount is
# writable; a workstation (or a node without the mount) keeps the default $PWD/logs/rl.
if [ "${METALAB_LOCATION:-local}" = "aws" ] && [ -z "${RL_LOG_ROOT:-}" ] && [ -w /opt/dlami/nvme ]; then
  export RL_LOG_ROOT=/opt/dlami/nvme/rl_logs
  mkdir -p "$RL_LOG_ROOT"
  log "AWS: RL_LOG_ROOT -> $RL_LOG_ROOT (local NVMe ~1.7T; keeps checkpoints off the small root EBS)"
fi

TRAIN_LOG="$(mktemp)"
# The trainer prints "[rl-trainer] log_dir=<...>/<run_name>" once it starts; both the sync loop and the
# final publish read the run dir from there (avoids guessing the run_name before the trainer builds it).
_run_dir(){ grep -oE '\[rl-trainer\] log_dir=\S+' "$TRAIN_LOG" 2>/dev/null | tail -1 | sed 's/.*log_dir=//'; }
# Mirror model_*.pt -> ckpts/<run_name>/. `sync` (not cp) skips already-uploaded files; NO --delete, so S3
# is never pruned — S3 IS the checkpoint store, everywhere. AFTER a successful upload the LOCAL model_*.pt
# are pruned to the NEWEST ONE, on a workstation exactly as on a node: one place to look for checkpoints
# instead of two divergent ones, and disks don't fill with copies of what is already durable.
#
# The newest is kept on purpose: it may still be mid-write (don't race torch.save), and the per-checkpoint
# video/report hook loads it. It is also what local/metalab_eval.sh picks up, so re-evaluating the LATEST
# checkpoint still needs no S3 pull — an OLDER one does (`aws s3 cp $CKPT_S3_ROOT/<run>/model_N.pt .`).
# KEEP_LOCAL_CKPTS=1 disables pruning. Nothing is deleted on a run without S3 (--no-s3-sync / no creds):
# _publish is only ever called when the mirror is on.
_publish(){ # $1=run dir
  local d="$1"; [ -n "$d" ] || return 0
  aws s3 sync "$d" "$CKPT_S3_ROOT/$(basename "$d")" --exclude '*' --include 'model_*.pt' < /dev/null >/dev/null 2>&1 || return 0
  # only reached when the upload SUCCEEDED (|| return 0 above) → safe to drop the now-on-S3 local copies
  if [ "${KEEP_LOCAL_CKPTS:-0}" != 1 ]; then
    ls -1t "$d"/model_*.pt 2>/dev/null | tail -n +2 | xargs -r rm -f   # keep newest, delete the rest
  fi
}

# MetaLab training goes through the RPC sim-service (team rule — RPC-only): rl_trainer spawns the engine's
# server.py as a separate process and drives it over a localhost socket (single-venv — both in this engine uv env).
# --no_wandb: turn wandb into a no-op process-wide (trainer logger + any video upload) without touching the
# trainer — WANDB_MODE=disabled makes wandb.init a no-op run (no network, no local files, no login needed).
[ "$NO_WANDB" = 1 ] && { export WANDB_MODE=disabled; log "wandb disabled (--no_wandb): WANDB_MODE=disabled"; }
# wandb ON (the default): require a login UP FRONT and fail loud — otherwise the run dies minutes later
# deep inside wandb.init. `wandb login` writes api.wandb.ai to ~/.netrc; WANDB_API_KEY / WANDB_MODE work too.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs 'api.wandb.ai' "$HOME/.netrc" 2>/dev/null; then
  echo "[train] wandb is ON by default but you are not logged in. Pick one:" >&2
  echo "[train]   • log in:  wandb login        (or: export WANDB_API_KEY=...)" >&2
  echo "[train]   • or skip: add --no_wandb     (Hub: check 'wandb 끄기')" >&2
  exit 1
fi
case "$SIMULATOR" in
  genesis|newton) ;;
  *) echo "[train] genesis|newton 만 지원 (SIMULATOR=$SIMULATOR). AWS 학습은 aws/metalab_train.sh." >&2; exit 2 ;;
esac
# Ensure the engine's uv env exists & matches uv.lock (clone the pinned sim source + `uv sync`;
# no-op if already current), so a freshly-cloned repo — local or on an AWS node — trains without
# manual setup. Then activate that venv (in-process trainer+sim run inside it).
# --viz rtx (newton) needs the OVRTX 'rtx' extra; sync it on demand (opt-in — kept out of the default env).
_extra=""; [ "$SIMULATOR" = newton ] && [ "$VIZ" = rtx ] && _extra=rtx
bash "$ROOT/learning/scripts/local/setup_env.sh" --sim "$SIMULATOR" ${_extra:+--extra "$_extra"}
_VENV="$(engine_venv "$SIMULATOR")"
source "$_VENV/bin/activate"
log "RPC sim-service: trainer(client) + engine server, both in uv env=$_VENV"
cd "$ROOT"

_BG_PIDS=()
if [ "$_do_sync" = 1 ]; then
  if [ "${KEEP_LOCAL_CKPTS:-0}" != 1 ]; then _ret="local pruned after upload (keep newest)"; else _ret="local copies kept (KEEP_LOCAL_CKPTS=1)"; fi
  log "S3 checkpoint mirror ON -> $CKPT_S3_ROOT/<run_name>/ (every ${SYNC_INTERVAL}s + final; ${_ret})"
  ( while :; do sleep "$SYNC_INTERVAL"; _publish "$(_run_dir)"; done ) &
  _BG_PIDS+=("$!")
fi
if [ "$RECORD" = 1 ]; then
  # Event-driven: on each checkpoint the trainer records + publishes a report, linked from the W&B run under
  # val/ (see rl_trainer._make_record_callback). BLOCKING (recording pauses the loop per checkpoint —
  # same behavior as main's isaaclab val-video path). Export the knobs it reads.
  export RECORD RECORD_ENVS RECORD_STEPS
  log "training-time eval recording ON (per checkpoint, blocking) -> W&B val/report (${RECORD_ENVS} env, ${RECORD_STEPS} steps, on the training GPU)"
fi
# Stop background jobs + do a final S3 sweep whatever happens (normal exit, Ctrl-C, error).
_cleanup(){
  for p in "${_BG_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  if [ "$_do_sync" = 1 ]; then
    local d; d="$(_run_dir)"
    [ -n "$d" ] && { echo "[train] final checkpoint publish -> $CKPT_S3_ROOT/$(basename "$d")/"; _publish "$d"; echo "[train] TRAIN_CKPTS_S3=$CKPT_S3_ROOT/$(basename "$d")/"; }
  fi
}
[ "${#_BG_PIDS[@]}" -gt 0 ] && trap _cleanup EXIT

# --task is parsed into TASK above (so the video recorder targets the same env); pass it explicitly.
# PASS holds everything unrecognized by our parser (--num_envs, --max_iterations, --device, --seed, and
# the --viz derived from --sim/--viz) → forwarded to the trainer verbatim.
# SIM selects the sim package (learning/rl/service.py dispatches to sim/$SIM/launch.py — learning itself
# is sim-agnostic); SIM_ENGINE is MetaLab's OWN engine knob read by sim/metalab/launch.py; EVAL_POLICY
# is the eval plugin the trainer's --record hook rolls checkpoints out with. SIMULATOR kept for run naming.
# ── run name — THIS SCRIPT'S OWN FORMAT (workstation runs) ──────────────────────────────────────────────────────
# EDIT `_run_name` BELOW AND NOTHING ELSE. Whatever it prints becomes, all at once:
#   • the wandb run name        (WandbSummaryWriter: os.path.split(log_dir)[-1])
#   • the S3 checkpoint folder  ($CKPT_S3_ROOT/<name>/model_*.pt)
#   • the local log dir         ($RL_LOG_ROOT/<experiment_name>/<name>/)
# The trainer takes $RUN_NAME verbatim and falls back to learning/rl/utils/run_naming.py only when it is
# unset, so this owns MetaLab naming while that shared default stays untouched for the isaaclab flow.
#
#   {yymmdd-HHMM}_{N}envs_{algo}_{engine}[_{recipe}][_{label}]_{sha}_{location}
#   260728-2108_16384envs_ppo_newton_privileged_f7e77ac_local   (no label given -> that segment is absent)
#   260728-2108_16384envs_sapg_newton_only-ycb_gain-sweep3_f7e77ac_local
# recipe   = --recipe; OMITTED for a single-file contract, which has no recipe axis. The TASK stays out
#            of the name (it always has) — it is on the W&B run config as train_cfg.task.
# label    = --run_label / RUN_LABEL; OMITTED when not given (the task is not substituted in)
# {N}envs  = OMITTED when --num_envs is left at the experiment.py default (this script cannot see it)
# algo     = $ALGO (what the Launchpad's algorithm selector exports), default ppo — resolved the SAME way
#            the experiment does (`os.environ.get("ALGO", "ppo")`), so the name cannot claim an algorithm
#            the run did not use. ALWAYS present: unlike the omitted segments this is never unknown.
# sha      = METALAB_GIT_SHA if set (a deployed tree has no .git), else this worktree's HEAD
# other    = aws/metalab_train.sh keeps its OWN copy of this format and exports the name it
#            built, so an AWS run arrives here with RUN_NAME already set and is NOT renamed.
#            The two are deliberately independent — change one without touching the other.
# Override the WHOLE name: RUN_NAME=<name> …   (wins over everything below)
_run_name() {
  # segments joined by "_", any that is unknown/not given is OMITTED (never a filler word)
  local envs="" seg=() i IFS
  for ((i = 0; i < ${#PASS[@]}; i++)); do
    [ "${PASS[i]}" = --num_envs ] && envs="${PASS[i + 1]}"
  done
  seg+=("$(date +%y%m%d-%H%M)")
  [ -n "$envs" ] && seg+=("${envs}envs")                 # omitted when left at the experiment.py default
  local algo="${ALGO:-ppo}"; seg+=("${algo,,}")           # ppo | sapg | …; lowercased like experiment.py does
  seg+=("$SIMULATOR")
  [ -n "$RECIPE" ] && seg+=("$RECIPE")                   # omitted for a single-file contract (no recipe axis)
  [ -n "$RUN_LABEL" ] && seg+=("$RUN_LABEL")             # omitted when --run_label / RUN_LABEL is not given
  seg+=("${METALAB_GIT_SHA:-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)}")
  seg+=("${METALAB_LOCATION:-local}")
  IFS=_; echo "${seg[*]}"
}
RUN_NAME="${RUN_NAME:-$(_run_name)}"
export RUN_NAME
log "run name = $RUN_NAME  (wandb run · S3 ckpt folder · local log dir)"

export SIM=metalab SIM_ENGINE="$SIMULATOR" EVAL_POLICY=actor
python -m learning.train --trainer rl --task "$TASK" ${RECIPE:+--recipe "$RECIPE"} \
  --experiment "$EXPERIMENT" "${PASS[@]}" 2>&1 | tee "$TRAIN_LOG"
exit "${PIPESTATUS[0]}"
