#!/usr/bin/env bash
# watch_and_cull.sh - "train to 5000 but kill as necessary": while WBT training runs on the node,
# periodically t=0-eval the newest checkpoint (concurrent, no video). If the eval final-frame reward
# drops below DROP x best on TWO consecutive probes (past MIN_ITER), kill training, publish checkpoints
# to S3, exit with a summary. Also exits when training ends naturally. Selection signal = the EVAL,
# not training curves (v5-long: training reward peaked at 2000 but eval peaked at 1000).
set -o pipefail   # NOT -u: lib.sh (sourced below, with output redirected) trips set -u on vars unbound
                  # in this context and would kill the script silently inside the source
trap 'echo "WATCH_EXIT rc=$? line=$LINENO"' EXIT   # a silent death is worse than a noisy one
LOG_TAG="${LOG_TAG:-watch}"
NODE="${NODE:?i-...}"
TRAIN_LOG="${TRAIN_LOG:?path to the local teed track_train log}"
AWS_DIR="${AWS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CKPT_S3_ROOT="${CKPT_S3_ROOT:-s3://wirobotics-internal/chrisryu/sim_rl/ckpts}"
MIN_ITER="${MIN_ITER:-1600}"
PROBE_GAP="${PROBE_GAP:-400}"
DROP="${DROP:-0.85}"
# Probe sample: SR quantization is 1/PROBE_EPS and binomial noise at p~0.15 is ~6% at 32
# episodes vs ~4.5% at 64 — probes run concurrently with training, so a larger sample mostly
# costs probe latency, not training throughput. Keep it well under the probe-gap wall-clock.
PROBE_ENVS="${PROBE_ENVS:-16}"
PROBE_EPS="${PROBE_EPS:-32}"
export AWS_PROFILE="${AWS_PROFILE-chrisryu-gpu}"   # dash, not :- — node-local mode passes AWS_PROFILE=
                                                   # (EMPTY) to use the instance role; a nonexistent
                                                   # named profile makes every aws call fail silently
                                                   # (v14: both checkpoint publishes died on it)
source "$AWS_DIR/lib.sh" >/dev/null 2>&1
SSH="${SSH:-$(ensure_ssm_ssh_wrapper 2>/dev/null)}"   # node-local mode: SSH=bash DEST=-c
DEST="${DEST:-ubuntu@$NODE}"

LOGDIR="${LOGDIR:-}"   # node-local mode passes it directly (no devbox train log to grep)
i=0
while [ -z "$LOGDIR" ] && [ $i -lt 60 ]; do
  LOGDIR=$(grep -oE "\[rl-trainer\] log_dir=\S+" "$TRAIN_LOG" 2>/dev/null | tail -1 | sed 's/.*log_dir=//')
  if [ -n "$LOGDIR" ]; then break; fi
  sleep 30; i=$((i+1))
done
if [ -z "$LOGDIR" ]; then echo "WATCH_ABORT no log_dir after 30min"; exit 1; fi
RUN_NAME=$(basename "$LOGDIR")
echo "WATCH_START logdir=$LOGDIR min_iter=$MIN_ITER probe_gap=$PROBE_GAP drop=$DROP"

last_eval=-1000; best=0; best_it=-1; declines=0
while true; do
  ALIVE=$(timeout 90 "$SSH" "$DEST" "pgrep -f 'learning.train --trainer rl' >/dev/null && echo yes || echo no" 2>/dev/null | tr -dc a-z)
  LATEST=$(timeout 90 "$SSH" "$DEST" "ls $LOGDIR/model_*.pt 2>/dev/null | sed 's/.*model_//' | sed 's/\.pt//' | sort -n | tail -1" 2>/dev/null | tr -dc 0-9)
  if [ "$ALIVE" != "yes" ]; then echo "WATCH_END training finished naturally, latest ckpt $LATEST"; break; fi
  if [ -n "$LATEST" ] && [ "$LATEST" -ge $((last_eval + PROBE_GAP)) ]; then
    # One eval at a time node-wide: under multi-run packing, concurrent 826-env probe sims
    # contend for the GPU and die at spawn (rc=245).
    while timeout 60 "$SSH" "$DEST" "pgrep -f '[e]val_service' >/dev/null" 2>/dev/null; do sleep 30; done
    echo "--- probe model_$LATEST at $(date -u +%H:%M) ---"
    OUT=$(timeout 500 "$SSH" "$DEST" "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && cd /home/ubuntu/learning_repo && SIM_ROOT=/home/ubuntu python -m learning.eval.eval_service --checkpoint $LOGDIR/model_$LATEST.pt --wbt --reference_dir /home/ubuntu/sim_references --num_envs $PROBE_ENVS --episodes $PROBE_EPS --seed 42 --experiment tracking --train probe-$LATEST 2>&1 | grep -E 'EVAL_OVER_SERVICE_OK|WBT_BREAKDOWN|TERM_BREAKDOWN'" 2>/dev/null)
    echo "$OUT"
    FINAL=$(echo "$OUT" | grep -oE "SR=[0-9.]+" | head -1 | sed 's/SR=//')   # decision metric = eval/SR (the KPI)
    LIFTS=$(echo "$OUT" | grep "final-frame" | sed 's/.*track_success[^=]*=//' | sed 's|/32.*||')
    last_eval=$LATEST
    if [ -n "$FINAL" ]; then
      better=$(awk -v f="$FINAL" -v b="$best" 'BEGIN { if (f > b) print 1; else print 0 }')
      # b > 0.02: a pre-ignition noise flicker (e.g. 0.0012 at iter 300) must not arm the decline
      # trap — two 0.000 probes past MIN_ITER would kill a run that simply hasn't ignited yet.
      dropped=$(awk -v f="$FINAL" -v b="$best" -v d="$DROP" 'BEGIN { if (b > 0.02 && f < b * d) print 1; else print 0 }')
      if [ "$better" = "1" ]; then
        best=$FINAL; best_it=$LATEST; declines=0
      elif [ "$dropped" = "1" ]; then
        declines=$((declines+1))
      else
        declines=0
      fi
      echo "PROBE it=$LATEST final=$FINAL lifts=$LIFTS best=$best at_it=$best_it declines=$declines"
      # Probe results onto the RUN PAGE (wandb-or-it-didn't-happen): summary-only update, safe
      # alongside the live training writer. Best-effort — a wandb hiccup must not stop the watch.
      # ${CONDA_SH:-…}: lib.sh leaves set -u ON — a bare unset $CONDA_SH here killed the watcher
      # mid-probe in node-local mode, and every respawn reset best/declines, disabling the auto-kill.
      timeout 90 "$SSH" "$DEST" "source ${CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}; conda activate isaaclab && python /home/ubuntu/learning_repo/learning/scripts/aws/wandb_probe_summary.py '$RUN_NAME' $LATEST $FINAL $best $best_it" 2>/dev/null | grep PROBE_SUMMARY || true
      if [ "$declines" -ge 2 ] && [ "$LATEST" -ge "$MIN_ITER" ]; then
        echo "WATCH_KILL confirmed decline: 2 probes below ${DROP}x best=$best at_it=$best_it, now it=$LATEST - killing training"
        # Publish BEFORE killing: the kill can sever this script's own SSM session (see pkill note),
        # and checkpoints must never depend on surviving it (v6+v7: every kill path lost the publish).
        echo "WATCH_PUBLISH checkpoints to $CKPT_S3_ROOT/$RUN_NAME/ (pre-kill)"
        timeout 600 "$SSH" "$DEST" "aws s3 cp $LOGDIR $CKPT_S3_ROOT/$RUN_NAME --recursive --exclude '*' --include 'model_*.pt' 2>&1 | tail -1" || echo "WATCH_PUBLISH_FAILED (publish manually from $LOGDIR)"
        # Kill ONLY this run's trainer + its sim server: under multi-run packing a bare pkill on
        # 'learning.train' murders the sibling run. The run tag (from the log-dir name) is matched
        # against each candidate's /proc/PID/environ (WBT_RUN_TAG rides the environment, not the
        # cmdline — [b]racket patterns keep the remote shell from self-matching).
        RUN_TAG=$(basename "$LOGDIR" | sed 's/^[0-9-]*_//; s/_[0-9]*it.*//')
        timeout 90 "$SSH" "$DEST" "for p in \$(pgrep -f '[l]earning.train --trainer rl'; pgrep -f '[s]erver.py'); do tr '\0' '\n' < /proc/\$p/environ 2>/dev/null | grep -q '^WBT_RUN_TAG=$RUN_TAG\$' && kill -9 \$p; done" 2>/dev/null
        sleep 10
        break
      fi
    else
      echo "PROBE it=$LATEST no breakdown parsed, eval hiccup - continuing"
    fi
  fi
  sleep 240
done

echo "WATCH_PUBLISH checkpoints to $CKPT_S3_ROOT/$RUN_NAME/"
timeout 600 "$SSH" "$DEST" "aws s3 cp $LOGDIR $CKPT_S3_ROOT/$RUN_NAME --recursive --exclude '*' --include 'model_*.pt' 2>&1 | tail -1" || echo "WATCH_PUBLISH_FAILED (publish manually from $LOGDIR)"
echo "WATCH_DONE best_eval=$best at model_$best_it run=$RUN_NAME"
