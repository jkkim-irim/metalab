#!/usr/bin/env bash
# Train the "S1" ACT variant on the canonical 132-D bolt-drilling dataset — the optimization recipe
# (see the S1 plan): short action horizon (chunk 16, to compare against GR00T's 16), pos-only state
# (state_keys=["q"] = 44-D), and a warmup->cosine LR schedule. STEP-based (not --epochs) so the cosine
# horizon == the run length. Clones train.sh's env/data setup verbatim; only the train recipe differs
# (train.sh is left untouched so the 5m3dzdwe baseline stays reproducible).
#
# Ablation results (curves, tables, compute):
#   https://d1iitptfxhu64e.cloudfront.net/chrisryu/docs/act-s1-ablations.html
#
# Ablate one axis at a time via EXTRA_FLAGS, e.g.:
#   EXTRA_FLAGS="--policy.use_vae false"                     # R4: drop CVAE -> plain DETR
#   EXTRA_FLAGS="--policy.n_obs_steps 2"                     # R1: observation history (PR2)
#   EXTRA_FLAGS="--policy.action_parameterization relative_arm"  # R6: relative arm actions (PR2)
# Other overrides: VENV, DATASET_S3/DATASET_ROOT/DATASET_REPO_ID, OUTPUT_DIR, STEPS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# 0) ensure the venv + pinned env (shared helper; creates $VENV once outside the repo dir, sets $PY)
source "$SCRIPT_DIR/_setup_venv.sh"

# 1) dataset — always sourced from S3. make_dataset syncs DATASET_S3 -> DATASET_ROOT (a local cache)
#    on start, guarded by a COMPLETE marker (learning/data/s3_sync.py), and injects ImageNet image
#    stats in-code (use_imagenet_stats), so no manual s3-sync or stats.json patch is needed here.
DATASET_S3="${DATASET_S3:-s3://wirobotics-internal/chrisryu/datasets/allex/lerobot/bolt_drilling/v3_132d_20260510_12_382ep}"
DATASET_NAME="$(basename "$DATASET_S3")"
DATASET_ROOT="${DATASET_ROOT:-/opt/dlami/nvme/data/$DATASET_NAME}"
DATASET_REPO_ID="${DATASET_REPO_ID:-chrisryu/$DATASET_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/dlami/nvme/outputs/act_s1_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-124000}"   # ~3 epochs at bs8 / val_ratio 0.2 — matches the baseline ablation's 124k

# 2) train — S1 recipe. Deltas vs train.sh: chunk 80->16, state_keys=["q"] (pos/q only, 44-D),
#    warmup->cosine LR (step-based --steps so the horizon aligns), rollout horizons rescaled < 16.
#    FK uses a PINNED, versioned URDF snapshot vendored under learning/assets/ (rides along in the
#    `learning` git archive). The ALLEX model is still in flux and can't be pegged to an
#    ALLEX_DESCRIPTION_DIR version yet, so we vendor a named snapshot for reproducible fk_mm — bump the
#    filename when the robot version changes. Override the path with URDF_PATH=.
# Durable checkpoints: periodically sync checkpoints/ to S3 so a killed run OR a node terminated for
# cost still keeps the rolling step_<N> checkpoints (+ the best-select step + the `last` symlink).
# (The ACT best-config checkpoint was lost exactly this way — the recipe never synced off the node.)
# --delete mirrors the bounded rolling window (pruned step dirs are removed from S3 too). Background
# loop + a final sync on exit (incl. the SIGTERM a node stop/terminate sends). Not `exec`, so the
# trap can run.
CKPT_S3="${CKPT_S3:-s3://wirobotics-internal/chrisryu/models/$(basename "$OUTPUT_DIR")}"
echo "durable checkpoints -> $CKPT_S3/checkpoints (every ${SYNC_EVERY:-600}s + on exit)"
( while true; do sleep "${SYNC_EVERY:-600}"; \
    aws s3 sync "$OUTPUT_DIR/checkpoints" "$CKPT_S3/checkpoints" --delete --only-show-errors || true; \
  done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; \
      aws s3 sync "$OUTPUT_DIR/checkpoints" "$CKPT_S3/checkpoints" --delete --only-show-errors || true; \
      echo "final checkpoint sync -> $CKPT_S3/checkpoints"' EXIT

"$PY" -m learning.train \
    --policy.type act \
    --dataset.repo_id "$DATASET_REPO_ID" \
    --dataset.s3_uri "$DATASET_S3" \
    --dataset.root "$DATASET_ROOT" \
    --dataset.video_backend torchcodec \
    --dataset.image_aug gpu \
    --policy.chunk_size 16 \
    --policy.n_action_steps 1 \
    --policy.temporal_ensemble_coeff 0.05 \
    --policy.state_keys '["q"]' \
    --policy.dim_model 512 \
    --policy.n_heads 8 \
    --policy.n_encoder_layers 4 \
    --policy.n_decoder_layers 1 \
    --policy.use_vae true \
    --policy.latent_dim 32 \
    --policy.kl_weight 0.5 \
    --policy.vision_backbone resnet18 \
    --policy.pretrained_backbone_weights ResNet18_Weights.IMAGENET1K_V1 \
    --policy.dropout 0.1 \
    --policy.optimizer_lr 1e-05 \
    --policy.optimizer_weight_decay 0.0001 \
    --policy.optimizer_lr_backbone 1e-05 \
    --policy.lr_scheduler cosine \
    --policy.warmup_steps 1000 \
    --policy.push_to_hub false \
    --tolerance_s 0.1 \
    --batch_size 8 \
    --num_workers 24 \
    --compile_mode reduce-overhead \
    --seed 1000 \
    --output_dir "$OUTPUT_DIR" \
    --wandb.enable true \
    --wandb.project "${WANDB_PROJECT:-chrisryu-s1}" \
    --wandb.entity wiin2-wirobotics-inc \
    --steps "$STEPS" \
    --val_every_n_steps "${VAL_EVERY:-5000}" \
    --val_max_batches "${VAL_MAX_BATCHES:-40}" \
    --save_every_n_steps "${SAVE_EVERY:-5000}" \
    --save_last_n "${SAVE_LAST_N:-5}" \
    --select_metric "${SELECT_METRIC:-l1_all_unnorm}" \
    --val_flow_matching_steps "${VAL_FM_STEPS:-0}" \
    --val_ratio 0.2 \
    --split_seed 42 \
    --rollout_horizons 0,4,8,15 \
    --fk_validation \
    --urdf_path "${URDF_PATH:-$REPO_ROOT/learning/assets/ALLEX_0.1.3.urdf}" \
    --fk_video_n 3 \
    --fk_video_every_n_steps "${FK_VIDEO_EVERY:-20000}" \
    ${EXTRA_FLAGS:-}
