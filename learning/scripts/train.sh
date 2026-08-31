#!/usr/bin/env bash
# Train ACT on the canonical 132-D bolt-drilling dataset — reproduces reference run 5m3dzdwe.
# Runs ON the node: ensures the venv + pinned deps, syncs the dataset from S3, runs the recipe.
# Override via env: VENV, DATASET_S3 / DATASET_ROOT / DATASET_REPO_ID, OUTPUT_DIR,
#   EXTRA_FLAGS (extra/override flags appended to train, e.g. a smoke test).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# 0) ensure the venv + pinned env (shared helper; creates $VENV once outside the repo dir, sets $PY)
source "$SCRIPT_DIR/_setup_venv.sh"

# 1) dataset — always sourced from S3; make_dataset syncs DATASET_S3 -> DATASET_ROOT (local cache) on
#    start (COMPLETE-marker guarded) and injects ImageNet image stats in-code (use_imagenet_stats).
DATASET_S3="${DATASET_S3:-s3://wirobotics-internal/chrisryu/datasets/allex/lerobot/bolt_drilling/v3_132d_20260510_12_382ep}"
DATASET_NAME="$(basename "$DATASET_S3")"                                  # e.g. v3_132d_20260510_12_382ep
DATASET_ROOT="${DATASET_ROOT:-/opt/dlami/nvme/data/$DATASET_NAME}"        # local cache of DATASET_S3
DATASET_REPO_ID="${DATASET_REPO_ID:-chrisryu/$DATASET_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/dlami/nvme/outputs/act_$(date +%Y%m%d_%H%M%S)}"

# 2) train — 5m3dzdwe recipe (chunk80 / n_enc4 / latent32 / kl0.5 / lr1e-5 / bs8 / seed1000, no aug).
#    image_aug is pinned "off" explicitly now that the default is "gpu", to keep this the no-aug reference.
#    EXTRA_FLAGS appends/overrides (e.g. a smoke test: --steps 50 --val_every_n_steps 25 --val_max_batches 5).
exec "$PY" -m learning.train \
    --policy.type act \
    --dataset.repo_id "$DATASET_REPO_ID" \
    --dataset.s3_uri "$DATASET_S3" \
    --dataset.root "$DATASET_ROOT" \
    --dataset.video_backend torchcodec \
    --dataset.image_aug off \
    --policy.chunk_size 80 \
    --policy.n_action_steps 1 \
    --policy.temporal_ensemble_coeff 0.05 \
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
    --policy.push_to_hub false \
    --tolerance_s 0.1 \
    --batch_size 8 \
    --num_workers 24 \
    --compile_mode reduce-overhead \
    --seed 1000 \
    --output_dir "$OUTPUT_DIR" \
    --wandb.enable true \
    --wandb.project chrisryu-dev \
    --wandb.entity wiin2-wirobotics-inc \
    --epochs 3 \
    --val_ratio 0.2 \
    --split_seed 42 \
    --rollout_horizons 0,20,40,79 \
    ${EXTRA_FLAGS:-}
