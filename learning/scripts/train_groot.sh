#!/usr/bin/env bash
# Finetune GR00T-N1.7 on the canonical 132-D bolt-drilling dataset, fully inside learning/
# (--policy.type groot). Runs ON the node. Reuses the same wir_v1 dataloader + gpu_aug + validation
# metrics + BCTrainer as the ACT recipe (learning/scripts/train.sh) — see learning/model/groot/README.md.
#
# Self-contained: the GR00T model code is vendored in-repo (learning/model/groot — no external gr00t
# package / Isaac-GR00T clone), and every artifact is pulled from S3, so no gated HuggingFace access
# or HF token is needed:
#   * a dedicated venv ($GROOT_VENV) built from learning/scripts/requirements-groot.txt — the torch
#     2.7.1 line (SEPARATE from the ACT venv's torch 2.10). Built here on first run.
#   * the N1.7-3B base ($GROOT_BASE) — self-contained (bundles the Qwen3-VL backbone's config+tokenizer
#     under backbone_assets/); synced from S3 and staged into an offline HF cache, so from_pretrained
#     resolves locally (HF_HUB_OFFLINE=1). No external Cosmos-Reason2-2B repo / gated token.
#   * the dataset — synced from S3.
#
# Override via env: GROOT_VENV, GROOT_BASE, GROOT_BASE_S3, HF_HOME, DATASET_S3,
#   DATASET_ROOT, DATASET_REPO_ID, OUTPUT_DIR, BATCH_SIZE, STEPS, VAL_EVERY, VAL_RATIO,
#   VAL_MAX_BATCHES, VAL_EVAL_MAX_BATCHES, IMAGE_AUG (off|cpu|gpu, default gpu), GPUS, ZERO_STAGE,
#   TUNE_LLM, TUNE_VISUAL, LR, EXTRA_FLAGS. (#7 trainer is step-driven: --steps is the total
#   optimizer-step budget; validation cadence is --val_every_n_steps.)
#   In-training CLOSED-LOOP sim eval (off unless SIM_EVAL_EVERY>0): SIM_EVAL_EVERY (steps between runs,
#   a multiple of VAL_EVERY), SIM_EVAL_SUITE (e.g. libero_90), SIM_EVAL_TASKS (e.g. 0 or 0,11),
#   SIM_EVAL_EPISODES, SIM_EVAL_SIM_PYTHON (the SIM venv's python), SIM_EVAL_RESOLUTION,
#   SIM_EVAL_MAX_EPISODE_STEPS, SIM_EVAL_REPLAN_STEPS, SIM_EVAL_BASE_SEED, SIM_EVAL_INFERENCE_TIMESTEPS,
#   SIM_EVAL_VIDEO_EPISODES (>0 -> log rollout MP4s to wandb: sim_eval/videos/*), SIM_EVAL_VIDEO_FPS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GROOT_VENV="${GROOT_VENV:-/opt/dlami/nvme/groot-venv}"
PY="$GROOT_VENV/bin/python"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/hf_cache}"
# GR00T at batch 2 sits ~38 GB on a 40 GB A100; without expandable segments the CUDA allocator
# fragments over the limit and the run is OOM-SIGKILLed mid-training (~step 150, no Python traceback).
# Default it on (override via env if you need the legacy allocator).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Launchers run as root over SSM with HOME unset -> wandb/torch kernel-cache + the uv install dir fall
# back oddly; pin a default so caches/binaries land somewhere sane.
export HOME="${HOME:-/root}"
export PATH="$HOME/.local/bin:$PATH"
# The vendored GR00T model code lives in-repo (learning.model.groot); only REPO_ROOT on the path.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Offline: the backbone's config+tokenizer are served from the local cache below (the backbone is built
# from config — no Cosmos weights); no huggingface_hub / network. transformers is no longer a dependency
# — the Qwen3-VL backbone + processor are vendored under learning/model/groot/_hf.
export HF_HUB_OFFLINE=1
GROOT_BASE="${GROOT_BASE:-/opt/dlami/nvme/groot-base-n1.7-3b}"
GROOT_BASE_S3="${GROOT_BASE_S3:-s3://wirobotics-internal/chrisryu/models/groot/groot-base-n1.7-3b}"
COSMOS_REV="9ce19a195e423419c349abfc86fd07178b230561"   # HF-cache snapshot id for the backbone config+tokenizer

# 0) venv — build once from the pinned requirements (uv; installs uv + ffmpeg libs if the box lacks them).
# Keep uv's cache on the big ephemeral disk (same fs as $GROOT_VENV): the torch 2.7.1 cu-wheels are
# ~5 GB and the DLAMI root fs is small, and a same-fs cache lets uv hardlink instead of full-copy.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/opt/dlami/nvme/uv-cache}"
if ! command -v uv >/dev/null 2>&1; then echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; fi
if ! ldconfig -p | grep -q 'libavutil\.so'; then   # torchcodec loads system FFmpeg shared libs
  echo "installing ffmpeg"; apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg
fi
if ! "$PY" -c 'import torch, flash_attn, diffusers' 2>/dev/null; then
  echo "building GR00T venv at $GROOT_VENV from requirements-groot.txt (torch 2.7.1 line)"
  uv venv --python 3.10 "$GROOT_VENV"
  # unsafe-best-match: pick the best version across BOTH indexes (torch/vision/codec from the cu128
  # index, everything else from PyPI) instead of uv's default first-index-only match.
  VIRTUAL_ENV="$GROOT_VENV" uv pip install --index-strategy unsafe-best-match -r "$SCRIPT_DIR/requirements-groot.txt"
fi
# verify the env + vendored code — fail loud, never silently fall back
[ -x "$PY" ] || { echo "FATAL: GR00T venv missing at $GROOT_VENV." >&2; exit 1; }
"$PY" -c "import learning.model.groot.model.gr00t_n1d7.gr00t_n1d7, flash_attn, pytorch_kinematics" \
  || { echo "FATAL: GR00T venv incomplete — need learning.model.groot + flash_attn + pytorch_kinematics." >&2; exit 1; }

# 1) N1.7-3B base — pull from S3 if absent. Self-contained: bundles the Qwen3-VL backbone's
#    config+tokenizer under backbone_assets/ (no external Cosmos-Reason2-2B repo).
if [ ! -f "$GROOT_BASE/config.json" ]; then
  echo "fetching N1.7-3B base from $GROOT_BASE_S3 -> $GROOT_BASE"
  mkdir -p "$GROOT_BASE"; aws s3 sync "$GROOT_BASE_S3/" "$GROOT_BASE" --only-show-errors
fi
[ -f "$GROOT_BASE/backbone_assets/config.json" ] || { echo "FATAL: base checkpoint missing backbone_assets/ (Qwen3-VL config+tokenizer)." >&2; exit 1; }
# Stage the bundled config+tokenizer (~11 MB, NO weights) into the offline HF cache so
# from_pretrained("nvidia/Cosmos-Reason2-2B") resolves the config+tokenizer locally. The backbone is
# built from config (Cosmos's 4.87 GB of weights are redundant — overwritten by the base checkpoint).
COSMOS_HUB="$HF_HOME/hub/models--nvidia--Cosmos-Reason2-2B"
if [ ! -f "$COSMOS_HUB/snapshots/$COSMOS_REV/config.json" ]; then
  echo "staging backbone_assets/ (Qwen3-VL config+tokenizer) -> offline HF cache"
  mkdir -p "$COSMOS_HUB/snapshots/$COSMOS_REV" "$COSMOS_HUB/refs"
  cp -a "$GROOT_BASE/backbone_assets/." "$COSMOS_HUB/snapshots/$COSMOS_REV/"
  printf '%s' "$COSMOS_REV" > "$COSMOS_HUB/refs/main"   # HF offline resolves ref -> snapshot
fi

# 2) dataset — always sourced from S3 (same source as the ACT recipe). make_dataset syncs
#    DATASET_S3 -> DATASET_ROOT (local cache) on start (COMPLETE-marker guarded) and injects ImageNet
#    image stats in-code (use_imagenet_stats), so no manual s3-sync or stats.json patch is needed.
DATASET_S3="${DATASET_S3:-s3://wirobotics-internal/chrisryu/datasets/allex/lerobot/bolt_drilling/v3_132d_20260510_12_382ep}"
DATASET_NAME="$(basename "$DATASET_S3")"
DATASET_ROOT="${DATASET_ROOT:-/opt/dlami/nvme/data/$DATASET_NAME}"
DATASET_REPO_ID="${DATASET_REPO_ID:-chrisryu/$DATASET_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/dlami/nvme/outputs/groot_$(date +%Y%m%d_%H%M%S)}"

# 3) finetune — action head only (frozen Cosmos backbone), bf16 autocast, FK validation (l1_all +
#    fingertip/wrist mm). val_eval_max_batches bounds the per-epoch val passes (GR00T's 4-step
#    denoising predict makes a full val pass costly). train.py sets bf16 mixed precision for groot.
# Multi-GPU: one process per visible GPU (8 on a p4d.24xlarge) via accelerate launch.
#   ZERO_STAGE>0 (default 2) -> DeepSpeed ZeRO: stage 2 shards optimizer+gradients so the FULL VLM
#   finetune (TUNE_LLM/TUNE_VISUAL) fits in 40 GB; ZERO_STAGE=0 -> plain DDP (only fits the
#   frozen-backbone recipe). TUNE_LLM/TUNE_VISUAL default true (unfreeze the Cosmos VLM); set both
#   false for the frozen-backbone baseline. eff_batch = batch_size x GPUS.
GPUS="${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; GPUS="${GPUS:-1}"
ZERO_STAGE="${ZERO_STAGE:-2}"
if [ "$ZERO_STAGE" -gt 0 ]; then
  ACC_ARGS="--use_deepspeed --zero_stage $ZERO_STAGE --zero3_init_flag false --offload_optimizer_device none --offload_param_device none"
  echo "launching GR00T finetune on $GPUS GPU(s) (DeepSpeed ZeRO-$ZERO_STAGE, tune_llm=${TUNE_LLM:-true} tune_visual=${TUNE_VISUAL:-true})"
else
  ACC_ARGS=""
  echo "launching GR00T finetune on $GPUS GPU(s) (DDP, tune_llm=${TUNE_LLM:-true} tune_visual=${TUNE_VISUAL:-true})"
fi
exec "$GROOT_VENV/bin/accelerate" launch --num_processes "$GPUS" --num_machines 1 \
    --mixed_precision bf16 $ACC_ARGS \
    -m learning.train \
    --policy.type groot \
    --policy.device cuda \
    --policy.chunk_size 16 \
    --policy.base_model_path "$GROOT_BASE" \
    --policy.tune_projector true \
    --policy.tune_diffusion_model true \
    --policy.tune_llm "${TUNE_LLM:-true}" \
    --policy.tune_visual "${TUNE_VISUAL:-true}" \
    --policy.gradient_checkpointing true \
    --policy.optimizer_lr "${LR:-2e-5}" \
    --dataset.repo_id "$DATASET_REPO_ID" \
    --dataset.s3_uri "$DATASET_S3" \
    --dataset.root "$DATASET_ROOT" \
    --dataset.video_backend torchcodec \
    --batch_size "${BATCH_SIZE:-2}" \
    --num_workers 4 \
    --output_dir "$OUTPUT_DIR" \
    --wandb.enable true \
    --wandb.project chrisryu-dev \
    --wandb.entity wiin2-wirobotics-inc \
    --dataset.image_aug "${IMAGE_AUG:-gpu}" \
    --steps "${STEPS:-3000}" \
    --val_every_n_steps "${VAL_EVERY:-1000}" \
    --val_ratio "${VAL_RATIO:-0.2}" \
    --val_max_batches "${VAL_MAX_BATCHES:-0}" \
    --val_eval_max_batches "${VAL_EVAL_MAX_BATCHES:-0}" \
    --rollout_horizons 0,8,15 \
    --log_every_n_steps 200 \
    --sim_eval_every_n_steps "${SIM_EVAL_EVERY:-0}" \
    --sim_eval_suite "${SIM_EVAL_SUITE:-}" \
    --sim_eval_tasks "${SIM_EVAL_TASKS:-}" \
    --sim_eval_episodes "${SIM_EVAL_EPISODES:-8}" \
    --sim_eval_sim_python "${SIM_EVAL_SIM_PYTHON:-}" \
    --sim_eval_resolution "${SIM_EVAL_RESOLUTION:-256}" \
    --sim_eval_max_episode_steps "${SIM_EVAL_MAX_EPISODE_STEPS:-0}" \
    --sim_eval_replan_steps "${SIM_EVAL_REPLAN_STEPS:-0}" \
    --sim_eval_base_seed "${SIM_EVAL_BASE_SEED:-0}" \
    --sim_eval_inference_timesteps "${SIM_EVAL_INFERENCE_TIMESTEPS:-0}" \
    --sim_eval_video_episodes "${SIM_EVAL_VIDEO_EPISODES:-0}" \
    --sim_eval_video_fps "${SIM_EVAL_VIDEO_FPS:-15}" \
    ${EXTRA_FLAGS:-}
