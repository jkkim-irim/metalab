#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# ### RUNS ON THE NODE — not on your machine. Shipped + dispatched by `setup_sim_node.sh provision`. ###
#
# remote/provision.sh — runs ON the node (as ubuntu) to provision the Newton stack.
# Crash-loud on guarded stages. Logs to ~/allex_setup.log. (Runnable by hand on the
# node for debugging; normally you just call `setup_sim_node.sh provision` from your machine.)
#
# Self-contained: restores a prebuilt env SNAPSHOT from S3 (pulled with the node's own
# IAM role — no creds, no reference to any specific/running box):
#   ENV_SNAPSHOT_S3/env_isaaclab.tar -> ~/miniconda3/envs/<env>   (conda env: kitless isaaclab
#                                                                  fork + newton + mujoco-warp)
#   ENV_SNAPSHOT_S3/isaaclab.tar     -> ~/IsaacLab                (the IsaacLab fork source)
# The Newton stack is KITLESS (newton + mujoco-warp; NO Isaac Sim / Omniverse), so isaacsim is
# intentionally absent. This snapshot is the ENV BASELINE ONLY — no project code. The sim + trainer
# code is deployed separately by learning/scripts/aws/rl_train.sh (rsync over SSM-SSH + an editable reinstall of
# the sim packages), so nothing project-specific is ever staged on S3.
# ──────────────────────────────────────────────────────────────────────────────
set -o pipefail   # NOT -u: conda activate trips on nounset
export DEBIAN_FRONTEND=noninteractive

ENV_SNAPSHOT_S3="${ENV_SNAPSHOT_S3:?set ENV_SNAPSHOT_S3=s3://.../envsnap (holds env_isaaclab.tar + isaaclab.tar)}"
ENVNAME="${CONDA_ENV:-isaaclab}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/ubuntu/IsaacLab}"
ENVDIR="$HOME/miniconda3/envs/$ENVNAME"
mark(){ echo "=== [setup] $* ($(date -u +%H:%M:%S)) ==="; }

mark "1/4 apt: Vulkan/GL/X libs + ffmpeg (Newton headless render deps)"
sudo apt-get update -y -qq
sudo apt-get install -y -qq --no-install-recommends curl \
  libvulkan1 vulkan-tools libgl1 libglu1-mesa libegl1 libgomp1 \
  libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 libsm6 libice6 \
  libxt6 libxrender1 libxkbcommon0 libxcb-cursor0 ffmpeg || { echo "[setup] apt FAILED"; exit 1; }
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json 2>/dev/null || true   # duplicate-ICD GPU double-enum bug

mark "2/4 miniconda (base only; the env itself comes from the snapshot)"
if [ ! -d "$HOME/miniconda3" ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3" || { echo "[setup] miniconda FAILED"; exit 1; }
fi

mark "3/4 restore conda env + IsaacLab fork from snapshot ($ENV_SNAPSHOT_S3) via node IAM role"
# Paths match across nodes (/home/ubuntu/...), so the conda env relocates without conda-unpack.
rm -rf "$ENVDIR"; mkdir -p "$HOME/miniconda3/envs"
# `zcat -f` handles a gzip-compressed snapshot (snapshot_env.sh uses pigz) AND passes an
# uncompressed .tar through — `tar -xf -` alone can't auto-detect gzip from a (non-seekable) pipe.
aws s3 cp "$ENV_SNAPSHOT_S3/env_isaaclab.tar" - | zcat -f | tar -C "$HOME/miniconda3/envs" -xf - || { echo "[setup] env restore FAILED (check ENV_SNAPSHOT_S3 + node IAM s3 read)"; exit 1; }
rm -rf "$ISAACLAB_DIR"
aws s3 cp "$ENV_SNAPSHOT_S3/isaaclab.tar" - | zcat -f | tar -C "$(dirname "$ISAACLAB_DIR")" -xf - || { echo "[setup] IsaacLab restore FAILED"; exit 1; }

mark "4/4 verify (kitless: isaaclab + newton resolve; isaacsim intentionally absent)"
P="$ENVDIR/bin/python"
"$P" --version
"$P" -c "import importlib.util as u, sys; sys.exit(0 if (u.find_spec('isaaclab') and u.find_spec('newton')) else 1)" \
  && echo "[setup] isaaclab + newton resolve OK" || { echo "[setup] CORE IMPORT FAILED"; exit 1; }
# Project code is NOT in this snapshot: learning/scripts/aws/rl_train.sh rsyncs sim/isaaclab + learning onto the
# node and editable-installs the sim packages (envs/robot/assets) into this env.

mark "SETUP_DONE"
