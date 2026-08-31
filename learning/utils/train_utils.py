"""Checkpoint save / last-symlink utilities (de-lerobot'd).

Reproduces the two functions ALLEX's BC trainer calls:
  * ``save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, scheduler, preprocessor,
    postprocessor)``
  * ``update_last_checkpoint(ckpt_dir)``

LeRobot persists optimizer/scheduler/rng state via safetensors + bespoke json. Here we use plain
``torch.save`` (torch + std-lib only) for those, and delegate the policy/processor configs to their
own ``save_pretrained`` (the de-lerobot'd ``ACTPolicy.save_pretrained`` writes ``model.pt`` +
``config.json``). The on-disk layout mirrors LeRobot's so a checkpoint round-trips (policy weights
reload via ``ACTPolicy.from_pretrained(checkpoint_dir / 'pretrained_model')``):

    <checkpoint_dir>/
    ├── pretrained_model/
    │   ├── config.json            # policy config            (ACTPolicy.save_pretrained)
    │   ├── model.pt               # policy weights            (ACTPolicy.save_pretrained)
    │   ├── train_config.json      # the TrainPipelineConfig.to_dict()
    │   ├── preprocessor_stats.pt  # Normalize stats     (Normalize.save_pretrained)
    │   └── postprocessor_stats.pt # Unnormalize stats   (Unnormalize.save_pretrained)
    └── training_state/
        ├── optimizer_state.pt     # optimizer.state_dict()
        ├── scheduler_state.pt     # scheduler.state_dict()    (if scheduler is not None)
        ├── rng_state.pt           # python/numpy/torch(/cuda) RNG state
        └── training_step.json     # {"step": <step>}
"""

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
TRAIN_CONFIG_NAME = "train_config.json"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.pt"
SCHEDULER_STATE = "scheduler_state.pt"
RNG_STATE = "rng_state.pt"
LAST_CHECKPOINT_LINK = "last"


def update_last_checkpoint(checkpoint_dir: Path) -> None:
    """Point ``<parent>/last`` at ``checkpoint_dir`` (LeRobot ``update_last_checkpoint``)."""
    checkpoint_dir = Path(checkpoint_dir)
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def _save_rng_state(save_dir: Path) -> None:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, save_dir / RNG_STATE)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
) -> None:
    """Save training step + optimizer/scheduler/rng state under ``training_state/``."""
    save_dir = Path(checkpoint_dir) / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / TRAINING_STEP, "w") as f:
        json.dump({"step": int(train_step)}, f)
    _save_rng_state(save_dir)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), save_dir / OPTIMIZER_STATE)
    if scheduler is not None:
        torch.save(scheduler.state_dict(), save_dir / SCHEDULER_STATE)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: Any,
    policy: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    preprocessor: Any | None = None,
    postprocessor: Any | None = None,
) -> None:
    """Save a full checkpoint (policy + processors + train config + training state).

    Args mirror LeRobot's ``save_checkpoint``. ``policy`` must expose ``save_pretrained``; the
    processors are saved via their own ``save_pretrained`` when provided.
    """
    checkpoint_dir = Path(checkpoint_dir)
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    policy.save_pretrained(pretrained_dir)
    with open(pretrained_dir / TRAIN_CONFIG_NAME, "w") as f:
        json.dump(cfg.to_dict(), f, indent=4)

    # Persist the (un)normalizer stats — fail loud if a provided processor can't save (a checkpoint
    # without its normalization stats would silently reload as IDENTITY).
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)

    save_training_state(checkpoint_dir, step, optimizer, scheduler)
