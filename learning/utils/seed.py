"""Seeding utility (de-lerobot'd).

Faithful copy of ``lerobot.utils.random_utils.set_seed`` for the call ALLEX makes:
``set_seed(cfg.seed, accelerator=accelerator)``. Seeds ``random``, ``numpy``, ``torch`` (and CUDA),
then defers to accelerate's ``set_seed`` so a distributed run matches LeRobot's RNG state exactly.
"""

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, accelerator: Any | None = None) -> None:
    """Set seed for reproducibility (random / numpy / torch / cuda, + accelerate if given)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if accelerator:
        from accelerate.utils import set_seed as _accelerate_set_seed

        _accelerate_set_seed(seed)
