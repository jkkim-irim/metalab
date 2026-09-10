from __future__ import annotations

import torch


def step_edge(env) -> tuple[torch.Tensor, torch.Tensor]:
    at = env.buffer("_at_step", fill=-1, dtype=torch.long)
    first = at < 0
    advanced = at != env.common_step_counter
    at.fill_(env.common_step_counter)
    return advanced, first
