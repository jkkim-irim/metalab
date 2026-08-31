"""Shared LR schedulers for all policies (ACT, GR00T, ...).

Pure-torch ``LambdaLR`` builders — no ``transformers`` dependency, so both the lean ACT venv and the
GR00T venv use the SAME schedule code (the cosine matches ``transformers.get_cosine_schedule_with_warmup``
exactly). Built in each policy's outer ``build(cfg, dataset)`` where ``cfg.steps`` (the run horizon) is
in scope.
"""
import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup for ``warmup_steps`` then cosine decay to 0 over ``total_steps``.

    Matches ``transformers.get_cosine_schedule_with_warmup``. ``warmup_steps=0`` -> pure cosine from
    step 0.
    """
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return LambdaLR(optimizer, lr_lambda)


def wsd_with_warmup(optimizer, warmup_steps: int, total_steps: int, decay_steps: int) -> LambdaLR:
    """Warmup-Stable-Decay: linear warmup -> constant peak LR -> cosine decay to 0 over the final
    ``decay_steps``.

    Unlike ``cosine_with_warmup`` (whose entire LR profile scales with ``total_steps``, so a longer
    run just decays more slowly *everywhere* and can't add effective training), WSD **decouples run
    length from the anneal**: extra steps extend the constant-LR body, then the same-shaped tail
    anneals to 0. So "train longer" is a real lever again. ``decay_steps`` = ``lr_decay_ratio *
    total_steps`` (set by the caller).
    """
    decay_start = max(warmup_steps, total_steps - decay_steps)
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return LambdaLR(optimizer, lr_lambda)
