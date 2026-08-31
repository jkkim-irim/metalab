"""ALLEX learning — ACT policy / processor / optimizer construction.

Builds a fresh policy + pre/post processors + optimizer for one training (or eval) run via the
internal ACT builder (learning.model.act) — no lerobot.
"""
from learning.model.act.build import build_act
from learning.model.lr_schedulers import cosine_with_warmup, wsd_with_warmup


def build(cfg, dataset):
    """Build a fresh policy, pre/post processors, optimizer and scheduler for one run.

    Returns ``(policy, preprocessor, postprocessor, optimizer, lr_scheduler)``. ``build_act`` returns
    a ``None`` scheduler (LeRobot ACT default); when ``cfg.policy.lr_scheduler`` is set we build it
    here, where ``cfg.steps`` (the run horizon) is in scope.
    """
    policy, preprocessor, postprocessor, optimizer, lr_scheduler = build_act(cfg.policy, dataset.meta)
    sched = cfg.policy.lr_scheduler
    if sched == "cosine":
        lr_scheduler = cosine_with_warmup(optimizer, cfg.policy.warmup_steps, cfg.steps)
    elif sched == "wsd":
        decay_steps = int(cfg.policy.lr_decay_ratio * cfg.steps)
        lr_scheduler = wsd_with_warmup(optimizer, cfg.policy.warmup_steps, cfg.steps, decay_steps)
    elif sched is not None:
        raise ValueError(f"Unknown ACT lr_scheduler {sched!r} (supported: None, 'cosine', 'wsd').")
    return policy, preprocessor, postprocessor, optimizer, lr_scheduler
