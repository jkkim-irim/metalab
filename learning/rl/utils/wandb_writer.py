# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""W&B logging writer for rsl_rl — no tensorboard dependency (internalized).

Reimplemented from rsl_rl's tensorboard-based ``WandbSummaryWriter`` as a W&B-only writer (no
``torch.utils.tensorboard.SummaryWriter`` base), keeping the same class name + method surface
(add_scalar / store_config / save_file / save_video / stop) so ``logger.py`` uses it unchanged —
which keeps the trainer venv free of the tensorboard package. Checkpoints are deliberately NOT
uploaded to W&B (no ``save_model``): the runner writes each ``model_*.pt`` to ``log_dir`` and pushes
it straight to S3 (the system of record), so duplicating them into W&B run files only inflated
storage cost. See ``OnPolicyRunner._upload_ckpt_s3``.
"""
from __future__ import annotations

from dataclasses import asdict
import os
import pathlib

import wandb

# Runner-cfg keys kept OUT of the run config. The nets (actor/critic/algorithm) arrive via store_config;
# the rest are constants or restate what the run page already says (the project, that the logger is W&B,
# the resume globs of a run that resumes nothing) — as columns they only crowd out the tuned ones.
_CFG_SKIP = frozenset({"actor", "critic", "algorithm",
                       "class_name", "logger", "neptune_project", "wandb_project",
                       "resume", "load_run", "load_checkpoint", "check_for_nan"})


class WandbSummaryWriter:
    """rsl_rl-compatible writer that logs to W&B only (no tensorboard base)."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        run_name = os.path.split(log_dir)[-1]
        project = cfg["wandb_project"]
        entity = os.environ.get("WANDB_USERNAME")
        # Run config = log_dir + every scalar/dict runner-cfg knob (experiment name, iterations,
        # dataset provenance, eval_cfg, ...) — otherwise which dataset a run trained on or what the
        # eval measured is untraceable from W&B. Dicts render as nested config groups; the big
        # nested specs still arrive via store_config.
        scalars = {k: v for k, v in cfg.items()
                   if isinstance(v, (str, int, float, bool, dict)) and k not in _CFG_SKIP}
        wandb.init(project=project, entity=entity, name=run_name,
                   config={"log_dir": log_dir, **scalars}, settings=wandb.Settings(start_method="thread"))
        # Continuations resume the ORIGINAL run (WANDB_RUN_ID + WANDB_RESUME env): the stored
        # config then differs from this launch's (run_name, notes, resume_ckpt, ...) and plain
        # config.update raises ConfigError - which killed the first w4nax0bt continuation at
        # startup. Resumed runs must update with allow_val_change.
        self.logged_videos: set[str] = set()

    def store_config(self, env_cfg, train_cfg) -> None:
        wandb.config.update({"train_cfg": train_cfg}, allow_val_change=True)
        try:
            cfg = env_cfg.to_dict()
        except Exception:
            cfg = asdict(env_cfg)
        # MetaLab ships the task contract's tuning knobs as `recipe` (sim/metalab/contract/recipe.py).
        # Lifted to the TOP level: nested under env_cfg every knob column reads `env_cfg.recipe.gate.…`,
        # and the recipe is the one thing a run is compared on.
        recipe = cfg.pop("recipe", None)
        if recipe:
            wandb.config.update({"recipe": recipe}, allow_val_change=True)
        wandb.config.update({"env_cfg": cfg}, allow_val_change=True)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False) -> None:
        wandb.log({tag: scalar_value}, step=global_step)

    def save_file(self, path) -> None:
        wandb.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it) -> None:
        if video.name not in self.logged_videos:
            wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self.logged_videos.add(video.name)

    def stop(self, exit_code: int = 0) -> None:
        # exit_code: 0 → the run finishes as "finished"; non-zero (255 = SIGINT/Stop) → wandb marks the
        # run "killed" instead of the "crashed" it would show if the process just died without finishing.
        wandb.finish(exit_code=exit_code)
