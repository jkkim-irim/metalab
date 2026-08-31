"""Weights & Biases logger (de-lerobot'd).

Reproduces the ``lerobot.rl.wandb_utils.WandBLogger`` surface ALLEX's trainer uses:
``WandBLogger(cfg)``, ``logger._wandb.run`` (a settable ``.name``), ``logger.log_dict(dict, step)``,
and ``logger._wandb.run.finish()``. ``wandb`` itself is the only extra dependency.

Dropped vs LeRobot: env-fps video logging, model-artifact upload, RL custom-step-key handling, and
the filesystem run-id resumption (ALLEX passes ``wandb.run_id`` explicitly when resuming). The
``cfg_to_group`` tag/group construction is kept (sans the ``env`` tag, which our config has no field
for) so run grouping/tags match the BC use.
"""

import datetime
import logging
import os
from typing import Any

from termcolor import colored

from learning.configs.config import policy_type_name


def cfg_to_group(cfg: Any, return_list: bool = False) -> list[str] | str:
    """Group/tag name for logging (LeRobot ``cfg_to_group`` minus the env tag)."""
    lst = [
        f"policy:{policy_type_name(cfg.policy)}",
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    lst = [tag[:64] for tag in lst]  # wandb rejects tags > 64 chars
    return lst if return_list else "-".join(lst)


class WandBLogger:
    """Helper to log metrics to wandb (the subset ALLEX's BC trainer uses)."""

    def __init__(self, cfg: Any, extras: Any = None):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = None
        self._group = cfg_to_group(cfg)

        os.environ["WANDB_SILENT"] = "True"
        import wandb

        # Run name = [name]-[UTC MMDD-HHMM]-[short SHA] (CLAUDE.md) so a run traces to exact code,
        # e.g. "act-0623-0117-30142e8". The SHA is passed in by the launcher (ALLEX_SHA); the node
        # runs from a git-archive with no .git, so there's no git fallback — absent => "nosha".
        sha = os.environ.get("ALLEX_SHA", "nosha")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%m%d-%H%M")
        run_name = f"{self.job_name}-{stamp}-{sha}"

        # Full run config = the parsed TrainPipelineConfig (incl. dataset.image_aug) + the ALLEX
        # "extras" (fk_*, val_*,
        # epochs, ... — parsed separately from argv) + the code SHA, so the wandb config is the single
        # source of truth for the run (no need to read the launch command to know what it did).
        wandb_config = cfg.to_dict()
        if extras is not None:
            wandb_config["extras"] = vars(extras) if not isinstance(extras, dict) else dict(extras)
        wandb_config["allex_sha"] = sha
        wandb_run_id = cfg.wandb.run_id if cfg.wandb.run_id else None
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=run_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True),
            dir=self.log_dir,
            config=wandb_config,
            save_code=False,
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        run_id = wandb.run.id
        cfg.wandb.run_id = run_id
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(
            f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}"
        )
        self._wandb = wandb

    def log_dict(self, d: dict, step: int | None = None) -> None:
        """Log a flat dict of scalar metrics at ``step``, keys logged verbatim.

        Keys are expected to already carry their wandb section (e.g. ``train/loss``, ``val/loss``,
        ``val/fk_mm_finger_mean``); the caller owns the section so train/val metrics group correctly
        (this wrapper no longer prepends a single ``mode/`` to everything).
        """
        if step is None:
            raise ValueError("step must be provided.")

        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'WandB logging of key "{k}" was ignored as its type "{type(v)}" is not '
                    "handled by this wrapper."
                )
                continue
            self._wandb.log(data={k: v}, step=step)
