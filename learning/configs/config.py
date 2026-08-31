"""Self-contained training-pipeline config (de-lerobot'd).

Reproduces the ``lerobot.configs.train.TrainPipelineConfig`` surface the ALLEX trainer uses,
dropping all the draccus / HF-hub machinery. Field names + defaults match LeRobot 0.4.4 for the
fields ``learning/train.py`` and ``learning/trainer/bc_trainer.py`` read, so a run parsed by our
parser produces the same values LeRobot's would for our argv (see ``parser.py``).

Pieces reproduced (LeRobot source in parens):
  * ``DatasetConfig``                  (lerobot.configs.default.DatasetConfig)
  * ``WandBConfig``                    (lerobot.configs.default.WandBConfig)
  * ``EvalConfig``                     (lerobot.configs.default.EvalConfig)
  * ``ImageAugConfig``                 (ALLEX — one device-agnostic aug; see learning/data/image_aug.py)
  * ``OptimizerConfig`` (adamw preset) (lerobot.optim.optimizers.AdamWConfig)
  * ``TrainPipelineConfig``            (lerobot.configs.train.TrainPipelineConfig)

The policy config is ``learning.model.act.configuration.ACTConfig`` (already de-lerobot'd); it is
selected by ``--policy.type act`` in the parser. ``cfg.optimizer`` is filled in by ``validate()``
from the policy's preset, exactly like LeRobot's ``use_policy_training_preset`` path.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import datetime as dt
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from learning.data.image_aug import ImageAugConfig
from learning.model.act.configuration import ACTConfig
from learning.model.groot.configuration import GrootConfig

# Map a policy config class -> its ``--policy.type`` choice name. The de-lerobot'd ``ACTConfig`` is
# not a draccus ChoiceRegistry, so it has no ``.type`` property; we derive the name here (mirroring
# what draccus's ``get_choice_name`` returned for the lerobot ACTConfig: "act"). Keep in sync with
# ``parser.POLICY_CHOICES``.
POLICY_TYPE_NAMES: dict[type, str] = {ACTConfig: "act", GrootConfig: "groot"}


def policy_type_name(policy: object) -> str:
    """Return the ``--policy.type`` name for a policy config instance ("act" for ACTConfig)."""
    name = POLICY_TYPE_NAMES.get(type(policy))
    if name is None:
        raise ValueError(f"Unknown policy config type {type(policy).__name__!r}.")
    return name


# LeRobot's DatasetConfig.video_backend defaults to get_safe_default_codec(), which returns
# "torchcodec" when torchcodec is importable else "pyav". Our argv always passes
# --dataset.video_backend explicitly (train.sh: torchcodec), so the default is never exercised; we
# pin the common "torchcodec" default to stay value-faithful when nothing is passed.
DEFAULT_VIDEO_BACKEND = "torchcodec"


@dataclass
class DatasetConfig:
    """Dataset config (LeRobot ``configs.default.DatasetConfig``)."""

    repo_id: str
    # Canonical dataset source — always an ``s3://`` URI. The trainer syncs it to ``root`` (a local
    # cache) before loading, guarded by a COMPLETE marker (``learning/data/s3_sync.py``), so the run
    # config records exactly where the data came from and an interrupted sync can never be used.
    s3_uri: str
    root: str | None = None   # local sync target / cache; derived from the s3_uri basename if None
    episodes: list[int] | None = None
    # The one image-augmentation knob: "off" (none), "cpu" (per-item in the dataloader workers), or
    # "gpu" (per-sample on-device in the train loop). cpu and gpu run the SAME augmentation
    # (``learning/data/image_aug.apply_image_aug``) — the choice is only WHERE it runs. Ranges +
    # fixed order live in ``image_aug_config``. ON ("gpu") by default.
    image_aug: str = "gpu"
    image_aug_config: ImageAugConfig = field(default_factory=ImageAugConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = DEFAULT_VIDEO_BACKEND
    streaming: bool = False

    def __post_init__(self) -> None:
        # Datasets are ALWAYS sourced from S3 — enforced at construction so both the CLI and
        # programmatic use fail loud, before any training starts.
        if not (isinstance(self.s3_uri, str) and self.s3_uri.startswith("s3://")):
            raise ValueError(f"dataset.s3_uri must be an s3:// URI (got {self.s3_uri!r}).")


@dataclass
class WandBConfig:
    """Weights & Biases config (LeRobot ``configs.default.WandBConfig``)."""

    enable: bool = False
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # 'online' | 'offline' | 'disabled'; None -> 'online'


@dataclass
class EvalConfig:
    """Eval config (LeRobot ``configs.default.EvalConfig``)."""

    n_episodes: int = 50
    batch_size: int = 50
    use_async_envs: bool = False

    def __post_init__(self) -> None:
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your "
                f"command to increase the number of episodes to match the batch size "
                f"(e.g. `eval.n_episodes={self.batch_size}`), or lower the batch size "
                f"(e.g. `eval.batch_size={self.n_episodes}`)."
            )


@dataclass
class OptimizerConfig:
    """AdamW optimizer preset (LeRobot ``optim.optimizers.AdamWConfig``, the ACT preset).

    LeRobot has an abstract ``OptimizerConfig`` with several registered subclasses; ACT's
    ``get_optimizer_preset()`` always returns an ``AdamWConfig``, which is the only optimizer our
    pipeline builds. We collapse that to this single dataclass. ``grad_clip_norm`` is the field the
    trainer reads (``cfg.optimizer.grad_clip_norm``); the rest match AdamW's defaults so ``build()``
    constructs an identical ``torch.optim.AdamW``.
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    grad_clip_norm: float = 10.0

    @property
    def type(self) -> str:
        return "adamw"

    def build(self, params: Any) -> Any:
        """Build a ``torch.optim.AdamW`` (matches LeRobot's ``AdamWConfig.build``)."""
        return torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


# LeRobot's ACTConfig.get_optimizer_preset(): AdamWConfig(lr=optimizer_lr, weight_decay=optimizer_
# weight_decay). betas/eps/grad_clip_norm keep AdamWConfig's defaults. We reproduce it here rather
# than adding a method to the (shared, untouched) ACTConfig.
def act_optimizer_preset(policy: ACTConfig) -> OptimizerConfig:
    return OptimizerConfig(lr=policy.optimizer_lr, weight_decay=policy.optimizer_weight_decay)


def act_scheduler_preset(policy: ACTConfig) -> None:
    """ACT uses no LR scheduler (LeRobot ``ACTConfig.get_scheduler_preset`` returns None)."""
    return None


def groot_optimizer_preset(policy: GrootConfig) -> OptimizerConfig:
    """GR00T AdamW preset — carries the policy's ``grad_clip_norm`` (the trainer reads
    ``cfg.optimizer.grad_clip_norm``). The actual optimizer/scheduler are built in
    ``groot_policy.build`` (cosine warmup); this only feeds ``validate()`` + the grad-clip read."""
    return OptimizerConfig(
        lr=policy.optimizer_lr,
        weight_decay=policy.optimizer_weight_decay,
        grad_clip_norm=policy.grad_clip_norm,
    )


def optimizer_preset(policy):
    """Dispatch the optimizer preset on policy type (ACT vs GR00T)."""
    if isinstance(policy, GrootConfig):
        return groot_optimizer_preset(policy)
    return act_optimizer_preset(policy)


@dataclass
class TrainPipelineConfig:
    """Top-level training config (LeRobot ``configs.train.TrainPipelineConfig``).

    Only the fields ALLEX's trainer / dataset path read are kept; the env / RL / PEFT / RA-BC /
    resume-from-hub machinery is dropped. ``policy`` is an ``ACTConfig`` (selected by
    ``--policy.type act``); ``optimizer`` is filled by ``validate()`` from the policy preset.
    """

    dataset: DatasetConfig
    policy: ACTConfig | GrootConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    resume: bool = False
    seed: int | None = 1000
    num_workers: int = 4
    # torch.compile the inner ACT model. None = off; "default" or "reduce-overhead" (CUDA graphs).
    # (Compiled on the inner model, not the ACTPolicy wrapper, which .item()-graph-breaks.)
    compile_mode: str | None = None
    batch_size: int = 8
    # Episode-wise train/val split (held-out validation set); split_seed makes it reproducible
    # independently of the training seed.
    val_ratio: float = 0.2
    split_seed: int = 42
    steps: int = 100_000
    eval_freq: int = 20_000
    log_freq: int = 200
    tolerance_s: float = 1e-4
    save_checkpoint: bool = True
    save_freq: int = 20_000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    # Rename map for the observation to override the image and state keys.
    rename_map: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Reproduces the subset of LeRobot ``TrainPipelineConfig.validate()`` our path uses.

        Fail-loud on a missing policy / pre-existing output dir, derive ``job_name`` and a default
        ``output_dir``, and (the load-bearing bit) fill ``optimizer`` from the policy training
        preset when ``use_policy_training_preset`` is set.
        """
        if self.policy is None:
            raise ValueError(
                "Policy is not configured. Please specify a policy with `--policy.type act`."
            )

        if not self.job_name:
            self.job_name = f"{policy_type_name(self.policy)}"

        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"Output directory {self.output_dir} already exists and resume is {self.resume}. "
                f"Please change your output directory so that {self.output_dir} is not overwritten."
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")

        # Image-aug selector is the single source of truth: "cpu" turns on the dataloader
        # transforms; "gpu" is read in the train loop (bc_trainer builds gpu_aug); "off" → neither.
        if self.dataset.image_aug not in ("off", "cpu", "gpu"):
            raise ValueError(
                f"--dataset.image_aug must be one of 'off' | 'cpu' | 'gpu' "
                f"(got {self.dataset.image_aug!r})."
            )

        if not self.use_policy_training_preset and (
            self.optimizer is None or self.scheduler is None
        ):
            raise ValueError(
                "Optimizer and Scheduler must be set when the policy presets are not used."
            )
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = optimizer_preset(self.policy)
            self.scheduler = act_scheduler_preset(self.policy)

    def to_dict(self) -> dict[str, Any]:
        """Plain-json-able nested dict of the config (LeRobot ``to_dict`` -> ``draccus.encode``).

        Matches draccus.encode for our fields: dataclasses -> nested dicts, enums -> their value,
        ``Path`` -> str, tuples -> lists. Policy enums (``FeatureType``/``NormalizationMode``) and
        ``PolicyFeature`` are dataclass/enum and so handled by the generic walk. The policy's
        ``type`` (a property, not a field) is injected to mirror draccus's ChoiceRegistry encoding.
        """
        out = _encode(self)
        if self.policy is not None and isinstance(out.get("policy"), dict):
            out["policy"]["type"] = policy_type_name(self.policy)
        if self.optimizer is not None and isinstance(out.get("optimizer"), dict):
            out["optimizer"]["type"] = self.optimizer.type
        return out


def _encode(obj: Any) -> Any:
    """draccus.encode-compatible recursive encoder (enums -> value, Path -> str, tuple -> list)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _encode(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


# Re-export so callers can asdict() the optimizer the same way LeRobot does, if needed.
__all__ = [
    "DatasetConfig",
    "EvalConfig",
    "ImageAugConfig",
    "OptimizerConfig",
    "TrainPipelineConfig",
    "WandBConfig",
    "act_optimizer_preset",
    "act_scheduler_preset",
    "asdict",
]
