"""make_dataset — lerobot-free reimplementation of lerobot 0.4.4's dataset factory.

Faithful port of ``lerobot/datasets/factory.py`` (0.4.4) for the single, non-streaming
``LeRobotDataset`` path the ALLEX trainer uses: it wires up ``delta_timestamps`` from the policy
config (``resolve_delta_timestamps``), optional image transforms, video backend, ``tolerance_s``,
and the imagenet-stats override for camera keys.
"""
import torch

from learning.data.contract import validate_wir_contract
from learning.data.image_aug import make_cpu_transform
from learning.data.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from learning.data.s3_sync import ensure_local_dataset

# Mirror of lerobot.utils.constants
ACTION = "action"
REWARD = "next.reward"
OBS_PREFIX = "observation."

# Mirror of lerobot.datasets.factory.IMAGENET_STATS
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(cfg, ds_meta: LeRobotDatasetMetadata) -> dict[str, list] | None:
    """Resolve delta_timestamps from the policy config's delta-index properties.

    Mirrors lerobot's ``resolve_delta_timestamps``: for the reward / action / observation keys the
    corresponding ``*_delta_indices`` (when not None) are converted to second-deltas via ``/ fps``.
    Returns ``None`` if the resulting dict is empty (e.g. ACT, where only ``action`` has indices ->
    a single-key dict). For ACT specifically: ``observation_delta_indices`` and
    ``reward_delta_indices`` are ``None`` and ``action_delta_indices == list(range(chunk_size))``.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(cfg) -> LeRobotDataset:
    """Set up delta timestamps and image transforms, then create a ``LeRobotDataset``.

    ``cfg`` is the ALLEX/lerobot ``TrainPipelineConfig`` (``cfg.dataset`` is a ``DatasetConfig`` and
    ``cfg.policy`` is a ``PreTrainedConfig``). Only the single-dataset, non-streaming branch is
    supported (the only path the ALLEX trainer exercises); streaming / multi-dataset raise.
    """
    # "cpu" aug runs per-item in the dataloader workers (the picklable make_cpu_transform callable,
    # applied to each camera image in __getitem__). "gpu"/"off" leave this None — gpu aug runs on the
    # batch in the train loop (bc_trainer), via the SAME apply_image_aug. The one knob is image_aug.
    image_transforms = (
        make_cpu_transform(cfg.dataset.image_aug_config)
        if cfg.dataset.image_aug == "cpu"
        else None
    )

    if not isinstance(cfg.dataset.repo_id, str):
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
    if getattr(cfg.dataset, "streaming", False):
        raise NotImplementedError("StreamingLeRobotDataset isn't supported here.")

    # Datasets are always sourced from S3: mirror s3_uri -> local cache (COMPLETE-marker guarded)
    # before loading, then load from that local root.
    root = ensure_local_dataset(cfg.dataset.s3_uri, cfg.dataset.root)

    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=root, revision=cfg.dataset.revision)
    # Every dataset entering the pipeline must satisfy the wir_v1 contract (fail loud at the boundary).
    validate_wir_contract(ds_meta)
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    # Camera keys use ImageNet mean/std (the converters omit image stats). ``setdefault`` creates the
    # entry when stats.json lacks it, so this holds whether or not image stats were pre-written.
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
