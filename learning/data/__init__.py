"""ALLEX learning — data layer.

Lerobot-free reimplementation of the LeRobot 0.4.4 v3.0 dataset machinery, plus ALLEX's
episode-wise split / dataloader:

  lerobot_dataset.py  LeRobotDataset / LeRobotDatasetMetadata (v3.0 parquet + torchcodec video)
  sampler.py          EpisodeAwareSampler
  factory.py          make_dataset (build a LeRobotDataset from a TrainPipelineConfig)
  transforms.py       image augmentation
  video_utils.py      torchcodec frame decode
  dataset.py          episode-wise train/val split + EpisodeAwareSampler DataLoader
                      (+ a make_dataset re-export)
  contract.py         the wir_v1 dataset contract + validate_wir_contract (see WIR_FORMAT.md)

Only torch / torchvision / torchcodec / pyarrow / pandas / numpy / PIL + std lib are used —
no ``import lerobot`` and no HuggingFace ``datasets``.

Datasets are read as LeRobot format **v3.0** on disk, but the stable interface models depend on is
versioned separately as **wir_v1** (``contract.py`` / ``WIR_FORMAT.md``) — so the contract can
evolve without overloading lerobot's codebase_version. (That is also why this is a flat package, not
a ``v3/`` namespace.)
"""
