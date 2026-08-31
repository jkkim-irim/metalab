"""Self-contained ACT configuration (de-lerobot'd).

`ACTConfig` mirrors LeRobot 0.4.4's `lerobot.policies.act.configuration_act.ACTConfig` field
for field (same defaults, same feature-derivation properties) so that:

* a model built from this config has the exact same architecture, and
* a LeRobot checkpoint's config json round-trips into it.

The LeRobot config inherits a lot of HF-hub / draccus machinery from `PreTrainedConfig`; here we
drop all of that and keep only the fields the ACT model, the (un)normalizers, and our trainer use.
The `FeatureType` / `NormalizationMode` enums and `PolicyFeature` dataclass are copied verbatim
from `lerobot.configs.types` (string-valued enums so json (de)serialization is trivial).
"""

from dataclasses import dataclass, field
from enum import Enum

from learning.model.act.constants import ACTION, OBS_ENV_STATE, OBS_STATE, OBS_STR


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


@dataclass
class ACTConfig:
    """Configuration class for the Action Chunking Transformers policy.

    Field names, defaults and the feature-derivation properties match LeRobot 0.4.4 exactly.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    # ALLEX extension (not in LeRobot): select which named blocks of the 132-D ``observation.state``
    # to consume, by group name from ``learning/data/allex_modality.STATE_GROUPS`` (q / dq / tau) —
    # the same way GR00T expresses its state (``state_keys``), sourced from the one modality map.
    # The selection must form a contiguous ``[0:N]`` prefix; ``build_act`` resolves it to N and slices
    # the leading N dims (+ truncates stats), and ``Normalize`` slices the incoming tensor to match.
    # ``["q"]`` = pos-only 44-D (the S1 recipe); ``None`` = the full state (baseline). WITHOUT
    # reconverting the dataset.
    state_keys: list[str] | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # The PolicyFeature dicts describing the model's inputs/outputs. In LeRobot these are derived
    # from the dataset metadata by the policy factory; here `build_act` fills them in the same way.
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    # Where to place the model (kept so the config is a faithful stand-in for LeRobot's).
    device: str | None = None

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in
    # the code that means only the first layer is used. Here we match the original implementation by
    # setting this to 1. See https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset.
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # ALLEX extension (LeRobot's ACT uses no scheduler): optional LR schedule, built in
    # ``learning/model/act_policy.build``. Options:
    #   "cosine" = linear warmup over ``warmup_steps`` then cosine decay to 0 across the whole run.
    #   "wsd"    = warmup -> constant peak -> cosine decay to 0 over the FINAL ``lr_decay_ratio`` of
    #              the run. Decouples run length from the anneal, so more ``--steps`` = more effective
    #              training (the constant body grows) rather than a slower decay everywhere.
    # The horizon is the run's total optimizer steps (``TrainPipelineConfig.steps``), so a scheduled
    # ACT run should be STEP-based (``--steps N``) — the builder can't see the length of an
    # ``--epochs`` run. None = no scheduler (baseline).
    lr_scheduler: str | None = None
    warmup_steps: int = 0
    lr_decay_ratio: float = 0.2  # "wsd" only: fraction of total steps for the final cosine anneal

    def __post_init__(self) -> None:
        """Input validation (not exhaustive); mirrors LeRobot's `ACTConfig.__post_init__`."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per model "
                f"invocation. Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} "
                "for `chunk_size`."
            )
        if self.n_obs_steps < 1:
            raise ValueError(f"n_obs_steps must be >= 1. Got {self.n_obs_steps}.")

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        if not self.input_features:
            return None
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        if not self.input_features:
            return {}
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        if not self.output_features:
            return None
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    @property
    def observation_delta_indices(self) -> list | None:
        # None for the single-frame case (baseline: the dataloader returns (B, ·) obs tensors,
        # byte-identical to before). For n_obs_steps>1 request an observation-history window of the
        # last n_obs_steps frames (oldest -> current, inclusive of 0) via LeRobot delta_timestamps;
        # the dataloader then returns (B, n_obs_steps, ·) obs tensors (clamped + padded at ep start).
        if self.n_obs_steps <= 1:
            return None
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert LeRobot dataset features to `PolicyFeature`s.

    Faithful copy of `lerobot.datasets.utils.dataset_to_policy_features` so `build_act` can accept
    a LeRobot dataset-metadata `features` dict and derive the same input/output feature shapes the
    LeRobot policy factory would.
    """
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            ftype = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            names = ft["names"]
            # Backward compatibility for "channel" (an error introduced in LeRobotDataset v2.0).
            if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == OBS_ENV_STATE:
            ftype = FeatureType.ENV
        elif key.startswith(OBS_STR):
            ftype = FeatureType.STATE
        elif key.startswith(ACTION):
            ftype = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(type=ftype, shape=shape)

    return policy_features
