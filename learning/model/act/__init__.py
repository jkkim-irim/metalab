"""Self-contained, lerobot-free ACT policy (faithful to LeRobot 0.4.4).

Public API:
    build_act          -- build (policy, preprocessor, postprocessor, optimizer, lr_scheduler)
    ACTPolicy, ACT     -- the policy wrapper and the underlying nn.Module
    ACTConfig          -- the policy configuration dataclass
    Normalize, Unnormalize  -- the (un)normalization modules
    PolicyFeature, FeatureType, NormalizationMode  -- config types
    dataset_to_policy_features  -- LeRobot dataset features -> PolicyFeature mapping
    OBS_STATE, OBS_IMAGES, OBS_ENV_STATE, ACTION  -- batch key constants
"""

from learning.model.act.build import build_act
from learning.model.act.configuration import (
    ACTConfig,
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    dataset_to_policy_features,
)
from learning.model.act.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
)
from learning.model.act.modeling import ACT, ACTPolicy
from learning.model.act.normalize import Normalize, Unnormalize

__all__ = [
    "ACT",
    "ACTION",
    "OBS_ENV_STATE",
    "OBS_IMAGES",
    "OBS_STATE",
    "ACTConfig",
    "ACTPolicy",
    "FeatureType",
    "Normalize",
    "NormalizationMode",
    "PolicyFeature",
    "Unnormalize",
    "build_act",
    "dataset_to_policy_features",
]
