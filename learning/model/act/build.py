"""Build an ACT policy + (un)normalizers + optimizer + scheduler (de-lerobot'd).

`build_act` is the in-house equivalent of LeRobot 0.4.4's
``make_policy`` + ``make_pre_post_processors`` + ``make_optimizer_and_scheduler`` for ACT, with
matching behavior:

* features: derived from a LeRobot dataset-metadata object (``ds_meta.features``) the same way
  ``make_policy`` does, or passed directly as an already-built ``input/output_features`` mapping.
  ``output_features`` = the ACTION feature(s); ``input_features`` = everything else.
* preprocessor / postprocessor: ``Normalize`` over ``{input_features, output_features}`` and
  ``Unnormalize`` over ``output_features`` only — same split as ``make_act_pre_post_processors``.
* optimizer: ``AdamW`` with ``lr=optimizer_lr``, ``weight_decay=optimizer_weight_decay`` and two
  param groups (everything-but-backbone at base lr; ``model.backbone.*`` at ``optimizer_lr_backbone``)
  — matching ``ACTConfig.get_optimizer_preset`` + ``ACTPolicy.get_optim_params``.
* scheduler: ``None`` (``ACTConfig.get_scheduler_preset`` returns ``None``).
"""

import torch

from learning.data import allex_modality
from learning.model.act.configuration import (
    ACTConfig,
    FeatureType,
    PolicyFeature,
    dataset_to_policy_features,
)
from learning.model.act.constants import OBS_STATE
from learning.model.act.modeling import ACTPolicy
from learning.model.act.normalize import Normalize, Unnormalize


def _features_from_ds_meta(ds_meta) -> dict[str, PolicyFeature]:
    return dataset_to_policy_features(ds_meta.features)


def build_act(cfg: ACTConfig, ds_meta_or_features):
    """Build ``(policy, preprocessor, postprocessor, optimizer, lr_scheduler)`` for one ACT run.

    Args:
        cfg: an ``ACTConfig``. Its ``input_features`` / ``output_features`` are (re)derived here, so
            they may be left empty when ``ds_meta_or_features`` carries dataset metadata.
        ds_meta_or_features: either a LeRobot dataset-metadata object (must expose ``.features`` and,
            for normalization, ``.stats``) or an already-built ``dict[str, PolicyFeature]``. When a
            bare feature dict is given, normalization stats are unavailable so the (un)normalizers are
            built without stats (they then act as IDENTITY, exactly like LeRobot with no stats).

    Returns:
        ``(policy, preprocessor, postprocessor, optimizer, lr_scheduler)`` where ``lr_scheduler`` is
        ``None``.
    """
    if isinstance(ds_meta_or_features, dict):
        features = ds_meta_or_features
        dataset_stats = None
    else:
        features = _features_from_ds_meta(ds_meta_or_features)
        dataset_stats = getattr(ds_meta_or_features, "stats", None)

    # Same feature split as make_policy: outputs are the ACTION features, inputs are the rest.
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    # ALLEX extension: consume only the named ``state_keys`` blocks of observation.state (e.g. ["q"] =
    # pos/q only -> leading 44 dims), resolved to a leading-N prefix via the canonical modality map.
    # Rewrite the state feature shape so the model's projections are sized to the slice, and truncate
    # the state normalization stats to match — on a shallow copy so ds_meta.stats (shared with the
    # GR00T path + validation) is never mutated. Normalize slices the incoming tensor (below).
    state_slice = None
    n_state = None if cfg.state_keys is None else allex_modality.state_prefix_dim(cfg.state_keys)
    if n_state is not None:
        state_ft = cfg.input_features.get(OBS_STATE)
        if state_ft is None:
            raise ValueError("state_keys is set but there is no observation.state input feature.")
        full = state_ft.shape[0]
        if not (0 < n_state <= full):
            raise ValueError(
                f"state_keys={cfg.state_keys} -> {n_state} dims out of range for "
                f"observation.state dim {full}."
            )
        if n_state < full:
            cfg.input_features[OBS_STATE] = PolicyFeature(type=state_ft.type, shape=(n_state,))
            state_slice = n_state
            if dataset_stats is not None and OBS_STATE in dataset_stats:
                dataset_stats = dict(dataset_stats)  # shallow copy — do not mutate ds_meta.stats
                dataset_stats[OBS_STATE] = {
                    k: (v[:n_state] if hasattr(v, "__len__") and len(v) == full else v)
                    for k, v in dataset_stats[OBS_STATE].items()
                }

    policy = ACTPolicy(cfg)
    if cfg.device is not None:
        policy.to(cfg.device)

    # Pre/post processors (normalization) — same feature sets as make_act_pre_post_processors.
    preprocessor = Normalize(
        features={**cfg.input_features, **cfg.output_features},
        norm_map=cfg.normalization_mapping,
        stats=dataset_stats,
        state_slice=state_slice,
    )
    postprocessor = Unnormalize(
        features=cfg.output_features,
        norm_map=cfg.normalization_mapping,
        stats=dataset_stats,
    )
    if cfg.device is not None:
        preprocessor.to(cfg.device)
        postprocessor.to(cfg.device)

    # Optimizer: AdamW with ACT's backbone-vs-rest param groups (matches get_optim_params + preset).
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=cfg.optimizer_lr,
        weight_decay=cfg.optimizer_weight_decay,
    )

    # ACT uses no LR scheduler (get_scheduler_preset returns None).
    lr_scheduler = None

    return policy, preprocessor, postprocessor, optimizer, lr_scheduler
