"""Self-contained (un)normalization for ACT (de-lerobot'd).

LeRobot 0.4.4 normalizes/unnormalizes inside its processor pipeline
(`NormalizerProcessorStep` / `UnnormalizerProcessorStep` in
`lerobot.processor.normalize_processor`). These two `nn.Module`s reproduce that math **exactly**:

* ``MEAN_STD``  :  forward ``(x - mean) / (std + eps)``      inverse ``x * std + mean``
* ``MIN_MAX``   :  forward ``2 * (x - min) / (max - min) - 1`` inverse ``(x + 1) / 2 * (max - min) + min``
* ``QUANTILES`` :  same shape as MIN_MAX using q01/q99
* ``QUANTILE10``:  same shape as MIN_MAX using q10/q90
* ``IDENTITY``  :  passthrough (also used when a key has no stats)

with ``eps = 1e-8`` and the ``denom == 0 -> eps`` guard for the (min_max / quantile) range, matching
LeRobot's `_apply_transform`.

Stats are converted to tensors and registered as buffers (so `.to(device)` / `state_dict` work),
preserving the stored shape — LeRobot stores image stats as ``(C, 1, 1)`` so they broadcast against
``(B, C, H, W)`` images, and vector stats as ``(D,)`` so they broadcast against ``(B, D)`` / ``(B, T, D)``.

`Normalize` is applied to the input batch (observations + action), `Unnormalize` to the model's
action output — the same split LeRobot's pre/post processors use.
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from learning.model.act.configuration import FeatureType, NormalizationMode, PolicyFeature
from learning.model.act.constants import ACTION, OBS_STATE

_EPS = 1e-8


def _to_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None,
) -> dict[str, dict[str, Tensor]]:
    """Build {key: {stat_name: float32 Tensor}} for keys whose norm mode needs stats.

    Mirrors which stat names each mode consumes in LeRobot's `_apply_transform`. Missing stats for a
    needed key are left out (the key then falls through to IDENTITY, exactly like LeRobot, which
    returns the tensor unchanged when ``key not in self._tensor_stats``).
    """
    needed: dict[NormalizationMode, tuple[str, ...]] = {
        NormalizationMode.MEAN_STD: ("mean", "std"),
        NormalizationMode.MIN_MAX: ("min", "max"),
        NormalizationMode.QUANTILES: ("q01", "q99"),
        NormalizationMode.QUANTILE10: ("q10", "q90"),
    }
    out: dict[str, dict[str, Tensor]] = {}
    if stats is None:
        return out
    for key, feature in features.items():
        mode = norm_map.get(feature.type.value, NormalizationMode.IDENTITY)
        if mode == NormalizationMode.IDENTITY or mode not in needed:
            continue
        if key not in stats:
            continue
        key_stats = {}
        for stat_name in needed[mode]:
            if stat_name not in stats[key]:
                # A required stat is absent -> skip the key so it behaves as IDENTITY (LeRobot does
                # the same; it never half-normalizes).
                key_stats = {}
                break
            key_stats[stat_name] = torch.as_tensor(stats[key][stat_name], dtype=torch.float32)
        if key_stats:
            out[key] = key_stats
    return out


def _apply(tensor: Tensor, mode: NormalizationMode, stats: dict[str, Tensor], inverse: bool) -> Tensor:
    """LeRobot `_apply_transform` for a single tensor/key (stats already on the right device/dtype)."""
    if mode == NormalizationMode.MEAN_STD:
        mean, std = stats["mean"], stats["std"]
        denom = std + _EPS
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / denom
    if mode == NormalizationMode.MIN_MAX:
        min_val, max_val = stats["min"], stats["max"]
        denom = max_val - min_val
        denom = torch.where(denom == 0, torch.tensor(_EPS, device=tensor.device, dtype=tensor.dtype), denom)
        if inverse:
            return (tensor + 1) / 2 * denom + min_val
        return 2 * (tensor - min_val) / denom - 1
    if mode == NormalizationMode.QUANTILES:
        q01, q99 = stats["q01"], stats["q99"]
        denom = q99 - q01
        denom = torch.where(denom == 0, torch.tensor(_EPS, device=tensor.device, dtype=tensor.dtype), denom)
        if inverse:
            return (tensor + 1.0) * denom / 2.0 + q01
        return 2.0 * (tensor - q01) / denom - 1.0
    if mode == NormalizationMode.QUANTILE10:
        q10, q90 = stats["q10"], stats["q90"]
        denom = q90 - q10
        denom = torch.where(denom == 0, torch.tensor(_EPS, device=tensor.device, dtype=tensor.dtype), denom)
        if inverse:
            return (tensor + 1.0) * denom / 2.0 + q10
        return 2.0 * (tensor - q10) / denom - 1.0
    return tensor


class _NormBase(nn.Module):
    """Shared buffer registration / mode lookup for `Normalize` and `Unnormalize`."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None,
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        # Per-key modes (default IDENTITY for any type not in the mapping).
        self._modes = {
            key: norm_map.get(ft.type.value, NormalizationMode.IDENTITY) for key, ft in features.items()
        }
        # Register stats as buffers so device moves / state_dict round-trips work. Buffer names can't
        # contain '.', so we sanitize and keep a key -> {stat_name: buffer_name} map.
        self._buffer_names: dict[str, dict[str, str]] = {}
        tensor_stats = _to_stats_buffers(features, norm_map, stats)
        for key, key_stats in tensor_stats.items():
            safe_key = key.replace(".", "_")
            self._buffer_names[key] = {}
            for stat_name, value in key_stats.items():
                buf_name = f"stats_{safe_key}_{stat_name}"
                self.register_buffer(buf_name, value, persistent=False)
                self._buffer_names[key][stat_name] = buf_name
        # ALLEX state-slice: leading dims of observation.state to keep (set by ``Normalize``; stays
        # None on the base / ``Unnormalize``). Persisted by save_pretrained so a reloaded checkpoint
        # slices the raw incoming state the same way (its stats + feature shape are already truncated).
        self._state_slice: int | None = None

    def _stats_for(self, key: str) -> dict[str, Tensor] | None:
        if key not in self._buffer_names:
            return None
        return {stat_name: getattr(self, buf) for stat_name, buf in self._buffer_names[key].items()}

    # ----------------------------------------------------------------- persistence (plain torch) --
    # Filename under the checkpoint's pretrained_model/ dir; each subclass uses its own so a
    # Normalize and an Unnormalize can both save into the same directory.
    _STATS_FILENAME = "normalizer_stats.pt"

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Persist features + per-type norm modes + stat tensors so this (un)normalizer reloads
        WITHOUT the dataset (LeRobot's processor steps persist their stats the same way). The stat
        buffers are ``persistent=False`` (excluded from ``state_dict``), so without this a checkpoint
        would carry no normalization stats and reload as IDENTITY."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": {
                k: {"type": ft.type.value, "shape": list(ft.shape)} for k, ft in self.features.items()
            },
            "norm_map": {t: m.value for t, m in self.norm_map.items()},
            "stats": {
                key: {sn: getattr(self, buf).detach().cpu() for sn, buf in bufs.items()}
                for key, bufs in self._buffer_names.items()
            },
            "state_slice": self._state_slice,
        }
        torch.save(payload, save_directory / self._STATS_FILENAME)

    @classmethod
    def from_pretrained(cls, save_directory: str | Path) -> "_NormBase":
        """Reconstruct the (un)normalizer saved by ``save_pretrained`` (stats included)."""
        payload = torch.load(Path(save_directory) / cls._STATS_FILENAME, weights_only=False)
        features = {
            k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
            for k, v in payload["features"].items()
        }
        norm_map = {t: NormalizationMode(m) for t, m in payload["norm_map"].items()}
        obj = cls(features, norm_map, payload["stats"])
        obj._state_slice = payload.get("state_slice")  # None for older checkpoints / Unnormalize
        return obj


class Normalize(_NormBase):
    """Normalize observation + action tensors in a batch dict (LeRobot `NormalizerProcessorStep`)."""

    _STATS_FILENAME = "preprocessor_stats.pt"

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None,
        state_slice: int | None = None,
    ):
        super().__init__(features, norm_map, stats)
        self._state_slice = state_slice

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key in self.features:
            if key not in batch:
                continue
            # ALLEX state-slice: keep only the leading dims of observation.state (e.g. pos/q) so the
            # model sees exactly its declared state dim. Applied before normalization (independent of
            # the norm mode), so it also runs when the state has no stats (falls through to IDENTITY).
            if self._state_slice is not None and key == OBS_STATE:
                batch[key] = batch[key][..., : self._state_slice]
            mode = self._modes[key]
            if mode == NormalizationMode.IDENTITY:
                continue
            key_stats = self._stats_for(key)
            if key_stats is None:
                continue
            batch[key] = _apply(batch[key], mode, key_stats, inverse=False)
        return batch


class Unnormalize(_NormBase):
    """Unnormalize the model's action output (LeRobot `UnnormalizerProcessorStep`).

    Built from `output_features` only, so it touches the ``action`` key. Works on either a batch dict
    (returns the dict with ``action`` unnormalized) or a bare action tensor (returns the tensor).
    """

    _STATS_FILENAME = "postprocessor_stats.pt"

    def forward(self, batch: dict[str, Tensor] | Tensor) -> dict[str, Tensor] | Tensor:
        if isinstance(batch, dict):
            batch = dict(batch)
            for key in self.features:
                if key not in batch:
                    continue
                mode = self._modes[key]
                if mode == NormalizationMode.IDENTITY:
                    continue
                key_stats = self._stats_for(key)
                if key_stats is None:
                    continue
                batch[key] = _apply(batch[key], mode, key_stats, inverse=True)
            return batch
        # Bare action tensor.
        mode = self._modes.get(ACTION, NormalizationMode.IDENTITY)
        key_stats = self._stats_for(ACTION)
        if mode == NormalizationMode.IDENTITY or key_stats is None:
            return batch
        return _apply(batch, mode, key_stats, inverse=True)
