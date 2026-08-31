"""Pure-numpy state/action slice + reassembly for the GR00T adapter (torch / gr00t-free).

Factored out of ``groot_policy`` so the slice/reassembly logic is importable — and unit-testable —
without pulling in torch or the vendored gr00t. ``groot_policy`` calls these, so a test that
exercises them IS exercising the production path (no re-implementation). Every function is
parameterized by a :class:`~learning.model.groot.modality_spec.ModalitySpec`, so the ALLEX 132/44
layout and the mikasa 7/7 layout flow through the same code.
"""
import numpy as np

from learning.model.groot.modality_spec import ModalitySpec


def split_state_row(spec: ModalitySpec, state_row: np.ndarray) -> dict[str, np.ndarray]:
    """One training sample: ``(D,)`` state -> ``{group: (1, dim)}`` (VLAStepData.states)."""
    return {k: state_row[a:b][None, :] for k, (a, b) in spec.state_groups.items()}


def split_state_batch(spec: ModalitySpec, state_batch: np.ndarray) -> dict[str, np.ndarray]:
    """Inference: ``(B, D)`` state -> ``{"state.<group>": (B, 1, dim)}`` (process_observation).

    The extra ``T=1`` axis is the state-history axis the model's state encoder expects; without it
    the processor emits ``(B, dim)`` and the model asserts ``current_T != state_history_length``.
    """
    return {f"state.{k}": state_batch[:, a:b][:, None, :] for k, (a, b) in spec.state_groups.items()}


def split_action_chunk(spec: ModalitySpec, action_chunk: np.ndarray) -> dict[str, np.ndarray]:
    """One training sample: ``(K, D)`` action -> ``{group: (K, dim)}`` (VLAStepData.actions)."""
    return {k: action_chunk[:, a:b] for k, (a, b) in spec.action_groups.items()}


def concat_action_groups(spec: ModalitySpec, per_key: dict[str, np.ndarray]) -> np.ndarray:
    """Reassemble the unapplied per-group action dict back into the flat action vector.

    ``per_key`` maps ``"action.<group>"`` -> ``(..., dim)`` (the ``processor.unapply`` output);
    groups are concatenated on the last axis in ``spec.action_keys`` order -> ``(..., sum dim)``.
    """
    return np.concatenate([per_key[f"action.{k}"] for k in spec.action_keys], axis=-1)
