"""GR00T-N1.7 policy config (lightweight — no torch / no gr00t imports).

Selected by ``--policy.type groot``. Kept dependency-free so ``learning.configs.parser`` /
``learning.configs.config`` can import it at parse time without pulling in torch or the vendored
Isaac-GR00T package (those load only when ``groot_policy.build`` runs on the training node). The
``modality`` selector below reads ``learning.model.groot.modality_spec`` — deliberately a
torch/gr00t/numpy-free stdlib module, so this stays parse-time safe.

Only the surface the framework reads is here:
  * the ``*_delta_indices`` properties the data factory turns into ``delta_timestamps`` (action
    chunk = ``range(chunk_size)``; obs/reward None) — same contract as ``ACTConfig``;
  * ``chunk_size`` (== the GR00T action horizon) which the validation metrics read as
    ``policy.config.chunk_size``;
  * ``device`` (read by ``train.py``), ``optimizer_lr`` / ``optimizer_weight_decay`` (the training
    preset, dispatched in ``configs/config.py``).

The rest are GR00T knobs consumed by ``groot_policy.build`` (base checkpoint, embodiment, model
dims, which sub-modules to fine-tune, …). The state/action layout is NOT hardcoded here: pick it
with ``--policy.modality`` (default ``allex`` = the wir_v1 132-D ``q/dq/tau`` state + 44-D
``r_/l_arm_cmd`` / ``r_/l_hand_cmd`` action contract; ``mikasa`` = a 7-D single-group embodiment).
``state_keys`` / ``max_state_dim`` / ``max_action_dim`` are DERIVED from the selected modality's
``ModalitySpec`` in ``__post_init__`` (single source of truth — see ``modality_spec.py``).
"""
from dataclasses import dataclass, field

from learning.model.groot.modality_spec import get_modality_spec


@dataclass
class GrootConfig:
    """Config for the GR00T-N1.7 VLA policy (adapter over the vendored Isaac-GR00T model)."""

    # ---- framework-required (read by train.py / data factory / validation) ----
    n_obs_steps: int = 1
    chunk_size: int = 16            # GR00T action horizon (<= max_action_horizon); drives the
    #                                 action-chunk delta_timestamps AND policy.config.chunk_size.
    device: str | None = None
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-5
    warmup_steps: int = 1000        # linear LR warmup steps, then cosine decay (build() wires it)
    grad_clip_norm: float = 1.0

    # ---- GR00T model / adapter knobs (consumed by groot_policy.build) ----
    base_model_path: str = "/opt/dlami/nvme/groot-base-n1.7-3b"
    embodiment_tag: str = "new_embodiment"
    # NOTE: max_state_dim / max_action_dim / state_keys are DERIVED from `modality` in __post_init__
    # (do not set directly — the selected ModalitySpec is authoritative). The defaults below are the
    # ALLEX values, so a default GrootConfig() is byte-for-byte the legacy config; --policy.modality
    # mikasa overrides them. max_state_dim / max_action_dim = the width the processor pads each
    # per-group state/action UP to (== the pretrained head width, 132 — see modality_spec.py).
    max_state_dim: int = 132
    max_action_dim: int = 132
    max_action_horizon: int = 40    # model-fixed (== base checkpoint config.action_horizon)
    # Which sub-modules to fine-tune (the diffusion action head + projector by default; the VLM
    # backbone frozen for a first, memory-friendly finetune — flip on for full finetune).
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    use_relative_action: bool = False   # v1: ABSOLUTE action targets (no state-key alignment needed)
    gradient_checkpointing: bool = True
    # ordered state groups fed to the model (derived from `modality`; ALLEX = q/dq/tau).
    state_keys: list[str] = field(default_factory=lambda: ["q", "dq", "tau"])
    # ---- embodiment selector + reduced/fast-profile knobs (appended; ALLEX keeps legacy behavior) --
    # Embodiment layout selector: "allex" (default, 132-D q/dq/tau state + 44-D 4-group action) or
    # "mikasa" (7-D single-group state + 7-D single-group action). Switches state_keys / max_state_dim
    # / max_action_dim via its ModalitySpec (see modality_spec.py). Fail loud on an unknown name.
    modality: str = "allex"
    # Reduced/fast-profile knob: diffusion inference denoising steps. None -> the checkpoint default
    # (4). Smaller = faster predict/validation (e.g. 1-2 for a smoke run); does not touch weights.
    num_inference_timesteps: int | None = None

    def __post_init__(self) -> None:
        # Resolve the embodiment layout (single source of truth) and derive the layout fields from
        # it, so --policy.modality alone switches the state slices + dims. allex reproduces the
        # legacy defaults byte-for-byte; mikasa switches to the 7-D single-group layout.
        spec = get_modality_spec(self.modality)
        self.state_keys = list(spec.state_keys)
        self.max_state_dim = spec.max_state_dim
        self.max_action_dim = spec.max_action_dim
        if self.chunk_size > self.max_action_horizon:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be <= max_action_horizon "
                f"({self.max_action_horizon}); the model predicts at most {self.max_action_horizon} steps."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"n_obs_steps must be 1 (got {self.n_obs_steps}).")
        if self.num_inference_timesteps is not None and self.num_inference_timesteps < 1:
            raise ValueError(
                f"num_inference_timesteps must be >= 1 (got {self.num_inference_timesteps})."
            )

    # --- delta-index properties consumed by learning.data.factory.resolve_delta_timestamps ---
    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
