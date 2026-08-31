"""GR00T-N1.7 policy — an adapter over the vendored Isaac-GR00T model, fully inside ``learning/``.

``build(cfg, dataset)`` returns the SAME tuple as ``model.act_policy.build``
(``policy, preprocessor, postprocessor, optimizer, lr_scheduler``) so the SAME ``BCTrainer`` +
validation metrics (``metrics/validation.py``) + ``image_aug`` (apply_image_aug) + wir_v1 dataloader drive GR00T with no
trainer fork — only a build dispatch on ``cfg.policy.type``.

The GR00T processor normalizes state/action, resizes/augments images and tokenizes the VLM prompt
*internally*, so:
  * ``preprocessor`` / ``postprocessor`` are IDENTITY (no ACT-style Normalize/Unnormalize);
  * ``forward(batch)`` builds per-sample ``VLAStepData`` -> processor -> collator -> ``model.forward``
    and returns ``(loss, out)`` — the flow-matching loss;
  * ``predict_action_chunk(batch)`` runs the obs-only ``process_observation`` -> ``model.get_action``
    (plain denoising) and ``unapply``s back to **un-normalized action radians** (44-D for the ALLEX
    modality, 7-D for mikasa — the flat action width of the selected ``ModalitySpec``), so it lines
    up with the raw ``batch[ACTION]`` the metrics compare against. Because the pre/post-processors are
    identity,
    GR00T's ``l1_*_norm`` and ``l1_*_unnorm`` are the SAME chunk-L1 in radians (whereas for ACT
    ``_norm`` is in normalized space); only ``_unnorm``/``fk_*`` are comparable ACT-vs-GR00T.

The model is loaded in fp32 and run under bf16 autocast (the BCTrainer's ``accelerator.autocast()``)
— matching the upstream finetune (``load_bf16=False`` + bf16 AMP): FlashAttention sees bf16 while the
flow-matching time sampler (Beta/dirichlet, unimplemented for bf16) stays fp32.

Vendored Isaac-GR00T is wrapped, never modified (CLAUDE.md). gr00t/torch import lazily here (this
module loads only when ``--policy.type groot`` selects it, on the training node with the GR00T venv).
"""
import logging

import numpy as np
import torch
from torch import nn

from learning.model.act.constants import ACTION, OBS_STATE
from learning.model.groot.data.embodiment_tags import EmbodimentTag
from learning.model.groot.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
    VLAStepData,
)
import learning.model.groot.hf_compat  # noqa: F401  # HF local-first + mistral from_pretrained patches (applied on import)
from learning.model.groot.modality_spec import ModalitySpec, get_modality_spec
from learning.model.groot.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from learning.model.groot.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from learning.model.groot.slicing import (
    concat_action_groups,
    split_action_chunk,
    split_state_batch,
    split_state_row,
)
from learning.model.lr_schedulers import cosine_with_warmup

DEFAULT_TASK = "perform the task"


class _Identity(nn.Module):
    """No-op pre/post processor — GR00T (un)normalizes internally, so the trainer's
    ``preprocessor(batch)`` / ``postprocessor(pred)`` calls pass straight through. There is no
    normalization state to persist (it lives in the GR00T processor), so save_pretrained is a no-op."""

    def forward(self, x):
        return x

    def save_pretrained(self, save_directory):  # noqa: D401 — checkpoint hook; nothing to save
        return


def _slices(stat, groups):
    """Slice a per-dim stats dict ({min,max,mean,std} as 1-D arrays) into named groups."""
    out = {}
    for g, (a, b) in groups.items():
        out[g] = {
            "min": stat["min"][a:b], "max": stat["max"][a:b],
            "mean": stat["mean"][a:b], "std": stat["std"][a:b],
            "q01": stat["min"][a:b], "q99": stat["max"][a:b],
        }
    return out


def _as_list(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float64).reshape(-1).tolist()


def _build_statistics(ds_meta, embodiment_value, spec):
    """Per-group min/max/mean/std for new_embodiment from the wir_v1 dataset stats.

    ``observation.state`` -> ``spec.state_groups`` slices; ``action`` -> ``spec.action_groups``. Uses
    min/max (``use_percentiles=False``) so dataset percentiles aren't required.
    """
    state_stat = {k: _as_list(ds_meta.stats[OBS_STATE][k]) for k in ("min", "max", "mean", "std")}
    action_stat = {k: _as_list(ds_meta.stats[ACTION][k]) for k in ("min", "max", "mean", "std")}
    return {embodiment_value: {
        "state": _slices(state_stat, spec.state_groups),
        "action": _slices(action_stat, spec.action_groups),
    }}


def _build_modality_config(cameras, chunk_size, spec):
    """The python ModalityConfig for the selected embodiment (v1: ABSOLUTE actions, no relative)."""
    action_keys = list(spec.action_keys)
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(cameras)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(spec.state_keys)),
        "action": ModalityConfig(
            delta_indices=list(range(chunk_size)),
            modality_keys=action_keys,
            action_configs=[
                ActionConfig(ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT)
                for _ in action_keys
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.action.task_description"]),
    }


class GrootPolicy(nn.Module):
    """Wraps the vendored Gr00tN1d7 model + its processor; exposes the ACT policy interface."""

    def __init__(self, cfg, model, processor, cameras, spec: ModalitySpec):
        super().__init__()
        self.config = cfg                       # so policy.config.chunk_size works in the metrics
        self.model = model
        self.processor = processor
        self.cameras = list(cameras)
        self.embodiment = EmbodimentTag.NEW_EMBODIMENT
        self.spec = spec                        # embodiment state/action layout (slices + concat order)
        self.state_keys = list(spec.state_keys)

    # ---- batch (wir_v1) -> GR00T inputs -------------------------------------------------
    def _images_uint8_hwc(self, batch, i):
        """Per-sample {cam: [HWC uint8 np]} from observation.images.<cam> (C,H,W) float[0,1]."""
        out = {}
        for cam in self.cameras:
            v = batch[f"observation.images.{cam}"]
            x = v[:, 0] if v.ndim == 5 else v       # (B,T,C,H,W)->(B,C,H,W) ; T=1
            img = x[i].float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            out[cam] = [img]                        # list = temporal stack of 1 (delta_indices=[0])
        return out

    def _task(self, batch, i):
        t = batch.get("task")
        if isinstance(t, (list, tuple)) and i < len(t) and isinstance(t[i], str) and t[i]:
            return t[i]
        if isinstance(t, str) and t:
            return t
        return DEFAULT_TASK

    def _batch_to_steps(self, batch):
        """List of VLAStepData (one per sample) for training (includes the action target)."""
        state = batch[OBS_STATE]                    # (B, state_dim)
        action = batch[ACTION]                      # (B, K, action_dim)
        B = state.shape[0]
        steps = []
        for i in range(B):
            st = state[i].float().cpu().numpy()
            states = split_state_row(self.spec, st)               # {group: (1, dim)}
            act = action[i].float().cpu().numpy()                 # (K, action_dim)
            actions = split_action_chunk(self.spec, act)          # {group: (K, dim)}
            steps.append(VLAStepData(
                images=self._images_uint8_hwc(batch, i), states=states, actions=actions,
                text=self._task(batch, i), embodiment=self.embodiment))
        return steps

    def _batch_to_observation(self, batch):
        """Batched obs dict for inference (no action): video.<cam> (B,T,H,W,C), state.<k> (B,1,dim)."""
        state = batch[OBS_STATE]                    # (B, state_dim)
        st = state.float().cpu().numpy()
        # {"state.<group>": (B, T=1, dim)} — the state-history axis the model's state encoder expects.
        obs = split_state_batch(self.spec, st)
        for cam in self.cameras:
            v = batch[f"observation.images.{cam}"]
            x = v[:, 0] if v.ndim == 5 else v       # (B,C,H,W)
            arr = x.float().clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            obs[f"video.{cam}"] = arr[:, None]      # (B,1,H,W,C) — T=1
        obs["annotation.human.action.task_description"] = [
            self._task(batch, i) for i in range(state.shape[0])]
        return obs

    # ---- ACT policy interface -----------------------------------------------------------
    def forward(self, batch):
        """Training step: returns (loss, out_dict). Runs under the trainer's bf16 autocast."""
        steps = self._batch_to_steps(batch)
        feats = [self.processor([{"content": s}]) for s in steps]
        inputs = self.processor.collator(feats)["inputs"]
        out = self.model(dict(inputs))
        return out["loss"], out

    def save_pretrained(self, save_directory):
        """Save the GR00T model + processor so the checkpoint reloads via the GR00T inference path
        (Gr00tN1d7.from_pretrained + Gr00tN1d7Processor.from_pretrained — e.g. groot_inference_node).
        Mirrors the ACT policy's save_pretrained hook that the trainer's save_checkpoint calls."""
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)

    def set_inference_steps(self, n):
        """Override the flow-matching sampler's Euler step count (``num_inference_timesteps``) and
        return the previous value (so the caller can restore it). Inference-only knob — training learns
        a continuous velocity field, so the sampling step count is free to change at eval. Used by the
        trainer's ``--val_flow_matching_steps``."""
        head = self.model.action_head
        old = head.num_inference_timesteps
        head.num_inference_timesteps = int(n)
        return old

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        """Predicted action chunk in un-normalized radians, shape (B, chunk_size, action_dim)."""
        obs = self._batch_to_observation(batch)
        feats = self.processor.process_observation(obs, self.embodiment)
        out = self.model.get_action(dict(feats))                 # obs-only -> plain denoising
        ap = out["action_pred"].float().cpu().numpy()            # (B, max_action_horizon, max_action_dim)
        unapplied = self.processor.unapply(ap, self.embodiment)  # {action.<key>: (B, chunk_size, dim)}
        chunk = concat_action_groups(self.spec, unapplied)       # (B, chunk_size, action_dim)
        return torch.from_numpy(chunk).to(batch[ACTION].device, dtype=torch.float32)

    def input_summary(self, batch):
        """Concise, model-specific lines describing how THIS GR00T policy actually consumes ``batch``.

        Logged by the trainer's first-batch viz (learning/trainer/viz.py) so the wandb panel documents
        the REAL model-input contract (cameras/resize, internal normalization, state/action group
        layout + padding, prompt tokenizer, eval denoise steps) rather than a generic guess. Reads
        ``self.spec`` / ``self.config`` / ``self.processor`` so it stays correct as the modality or
        model changes; the viz just renders whatever this returns (it is model-agnostic).
        """
        p = self.processor
        tgt = "x".join(str(s) for s in p.image_target_size)   # e.g. "256x256"
        crop = "x".join(str(s) for s in p.image_crop_size)    # e.g. "230x230"
        state_dim = (batch[OBS_STATE].shape[-1] if OBS_STATE in batch
                     else max(e for _, e in self.spec.state_groups.values()))
        action_dim = (batch[ACTION].shape[-1] if ACTION in batch
                      else sum(e - s for s, e in self.spec.action_groups.values()))
        steps = self.config.num_inference_timesteps
        steps = "checkpoint default (4)" if steps is None else steps
        return [
            f"GR00T-N1.7 | modality={self.spec.name} | embodiment={self.embodiment.value}",
            f"images: cameras {self.cameras} -> resized {tgt} (crop {crop}), "
            f"normalized INTERNALLY by the processor (pre/post are identity)",
            f"state: {state_dim}-D via groups {dict(self.spec.state_groups)} "
            f"(order {list(self.spec.state_keys)}) -> padded to max_state_dim ({self.spec.max_state_dim})",
            f"action: {action_dim}-D via groups {dict(self.spec.action_groups)} "
            f"(order {list(self.spec.action_keys)}) -> padded to max_action_dim "
            f"({self.spec.max_action_dim}), chunk_size={self.config.chunk_size}",
            f"prompt: per-sample task string, tokenized by the {p.model_name} VLM; "
            f"num_inference_timesteps={steps} (eval denoise)",
        ]


def build(cfg, dataset):
    """Build ``(policy, preprocessor, postprocessor, optimizer, lr_scheduler)`` for a GR00T run.

    ``cfg`` is the full ``TrainPipelineConfig`` (like ``act_policy.build``); the GR00T knobs live on
    ``cfg.policy`` (a ``GrootConfig``).
    """
    pcfg = cfg.policy
    ds_meta = dataset.meta
    cameras = [k.replace("observation.images.", "")
               for k in ds_meta.features if k.startswith("observation.images.")]
    if not cameras:
        raise ValueError("GR00T policy needs at least one observation.images.<cam> in the dataset.")
    emb = EmbodimentTag.NEW_EMBODIMENT
    # Embodiment layout (state/action slices, concat order, pad dims) — single source of truth.
    spec = get_modality_spec(pcfg.modality)
    logging.info(
        f"GR00T: base={pcfg.base_model_path} modality={spec.name} cameras={cameras} "
        f"chunk_size={pcfg.chunk_size} state_keys={list(spec.state_keys)} "
        f"action_keys={list(spec.action_keys)} max_state_dim={spec.max_state_dim} "
        f"max_action_dim={spec.max_action_dim}")

    modality = {emb.value: _build_modality_config(cameras, pcfg.chunk_size, spec)}
    statistics = _build_statistics(ds_meta, emb.value, spec)

    # Processor: the base model's image/dims settings + ONLY new_embodiment stats (so the action_dim
    # bookkeeping never touches the base embodiments, which carry a relative_action modality we omit).
    # max_state_dim/max_action_dim come from the spec and MUST equal the base checkpoint's
    # config.max_*_dim (the processor pads each per-group state/action UP to these widths — see
    # modality_spec.py); a 7-D embodiment pads 7 -> 132, it does not shrink the pretrained head.
    processor = Gr00tN1d7Processor(
        modality_configs=modality, statistics=statistics,
        max_state_dim=spec.max_state_dim, max_action_dim=spec.max_action_dim,
        max_action_horizon=pcfg.max_action_horizon,
        image_crop_size=[230, 230], image_target_size=[256, 256],
        shortest_image_edge=256, crop_fraction=0.95, use_albumentations=True,
        color_jitter_params=None, random_rotation_angle=0,   # photometric aug off -> image_aug owns it
        apply_sincos_state_encoding=False, use_percentiles=False, use_mean_std=False,
        clip_outliers=True, use_relative_action=pcfg.use_relative_action, formalize_language=True,
        state_dropout_prob=0.0, exclude_state=False, model_name="nvidia/Cosmos-Reason2-2B",
    )

    # Model: fp32 weights (bf16 via the trainer's autocast). Freeze the VLM backbone, train the
    # diffusion action head + projectors (memory-friendly first finetune on a single A100).
    model = Gr00tN1d7.from_pretrained(pcfg.base_model_path)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.action_head.parameters():
        p.requires_grad = (pcfg.tune_diffusion_model or pcfg.tune_projector)
    if pcfg.tune_llm or pcfg.tune_visual:
        for p in model.backbone.parameters():
            p.requires_grad = True
    if pcfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # Reduced/fast-profile knob: override the flow-matching inference denoising steps (default 4).
    # Fewer steps -> faster predict_action_chunk (validation) at some accuracy cost; weights untouched.
    if pcfg.num_inference_timesteps is not None:
        model.config.num_inference_timesteps = pcfg.num_inference_timesteps
        model.action_head.num_inference_timesteps = pcfg.num_inference_timesteps
        logging.info(f"GR00T: num_inference_timesteps={pcfg.num_inference_timesteps} (fast profile)")
    if pcfg.device is not None:
        model = model.to(pcfg.device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"GR00T: {n_train:,} trainable / {sum(p.numel() for p in model.parameters()):,} params")

    policy = GrootPolicy(pcfg, model, processor, cameras, spec)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=pcfg.optimizer_lr, weight_decay=pcfg.optimizer_weight_decay,
    )
    # Warmup -> cosine LR over the run (total_steps = cfg.steps). Warmup is essential once the VLM
    # backbone is unfrozen: a cold high LR on the pretrained weights degrades them. Shared pure-torch
    # builder (learning.model.lr_schedulers) — identical schedule to the former transformers cosine,
    # no transformers dependency (wsd_with_warmup is available there for a later follow-up).
    lr_scheduler = cosine_with_warmup(optimizer, pcfg.warmup_steps, cfg.steps)

    return policy, _Identity(), _Identity(), optimizer, lr_scheduler
