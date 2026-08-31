"""Self-contained Action Chunking Transformer (ACT) model + policy (de-lerobot'd).

Faithful reimplementation of LeRobot 0.4.4's `lerobot.policies.act.modeling_act` with **no lerobot
imports**. The module / parameter structure (every attribute and submodule name) matches LeRobot's
exactly, so a LeRobot ACT `state_dict` loads here with `load_state_dict(strict=True)` and the model
produces identical outputs.

Differences from LeRobot (all non-numerical):
* `ACTPolicy` subclasses `torch.nn.Module` directly instead of `PreTrainedPolicy` (the HF-hub mixin
  is dropped). `save_pretrained` / `from_pretrained` are plain torch.save/torch.load of the
  state_dict plus a config json — no safetensors, no huggingface_hub.
* `(un)normalization` is NOT inside the policy (LeRobot 0.4.4 also keeps it outside, in its processor
  pipeline); it lives in `learning.model.act.normalize` and is wired up by `build_act`.

Everything inside `ACT` (VAE encoder, transformer encoder/decoder, ResNet backbone, positional
embeddings, init, forward math, VAE reparameterization) is copied from LeRobot verbatim.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from itertools import chain
import json
import math
from pathlib import Path

import einops
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F  # noqa: N812
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from learning.model.act.configuration import (
    ACTConfig,
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)
from learning.model.act.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTPolicy(nn.Module):
    """Action Chunking Transformer Policy (paper: https://huggingface.co/papers/2304.13705)."""

    name = "act"

    def __init__(self, config: ACTConfig):
        super().__init__()
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> list:
        # Backbone params get `optimizer_lr_backbone`; everything else uses the optimizer's base lr.
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        Manages an action queue (or temporal ensembler) and only queries the model when needed.
        """
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # (batch_size, n_action_steps, action_dim) -> queue holds (n_action_steps, batch_size, *).
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # Dₖₗ(latent_pdf || standard_normal), summed over latent dim then mean over batch.
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict

    # ------------------------------------------------------------------ persistence (plain torch) --
    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save state_dict (`model.safetensors` replacement) + a `config.json`, no HF hub.

        Writes `model.pt` (torch.save of the state_dict) and `config.json` (the ACTConfig fields).
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_directory / "model.pt")
        with open(save_directory / "config.json", "w") as f:
            json.dump(_config_to_dict(self.config), f, indent=4)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        *,
        config: ACTConfig | None = None,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> "ACTPolicy":
        """Load a policy saved by `save_pretrained` (or a raw `model.pt` + matching config)."""
        pretrained_path = Path(pretrained_path)
        if config is None:
            with open(pretrained_path / "config.json") as f:
                config = _config_from_dict(json.load(f))
        policy = cls(config)
        state_dict = torch.load(pretrained_path / "model.pt", map_location=map_location)
        policy.load_state_dict(state_dict, strict=strict)
        policy.eval()
        return policy


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of the ACT paper.

        Weights wᵢ = exp(-temporal_ensemble_coeff * i), normalized to sum to 1. An online running
        average is maintained so we never cache the full action history.
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """Update the ensemble with a (batch, chunk_size, action_dim) chunk, return the next action."""
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: the underlying neural network for ACTPolicy."""

    def __init__(self, config: ACTConfig):
        # BERT-style VAE encoder with input tokens [cls, robot_state, *action_sequence]. The cls token
        # forms parameters of the latent's distribution ([*means, *log_variances]).
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # Projection layer for joint-space configuration to hidden dimension.
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # Projection layer for action (joint-space target) to hidden dimension.
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # Projection from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder.
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Assumption: a ResNet model (so layer4 is the final feature map). The IntermediateLayerGetter
            # forward returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. Tokens are structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 1  # for the latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        # Learned temporal positional embedding for observation history (n_obs_steps frames), added on
        # top of the per-token spatial/1D positional embed so the encoder can tell the frames apart.
        # Created only when history is on, so n_obs_steps=1 keeps the module list / state_dict
        # byte-identical to the original single-frame ACT.
        if config.n_obs_steps > 1:
            self.encoder_temporal_pos_embed = nn.Embedding(config.n_obs_steps, config.dim_model)

        # Transformer decoder.
        # Learnable positional embedding for the decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        Returns:
            (B, chunk_size, action_dim) batch of action sequences and a tuple of the latent PDF's
            parameters (mean, log(σ²)), both (B, L) where L is the latent dimension (or (None, None)).
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # VAE encoder input: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                # With obs history (n_obs_steps>1) the state is (B, T, D); the VAE encoder is a
                # current-timestep model, so it consumes only the most-recent frame -> identical to
                # the single-frame case.
                state_cur = batch[OBS_STATE][:, -1] if self.config.n_obs_steps > 1 else batch[OBS_STATE]
                robot_state_embed = self.vae_encoder_robot_state_input_proj(state_cur)
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Fixed positional embedding (detach() to match the original code).
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Key padding mask for the VAE encoder. 1 or 2 extra non-pad tokens at the start (cls and
            # robot state). False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, set the latent to all zeros.
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs. Token order: [latent, (robot_state × n_obs),
        # (env_state), (image pixels × n_obs)]. The 1D positional embeddings index
        # [latent, (robot_state), (env_state)] in that order; with obs history each state / image
        # frame reuses its 1D / 2D positional embed plus a learned per-frame temporal embed so the
        # encoder can tell frames apart. n_obs_steps=1 takes the original single-frame path (no
        # temporal embed, one token per feature) and is byte-identical to the pre-history model.
        n_obs = self.config.n_obs_steps
        pos_1d = self.encoder_1d_feature_pos_embed.weight  # (n_1d_tokens, D)
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = [pos_1d[0].unsqueeze(0)]  # latent
        idx = 1
        # Robot state token(s).
        if self.config.robot_state_feature:
            state_pos = pos_1d[idx].unsqueeze(0)  # (1, D)
            idx += 1
            if n_obs > 1:
                state_hist = self.encoder_robot_state_input_proj(batch[OBS_STATE])  # (B, T, D)
                temporal = self.encoder_temporal_pos_embed.weight  # (T, D)
                for t in range(n_obs):
                    encoder_in_tokens.append(state_hist[:, t])                        # (B, D)
                    encoder_in_pos_embed.append(state_pos + temporal[t].unsqueeze(0))
            else:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
                encoder_in_pos_embed.append(state_pos)
        # Environment state token (single-frame only).
        if self.config.env_state_feature:
            if n_obs > 1:
                raise NotImplementedError(
                    "Observation history (n_obs_steps>1) with env_state is not supported."
                )
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
            encoder_in_pos_embed.append(pos_1d[idx].unsqueeze(0))
            idx += 1

        if self.config.image_features:
            # For a list of images, H and W may vary but H*W is constant.
            for img in batch[OBS_IMAGES]:
                if n_obs > 1:
                    # img: (B, T, C, H, W). Fold time into batch for the backbone, then emit one set
                    # of pixel tokens per frame with the shared spatial PE + that frame's temporal PE.
                    b, t = img.shape[0], img.shape[1]
                    flat = einops.rearrange(img, "b t c h w -> (b t) c h w")
                    feat = self.backbone(flat)["feature_map"]
                    # Spatial 2D PE is batch-independent -> (1, D, h, w); features are per-sample.
                    pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)  # (1, D, h, w)
                    feat = self.encoder_img_feat_input_proj(feat)                     # (b*t, D, h, w)
                    feat = einops.rearrange(feat, "(b t) c h w -> t (h w) b c", b=b, t=t)  # (T,h*w,b,D)
                    pos_hw = einops.rearrange(pos, "1 c h w -> (h w) 1 c")            # (h*w, 1, D)
                    temporal = self.encoder_temporal_pos_embed.weight                 # (T, D)
                    for ti in range(t):
                        encoder_in_tokens.extend(list(feat[ti]))                      # (h*w) × (b, D)
                        pos_ti = pos_hw + temporal[ti].to(pos_hw.dtype).view(1, 1, -1)  # (h*w, 1, D)
                        encoder_in_pos_embed.extend(list(pos_ti))                     # (h*w) × (1, D)
                else:
                    cam_features = self.backbone(img)["feature_map"]
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)

                    # Rearrange features to (sequence, batch, dim).
                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                    # Extend immediately instead of accumulating and concatenating.
                    encoder_in_tokens.extend(list(cam_features))
                    encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last encoder layer.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Returns: (num_positions, dimension) position embeddings.
    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to Attention Is All You Need.

    Position indices are normalized in [0, 2π] (the lower bound is 1/H vertically and 1/W
    horizontally).
    """

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # These are like range(1, H+1) and range(1, W+1). Kept as-is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # This stack-then-flatten results in interleaved sine and cosine terms.
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


# --------------------------------------------------------------------------- config (de)serialize --
def _feature_to_dict(ft: PolicyFeature) -> dict:
    return {"type": ft.type.value, "shape": list(ft.shape)}


def _feature_from_dict(d: dict) -> PolicyFeature:
    return PolicyFeature(type=FeatureType(d["type"]), shape=tuple(d["shape"]))


def _config_to_dict(config: ACTConfig) -> dict:
    """Serialize an ACTConfig to plain json-able types (enums -> str, PolicyFeature -> dict)."""
    out = asdict(config)
    out["normalization_mapping"] = {k: v.value for k, v in config.normalization_mapping.items()}
    out["input_features"] = {k: _feature_to_dict(v) for k, v in config.input_features.items()}
    out["output_features"] = {k: _feature_to_dict(v) for k, v in config.output_features.items()}
    return out


def _config_from_dict(d: dict) -> ACTConfig:
    """Inverse of `_config_to_dict`."""
    d = dict(d)
    d["normalization_mapping"] = {k: NormalizationMode(v) for k, v in d["normalization_mapping"].items()}
    d["input_features"] = {k: _feature_from_dict(v) for k, v in d["input_features"].items()}
    d["output_features"] = {k: _feature_from_dict(v) for k, v in d["output_features"].items()}
    return ACTConfig(**d)
