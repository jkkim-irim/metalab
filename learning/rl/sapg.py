# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# SAPG (Split-and-Aggregate Policy Gradients, Singla et al., ICML 2024) built additively on our vendored
# PPO. This is a reimplementation of the algorithm (the upstream jayeshs999/sapg is bound to rl_games +
# IsaacGymEnvs); only the mechanism is ported. See SAPG_INTEGRATION_PLAN.md for the full design.
#
# Design invariants (do not break):
#   * ppo.py gets only an additive storage-factory hook (_make_storage); on_policy_runner.py and
#     rollout_storage.py are untouched. SAPG is a subclass + new models + a storage subclass.
#   * PPO byte-parity: with num_expl_coef_blocks == 1 and all other knobs off, SAPG is inactive and every
#     override short-circuits to the PPO code path (same actor/critic/distribution/storage).
#
# Mechanism (see the plan's Phase-0/1 findings):
#   * SPLIT: num_envs -> N contiguous blocks. A per-env block id is carried in the obs TensorDict under the
#     key ``block_id`` (NOT an obs_groups member, so the normalizer/obs_dim ignore it). SAPGActor/SAPGCritic
#     read it to append a learned per-block embedding AFTER normalization; the actor's std is per-block.
#   * AGGREGATE: after the on-policy rollout, build ``off_policy_ratio`` relabeled copies — roll the block id
#     to a target block, recompute values/returns/advantages with the (block-conditioned) critic, but keep
#     the behavior ``old_logp`` unchanged. Concatenate leader + relabeled copies into one PPO buffer.
#   * UPDATE: the ordinary PPO clipped-ratio loss; the ratio exp(old_logp - new_logp) is the off-policy
#     correction for free, because new_logp is evaluated under the relabeled conditioning.


from __future__ import annotations

from collections.abc import Generator
import copy

from tensordict import TensorDict
import torch
import torch.nn as nn

from learning.rl.models import MLPModel, RNNModel
from learning.rl.ppo import PPO
from learning.rl.rollout_storage import RolloutStorage
from learning.rl.utils import unpad_trajectories
from learning.rl.vec_env import VecEnv

# Obs key carrying the per-env block id (see module docstring). Not an obs_groups member.
_BLOCK_KEY = "block_id"


def _sapg_is_active(algo_cfg: dict) -> bool:
    """Whether any SAPG knob is on. All-off ⇒ inactive ⇒ pure-PPO path (byte-parity)."""
    return (
        algo_cfg.get("num_expl_coef_blocks", 1) > 1
        or algo_cfg.get("off_policy_ratio", 0.0) > 0.0
        or algo_cfg.get("use_others_experience", "none") != "none"
        or algo_cfg.get("ir_type", "none") != "none"
    )


class SAPGActor(MLPModel):
    """Actor that appends a learned per-block embedding to the (normalized) latent and samples with a
    per-block std. Reduces to :class:`MLPModel` behaviour at ``num_blocks == 1`` (constant embedding row)."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        num_blocks: int,
        embed_dim: int,
        **kwargs,
    ) -> None:
        # _embed_dim must exist before super().__init__ (it calls _get_latent_dim to size the MLP head).
        self._embed_dim = embed_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.block_embed = nn.Embedding(num_blocks, embed_dim)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._embed_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        # PADDED grid: this model does not unpad in get_latent (MLPModel.forward does, once, on the whole
        # latent), so the embedding rows must stay padded too or the concatenation misaligns.
        latent = super().get_latent(obs, masks, hidden_state)
        return torch.cat([latent, self.block_embed(_block_ids(obs, None))], dim=-1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        latent = self.get_latent(obs, masks, hidden_state)
        if masks is not None:            # recurrent minibatch, feedforward model — same step MLPModel takes
            latent = unpad_trajectories(latent, masks)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                # ids for the UNPADDED rows now: mlp_output has been unpadded above
                self.distribution.update_with_blocks(mlp_output, _block_ids(obs, masks))  # per-block std
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output


def _block_ids(obs, masks) -> torch.Tensor:
    """Long block ids shaped like the model's LATENT, minus its feature axis.

    Pass ``masks=None`` to stay on the PADDED grid (what a feedforward model's ``get_latent`` works on) and
    the batch's masks to get the UNPADDED grid (what an RNN's latent, and any unpadded MLP output, is on).

    Feedforward / rollout: ``[batch]`` for a ``[batch, latent]`` latent. Recurrent UPDATE batches are PADDED
    trajectories ``[T_pad, n_traj, ...]`` and ``RNN.forward`` unpads its output back to ``[T, envs, hidden]``,
    so the ids must be unpadded with the same masks — otherwise the embedding rows (and the per-block std)
    line up with the wrong transitions. The leading dims are kept so both cases broadcast on ``dim=-1``."""
    b = obs[_BLOCK_KEY]
    if masks is not None:
        b = unpad_trajectories(b, masks)
    return b.squeeze(-1).long()


class SAPGActorRNN(RNNModel):
    """:class:`SAPGActor` with an LSTM/GRU in FRONT of the MLP (rl_games' ``rnn.before_mlp: True``).

    The block embedding is concatenated to the RNN's OUTPUT, not to its input, so the recurrent state stays
    a function of the observations alone. That is what lets SAPG's off-policy augmentation reuse the stored
    hidden states verbatim when it relabels another block's transitions to the leader block: the relabeling
    changes the embedding, and the embedding is not part of what the LSTM integrated. (rl_games appends the
    coefficient embedding to the observation vector instead and copies the hidden states anyway, which is
    only approximately right; this ordering makes it exact.)"""

    def __init__(self, obs, obs_groups, obs_set, output_dim, *, num_blocks: int, embed_dim: int, **kwargs):
        self._embed_dim = embed_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.block_embed = nn.Embedding(num_blocks, embed_dim)

    def _get_latent_dim(self) -> int:
        return self.latent_dim + self._embed_dim          # RNN hidden ⊕ block embedding

    def get_latent(self, obs, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)          # [valid, rnn_hidden]
        return torch.cat([latent, self.block_embed(_block_ids(obs, masks))], dim=-1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        mlp_output = self.mlp(self.get_latent(obs, masks, hidden_state))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update_with_blocks(mlp_output, _block_ids(obs, masks))   # per-block std
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output


class SAPGCriticRNN(RNNModel):
    """:class:`SAPGCritic` with the RNN in front of the MLP — same embedding placement as
    :class:`SAPGActorRNN`."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, *, num_blocks: int, embed_dim: int, **kwargs):
        self._embed_dim = embed_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.block_embed = nn.Embedding(num_blocks, embed_dim)

    def _get_latent_dim(self) -> int:
        return self.latent_dim + self._embed_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)
        return torch.cat([latent, self.block_embed(_block_ids(obs, masks))], dim=-1)


class SAPGCritic(MLPModel):
    """Critic conditioned on the same per-block embedding (required so relabeled values under a rolled
    block_id actually differ — see :meth:`SAPG._aggregate`). No distribution / no per-block std."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        *,
        num_blocks: int,
        embed_dim: int,
        **kwargs,
    ) -> None:
        self._embed_dim = embed_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.block_embed = nn.Embedding(num_blocks, embed_dim)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._embed_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        latent = super().get_latent(obs, masks, hidden_state)      # PADDED grid — see SAPGActor.get_latent
        return torch.cat([latent, self.block_embed(_block_ids(obs, None))], dim=-1)


class SAPGRolloutStorage(RolloutStorage):
    """RolloutStorage that can serve an augmented (leader + relabeled) flat batch.

    Additive over :class:`RolloutStorage`: a transient ``_relabel`` dict set by :meth:`SAPG._aggregate`. When
    it is ``None`` (blocks==1 / off_policy_ratio==0) the minibatch generators delegate byte-exactly to the
    base implementation, guaranteeing PPO parity.

    Two layouts, because the two generators want different shapes:

    * FEEDFORWARD — already flat ``[(1+F)*T*envs, ...]``; the base generator shuffles rows freely.
    * RECURRENT — TIME-MAJOR ``[T, (1+F)*envs, ...]`` plus ``dones`` and the saved hidden states, because a
      recurrent minibatch is whole trajectories, not rows. The relabeled copies keep their own dones and
      their SOURCE block's hidden states (valid verbatim: the block embedding sits after the RNN, see
      :class:`SAPGActorRNN`), so the augmented copies extend the ENV axis and the base generator's
      trajectory splitting works on them unchanged.
    """

    def __init__(self, *args, num_blocks: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_blocks = num_blocks
        self.block_size = self.num_envs // num_blocks
        self._relabel: dict | None = None

    def clear(self) -> None:
        super().clear()
        self._relabel = None

    def set_relabel_batch(self, relabel: dict) -> None:
        self._relabel = relabel

    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Trajectory minibatches over the augmented pool. The pool is laid out exactly like a rollout with
        more envs, so this runs the BASE generator over a shallow view of this storage whose per-env tensors
        are the concatenated ones — no second copy of the trajectory-splitting or hidden-state gather."""
        if self._relabel is None:  # inactive / no augmentation → exact PPO path
            yield from super().recurrent_mini_batch_generator(num_mini_batches, num_epochs)
            return
        rel = self._relabel
        view = copy.copy(self)                      # same object, per-env tensors swapped for the pool's
        view._relabel = None
        view.num_envs = rel["observations"].shape[1]
        view.observations = rel["observations"]
        view.dones = rel["dones"]
        view.actions = rel["actions"]
        view.values = rel["values"]
        view.advantages = rel["advantages"]
        view.returns = rel["returns"]
        view.actions_log_prob = rel["old_actions_log_prob"]
        view.distribution_params = rel["old_distribution_params"]
        view.saved_hidden_state_a = rel["hidden_a"]
        view.saved_hidden_state_c = rel["hidden_c"]
        yield from RolloutStorage.recurrent_mini_batch_generator(view, num_mini_batches, num_epochs)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[RolloutStorage.Batch, None, None]:
        if self._relabel is None:  # inactive / no augmentation → exact PPO path
            yield from super().mini_batch_generator(num_mini_batches, num_epochs)
            return
        rel = self._relabel
        total = rel["actions"].shape[0]
        mini_batch_size = total // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield RolloutStorage.Batch(
                    observations=rel["observations"][idx],
                    actions=rel["actions"][idx],
                    values=rel["values"][idx],
                    advantages=rel["advantages"][idx],
                    returns=rel["returns"][idx],
                    old_actions_log_prob=rel["old_actions_log_prob"][idx],
                    old_distribution_params=tuple(p[idx] for p in rel["old_distribution_params"]),
                )


class SAPG(PPO):
    """Split-and-Aggregate Policy Gradients on top of PPO."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        num_expl_coef_blocks: int = 1,
        ir_type: str = "none",
        ir_coef_scale: float = 0.0,
        off_policy_ratio: float = 0.0,
        use_others_experience: str = "none",
        **ppo_kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **ppo_kwargs)

        assert ir_type in ("entropy", "none"), ir_type
        assert use_others_experience in ("lf", "all", "none"), use_others_experience
        assert num_expl_coef_blocks >= 1, num_expl_coef_blocks
        assert storage.num_envs % num_expl_coef_blocks == 0, (
            f"num_envs ({storage.num_envs}) must be divisible by num_expl_coef_blocks ({num_expl_coef_blocks})"
        )
        # Recurrent SAPG = rl_games' `rnn.before_mlp: True`. Supported, with one structural requirement:
        # the block embedding must be applied AFTER the RNN, so the hidden states the augmentation carries
        # over to a relabeled copy belong to the same recurrent computation. Enforced below once
        # `_sapg_active` is known (inactive SAPG is plain PPO and any recurrent model is fine).
        self._recurrent = bool(actor.is_recurrent or critic.is_recurrent)

        self.num_expl_coef_blocks = num_expl_coef_blocks
        self.ir_type = ir_type
        self.ir_coef_scale = ir_coef_scale
        self.off_policy_ratio = off_policy_ratio
        self.use_others_experience = use_others_experience

        assert not (self._recurrent and isinstance(actor, SAPGActor) and not isinstance(actor, SAPGActorRNN)), \
            "a recurrent SAPG actor must be SAPGActorRNN (the block embedding has to sit after the RNN)"

        self._sapg_active = _sapg_is_active(
            {
                "num_expl_coef_blocks": num_expl_coef_blocks,
                "ir_type": ir_type,
                "off_policy_ratio": off_policy_ratio,
                "use_others_experience": use_others_experience,
            }
        )

        if self._sapg_active:
            # num_envs must divide evenly: the Launchpad builds it as envs_per_block * num_blocks, so this is
            # its guarantee, ENFORCED rather than assumed. A floor divide on an indivisible count sends the
            # tail envs to block id num_blocks -- one block more than exists -- and that id then indexes past
            # both the entropy-coef table and the block embedding. A hand-written --num_envs is the way in.
            assert storage.num_envs % num_expl_coef_blocks == 0, (
                f"SAPG: num_envs={storage.num_envs} must be a multiple of num_expl_coef_blocks="
                f"{num_expl_coef_blocks} (the Launchpad builds num_envs as envs_per_block * num_blocks); "
                f"{storage.num_envs % num_expl_coef_blocks} envs would spill into a block that does not exist")
            block_size = storage.num_envs // num_expl_coef_blocks
            # Contiguous env→block map, constant across rollouts. [num_envs, 1] float (obs dtype).
            self._block_ids = (
                torch.arange(storage.num_envs, device=self.device) // block_size
            ).view(storage.num_envs, 1).float()
            # Per-block entropy coefficient (used only when ir_type='entropy'): block 0 = most exploratory
            # (0.5*scale) → block N-1 = pure exploit (0.0). Matches original a2c_common.py:328.
            self._entropy_coef_table = (
                torch.linspace(0.5, 0.0, num_expl_coef_blocks, device=self.device) * ir_coef_scale
            )
            # Original SAPG normalizes advantages ONCE, globally, over the whole mixed (leader + relabeled)
            # batch — done in _aggregate / compute_returns. Keep PPO's per-minibatch flag OFF so the update
            # loop does not re-normalize on top of that.
            self.normalize_advantage_per_mini_batch = False
        else:
            self._block_ids = None
            self._entropy_coef_table = None

    # ----- rollout / returns (block conditioning + aggregate) -----

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self._sapg_active:
            obs[_BLOCK_KEY] = self._block_ids
        return super().act(obs)

    def _entropy_loss(self, entropy: torch.Tensor, batch) -> torch.Tensor:
        """Per-sample entropy bonus weighted by each sample's block coefficient when ir_type='entropy'
        (original a2c_continuous.py:140-160); otherwise the uniform PPO term. The block of each sample is
        recovered from its block_id, which rides in the (relabeled) observation and survives shuffling."""
        if self._sapg_active and self.ir_type == "entropy":
            if getattr(batch, "masks", None) is None:
                # Flat batch: [batch] ids for [batch] entropies. The slice is for symmetry augmentation,
                # which repeats the batch but keeps the entropy of the original copy only.
                block_ids = batch.observations[_BLOCK_KEY].reshape(-1).long()[: entropy.shape[0]]
            else:
                # Recurrent batch: the observations are PADDED trajectories while `entropy` comes from the
                # actor's UNPADDED output, so the ids must be unpadded the same way (see _block_ids) —
                # flattening here would weight each transition with another one's block coefficient.
                block_ids = _block_ids(batch.observations, batch.masks)
                assert block_ids.shape == entropy.shape[: block_ids.dim()], (block_ids.shape, entropy.shape)
            coef = self._entropy_coef_table[block_ids]
            return (coef * entropy).mean()
        return super()._entropy_loss(entropy, batch)

    def compute_returns(self, obs: TensorDict) -> None:
        if not self._sapg_active:
            super().compute_returns(obs)
            return
        obs[_BLOCK_KEY] = self._block_ids
        st = self.storage
        # Leader (evaluation policy) = full GAE(lambda), computed raw here; normalization happens once,
        # globally, over the mixed batch (in _aggregate) or over the leader batch (no-relabel branch).
        last_values = self.critic(obs).detach()
        returns, advantages = self._compute_gae(st.rewards, st.values, st.dones, last_values, self.gamma, self.lam)
        st.returns[:] = returns
        st.advantages[:] = advantages
        if int(self.off_policy_ratio) > 0 and self.use_others_experience != "none":
            self._aggregate(obs)  # concat leader + relabeled copies, then global-normalize advantages
        else:
            # Multi-block, no relabeling: global-normalize like vanilla PPO / the original.
            a = st.advantages
            st.advantages = (a - a.mean()) / (a.std() + 1e-8)

    @staticmethod
    def _compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure GAE over ``[T, B, 1]`` tensors — identical recurrence to PPO.compute_returns, no mutation."""
        num_steps = rewards.shape[0]
        returns = torch.zeros_like(values)
        advantage = torch.zeros_like(values[0])
        for step in reversed(range(num_steps)):
            next_values = last_values if step == num_steps - 1 else values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]
        return returns, returns - values

    def _aggregate(self, last_obs: TensorDict) -> None:
        """Build leader + relabeled copies and hand the concatenated, globally-normalized flat batch to the
        storage. Mirrors the original ``augment_batch_for_mixed_expl`` + ``filter_leader``:

        - leader = the full on-policy batch (evaluation policy); its returns are the full-GAE returns.
        - the leader block is the LAST block (N-1, the exploit/eval block).
        - followers = a random distinct subset ``r`` of ``1..N-1`` of size ``min(N-1, off_policy_ratio)``.
          For each ``r`` we take physical block ``(r-1)``'s transitions, relabel their block id to the leader
          block (N-1), recompute values with the block-conditioned critic and returns with a **1-step TD
          bootstrap (GAE lambda=0)**, and keep the behavior ``old_logp`` unchanged.
        - advantages are normalized ONCE, globally, over the whole mixed batch.
        """
        st = self.storage
        num_envs, num_blocks = st.num_envs, self.num_expl_coef_blocks
        block_size = st.block_size
        leader_block = float(num_blocks - 1)  # original: leader = last block (exploit / eval)

        # Parts are kept TIME-MAJOR [T, envs, ...] here and laid out at the end: flattened for the
        # feedforward generator, concatenated along the ENV axis for the recurrent one (which needs whole
        # trajectories, their dones and their hidden states).
        obs_parts = [st.observations]
        act_parts = [st.actions]
        val_parts = [st.values]
        adv_parts = [st.advantages]
        ret_parts = [st.returns]
        logp_parts = [st.actions_log_prob]
        dp_parts = [list(st.distribution_params)]
        done_parts = [st.dones]
        # Hidden states are per (time, layer, env, hidden); a relabeled copy reuses its SOURCE envs' states
        # verbatim — exact here because the block embedding is applied AFTER the RNN (see SAPGActorRNN).
        ha_parts = [list(st.saved_hidden_state_a)] if st.saved_hidden_state_a is not None else None
        hc_parts = [list(st.saved_hidden_state_c)] if st.saved_hidden_state_c is not None else None

        # Original: num_repeat = min(N, int(ratio)+1); followers = random distinct subset of 1..N-1.
        num_followers = min(num_blocks - 1, int(self.off_policy_ratio))
        follower_r = (torch.randperm(num_blocks - 1, device=self.device)[:num_followers] + 1) if num_followers > 0 else []

        for r in follower_r:
            r = int(r)
            if self.use_others_experience == "lf":
                # physical block (r-1) relabeled to the leader block (N-1); keep only that block's slice.
                src = r - 1
                sl = slice(src * block_size, (src + 1) * block_size)
                robs = st.observations[:, sl].clone()
                width = block_size
                robs[_BLOCK_KEY] = torch.full((robs.shape[0], width, 1), leader_block, device=self.device)
                rlast = last_obs[sl].clone()
                rlast[_BLOCK_KEY] = torch.full((width, 1), leader_block, device=self.device)
                rewards_s, dones_s = st.rewards[:, sl], st.dones[:, sl]
                act_s, logp_s = st.actions[:, sl], st.actions_log_prob[:, sl]
                dp_s = [p[:, sl] for p in st.distribution_params]
            else:  # "all": roll every env's block id by -r (original torch.roll(embd, bs*r)); keep full batch.
                robs = st.observations.clone()
                width = num_envs
                new_bid = ((self._block_ids.view(-1) - r) % num_blocks).view(num_envs, 1)
                robs[_BLOCK_KEY] = new_bid.unsqueeze(0).expand(robs.shape[0], num_envs, 1)
                rlast = last_obs.clone()
                rlast[_BLOCK_KEY] = new_bid
                rewards_s, dones_s = st.rewards, st.dones
                act_s, logp_s = st.actions, st.actions_log_prob
                dp_s = list(st.distribution_params)

            num_steps = robs.shape[0]
            env_sl = sl if self.use_others_experience == "lf" else slice(None)
            with torch.inference_mode():
                if self.critic.is_recurrent:
                    # A recurrent critic cannot be re-evaluated on a flattened batch: V(s_t) depends on the
                    # state the rollout carried into t. Roll it through time from the hidden state SAVED at
                    # t=0 for these envs, zeroing it at each episode boundary exactly as the rollout did.
                    keep = self.critic.get_hidden_state()          # the live rollout state — restored below
                    h0 = tuple(h[0][:, env_sl].contiguous() for h in st.saved_hidden_state_c)
                    self.critic.reset(hidden_state=h0 if len(h0) > 1 else h0[0])
                    vs = []
                    for t in range(num_steps):
                        vs.append(self.critic(robs[t]))
                        self.critic.reset(dones=dones_s[t].squeeze(-1))
                    v_new = torch.stack(vs).view(num_steps, width, 1)
                    v_last = self.critic(rlast).view(width, 1)
                    self.critic.reset(hidden_state=keep)
                else:
                    v_new = self.critic(robs.flatten(0, 1)).view(num_steps, width, 1)
                    v_last = self.critic(rlast).view(width, 1)
            # Relabeled copies use a 1-step TD bootstrap (GAE lambda=0), matching the original (line 1026).
            ret_new, adv_new = self._compute_gae(rewards_s, v_new, dones_s, v_last, self.gamma, 0.0)

            obs_parts.append(robs)
            act_parts.append(act_s)
            val_parts.append(v_new)
            adv_parts.append(adv_new)
            ret_parts.append(ret_new)
            logp_parts.append(logp_s)
            dp_parts.append(list(dp_s))
            done_parts.append(dones_s)
            if ha_parts is not None:
                ha_parts.append([h[:, :, env_sl] for h in st.saved_hidden_state_a])
            if hc_parts is not None:
                hc_parts.append([h[:, :, env_sl] for h in st.saved_hidden_state_c])

        # Original: global advantage normalization over the whole mixed (leader + relabeled) batch, once —
        # the same set of numbers either way, so the layout below does not change the statistic.
        adv_all = torch.cat([a.reshape(-1) for a in adv_parts])
        adv_mean, adv_std = adv_all.mean(), adv_all.std()
        adv_parts = [(a - adv_mean) / (adv_std + 1e-8) for a in adv_parts]

        num_params = len(dp_parts[0])
        if self._recurrent:                 # trajectory batches: extend the ENV axis, keep time
            batch = {
                "observations": torch.cat(obs_parts, dim=1),
                "actions": torch.cat(act_parts, dim=1),
                "values": torch.cat(val_parts, dim=1),
                "advantages": torch.cat(adv_parts, dim=1),
                "returns": torch.cat(ret_parts, dim=1),
                "old_actions_log_prob": torch.cat(logp_parts, dim=1),
                "old_distribution_params": tuple(
                    torch.cat([dp[i] for dp in dp_parts], dim=1) for i in range(num_params)
                ),
                "dones": torch.cat(done_parts, dim=1),
                # hidden layout is [time, layers, envs, hidden] → the env axis is 2
                "hidden_a": ([torch.cat([h[i] for h in ha_parts], dim=2) for i in range(len(ha_parts[0]))]
                             if ha_parts is not None else None),
                "hidden_c": ([torch.cat([h[i] for h in hc_parts], dim=2) for i in range(len(hc_parts[0]))]
                             if hc_parts is not None else None),
            }
        else:                               # row batches: flatten time into the batch axis, as before
            batch = {
                "observations": torch.cat([o.flatten(0, 1) for o in obs_parts], dim=0),
                "actions": torch.cat([a.flatten(0, 1) for a in act_parts], dim=0),
                "values": torch.cat([v.flatten(0, 1) for v in val_parts], dim=0),
                "advantages": torch.cat([a.flatten(0, 1) for a in adv_parts], dim=0),
                "returns": torch.cat([r.flatten(0, 1) for r in ret_parts], dim=0),
                "old_actions_log_prob": torch.cat([lp.flatten(0, 1) for lp in logp_parts], dim=0),
                "old_distribution_params": tuple(
                    torch.cat([dp[i].flatten(0, 1) for dp in dp_parts], dim=0) for i in range(num_params)
                ),
            }
        st.set_relabel_batch(batch)

    # ----- construction (storage swap + block conditioning, additive) -----

    @classmethod
    def _make_storage(cls, env: VecEnv, cfg: dict, obs: TensorDict, device: str) -> RolloutStorage:
        algo = cfg["algorithm"]
        if _sapg_is_active(algo):
            return SAPGRolloutStorage(
                "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device,
                num_blocks=algo["num_expl_coef_blocks"],
            )
        return RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "SAPG":
        """Inactive ⇒ delegate to PPO's factory (byte-parity). Active ⇒ inject the block_id obs key and swap
        actor/critic/distribution to the SAPG-conditioned variants, then reuse PPO's factory (ppo.py stays
        untouched apart from the _make_storage hook)."""
        algo = cfg["algorithm"]
        active = _sapg_is_active(algo)
        # embed_dim is a SAPG-only key; pop it so it never reaches PPO.__init__ via the algorithm splat.
        embed_dim = algo.pop("embed_dim", 32)  # original SAPG default (network_builder param_size=32)

        if active:
            blocks = algo["num_expl_coef_blocks"]
            assert env.num_envs % blocks == 0, (
                f"num_envs ({env.num_envs}) must be divisible by num_expl_coef_blocks ({blocks})"
            )
            block_size = env.num_envs // blocks
            obs[_BLOCK_KEY] = (torch.arange(env.num_envs, device=device) // block_size).view(env.num_envs, 1).float()

            # RNNModel in the contract → the recurrent SAPG variants (LSTM/GRU in front of the MLP).
            recurrent = {"actor": cfg["actor"].get("class_name") == "RNNModel",
                         "critic": cfg["critic"].get("class_name") == "RNNModel"}
            cfg["actor"]["class_name"] = "SAPGActorRNN" if recurrent["actor"] else "SAPGActor"
            cfg["actor"]["num_blocks"] = blocks
            cfg["actor"]["embed_dim"] = embed_dim
            cfg["actor"]["distribution_cfg"]["class_name"] = "PerBlockGaussianDistribution"
            cfg["actor"]["distribution_cfg"]["num_blocks"] = blocks

            cfg["critic"]["class_name"] = "SAPGCriticRNN" if recurrent["critic"] else "SAPGCritic"
            cfg["critic"]["num_blocks"] = blocks
            cfg["critic"]["embed_dim"] = embed_dim

        return super().construct_algorithm(obs, env, cfg, device)  # type: ignore[return-value]


def collapse_sapg_actor(state: dict, obs_dim: int) -> dict:
    """Fold a SAPG actor checkpoint into a PLAIN ``MLPModel`` state_dict by baking in the LEADER block.

    SAPG conditions the actor on a per-block learned embedding (concatenated to the normalized obs) and
    samples with a per-block std. Eval/deploy always runs the **leader block (N-1)** — the exploit/eval
    policy (see :meth:`SAPG._aggregate`). Because the leader embedding is a *constant* vector, its
    contribution to the first MLP layer is a constant bias, so we can eliminate the SAPG structure:

      * drop ``block_embed.weight``;
      * slice ``mlp.0.weight`` back to obs-only inputs and fold the leader embedding into ``mlp.0.bias``
        (``mlp.0`` input layout is ``[normalized_obs (obs_dim), block_embed (embed_dim)]``);
      * collapse the per-block std ``[num_blocks, out]`` to the leader row ``[out]``.

    The result is structurally identical to a plain ``MLPModel`` actor, so it loads strictly and exports
    through the normal path (no block/embedding concept leaks into the deployable graph). A plain PPO
    checkpoint has no ``block_embed.weight`` — callers check for that key before calling this. Used by the
    MetaLab eval to run a SAPG checkpoint as a deployable MLP (no eval-side algorithm flag needed)."""
    embed = state["block_embed.weight"]                 # [num_blocks, embed_dim]
    num_blocks, embed_dim = embed.shape
    leader = num_blocks - 1
    leader_embed = embed[leader]                         # [embed_dim] — constant → folds into mlp.0.bias
    w0 = state["mlp.0.weight"]                           # [hidden0, latent_dim + embed_dim]
    # The MLP's non-embedding input is the model's LATENT: the normalized obs for an MLP actor, the RNN's
    # hidden width for a recurrent one. Read it off the checkpoint rather than assuming obs_dim, so the same
    # fold serves both; obs_dim is still checked for the MLP case, where the two coincide.
    latent_dim = w0.shape[1] - embed_dim
    assert latent_dim == obs_dim or any(k.startswith("rnn.") for k in state), (w0.shape, obs_dim, embed_dim)
    out = {k: v for k, v in state.items() if k != "block_embed.weight"}
    out["mlp.0.weight"] = w0[:, :latent_dim].contiguous()
    out["mlp.0.bias"] = state["mlp.0.bias"] + w0[:, latent_dim:] @ leader_embed
    for key in ("distribution.log_std_param", "distribution.std_param"):  # per-block std → leader row
        if key in state:
            out[key] = state[key][leader].contiguous()
    kind = "RNN" if any(k.startswith("rnn.") for k in state) else "MLP"
    print(f"[eval] SAPG checkpoint detected (blocks={num_blocks}, embed_dim={embed_dim}) → collapsed to "
          f"leader block {leader} (plain {kind} policy)", flush=True)
    return out
