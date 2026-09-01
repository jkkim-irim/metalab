"""Unit tests for SAPG (learning/rl/sapg.py) — CPU-only, no sim and no GPU.

Pins the three properties SAPG's correctness rests on, each against the real code path:

* the **PPO-parity gate** (``_sapg_is_active``) — which knob combinations fall back to plain PPO;
* the **per-block entropy schedule** — the table a real ``SAPG`` builds, the weighting
  ``SAPG._entropy_loss`` applies, and the fact that ``PPO.update`` calls that hook at all (it used to
  inline the uniform term, which silently disabled the schedule);
* **``collapse_sapg_actor``** — a SAPG checkpoint folds into a plain ``MLPModel`` that loads strictly
  and reproduces the leader block's output (the eval/deploy path).

Run in the engine env from the repo root:  python -m pytest learning/rl -q
"""
import inspect

import pytest
from tensordict import TensorDict
import torch

from learning.rl.models import MLPModel
from learning.rl.ppo import PPO  # noqa: E402
from learning.rl.sapg import (  # noqa: E402
    _BLOCK_KEY,
    SAPG,
    SAPGActor,
    SAPGActorRNN,
    SAPGCritic,
    SAPGCriticRNN,
    SAPGRolloutStorage,
    _sapg_is_active,
    collapse_sapg_actor,
)

NUM_ENVS, OBS_DIM, NUM_ACTIONS, NUM_BLOCKS, EMBED_DIM = 8, 5, 3, 4, 2
HIDDEN = dict(hidden_dims=[16, 16], activation="elu")

# Every SAPG knob off — the configuration that must take the untouched PPO path.
ALL_OFF = {
    "num_expl_coef_blocks": 1,
    "ir_type": "none",
    "off_policy_ratio": 0.0,
    "use_others_experience": "none",
}


def _obs(block_ids: torch.Tensor | None = None) -> TensorDict:
    """Observation dict with the block id SAPG conditions on (contiguous blocks unless given)."""
    if block_ids is None:
        block_ids = (torch.arange(NUM_ENVS) // (NUM_ENVS // NUM_BLOCKS)).view(NUM_ENVS, 1).float()
    return TensorDict({"policy": torch.randn(NUM_ENVS, OBS_DIM), _BLOCK_KEY: block_ids},
                      batch_size=[NUM_ENVS])


def _actor(obs: TensorDict) -> SAPGActor:
    return SAPGActor(
        obs, {"actor": ["policy"]}, "actor", NUM_ACTIONS,
        num_blocks=NUM_BLOCKS, embed_dim=EMBED_DIM,
        distribution_cfg={"class_name": "PerBlockGaussianDistribution", "num_blocks": NUM_BLOCKS},
        **HIDDEN,
    )


def _algorithm(ir_type: str = "entropy", ir_coef_scale: float = 0.1) -> SAPG:
    """A real SAPG — its __init__ is what builds the per-block entropy coefficient table."""
    obs = _obs()
    groups = {"actor": ["policy"], "critic": ["policy"]}
    critic = SAPGCritic(obs, groups, "critic", 1, num_blocks=NUM_BLOCKS, embed_dim=EMBED_DIM, **HIDDEN)
    storage = SAPGRolloutStorage("rl", NUM_ENVS, 4, obs, [NUM_ACTIONS], "cpu", num_blocks=NUM_BLOCKS)
    return SAPG(
        _actor(obs), critic, storage,
        num_expl_coef_blocks=NUM_BLOCKS, ir_type=ir_type, ir_coef_scale=ir_coef_scale,
        off_policy_ratio=1.0, use_others_experience="lf", device="cpu",
    )


class _Batch:
    """Minibatch stub carrying the only field _entropy_loss reads."""

    def __init__(self, observations: TensorDict) -> None:
        self.observations = observations


# --------------------------------------------------------------------------------------------------
# PPO-parity gate
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override, active",
    [
        ({}, False),                                    # all knobs off -> plain PPO
        ({"num_expl_coef_blocks": 4}, True),
        ({"ir_type": "entropy"}, True),
        ({"off_policy_ratio": 1.0}, True),
        ({"use_others_experience": "lf"}, True),
    ],
)
def test_ppo_parity_needs_every_knob_off(override: dict, active: bool):
    assert _sapg_is_active({**ALL_OFF, **override}) is active


def test_one_block_alone_is_not_the_ppo_path():
    """SAPG_BLOCKS=1 does NOT restore plain PPO: ALGO=sapg leaves ir_type/off_policy_ratio/mode on,
    and the gate ORs all four knobs. Guards the claim in hammer_lift_teacher/experiment.py."""
    algo_sapg_defaults = {
        "num_expl_coef_blocks": 1,      # SAPG_BLOCKS=1
        "ir_type": "entropy",           # SAPG_IR_TYPE default
        "off_policy_ratio": 1.0,        # SAPG_OFF_POLICY_RATIO default
        "use_others_experience": "lf",  # SAPG_MODE default
    }
    assert _sapg_is_active(algo_sapg_defaults)


# --------------------------------------------------------------------------------------------------
# Per-block entropy schedule
# --------------------------------------------------------------------------------------------------


def test_indivisible_num_envs_fails_loud():
    """The Launchpad builds num_envs as envs_per_block * num_blocks. A floor divide on an indivisible count
    would send the tail envs to block id num_blocks -- past both the coef table and the block embedding."""
    bad = NUM_ENVS + 1                                  # not a multiple of NUM_BLOCKS
    obs = TensorDict({"policy": torch.randn(bad, OBS_DIM), _BLOCK_KEY: torch.zeros(bad, 1)}, batch_size=[bad])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    critic = SAPGCritic(obs, groups, "critic", 1, num_blocks=NUM_BLOCKS, embed_dim=EMBED_DIM, **HIDDEN)
    storage = SAPGRolloutStorage("rl", bad, 4, obs, [NUM_ACTIONS], "cpu", num_blocks=NUM_BLOCKS)
    with pytest.raises(AssertionError, match="must be divisible by"):
        SAPG(_actor(obs), critic, storage, num_expl_coef_blocks=NUM_BLOCKS, ir_type="entropy",
             ir_coef_scale=0.1, off_policy_ratio=1.0, use_others_experience="lf", device="cpu")


def test_block_map_covers_exactly_the_requested_blocks():
    """Every env lands in [0, num_blocks) and every block gets the same count."""
    algo = _algorithm()
    ids = algo._block_ids.reshape(-1).long()
    assert ids.min().item() == 0 and ids.max().item() == NUM_BLOCKS - 1
    counts = torch.bincount(ids, minlength=NUM_BLOCKS)
    assert counts.tolist() == [NUM_ENVS // NUM_BLOCKS] * NUM_BLOCKS


def test_entropy_hook_is_called_from_the_update_loss():
    """The per-block schedule only applies because PPO.update goes through the hook. It used to
    compute `- self.entropy_coef * entropy.mean()` inline, which left SAPG._entropy_loss dead."""
    source = inspect.getsource(PPO.update)
    assert "self._entropy_loss(entropy, batch)" in source
    assert "self.entropy_coef * entropy.mean()" not in source
    assert SAPG._entropy_loss is not PPO._entropy_loss


def test_entropy_coef_table_ramps_to_zero_at_the_leader():
    """Upstream schedule (rl_games a2c_common.py:328): linspace(0.5, 0) * scale, so the leader block
    gets exactly 0 (pure exploit) and the maximum is half the scale, not the scale itself."""
    alg = _algorithm(ir_coef_scale=0.1)
    table = alg._entropy_coef_table
    assert torch.allclose(table, torch.linspace(0.5, 0.0, NUM_BLOCKS) * 0.1)
    assert table[-1].item() == 0.0
    assert table[0].item() == pytest.approx(0.05)


def test_zero_scale_disables_the_entropy_bonus_entirely():
    """SAPG_IR_COEF_SCALE=0.0 (the ALGO=sapg default, matching the paper) zeroes every block."""
    alg = _algorithm(ir_coef_scale=0.0)
    entropy = torch.rand(NUM_ENVS)
    batch = _Batch(_obs())
    assert torch.count_nonzero(alg._entropy_coef_table) == 0
    assert alg._entropy_loss(entropy, batch).item() == 0.0


def test_entropy_bonus_is_weighted_per_block():
    alg = _algorithm(ir_coef_scale=0.1)
    entropy = torch.rand(NUM_ENVS)
    obs = _obs()
    expected = (alg._entropy_coef_table[obs[_BLOCK_KEY].reshape(-1).long()] * entropy).mean()

    weighted = alg._entropy_loss(entropy, _Batch(obs))

    assert torch.allclose(weighted, expected)
    # The whole point of the schedule: it is NOT the uniform PPO term.
    assert not torch.allclose(weighted, alg.entropy_coef * entropy.mean())


def test_entropy_falls_back_to_the_uniform_term_without_the_entropy_reward():
    """ir_type='none' keeps the block split and off-policy mixing but drops the per-block weighting,
    exactly like upstream's `else: entropy_coef = self.entropy_coef` branch."""
    alg = _algorithm(ir_type="none")
    entropy = torch.rand(NUM_ENVS)

    uniform = alg._entropy_loss(entropy, _Batch(_obs()))

    assert torch.allclose(uniform, alg.entropy_coef * entropy.mean())


# --------------------------------------------------------------------------------------------------
# Checkpoint collapse (eval / deploy)
# --------------------------------------------------------------------------------------------------


def test_collapsed_checkpoint_reproduces_the_leader_block():
    """Eval always runs the leader block, whose embedding is constant and folds into mlp.0.bias — so a
    SAPG checkpoint must load strictly into a plain MLPModel and give the same actions."""
    obs = _obs()
    sapg_actor = _actor(obs)
    sapg_actor.eval()
    leader_obs = TensorDict(
        {"policy": obs["policy"], _BLOCK_KEY: torch.full((NUM_ENVS, 1), float(NUM_BLOCKS - 1))},
        batch_size=[NUM_ENVS],
    )
    with torch.no_grad():
        expected = sapg_actor(leader_obs)

    collapsed = collapse_sapg_actor(dict(sapg_actor.state_dict()), OBS_DIM)

    plain_obs = TensorDict({"policy": obs["policy"]}, batch_size=[NUM_ENVS])
    plain_actor = MLPModel(plain_obs, {"actor": ["policy"]}, "actor", NUM_ACTIONS,
                           distribution_cfg={"class_name": "GaussianDistribution"}, **HIDDEN)
    plain_actor.load_state_dict(collapsed, strict=True)   # no block/embedding concept may leak out
    plain_actor.eval()
    with torch.no_grad():
        got = plain_actor(plain_obs)

    assert torch.allclose(expected, got, atol=1e-6)


def test_collapse_drops_the_block_embedding_and_picks_the_leader_std():
    obs = _obs()
    state = dict(_actor(obs).state_dict())

    collapsed = collapse_sapg_actor(dict(state), OBS_DIM)

    assert "block_embed.weight" not in collapsed
    assert collapsed["mlp.0.weight"].shape[1] == OBS_DIM   # embedding columns folded into the bias
    assert torch.equal(collapsed["distribution.std_param"], state["distribution.std_param"][NUM_BLOCKS - 1])


# --------------------------------------------------------------------------------------------------
# recurrent SAPG (LSTM in front of the MLP — rl_games' rnn.before_mlp)
# --------------------------------------------------------------------------------------------------

RNN_HIDDEN, RNN_LAYERS, HORIZON = 12, 1, 6


def _rnn_actor(obs: TensorDict) -> SAPGActorRNN:
    return SAPGActorRNN(
        obs, {"actor": ["policy"]}, "actor", NUM_ACTIONS,
        num_blocks=NUM_BLOCKS, embed_dim=EMBED_DIM,
        distribution_cfg={"class_name": "PerBlockGaussianDistribution", "num_blocks": NUM_BLOCKS},
        rnn_type="lstm", rnn_hidden_dim=RNN_HIDDEN, rnn_num_layers=RNN_LAYERS, **HIDDEN,
    )


def _rnn_critic(obs: TensorDict) -> SAPGCriticRNN:
    return SAPGCriticRNN(
        obs, {"actor": ["policy"], "critic": ["policy"]}, "critic", 1,
        num_blocks=NUM_BLOCKS, embed_dim=EMBED_DIM,
        rnn_type="lstm", rnn_hidden_dim=RNN_HIDDEN, rnn_num_layers=RNN_LAYERS, **HIDDEN,
    )


def test_rnn_actor_puts_the_lstm_in_front_of_the_mlp():
    """The MLP head must consume [rnn_hidden + embed], i.e. the recurrent layer is FIRST and the block
    embedding rides behind it (which is what makes the stored hidden states valid after relabeling)."""
    actor = _rnn_actor(_obs())
    assert actor.is_recurrent
    assert actor.rnn.rnn.input_size == OBS_DIM, "the LSTM must read the observation, not a hidden layer"
    assert actor.mlp[0].in_features == RNN_HIDDEN + EMBED_DIM


def test_rnn_actor_block_embedding_changes_the_output_not_the_hidden_state():
    """Same observations, different block ids: the action changes (per-block conditioning still works) while
    the recurrent state does not (it is a function of the observations alone)."""
    obs = _obs()
    actor = _rnn_actor(obs)
    actor.reset()
    actor(obs, stochastic_output=False)
    h_a = tuple(h.clone() for h in actor.get_hidden_state())
    out_a = actor.mlp(actor.get_latent(obs)).clone()

    shifted = obs.clone()
    shifted[_BLOCK_KEY] = (obs[_BLOCK_KEY] + 1) % NUM_BLOCKS
    actor.reset()
    actor(shifted, stochastic_output=False)
    h_b = actor.get_hidden_state()
    out_b = actor.mlp(actor.get_latent(shifted))

    assert not torch.allclose(out_a, out_b), "the block embedding must reach the output"
    for x, y in zip(h_a, h_b):
        assert torch.allclose(x, y), "the block id must NOT enter the recurrent state"


def test_recurrent_batches_align_block_ids_with_unpadded_latents():
    """In an update batch the observations are PADDED trajectories and the RNN returns only the VALID steps.
    The embedding rows have to be unpadded the same way, or every transition gets another one's block."""
    from learning.rl.sapg import _block_ids
    from learning.rl.utils import split_and_pad_trajectories

    block = (torch.arange(NUM_ENVS) // (NUM_ENVS // NUM_BLOCKS)).view(1, NUM_ENVS, 1).float()
    obs = TensorDict({_BLOCK_KEY: block.expand(HORIZON, NUM_ENVS, 1).clone()}, batch_size=[HORIZON, NUM_ENVS])
    dones = torch.zeros(HORIZON, NUM_ENVS, 1, dtype=torch.bool)
    dones[HORIZON // 2, 0] = True                      # env 0 gets TWO trajectories, the rest one each
    padded, masks = split_and_pad_trajectories(obs, dones)
    assert padded.shape[1] > NUM_ENVS, "the fixture must actually split (and therefore pad) something"

    ids = _block_ids(padded, masks)
    assert tuple(ids.shape) == (HORIZON, NUM_ENVS), "unpadded back to the rollout layout"
    assert torch.equal(ids, obs[_BLOCK_KEY].squeeze(-1).long()), "each step keeps ITS OWN env's block id"


def _recurrent_algorithm() -> SAPG:
    obs = _obs()
    storage = SAPGRolloutStorage("rl", NUM_ENVS, HORIZON, obs, [NUM_ACTIONS], "cpu", num_blocks=NUM_BLOCKS)
    return SAPG(
        _rnn_actor(obs), _rnn_critic(obs), storage,
        num_expl_coef_blocks=NUM_BLOCKS, ir_type="entropy", ir_coef_scale=0.1,
        off_policy_ratio=1.0, use_others_experience="lf", device="cpu",
    )


def test_recurrent_sapg_is_accepted_and_feedforward_sapg_models_are_not():
    alg = _recurrent_algorithm()
    assert alg._recurrent
    obs = _obs()
    storage = SAPGRolloutStorage("rl", NUM_ENVS, HORIZON, obs, [NUM_ACTIONS], "cpu", num_blocks=NUM_BLOCKS)
    with pytest.raises(AssertionError, match="SAPGActorRNN"):
        SAPG(_actor(obs), _rnn_critic(obs), storage, num_expl_coef_blocks=NUM_BLOCKS, ir_type="entropy",
             ir_coef_scale=0.1, off_policy_ratio=1.0, use_others_experience="lf", device="cpu")


def test_augmented_recurrent_batches_keep_time_and_carry_hidden_states():
    """The off-policy pool must stay TIME-MAJOR and bring dones + hidden states, or the trajectory splitting
    that a recurrent update depends on has nothing to split and no state to start from."""
    alg = _recurrent_algorithm()
    st = alg.storage
    envs, layers = NUM_ENVS, RNN_LAYERS
    st.observations = TensorDict(
        {"policy": torch.randn(HORIZON, envs, OBS_DIM),
         _BLOCK_KEY: (torch.arange(envs) // (envs // NUM_BLOCKS)).view(1, envs, 1).expand(HORIZON, envs, 1).float()},
        batch_size=[HORIZON, envs])
    st.actions = torch.randn(HORIZON, envs, NUM_ACTIONS)
    st.rewards = torch.randn(HORIZON, envs, 1)
    st.dones = torch.zeros(HORIZON, envs, 1, dtype=torch.bool)
    st.dones[HORIZON // 2] = True                                  # one episode boundary, all envs
    st.values = torch.randn(HORIZON, envs, 1)
    st.returns = torch.randn(HORIZON, envs, 1)
    st.advantages = torch.randn(HORIZON, envs, 1)
    st.actions_log_prob = torch.randn(HORIZON, envs, 1)
    st.distribution_params = (torch.randn(HORIZON, envs, NUM_ACTIONS), torch.randn(HORIZON, envs, NUM_ACTIONS))
    st.saved_hidden_state_a = [torch.randn(HORIZON, layers, envs, RNN_HIDDEN) for _ in range(2)]  # LSTM (h, c)
    st.saved_hidden_state_c = [torch.randn(HORIZON, layers, envs, RNN_HIDDEN) for _ in range(2)]

    last_obs = TensorDict({"policy": torch.randn(envs, OBS_DIM),
                           _BLOCK_KEY: torch.zeros(envs, 1)}, batch_size=[envs])
    alg._aggregate(last_obs)
    rel = st._relabel

    assert rel["observations"].shape[0] == HORIZON, "time axis must survive"
    pool_envs = rel["observations"].shape[1]
    assert pool_envs > envs, "the augmented copies extend the ENV axis"
    assert rel["dones"].shape == (HORIZON, pool_envs, 1)
    for h in rel["hidden_a"] + rel["hidden_c"]:
        assert h.shape == (HORIZON, layers, pool_envs, RNN_HIDDEN)

    batches = list(st.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
    assert batches, "the recurrent generator must yield the augmented pool"
    b = batches[0]
    assert b.masks is not None and b.hidden_states[0] is not None and b.hidden_states[1] is not None
    assert b.observations.shape[0] == HORIZON


def test_recurrent_sapg_completes_a_real_update_and_moves_the_lstm():
    """End-to-end: drive a rollout through the REAL act/process_env_step path (so hidden states are saved by
    the code that saves them in training), aggregate, and run PPO.update. The LSTM weights must receive
    gradient — the whole point is that the recurrent layer is being trained, not carried along."""
    alg = _recurrent_algorithm()
    torch.manual_seed(0)
    obs = _obs()
    alg.actor.reset()
    alg.critic.reset()
    with torch.inference_mode():          # the rollout phase, exactly as OnPolicyRunner wraps it — the stored
        for t in range(HORIZON):          # hidden states must carry no graph or the epochs re-backward it
            alg.act(obs)
            rewards = torch.randn(NUM_ENVS)
            dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
            if t == HORIZON // 2:
                dones[: NUM_ENVS // 2] = True                # a mid-rollout episode boundary
            obs = _obs()
            alg.process_env_step(obs, rewards, dones, {})
        alg.compute_returns(obs)          # SAPG's override also builds the relabel pool — still in-mode,
                                          # which is where OnPolicyRunner calls it too
    st = alg.storage
    assert st.saved_hidden_state_a is not None and st.saved_hidden_state_c is not None, \
        "the rollout must have recorded hidden states for both models"
    assert st.saved_hidden_state_a[0].shape == (HORIZON, RNN_LAYERS, NUM_ENVS, RNN_HIDDEN)
    assert st._relabel is not None and st._relabel["observations"].shape[0] == HORIZON

    before = alg.actor.rnn.rnn.weight_hh_l0.detach().clone()
    losses = alg.update()
    after = alg.actor.rnn.rnn.weight_hh_l0

    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
    assert not torch.allclose(before, after), "the LSTM must be getting gradient from the update"


def test_lstm_actor_with_a_feedforward_critic_updates():
    """The shipped configuration: LSTM actor, MLP critic. The two models land on different grids inside a
    recurrent minibatch (the RNN unpads, the MLP does not), so this pins that SAPG reconciles them."""
    obs = _obs()
    critic = SAPGCritic(obs, {"actor": ["policy"], "critic": ["policy"]}, "critic", 1,
                        num_blocks=NUM_BLOCKS, embed_dim=EMBED_DIM, **HIDDEN)
    storage = SAPGRolloutStorage("rl", NUM_ENVS, HORIZON, obs, [NUM_ACTIONS], "cpu", num_blocks=NUM_BLOCKS)
    alg = SAPG(_rnn_actor(obs), critic, storage, num_expl_coef_blocks=NUM_BLOCKS, ir_type="entropy",
               ir_coef_scale=0.1, off_policy_ratio=1.0, use_others_experience="lf", device="cpu")
    assert alg._recurrent and not alg.critic.is_recurrent

    torch.manual_seed(0)
    o = _obs()
    alg.actor.reset()
    with torch.inference_mode():
        for t in range(HORIZON):
            alg.act(o)
            dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
            if t == HORIZON // 2:
                dones[: NUM_ENVS // 2] = True
            o = _obs()
            alg.process_env_step(o, torch.randn(NUM_ENVS), dones, {})
        alg.compute_returns(o)

    before = alg.actor.rnn.rnn.weight_hh_l0.detach().clone()
    losses = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), losses
    assert not torch.allclose(before, alg.actor.rnn.rnn.weight_hh_l0)


def test_collapsed_recurrent_checkpoint_loads_as_a_plain_rnn_and_exports():
    """The eval/deploy path for an LSTM SAPG run — it fires on EVERY checkpoint (the trainer's per-checkpoint
    recording loads through it), so a break here stops a training run mid-flight, not just an eval."""
    from learning.eval.protocol import build_actor

    obs = _obs()
    dist = {"class_name": "PerBlockGaussianDistribution", "num_blocks": NUM_BLOCKS,
            "init_std": 1.0, "std_type": "log"}
    sapg = SAPGActorRNN(obs, {"actor": ["policy"]}, "actor", NUM_ACTIONS,
                        num_blocks=NUM_BLOCKS, embed_dim=EMBED_DIM, distribution_cfg=dist,
                        rnn_type="lstm", rnn_hidden_dim=RNN_HIDDEN, rnn_num_layers=RNN_LAYERS, **HIDDEN)
    state = dict(sapg.state_dict())
    obs_dim = state["mlp.0.weight"].shape[1] - state["block_embed.weight"].shape[1]   # what actor.py computes
    collapsed = collapse_sapg_actor(dict(state), obs_dim)
    assert "block_embed.weight" not in collapsed
    assert any(k.startswith("rnn.") for k in collapsed), "the recurrent weights must survive the fold"

    spec = {"class_name": "RNNModel", "rnn_type": "lstm", "rnn_hidden_dim": RNN_HIDDEN,
            "rnn_num_layers": RNN_LAYERS, "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
            **HIDDEN}
    plain = build_actor(obs, spec, {"actor": ["policy"]}, NUM_ACTIONS, "cpu", collapsed)   # strict load

    leader = _obs()
    leader[_BLOCK_KEY] = torch.full((NUM_ENVS, 1), float(NUM_BLOCKS - 1))
    sapg.eval(), plain.eval()
    with torch.no_grad():
        sapg.reset(); a = sapg(leader, stochastic_output=False)
        plain.reset(); b = plain(leader, stochastic_output=False)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()

    torch.jit.script(plain.as_jit())          # the deploy export the eval writes next to the checkpoint
