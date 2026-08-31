# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""The hammer-lift-teacher training experiment (MetaLab spoke) — trainer-owned and Isaac-free.

Holds two things:

* ``EXP`` — the experiment config (OnPolicyRunner cfg): RL algorithm (PPO; ``ALGO=sapg`` swaps to SAPG),
  critic, runner (networks, optimizer, rollout). Built in-process by the trainer (``learning/rl``).
* ``POLICY`` — the actor spec (obs groups + actor net), derived from ``EXP`` for eval/deploy.

Task knobs (reward weights, thresholds, action scales, DR ranges, curriculum) are SIM-OWNED and
live inline in the task contract (``sim/metalab/contract/tasks/rl/hammer_lift_teacher/``) — team
decision 2026-07-23; the trainer keeps only the RL config.

Separate from ``learning/rl/dexblind/hammer_lift/experiment.py`` (chris' isaaclab spoke) on purpose:
each spoke keeps its own values; they are not force-unified.
"""
from __future__ import annotations

import os

# Rollout length (env steps / PPO iteration), consumed by the runner (EXP) below and the curriculum.
NUM_STEPS_PER_ENV = 24

# ---------------------------------------------------------------------------
# EXP — the experiment config (OnPolicyRunner cfg): the RL algorithm (currently PPO; swap the
# ``algorithm`` sub-cfg for e.g. fast_sac), critic, runner. Consumed in-process by the trainer; the
# runtime fields seed / device / max_iterations / logger / wandb_project / run_name are overridden from
# CLI args. The actor/critic/algorithm sub-cfgs follow the vendored rsl_rl (v5.0.1) schema.
# ---------------------------------------------------------------------------
EXP: dict = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": NUM_STEPS_PER_ENV,
    "max_iterations": 5000,
    "empirical_normalization": {},
    "obs_groups": {"actor": ["actor"], "critic": ["privileged"]},
    "clip_actions": None,
    "check_for_nan": False,
    "save_interval": 200,
    "experiment_name": "jkkim-dexblind-hammer-teacher",
    "run_name": "",
    "logger": "wandb",
    "neptune_project": "metalab",
    "wandb_project": "jkkim-dexblind-hammer-teacher",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "class_name": "OnPolicyRunner",
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [1024, 1024, 512, 512],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
    },
    # Critic: same MLP as the actor (privileged, symmetric here; only the Gaussian head differs).
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [1024, 1024, 512, 512],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": None,
    },
    "algorithm": {
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-4,
        "schedule": "adaptive",
        "gamma": 0.995,
        "lam": 0.95,
        "entropy_coef": 0.005,
        "desired_kl": 0.016,
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        # Constant on the env reward, SimToolReal's `reward_shaper.scale_value`. The contract states rewards
        # per step, so a term paid every step totals weight x 600 (10.0 s at 60 Hz) — 1/600 puts the critic's
        # target back on the scale `value_loss_coef` and `clip_param`'s value branch are sized for.
        "reward_scale": 1.0 / 600,
        "share_cnn_encoders": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
    "policy": {},
}

# --- SAPG (optional, additive) — ALGO=sapg swaps the algorithm to Split-and-Aggregate PG; PPO is the
# default (ALGO unset). SAPG reuses the same PPO hyperparameters and adds knobs via env:
#   SAPG_BLOCKS (N contiguous env-blocks; num_envs must be divisible by it),
#   SAPG_MODE (lf|all), SAPG_OFF_POLICY_RATIO (extra relabeled copies), SAPG_EMBED_DIM.
# With SAPG_BLOCKS=1 SAPG is byte-identical to PPO. See learning/rl/sapg.py + SAPG_INTEGRATION_PLAN.md.
if os.environ.get("ALGO", "ppo").lower() == "sapg":
    _ppo_algo = EXP["algorithm"]
    EXP["algorithm"] = {
        **{k: v for k, v in _ppo_algo.items() if k != "class_name"},
        "class_name": "SAPG",
        "num_expl_coef_blocks": int(os.environ.get("SAPG_BLOCKS", "4")),
        # Per-block entropy schedule. Follower block k coef = linspace(0.5->0,N)[k] * ir_coef_scale; the
        # leader block (N-1) is always 0 (no entropy loss). ir_coef_scale (= lambda_ent = sigma) is a TUNED
        # hyperparameter: SAPG paper default is 0 (best for AllegroHand/Regrasping/Throw; Tables 2-4 + §5.1),
        # 0.005 helps Reorientation/ShadowHand. Default 0 here (paper); set SAPG_IR_COEF_SCALE for the rest.
        "ir_type": os.environ.get("SAPG_IR_TYPE", "entropy"),
        "ir_coef_scale": float(os.environ.get("SAPG_IR_COEF_SCALE", "0.0")),
        "off_policy_ratio": float(os.environ.get("SAPG_OFF_POLICY_RATIO", "1.0")),
        # 'lf' (leader-follower) = canonical --sapg (paper §4.3, Figure 3): off-policy copies are block
        # slices → update batch ~ (1 + ratio/N_blocks)x. 'all' (symmetric §4.2) keeps FULL copies → ~2x
        # update batch AND performs worse per the paper's ablation — so 'lf' is the default.
        "use_others_experience": os.environ.get("SAPG_MODE", "lf"),
        "embed_dim": int(os.environ.get("SAPG_EMBED_DIM", "32")),
    }

# POLICY — the actor subset the EVAL path rebuilds from (`learning/eval/eval_service.py`): eval needs
# only the actor architecture + its obs groups (it loads the checkpoint's `actor_state_dict`), not the
# critic / algorithm / runner. Derived from EXP so the actor spec stays single-sourced.
POLICY: dict = {"obs_groups": {"actor": EXP["obs_groups"]["actor"]}, "actor": EXP["actor"]}
