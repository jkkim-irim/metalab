# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""The hammer-lift training experiment — trainer-owned and Isaac-free. This is the file you edit to
tune a run.

It holds two things:

* ``EXP`` — the experiment config: the RL algorithm (currently PPO — swap the ``algorithm`` sub-cfg
  for e.g. fast_sac), critic, and runner (networks, optimizer, rollout). The trainer builds
  ``OnPolicyRunner`` directly from it (``learning/rl``); it never leaves this venv.
Task knobs (reward weights, curriculum schedule, success thresholds, action scales, DR ranges,
observation noise) are SIM-OWNED and live in ``sim/isaaclab/envs/hammer_lift/knobs.py`` — team
decision 2026-07-23; the trainer keeps only the RL config. The five ``TASK_SUCCESS_*_GATE``
constants imported below are a RE-EXPORT of the task-owned success definition
(``envs/hammer_lift/gate.py``, the SoT) for the trainer-side consumers: the eval-config gate stamp
(wandb run config) and the eval gate forensics.

The sim owns the env *mechanism* (how rewards/terminations/curriculum are computed, the scene, the
robot, the physics); this file owns the *experiment* (the values plugged into that mechanism). The
values here are the single source of truth — the sim keeps none of its own.
"""
from __future__ import annotations

# The success definition is TASK-OWNED: sim/isaaclab/envs/hammer_lift/gate.py (pure constants,
# no Isaac imports). This recipe re-exports the values as module constants so they stamp into
# every run config, but the definition lives with the env — recipes cannot redefine success.
from learning.rl.service import ensure_transport_importable

ensure_transport_importable()
from envs.hammer_lift.gate import (  # noqa: E402
    TASK_SUCCESS_CONTACT_COUNT_GATE,
    TASK_SUCCESS_HOLD_STEPS_GATE,
    TASK_SUCCESS_PALM_THRESHOLD_GATE,
    TASK_SUCCESS_POS_THRESHOLD_GATE,
    TASK_SUCCESS_ROT_THRESHOLD_GATE,
)

# Rollout length (env steps / PPO iteration), consumed by the runner (EXP) below. The env's
# iteration-based auto-curriculum keeps its own copy (sim/isaaclab/envs/hammer_lift/knobs.py
# NUM_STEPS_PER_ENV) — the ONE value shared across the trainer/sim boundary; keep them in sync.
NUM_STEPS_PER_ENV = 24

# ---------------------------------------------------------------------------
# EXP — the experiment config (OnPolicyRunner cfg): the RL algorithm (currently PPO; swap the
# ``algorithm`` sub-cfg for e.g. fast_sac), critic, runner. Consumed in-process by the trainer; the
# runtime fields seed / device / max_iterations / logger / wandb_project / run_name are overridden from
# CLI args. The actor/critic/algorithm sub-cfgs follow the vendored rsl_rl (v5.0.1) schema — the exact
# keys learning/rl/{models,nn} accept (no isaaclab_rl cfg wrapper on the trainer side).
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
    "experiment_name": "dexblind_newton_allex",
    "run_name": "",
    "logger": "wandb",
    "neptune_project": "isaaclab",
    "wandb_project": "dexblind_newton_allex",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "class_name": "OnPolicyRunner",
    # Actor: LSTM + MLP (recurrent); obs normalization on.
    "actor": {
        "class_name": "RNNModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
        "rnn_type": "lstm",
        "rnn_hidden_dim": 512,
        "rnn_num_layers": 1,
    },
    # Critic: MLP only (no recurrence); obs normalization on.
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
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
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        "share_cnn_encoders": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
    "policy": {},
}

# POLICY — the actor subset the EVAL path rebuilds from (`learning/eval/eval_service.py`): eval needs
# only the actor architecture + its obs groups (it loads the checkpoint's `actor_state_dict`), not the
# critic / algorithm / runner. Derived from EXP so the actor spec stays single-sourced.
POLICY: dict = {"obs_groups": {"actor": EXP["obs_groups"]["actor"]}, "actor": EXP["actor"]}

