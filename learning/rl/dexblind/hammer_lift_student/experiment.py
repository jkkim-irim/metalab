# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""The hammer-lift-student training experiment (MetaLab spoke) — trainer-owned and Isaac-free.

ASYMMETRIC actor-critic: the actor reads the compact ``actor`` obs group (history-stacked, real-robot
producible), the critic reads the ``privileged`` obs group (full state). The split lives in the TASK contract
(``sim/metalab/contract/tasks/rl/hammer_lift_student/``); ``EXP["obs_groups"]`` below only WIRES the model's
actor/critic to those task groups.

Holds two things:

* ``EXP`` — the experiment config (OnPolicyRunner cfg): RL algorithm (PPO; ``ALGO=sapg`` swaps to SAPG),
  critic, runner (networks, optimizer, rollout). Built in-process by the trainer (``learning/rl``).
* ``POLICY`` — the actor spec (obs groups + actor net), derived from ``EXP`` for eval/deploy.

Task knobs (reward weights, thresholds, action scales, DR ranges, curriculum) are SIM-OWNED and live
inline in the task contract (``sim/metalab/contract/tasks/rl/hammer_lift_student/``) — team decision
2026-07-23; the trainer keeps only the RL config.

Separate from ``hammer_lift_teacher``: the teacher's actor sees privileged obs (object realtime pose), so it
is a fast-learning reference; the student's actor is deployable (compact proprioception + reset-latched object
pose only). Same task / reward / physics.
"""
from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# EXP — the experiment config (OnPolicyRunner cfg): the RL algorithm (currently PPO; swap the
# ``algorithm`` sub-cfg for e.g. fast_sac), critic, runner. Consumed in-process by the trainer; the
# runtime fields seed / device / max_iterations / logger / wandb_project / run_name are overridden from
# CLI args. The actor/critic/algorithm sub-cfgs follow the vendored rsl_rl (v5.0.1) schema.
# ---------------------------------------------------------------------------
EXP: dict = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 48,
    "max_iterations": 20000,
    "empirical_normalization": {},
    "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
    "clip_actions": None,
    "check_for_nan": False,
    "save_interval": 200,
    "experiment_name": "jkkim-hammer-lift",
    "run_name": "",
    "logger": "wandb",
    "neptune_project": "metalab",
    "wandb_project": "jkkim-hammer-lift",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "class_name": "OnPolicyRunner",
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
    },
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
        "reward_scale": 0.01,
        "share_cnn_encoders": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
    "policy": {},
}

# --- SAPG (optional, additive) — ALGO=sapg swaps the algorithm to Split-and-Aggregate PG; PPO is the
if os.environ.get("ALGO", "ppo").lower() == "sapg":
    _ppo_algo = EXP["algorithm"]
    EXP["algorithm"] = {
        **{k: v for k, v in _ppo_algo.items() if k != "class_name"},
        "class_name": "SAPG",
        "num_expl_coef_blocks": int(os.environ.get("SAPG_BLOCKS", "4")),
        "ir_type": os.environ.get("SAPG_IR_TYPE", "entropy"),
        "ir_coef_scale": float(os.environ.get("SAPG_IR_COEF_SCALE", "0.0")),
        "off_policy_ratio": float(os.environ.get("SAPG_OFF_POLICY_RATIO", "1.0")),
        "use_others_experience": os.environ.get("SAPG_MODE", "lf"),
        "embed_dim": int(os.environ.get("SAPG_EMBED_DIM", "32")),
    }

# POLICY — the actor subset the EVAL path rebuilds from (`learning/eval/eval_service.py`): eval needs only the
POLICY: dict = {"obs_groups": {"actor": EXP["obs_groups"]["actor"]}, "actor": EXP["actor"]}
