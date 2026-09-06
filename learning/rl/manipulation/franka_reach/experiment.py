"""franka_reach training experiment — trainer-owned PPO config for the end-effector reach task.

``EXP`` is the OnPolicyRunner cfg (PPO, actor/critic MLPs, runner); ``POLICY`` the actor subset the eval
path rebuilds from. Every task knob (rewards, goal box, gate, action scale) lives in the sim contract
``sim/metalab/contract/tasks/rl/manipulation/franka_reach/``.
"""
from __future__ import annotations

NUM_STEPS_PER_ENV = 24

# Steps per full episode (15.0 s at 60 Hz) — the contract states rewards per step, so a term paid every
# step totals weight x 900; 1/900 puts the critic target back on the scale PPO's value branch is sized for.
EPISODE_STEPS = 900

EXP: dict = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": NUM_STEPS_PER_ENV,
    "max_iterations": 1000,
    "empirical_normalization": {},
    "obs_groups": {"actor": ["actor"], "critic": ["privileged"]},
    "clip_actions": None,
    "check_for_nan": False,
    "save_interval": 100,
    "experiment_name": "manipulation-franka-reach",
    "run_name": "",
    "logger": "wandb",
    "neptune_project": "metalab",
    "wandb_project": "manipulation-franka-reach",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "class_name": "OnPolicyRunner",
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": None,
    },
    "algorithm": {
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.005,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        "reward_scale": 1.0 / EPISODE_STEPS,
        "share_cnn_encoders": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
    "policy": {},
}

POLICY: dict = {"obs_groups": {"actor": EXP["obs_groups"]["actor"]}, "actor": EXP["actor"]}
