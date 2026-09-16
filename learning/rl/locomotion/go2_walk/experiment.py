"""The go2-walk training experiment (MetaLab spoke) — trainer-owned and engine-free.

``EXP`` is the OnPolicyRunner cfg, ``POLICY`` the actor subset eval rebuilds from. The values are the Genesis
``examples/locomotion/go2_train.py`` ``get_train_cfg`` (PPO, 3-layer MLP 512/256/128, lr 1e-3, 24 steps per
env), mapped onto the vendored rsl_rl (v5.0.1) schema. Task knobs (rewards, action scale, command ranges,
termination) are SIM-OWNED and live in ``sim/metalab/contract/tasks/rl/locomotion/go2_walk/``.
"""
from __future__ import annotations

NUM_STEPS_PER_ENV = 24
RECORD_ENVS = 1

EXP: dict = {
    "seed": 1,
    "device": "cuda:0",
    "num_steps_per_env": NUM_STEPS_PER_ENV,
    "max_iterations": 101,
    "empirical_normalization": {},
    # The contract's obs_groups: actor == privileged for go2_walk (Genesis feeds the critic the policy obs).
    "obs_groups": {"actor": ["actor"], "critic": ["privileged"]},
    "clip_actions": None,
    "check_for_nan": False,
    "save_interval": 100,
    "experiment_name": "go2-walk",
    "run_name": "",
    "logger": "wandb",
    "neptune_project": "metalab",
    "wandb_project": "go2-walk",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
    "class_name": "OnPolicyRunner",
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
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
        "entropy_coef": 0.01,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        # The recipe already folds the 0.02 s policy dt into its weights (Genesis `reward_scales *= dt`), so
        # the env reward is on the reference scale as-is.
        "reward_scale": 1.0,
        "share_cnn_encoders": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
    "policy": {},
}

POLICY: dict = {"obs_groups": {"actor": EXP["obs_groups"]["actor"]}, "actor": EXP["actor"]}
