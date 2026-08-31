# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""whole-body-tracking (WBT) experiment for the hammer-lift task.

A clean, tracking-only experiment (separate from dexblind's lift): learn to
reproduce the FULL configuration — object pose+vel, hand keypoints (palm + fingertips), arm+hand joints,
and velocities — of reference lift motions, via Gaussian-kernel rewards
(``exp(-‖e‖²/σ²)``, weights ~5, unnormalized). Sim-side terms live in
``sim/isaaclab/envs/hammer_lift/mdp/wbt.py`` (env variants ``HammerLiftEnvCfg_WBT`` / ``_WBTCOLLECT``);
driven by ``learning/train.py --trainer rl --experiment tracking --wbt``. Algorithm: PPO (the owned runner).
"""
