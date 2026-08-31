# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""WBT hammer-lift experiment (trainer-owned, Isaac-free).

Reuses the dexblind hammer-lift RL config wholesale: the env *mechanism* is identical
(same scene, robot, physics, actions), and the actor infers its input dims from the obs — so the extra
``wbt_goal`` goal-conditioning obs the WBT env adds is absorbed automatically without any change to the
net spec. The behavioural difference (Gaussian-kernel reward, full-state reference, exact RSI,
bad-tracking termination) lives entirely in the sim's WBT env variant, selected at the server
(``server.py --wbt --reference_dir``) and driven by
``learning/train.py --trainer rl --experiment tracking --wbt``.

Only ``experiment_name`` differs, so WBT runs get their own log/checkpoint namespace. Algorithm = PPO
(the in-repo OnPolicyRunner; the config is algorithm-agnostic ``EXP``)."""
from __future__ import annotations

import copy as _copy

# Task knobs are SIM-OWNED (sim/isaaclab/envs/hammer_lift/knobs.py). The gate re-export below is
# for the trainer-side consumers (eval-config gate stamp + eval gate forensics) — the SoT stays
# envs/hammer_lift/gate.py.
from learning.rl.dexblind.hammer_lift.experiment import (  # noqa: F401
    TASK_SUCCESS_CONTACT_COUNT_GATE,
    TASK_SUCCESS_HOLD_STEPS_GATE,
    TASK_SUCCESS_PALM_THRESHOLD_GATE,
    TASK_SUCCESS_POS_THRESHOLD_GATE,
    TASK_SUCCESS_ROT_THRESHOLD_GATE,
)
from learning.rl.dexblind.hammer_lift.experiment import EXP as _DEXBLIND_EXP

# Same RL cfg as dexblind's lift, renamed for a distinct logs/checkpoints namespace.
EXP: dict = _copy.deepcopy(_DEXBLIND_EXP)
EXP["experiment_name"] = "tracking_hammer_wbt"
# Full-batch PPO updates (batch == minibatch): the nets are small and the GPU sits near-idle during
# learning — 24k samples fit trivially, gradient noise drops, and the adaptive-KL controller re-tunes
# the LR to the cleaner steps. Scoped to the tracking experiment (dexblind's cfg is untouched).
EXP["algorithm"]["num_mini_batches"] = 1
# Entropy bonus tamed 0.005 -> 0.001: with the tracking reward, the measured failure arc of every
# long run was an action-std spiral (Policy/mean_std 1.1 -> 5.0 by ~iter 2000) — past the SR peak
# the advantage gradient flattens and the entropy term dominates, noising rollouts until the
# training-distribution success collapses. PPO keeps adequate exploration from init_std=1.0.
EXP["algorithm"]["entropy_coef"] = float(__import__("os").environ.get("WBT_ENTROPY_COEF", "0.005"))
# Exploit-stage LR pin: adaptive-KL steps (~3e-4) erode a resumed peak; a small fixed LR lets
# the critic re-fit around the good mean without displacing the actor.
if __import__("os").environ.get("WBT_LR"):
    EXP["algorithm"]["learning_rate"] = float(__import__("os").environ["WBT_LR"])
    EXP["algorithm"]["schedule"] = "fixed"

# ── improvement-package knobs — W&B provenance stamp ONLY ────────────────────────────────────────────
# The ENV reads these levers from the same env vars in sim/isaaclab/envs/hammer_lift/knobs.py (the
# server inherits this process' environment); here they are parsed again only to stamp EXP["wbt"]
# into the run config, so a run's recipe is readable off its W&B page.
_env = __import__("os").environ
_HOLD_PARTIAL_WEIGHT = float(_env.get("WBT_HOLD_PARTIAL_WEIGHT", "0"))  # 0 = pure streak ramp
_HOLD_PARTIAL_POWER = float(_env.get("WBT_HOLD_PARTIAL_POWER", "4"))
_PHASE_GUARD = _env.get("WBT_PHASE_GUARD", "0") not in ("", "0")
_EVAL_FORCE_GRACE = int(_env.get("WBT_EVAL_FORCE_GRACE", "0"))  # diagnostic: exempt first N steps
# The omniretarget set is the ONLY reward set (superseded sets removed 2026-07-14; the env cfg
# rejects other values loudly). The knob remains for provenance stamping; contact via
# WBT_CONTACT_WEIGHT, hold via WBT_HOLD_WEIGHT.
_REWARD_SET = _env.get("WBT_REWARD_SET", "")
# Contact defaults ON at the record weight: a knob-less eval used to silently build a 5-term
# reward (no contact) while training ran 6 — the derived legend exposed the mismatch (its second
# catch). Explicit WBT_CONTACT_WEIGHT=0 still disables for ablations.
_CONTACT_WEIGHT = float(_env.get("WBT_CONTACT_WEIGHT", "1.0"))
_HOLD_WEIGHT = float(_env.get("WBT_HOLD_WEIGHT", "0"))
# Minus-one ablation knobs (default = the record weights; 0 removes the term entirely).
_JOINT_WEIGHT = float(_env.get("WBT_JOINT_WEIGHT", "0.5"))
_JOINT_VEL_WEIGHT = float(_env.get("WBT_JOINT_VEL_WEIGHT", "0.25"))
# REMOVED LEVERS (2026-07-14, user-directed): the task-init fallback (WBT_NO_RSI_PROB) and its
# eval twin (WBT_EVAL_RANDOM_START) — both exploited this task's accidentally-narrow init
# distribution; the RSI tube is the transferable replacement. Fail loudly instead of silently
# ignoring a knob that no longer exists (the silent-ignore path is how zero-delta runs happen).
for _dead in ("WBT_NO_RSI_PROB", "WBT_EVAL_RANDOM_START"):
    if _env.get(_dead, "") not in ("", "0", "0.0"):
        raise RuntimeError(f"{_dead} was removed 2026-07-14 - the RSI tube (WBT_RSI_NOISE / "
                           "WBT_EVAL_RSI_NOISE) replaced task-init starts")
# RSI-tube DR (the transferable form of start diversity — perturb around the reference's own
# start instead of borrowing the task init): master σ-scale for training / for fixed-refs
# graded-ε robustness evals. Canonical σs live in wbt.py (RSI_NOISE_*).
_RSI_NOISE = float(_env.get("WBT_RSI_NOISE", "0"))          # init (t=0) scale
_RSI_NOISE_MID = _env.get("WBT_RSI_NOISE_MID", "")           # phase-entry scale; "" = 0.25x init
_EVAL_RSI_NOISE = float(_env.get("WBT_EVAL_RSI_NOISE", "0"))
EXP["wbt"] = {"hold_partial_weight": _HOLD_PARTIAL_WEIGHT, "hold_partial_power": _HOLD_PARTIAL_POWER,
              "phase_guard": _PHASE_GUARD,
              "reward_set": (_REWARD_SET or "omniretarget"),
              "contact_weight": _CONTACT_WEIGHT,
              "hold_weight": _HOLD_WEIGHT,
              "joint_weight": _JOINT_WEIGHT,
              "joint_vel_weight": _JOINT_VEL_WEIGHT,
              "rsi_noise": _RSI_NOISE,
              "rsi_noise_mid": (_RSI_NOISE_MID or "0.25x-init"),
              "eval_rsi_noise": _EVAL_RSI_NOISE}
# PRIVILEGED TEACHER (WBT_TEACHER_PRIV): the teacher ACTOR additionally reads the live-object
# group (env_cfg WbtTeacherPrivObsCfg — egocentric hammer pose, actual-vs-ref residual, true
# contacts, hammer velocity, variant), so closed-loop correction is demonstrable under the tube.
# The deployable actor group is untouched; a distilled student never sees this group.
_TEACHER_PRIV = _env.get("WBT_TEACHER_PRIV", "")   # "" / "0" = off (the env reads it via knobs.py)
if _TEACHER_PRIV not in ("", "0"):
    EXP["obs_groups"]["actor"] = list(EXP["obs_groups"]["actor"]) + ["teacher_priv"]
    EXP["obs_groups"]["critic"] = list(EXP["obs_groups"]["critic"]) + ["teacher_priv"]
    EXP["wbt"]["teacher_priv"] = True
if _env.get("WBT_RUN_TAG"):  # explicit W&B display name: {ts}_{tag}_{it}it_{envs}envs-{sha}
    EXP["run_name"] = _env["WBT_RUN_TAG"]
if _env.get("WBT_NOTES"):  # free-text run annotation, verbatim into wandb.config["experiment_notes"]
    EXP["experiment_notes"] = _env["WBT_NOTES"]

# Everything an eval needs, module-owned (mirrors dexblind's export): eval_service resolves the
# experiment MODULE from --experiment, so the actor spec follows the experiment instead of riding
# a hardwired dexblind import (the nets are identical today; this keeps any future divergence
# from silently loading the wrong spec).
POLICY: dict = {"obs_groups": {"actor": EXP["obs_groups"]["actor"]}, "actor": EXP["actor"]}
