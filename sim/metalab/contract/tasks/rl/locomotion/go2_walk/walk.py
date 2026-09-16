"""go2_walk RECIPE — the Genesis ``go2_train.py`` reward table ("walk").

Weights are the Genesis ``reward_scales`` times its 0.02 s policy dt: Genesis multiplies every scale by dt
before summing, and the metalab driver does not (``reward = weight * term``), so this keeps the per-step
return equal to the reference. The shared contract (obs / scene / robot / sim / action / events /
termination) lives in ``_base.py`` beside this file.
"""
from __future__ import annotations

from sim.metalab.contract.spec import Rew
from sim.metalab.terms import reward

from . import _base as base

_DT = 0.02                 # [s] Genesis policy dt folded into the reward scales
_TRACKING_SIGMA = 0.25     # Genesis tracking_sigma


class REWARD:
    tracking_lin_vel   = Rew(reward.tracking_lin_vel_xy, weight=1.0 * _DT, body="@frames.base", std=_TRACKING_SIGMA)
    tracking_ang_vel   = Rew(reward.tracking_ang_vel_z,  weight=0.2 * _DT, body="@frames.base", std=_TRACKING_SIGMA)
    lin_vel_z          = Rew(reward.body_lin_vel_z_l2,   weight=-1.0 * _DT, body="@frames.base")
    base_height        = Rew(reward.body_height_l2,      weight=-50.0 * _DT, body="@frames.base", target_height=0.3)
    action_rate        = Rew(reward.action_rate_l2,      weight=-0.005 * _DT)
    similar_to_default = Rew(reward.joint_pose_l1_from_init, weight=-0.1 * _DT, names="@joints.legs")


TASK = base.build_task("go2_walk_walk", reward=REWARD)
