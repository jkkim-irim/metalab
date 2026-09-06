from __future__ import annotations

from sim.metalab.contract.spec import Rew
from sim.metalab.terms import gate, reward

from . import _base as base


class REWARD:
    ee_goal_proximity_coarse = Rew(reward.body_goal_proximity, weight=1.0, body="@frames.palm", std=0.2)
    ee_goal_proximity_fine   = Rew(reward.body_goal_proximity, weight=2.0, body="@frames.palm", std=0.05)
    goal_reach_bonus         = Rew(reward.object_goal_reach_bonus, weight=300.0)
    arm_joint_vel            = Rew(reward.joint_vel_l1, weight=-0.01, names="@joints.arm")
    action_rate              = Rew(reward.action_rate_l2, weight=-0.01)


class GATE:
    predicate = gate.body_at_goal
    goal_dist_tol = 0.02
    hold_steps = 10


TASK = base.build_task("franka_reach_position", reward=REWARD, gate=GATE)
