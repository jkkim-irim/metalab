from __future__ import annotations

from sim.metalab.contract.spec import Done, Event, Obs, Rew
from sim.metalab.terms import events, obs, reward, terminate

from . import _base as base
from .parity_objects import OBJECTS

GRIPPER = "panda0_gripper"
FINGERS = ["panda0_leftfinger", "panda0_rightfinger"]


class ACTION:
    class arm:
        scale = 0.1
        ema_tau = 0.5   # [s]

    class gripper:
        scale = 0.02


class OBS:
    joint_pos = Obs(obs.joint_positions, names="@joints.ctrl")
    joint_vel = Obs(obs.joint_velocities, names="@joints.ctrl")
    joint_acc = Obs(obs.joint_accelerations, names="@joints.ctrl")
    joint_torque = Obs(obs.joint_torque_obs, names="@joints.ctrl")
    prev_action_targets = Obs(obs.prev_action_targets)
    last_action = Obs(obs.last_action)
    object_state = Obs(obs.object_state_world)
    object_lin_vel = Obs(obs.object_linear_velocity)
    object_ang_vel = Obs(obs.object_angular_velocity)
    gripper_lin_vel = Obs(obs.body_linear_velocity, body=GRIPPER)
    gripper_ang_vel = Obs(obs.body_angular_velocity, body=GRIPPER)
    finger_contact = Obs(obs.body_contact_flags, bodies=FINGERS)
    finger_object_force = Obs(obs.hand_object_force_magnitude, bodies=FINGERS)
    episode_step = Obs(obs.episode_step)
    instantaneous_reward = Obs(obs.instantaneous_reward)


class REWARD:
    gripper_object_proximity = Rew(reward.palm_object_proximity, weight=1.0, palm_body=GRIPPER, std=0.2)
    joint_torque = Rew(reward.joint_torque_penalty, weight=-1e-4, names="@joints.ctrl")
    joint_vel = Rew(reward.joint_vel_l1, weight=-1e-2, names="@joints.ctrl")
    action_rate = Rew(reward.action_rate_l2, weight=-1e-2)


class EVENTS:
    reset_joints = Event(events.reset_joints_by_offset, "reset", joints="@joints.ctrl",
                         position_range=[-0.05, 0.05])
    object_friction = Event(events.set_shape_friction, "reset", target="object", mu_range=[0.6, 0.9])


class TERMINATE:
    object_below_height = Done(terminate.object_below_height, min_height=0.3)
    object_far_from_gripper = Done(terminate.object_far_from_body, body=GRIPPER, max_distance=2.0)
    object_velocity_exceeded = Done(terminate.object_velocity_exceeded)


TASK = base.build_task(
    "parity_mdp", objects=OBJECTS,
    physics={**base.values(base.PHYSICS), "gravity": [0.0, 0.0, 0.0]},
    episode_length_s=3.0,
    action=ACTION,
    obs=OBS, obs_groups={"actor": "all", "critic": "all"}, obs_history_length={"critic": 2},
    reward=REWARD, events=EVENTS, terminate=TERMINATE,
)
