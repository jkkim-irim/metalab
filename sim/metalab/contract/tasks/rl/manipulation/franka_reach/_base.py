from __future__ import annotations

from sim.metalab.contract.spec import Done, Event, Obs, TaskSpec, values
from sim.metalab.terms import action, events, obs, terminate


class PHYSICS:
    hz = 120
    substeps = 2
    decimation = 2


class SCENE:
    ground = True

    class robot:
        name = "franka"
        base_pos = [0.0, 0.0, 0.0]
        base_quat = [1.0, 0.0, 0.0, 0.0]
        fixed_base = True
        init_pose = {
            "panda0_joint1": 0.0, "panda0_joint2": 20.0, "panda0_joint3": 0.0, "panda0_joint4": -110.0,
            "panda0_joint5": 0.0, "panda0_joint6": 130.0, "panda0_joint7": 45.0,
            "panda0_finger_joint1": 0.0, "panda0_finger_joint2": 0.0,
        }

    class camera:
        eye = [1.9, -1.5, 1.4]
        lookat = [0.5, 0.0, 0.4]
        fov = 40.0

    class goal:
        pos = [0.5, 0.0, 0.4]
        goal_dist_tol = 0.02


GOAL_X = [0.35, 0.65]
GOAL_Y = [-0.25, 0.25]
GOAL_Z = [0.2, 0.6]


class ACTION:
    arm = action.JointDeltaPosition(scale=0.5, ema_tau=0.1)


class OBS:
    joint_pos      = Obs(obs.joint_positions, names="@joints.arm")
    joint_vel      = Obs(obs.joint_velocities, names="@joints.arm")
    ee_pose        = Obs(obs.body_pose_in_chest, chest_body="@frames.base", target_body="@frames.palm")
    goal_pos       = Obs(obs.goal_position)
    ee_goal_error  = Obs(obs.body_goal_error, body="@frames.palm")
    last_action    = Obs(obs.last_action)


class EVENTS:
    sample_goal_position = Event(events.sample_goal_position, "interval",
                                 x_range=GOAL_X, y_range=GOAL_Y, z_range=GOAL_Z, interval_range_s=[3.0, 5.0])
    reset_joints_by_offset = Event(events.reset_joints_by_offset, "reset", joints="@joints.arm",
                                   position_range=[-0.1, 0.1])


class TERMINATE:
    time_out = Done(terminate.time_out, time_out=True)


def build_task(name: str, *, reward, gate, action=None, events=None) -> TaskSpec:
    return TaskSpec(
        name=name,
        num_envs=4096,
        env_spacing=1.5,
        episode_length_s=15.0,
        physics=PHYSICS,
        scene=values(SCENE),
        action=action if action is not None else ACTION,
        obs=OBS,
        obs_groups={"actor": "all", "privileged": "all"},
        reward=reward,
        events=events if events is not None else EVENTS,
        terminate=TERMINATE,
        gate=gate,
    )
