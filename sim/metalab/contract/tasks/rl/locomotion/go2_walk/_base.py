"""Shared core of the go2_walk family — a Unitree Go2 on flat ground tracking a base-velocity command.

Ported from the Genesis locomotion example (``examples/locomotion/go2_env.py`` + ``go2_train.py``): same
standing pose, 50 Hz policy, 1-step command latency, actions = 0.25 rad offsets from the standing pose,
observation layout (base ang vel · projected gravity · command · joint offset · joint vel · last action),
4 s command resampling and the 10 deg roll/pitch fall termination. A recipe supplies the reward table (the
Genesis ``reward_scales``) via :func:`build_task`.

Differences from the Genesis example, on purpose:
- Physics runs at 100 Hz with 2 substeps and ``decimation=2`` (Genesis: 50 Hz with 2 substeps). The policy
  rate is the same 50 Hz; the finer step is for newton parity.
- PD gains come from the robot contract (``robot/go2/go2.yaml``: kp 25 / kv 0.5) rather than the task
  (Genesis: 20 / 0.5).

Authoring rules → sim/metalab/contract/tasks/README.md.
"""
from __future__ import annotations

from sim.metalab.contract.spec import Done, Event, Obs, TaskSpec, values
from sim.metalab.terms import action, events, obs, terminate
from sim.metalab.terms.events.locomotion import COMMAND_KEYS


class PHYSICS:
    hz = 100            # physics rate — dt = 1/hz
    substeps = 2        # physics substeps per physics step
    decimation = 2      # physics steps per POLICY step → policy runs at 50 Hz (Genesis: 0.02 s)
    self_collision = False


class SCENE:
    ground = True

    class robot:
        name = "go2"
        base_pos = [0.0, 0.0, 0.42]
        base_quat = [1.0, 0.0, 0.0, 0.0]
        fixed_base = False
        # Standing pose [deg] — the Genesis default_joint_angles (hip 0, thigh 0.8 / 1.0 rad, calf -1.5 rad).
        # Actions are offsets from this pose (JointDeltaPosition), and it is the reference of
        # ``joint_positions_relative_to_init``.
        init_pose = {
            "FL_hip_joint": 0.0, "FL_thigh_joint": 45.837, "FL_calf_joint": -85.944,
            "FR_hip_joint": 0.0, "FR_thigh_joint": 45.837, "FR_calf_joint": -85.944,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 57.296, "RL_calf_joint": -85.944,
            "RR_hip_joint": 0.0, "RR_thigh_joint": 57.296, "RR_calf_joint": -85.944,
        }

    class camera:
        eye = [2.0, 0.0, 2.5]
        lookat = [0.0, 0.0, 0.5]
        fov = 40.0


# Velocity command ranges (the Genesis command_cfg: walk forward at 0.5 m/s, no lateral / yaw command).
CMD_LIN_VEL_X = [0.5, 0.5]        # [m/s]
CMD_LIN_VEL_Y = [0.0, 0.0]        # [m/s]
CMD_ANG_VEL_Z = [0.0, 0.0]        # [rad/s]
CMD_RESAMPLE_S = [4.0, 4.0]       # [s] Genesis resampling_time_s


class ACTION:
    legs = action.JointDeltaPosition(scale=0.25)   # target = init_pose + 0.25 * a [rad]
    min_delay = 1                                  # Genesis simulate_action_latency: 1 policy step
    max_delay = 1


class OBS:
    base_ang_vel      = Obs(obs.body_angular_velocity_local, body="@frames.base", scale=0.25)
    projected_gravity = Obs(obs.projected_gravity, body="@frames.base")
    command_lin_vel   = Obs(obs.dr_params, keys=list(COMMAND_KEYS[:2]), labels=list(COMMAND_KEYS[:2]), scale=2.0)
    command_ang_vel   = Obs(obs.dr_params, keys=list(COMMAND_KEYS[2:]), labels=list(COMMAND_KEYS[2:]), scale=0.25)
    joint_pos_offset  = Obs(obs.joint_positions_relative_to_init, names="@joints.legs")
    joint_vel         = Obs(obs.joint_velocities, names="@joints.legs", scale=0.05)
    last_action       = Obs(obs.last_action)


class EVENTS:
    # A command at reset, then a fresh one every CMD_RESAMPLE_S (Genesis: _resample_commands on reset + every 4 s).
    command_on_reset = Event(events.sample_velocity_command, "reset",
                             lin_vel_x_range=CMD_LIN_VEL_X, lin_vel_y_range=CMD_LIN_VEL_Y,
                             ang_vel_z_range=CMD_ANG_VEL_Z)
    command_resample = Event(events.sample_velocity_command, "interval",
                             lin_vel_x_range=CMD_LIN_VEL_X, lin_vel_y_range=CMD_LIN_VEL_Y,
                             ang_vel_z_range=CMD_ANG_VEL_Z, interval_range_s=CMD_RESAMPLE_S)


class TERMINATE:
    base_tilt = Done(terminate.body_tilt_exceeded, body="@frames.base", max_roll_deg=10.0, max_pitch_deg=10.0)
    time_out  = Done(terminate.time_out, time_out=True)


def build_task(name: str, *, reward, action=None, events=None, terminate=None) -> TaskSpec:
    """Assemble a go2_walk contract: this base plus a recipe's ``reward`` table. Optional overrides — ``None``
    takes this base's default."""
    return TaskSpec(
        name=name,
        num_envs=4096,
        episode_length_s=20.0,
        physics=PHYSICS,
        scene=values(SCENE),
        action=action if action is not None else ACTION,
        obs=OBS,
        obs_groups={"actor": "all", "privileged": "all"},
        reward=reward,
        events=events if events is not None else EVENTS,
        terminate=terminate if terminate is not None else TERMINATE,
    )
