"""Observation / action key strings for the self-contained ACT module.

These mirror the constants used by LeRobot 0.4.4 (`lerobot.utils.constants`) so that a
LeRobot ACT checkpoint and the batches the trainer feeds in line up exactly. Only the keys
the ACT model and its (un)normalization touch are reproduced here.
"""

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"

ACTION = "action"
