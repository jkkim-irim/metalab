"""Single source of conventions and units for the whole pipeline (engine-agnostic).

Every parser, state_adapter, term and contract must follow these. Implicit
conventions (quaternion order, angle unit, coordinate frame) that drift per
engine silently break sim2sim parity, so they are pinned here and violations
fail loud.

Imports no engine and no heavy dependency (pure constants/enums).
"""
from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# Units — SI, no exceptions
# ---------------------------------------------------------------------------
LENGTH_UNIT = "m"      # length [meter]
MASS_UNIT = "kg"       # mass [kilogram]
TIME_UNIT = "s"        # time [second]
FORCE_UNIT = "N"       # force [newton]
ANGLE_UNIT = "rad"     # angle [radian] — all angles/rotations in radians; no degrees.


# ---------------------------------------------------------------------------
# Quaternion order
# ---------------------------------------------------------------------------
class QuatOrder(str, Enum):
    """Quaternion component order."""

    WXYZ = "wxyz"
    XYZW = "xyzw"


#: Canonical pipeline order. ``state_adapter`` always returns in this order and
#: contracts/terms assume it. (Genesis and IsaacLab/Newton internal state are both wxyz.)
#: A term needing a different order (e.g. a specific obs form) must convert explicitly.
CANONICAL_QUAT = QuatOrder.WXYZ


# ---------------------------------------------------------------------------
# Coordinate frames
# ---------------------------------------------------------------------------
class Frame(str, Enum):
    """Reference frame a quantity is expressed in."""

    #: z-up world frame. Reference for absolute pose/velocity.
    WORLD = "world"
    #: Frame relative to the robot torso. Expressing hand/fingertip/object
    #: relative quantities here makes them invariant to robot motion. The actual
    #: origin body/offset is defined by the robot component
    #: (:class:`sim.metalab.contract.spec.RobotSpec` ``chest_origin_body`` / ``chest_origin_offset``).
    CHEST_ORIGIN = "chest_origin"


#: World up axis.
UP_AXIS = "z"

#: Standard gravity [m/s^2], world frame. Used unless a contract overrides it.
GRAVITY = (0.0, 0.0, -9.81)
