"""EnvSpec — engine-agnostic environment contract schema.

Role split:
- **Data parts** (physics/robot/object/fixture/camera) are Pydantic-validated
  (robot/object from YAML components, the rest from the task .py). Invalid/unknown values fail loud.
- **Logic parts** (obs/reward term, action) reference callables — functions, not values,
  so they aren't Pydantic-validated; the contract .py binds them directly.

A contract (``sim.metalab.contract.tasks.<task>``) composes these into an :class:`EnvSpec`.
**This file never imports an engine (gs/isaaclab).** Conventions: :mod:`sim.metalab.conventions`.
"""
from __future__ import annotations

from collections.abc import Callable
import math
import os
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sim.metalab.conventions import GRAVITY

# Type aliases — fixed-length tuples let Pydantic validate length too (fail-loud).
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z) — CANONICAL_QUAT


def rpy_deg_to_quat(rpy: Vec3) -> Quat:
    """(roll, pitch, yaw) in DEGREES → wxyz quaternion (CANONICAL_QUAT). Rotation
    R = Rz(yaw)·Ry(pitch)·Rx(roll) (intrinsic Z-Y-X = extrinsic X-Y-Z), the standard robotics RPY."""
    r, p, y = (math.radians(a) for a in rpy)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    return (cr * cp * cy + sr * sp * sy,   # w
            sr * cp * cy - cr * sp * sy,   # x
            cr * sp * cy + sr * cp * sy,   # y
            cr * cp * sy - sr * sp * cy)   # z


class _Data(BaseModel):
    """Data-part base: reject unknown fields (no silently-passed typos/bad knobs)."""

    model_config = ConfigDict(extra="forbid")


class _Logic(BaseModel):
    """Logic-part base: holds callables, so allow arbitrary types + reject unknown fields."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Data parts (YAML components / task-.py dicts → Pydantic)
# ---------------------------------------------------------------------------
class PhysicsSpec(_Data):
    """Physics integration/solver/default contact params. Per-engine parity knobs go in
    ``EnvSpec.overrides``; only engine-agnostic defaults live here."""

    hz: float = Field(gt=0.0)                       # control step frequency [Hz] (contract input). dt = 1/hz.
    substeps: int = Field(default=1, ge=1)          # physics substeps per control step
    decimation: int = Field(default=1, ge=1)        # physics steps per policy step
    gravity: Vec3 = GRAVITY
    solver_iterations: int = Field(default=100, ge=1)
    friction: float = Field(default=1.0, ge=0.0)    # default friction (parts may override)
    restitution: float = Field(default=0.0, ge=0.0)
    # Robot self-collision (robot↔robot). On by default (engine auto-excludes adjacent pairs,
    # keeps non-adjacent). For speed, drop collision geoms of unused bodies (left arm/head) via
    # RobotSpec.collision mask instead of disabling this (per body OR per collision mesh).
    self_collision: bool = True

    @property
    def dt(self) -> float:
        """control step dt [s] = 1/hz. Downstream (parser, env_driver) uses dt."""
        return 1.0 / self.hz


#: Override value: a float (overwrite) or "default" (keep XML value).
Override = float | Literal["default"]


class JointOverrideSpec(_Data):
    """MJCF-value override for an active joint. A float overwrites; ``"default"`` (or omitted) keeps XML.

    MJCF is the source of truth for most (equality/armature), so only set what you want to
    change in training (mainly kp/kv). ``effort`` = per-joint actuator torque limit [Nm] → dof force range;
    Genesis does not import the MJCF actuatorfrcrange, so set it here. Scalar ``e`` = symmetric (-e, +e);
    ``[lower, upper]`` = asymmetric (e.g. different bend vs extend limits).

    NOT here, deliberately: **frictionloss**. It is Coulomb (dry) friction — a PHYSICAL property of the
    joint that is always in effect, in either control mode, and both engines solve it as a constraint (an
    ``efc`` row, set-valued: in stiction it takes whatever value within ±frictionloss holds the joint).
    So it is read from the MJCF ONLY and is not a per-task knob. Listing it here fails loud (extra=forbid)
    rather than silently applying on one engine, which is what used to happen (genesis only)."""

    kp: Override = "default"
    kv: Override = "default"
    armature: Override = "default"
    effort: float | list[float] | Literal["default"] = "default"   # [Nm]: scalar=±e, [lo,hi]=asymmetric

    @field_validator("effort")
    @classmethod
    def _check_effort(cls, v):
        if isinstance(v, list):
            assert len(v) == 2 and v[0] <= v[1], f"effort [lower, upper] needs lower<=upper — got {v}"
        return v


class GravCompSpec(_Data):
    """Gravity-compensation contract — mirrors the real robot's hardware so sim2real matches.

    The real robot holds these joints against gravity so they do not sag:
    - ``actuator_joints`` — motor-supplied gravcomp. The compensating torque is part of the
      **measured joint torque** (the real torque sensor reports it), so sim folds the per-joint
      gravcomp torque ``g_j(q)`` into the ``joint_torque`` readout for these joints.
    - ``passive_joints`` — mechanical-spring gravcomp (e.g. waist/neck pitch). External to the
      motor, so it is **not** in the joint-torque readout.

    Physics is identical for both channels: the backend applies a constant anti-gravity body force
    (``m_i * g`` up at each link's COM) to every listed joint's child link, making that link
    weightless so the joint does not sag. The actuator/passive split only decides torque-readout
    folding. Hands/fingers are intentionally omitted — their weight loads the wrist and is held by
    PD, matching the real arm gravcomp model (which does not include the hand payload)."""

    actuator_joints: list[str] = Field(default_factory=list)
    passive_joints: list[str] = Field(default_factory=list)
    # Extra link (leaf body) names to also make weightless for the PASSIVE channel. A joint is fully
    # gravity-compensated only when ALL mass distal to it is weightless; each listed joint contributes just
    # its own child link, so the trunk/neck/head segments ABOVE the passive pitch joints (which are children
    # of non-gravcomp joints) must be named here or the pitch still sags under them. Arms are already covered
    # by actuator_joints; hands are intentionally left loaded (held by wrist PD). Matched by leaf name.
    passive_bodies: list[str] = Field(default_factory=list)

    def joints(self) -> list[str]:
        """All gravcomp joints (actuator + passive), in declaration order."""
        return [*self.actuator_joints, *self.passive_joints]


class MotorCouplingSpec(_Data):
    """Motor-to-joint coupled PD contract — one coupled group (robot HW fact).

    The group's joints are driven by PD in MOTOR space (firmware J2M map, τ_q = Gᵀ·τ_m) instead of the
    native diagonal PD, on newton the backend replaces it at physics-substep rate; genesis keeps native
    PD. ``METALAB_MOTOR_COUPLING=0`` disables at build time. Three structural ``kind``s (→ which
    kernel/loader the backend dispatches to):

    - **hand**     — 3-DOF, lower-triangular (finger ABAD/MCP/PIP, thumb Yaw/CMC/MCP). Gains keyed directly.
    - **arm**      — 2-DOF, FULL 2×2 (wrist Roll/Pitch ballscrew; elbow+wristYaw differential pulley).
                     Gains SLICE the 7-DOF arm group (``arm_slice``: elbow (3, 5), wrist (5, 7)).
    - **shoulder** — 3-DOF, DIAGONAL per-joint scalar ratio (shoulder Pitch/Roll/Yaw) — a degenerate
                     lower-triangular map driven by the hand kernel. Gains SLICE the arm group ((0, 3)).

    Not authored in YAML — :meth:`RobotSpec.coupled_groups` derives these from ``control_mode``."""

    kind: Literal["hand", "arm", "shoulder"] = "hand"
    params_key: str          # motor gains key in robot_model.json ("index_r"…"little_l" / "arm_r"/"arm_l")
    joints: list[str]        # active joints in fit-variable order: 3 (hand/shoulder) or 2 (arm)
    model_file: Optional[str] = None            # transmission json (None → finger.json)
    arm_slice: Optional[tuple[int, int]] = None  # arm/shoulder kinds: [start, end) into the 7-DOF arm gains

    @model_validator(mode="after")
    def _shape(self) -> "MotorCouplingSpec":
        need = 2 if self.kind == "arm" else 3
        assert len(self.joints) == need, f"{self.kind} group needs {need} joints — got {len(self.joints)}"
        assert (self.arm_slice is not None) == (self.kind != "hand"), \
            f"{self.params_key}: arm_slice must be set iff the gains slice the 7-DOF arm (kind != 'hand')"
        return self


# Canonical coupling groups for ``control_mode: motor``. Every group keys its own motor gains in
# robot_model.json by ``<part>_<hand>``. All four fingers on both hands share one transmission
# (finger.json); both thumbs share another (thumb.json) — the thumb is a different mechanism (Yaw
# decoupled, higher-order CMC/MCP) so it can't reuse the finger map. Fit-variable order per part:
# fingers ABAD/MCP/PIP, thumb Yaw/CMC/MCP (the thumb IP joint is an equality follower, not coupled).
_FINGER_NAMES = ("Index", "Middle", "Ring", "Little")
_FINGER_ROLES = ("ABAD", "MCP", "PIP")
_THUMB_ROLES = ("Yaw", "CMC", "MCP")
_THUMB_MODEL = "thumb.json"                       # basename → resolved under mj_mapping/ by load_hand_group
# Arm-sliced groups: gains slice the arm_{r,l} 7-DOF group at the motor indices.
# - shoulder Pitch/Roll/Yaw (per-joint scalar ratio, DIAGONAL 3-DOF map shoulder.json) = motors 0-2.
# - elbow+wristYaw (differential pulley, CONSTANT 2×2 map elbow.json) = motors 3,4.
# - wrist Roll+Pitch (ballscrew, nonlinear FULL 2×2 map wrist.json) = motors 5,6.
# Both arms share each map. Fit-variable order per map: shoulder [Pitch, Roll, Yaw],
# elbow [Elbow, Yaw], wrist [Roll, Pitch].
_SHOULDER_ROLES = ("Pitch", "Roll", "Yaw")
_SHOULDER_MODEL = "shoulder.json"
_ARM_SHOULDER_SLICE = (0, 3)
_ELBOW_MODEL = "elbow.json"
_ARM_ELBOW_SLICE = (3, 5)
_WRIST_ROLES = ("Roll", "Pitch")
_WRIST_MODEL = "wrist.json"
_ARM_WRIST_SLICE = (5, 7)
_HANDS = (("R", "r"), ("L", "l"))


class NailFrictionSpec(_Data):
    """A fixed contact friction for a named set of robot bodies (``RobotSpec.nail_friction``).

    The hand's NAIL shells are the case this exists for: a nail that grips is a grasp the real hand cannot
    make, so it is pinned slick while the pad keeps the robot's own (randomized) friction. ``mu=0`` reaches
    the solver as ``FRICTION_EPS`` — the contact mixing is a geometric mean
    (``sim.metalab.runtime.physics.friction``), so one slick side is enough to make the pair slick, which is
    what makes pinning ONE side meaningful at all."""

    mu: float = Field(ge=0.0)                    # absolute friction of those shapes
    bodies: list[str] = Field(min_length=1)      # MJCF body names


class RobotSpec(_Data):
    """Robot part. **Loads MJCF (source of truth) as-is** and layers only hub declarations on top:
    (a) joint active mask, (b) policy action groups, (c) frame bodies, (d) init pose, (e) joint_mode_param,
    (f) gravity compensation, (g) hand control mode (joint vs motor-coupled PD). Physics values
    (gains/armature/equality/limits) live in the MJCF, not here."""

    asset: dict[str, str] = Field(min_length=1)     # {"mjcf": "..."} — MJCF path (source of truth)
    # Base placement is task-owned (EnvSpec.base_pos/base_quat) — the robot yaml no longer sets it.
    # The loader overlays the task's values; base_pos is required there (fail-loud if unset).
    base_pos: Optional[Vec3] = None
    base_quat: Quat = (1.0, 0.0, 0.0, 0.0)          # wxyz (task overrides; identity default)
    fixed_base: bool = True
    # Initial joint pose [rad] — task-owned like the base placement, so it is authored in the task's robot
    # block (in DEGREES) and converted here by the loader. Omitted joint = 0. Applied right after model load;
    # mask-0 joints listed here are WELDED at that angle by mjcf_prep instead of at 0.
    init_pose: dict[str, float] = Field(default_factory=dict)
    # Full joint active mask: 1=active, 0=0°FIXED (parser strips <joint>+actuator+equality). List every joint.
    joints: dict[str, int] = Field(min_length=1)
    # Collision mask (optional): body name OR collision-MESH name → 1 (keep) / 0 (remove). Unlisted = 1 (keep).
    # A body key covers all of its class="collision" geoms; a mesh key covers only the geoms using that mesh and
    # WINS over the body key, so a kept body can still drop individual shells (`R_Palm_Link: 1` +
    # `R_Palm_Back_Collision1: 0`). The ALLEX geoms carry no `name`, so the mesh name (1:1 with the .stl) is the
    # per-geom handle. A mesh key hits every body using that mesh, and 17/69 ALLEX collision meshes are shared
    # (Finger_*x8 both hands, arm/wrist x2 L+R) — use the BODY key for one-sided control; the 52 1:1 meshes
    # (R_Palm_*, R_Thumb_*, torso, head) are safe individually. Dropped geoms leave self-collision and save
    # collision-mesh load; visual geoms are never touched (render unchanged). Same on both engines = parity.
    collision: dict[str, int] = Field(default_factory=dict)
    # Action groups the policy controls (active joint names) — contract references these to build EnvSpec.action.
    action_groups: dict[str, list[str]] = Field(min_length=1)
    # READ-ONLY joint groups: named joint sets a task references for SENSING, never for actuation. They join
    # the same "@joints.*" vocabulary as the action groups but are NOT selectable as an action group, so a
    # contract can observe a joint it does not command. That gap is real HW: the finger ABADs have an
    # actuator (kp/effort in joint_mode_param) and are PD-held at 0 rather than policy-driven, yet their
    # torque still reports what the object pushes back with. A name that collides with an action group (or
    # the derived "ctrl") fails loud in the loader rather than shadowing it.
    joint_groups: dict[str, list[str]] = Field(default_factory=dict)
    # Frames = MJCF body names (e.g. chest_origin, palm). Native bodies, so no offset.
    frames: dict[str, str] = Field(default_factory=dict)
    # Fingertip bodies (MJCF names), in hand order — a HW fact of this robot, like `frames`, not a task knob:
    # "which bodies touch the object when the hand grasps" cannot differ per task. Read by GateSpec's grip
    # requirement (env_driver): a contract's GATE states how many fingertips must grip, and names which ones
    # out of THIS list when the task cares.
    fingertips: list[str] = Field(default_factory=list)
    # The joint that CLOSES each fingertip, same order as `fingertips` — the finger's PIP (the thumb's MCP).
    # A grip condition reads its torque: pressing into the object shows up as a sustained FLEXING torque,
    # which is what separates a wrapped grip from a fingertip resting on the object in some other pose.
    # Empty = the condition degrades to contact-only (a hand that does not declare its flexing joints).
    fingertip_flex_joints: list[str] = Field(default_factory=list)
    # Bodies whose collision shapes carry a FIXED friction instead of the robot's: written once at build and
    # left out of the "robot" set the friction DR writes, so a reset cannot overwrite them. Named by BODY
    # because a shape has no name of its own in either engine — which is exact here, the nail shell and the
    # pad shell sit on different bodies (``*_Distal_Link`` vs ``*_Fingertip``), and it is also the finest
    # granularity genesis can address.
    nail_friction: Optional["NailFrictionSpec"] = None

    @model_validator(mode="after")
    def _fingertip_lists_align(self) -> "RobotSpec":
        assert not self.fingertip_flex_joints or len(self.fingertip_flex_joints) == len(self.fingertips), (
            f"fingertip_flex_joints ({len(self.fingertip_flex_joints)}) must be one per fingertip "
            f"({len(self.fingertips)}), in the same order")
        return self
    # Overrides on top of MJCF (active joints only). Omitted = XML as-is.
    # Per-joint drive parameters: kp / kv / armature / effort(torque limit). Named for what it
    # IS rather than "overrides" — the task's ``overrides`` means per-engine corrections, a different thing.
    # Only list what differs from the MJCF; every value here is the joint-level drive contract.
    joint_mode_param: dict[str, JointOverrideSpec] = Field(default_factory=dict)
    # Gravity compensation (robot HW fact; applies to every task using this robot). None = off.
    gravcomp: Optional[GravCompSpec] = None
    # Control mode (robot HW fact): "joint" = native diagonal PD on every joint (default); "motor" = PD
    # solved in MOTOR space through the firmware transmission (J2M map + analytic Jacobian G:
    # tau_q = G^T . tau_m, single clamp on the motor-space sum), then mapped back to joints. Covers thumb +
    # 4 fingers + shoulder + elbow/wristYaw + wrist Roll/Pitch = the whole active hand and arm; see
    # `coupled_groups` for the exact groups (allex_right: 8 groups / 22 joints). Implemented on BOTH engines
    # (newton warp kernel, genesis torch mirror against the same oracle — sim/metalab/tests/
    # test_motor_coupling.py pins the parity). METALAB_MOTOR_COUPLING=0 forces "joint" at build time.
    control_mode: Literal["joint", "motor"] = "joint"
    # NOTE: init_pose is owned by the **task (EnvSpec.init_pose)**, not the robot — it varies per task.

    def active_joints(self) -> set[str]:
        return {n for n, v in self.joints.items() if v == 1}

    def coupled_groups(self) -> list[MotorCouplingSpec]:
        """Groups to drive with motor-space coupled PD — empty unless ``control_mode`` is "motor".
        Per hand: **hand** groups (thumb + 4 fingers, 3-DOF, thumb.json/finger.json), then the
        **shoulder** group (diagonal 3-DOF, shoulder.json), then the **arm** groups (elbow+wristYaw
        constant 2×2 elbow.json, then wrist Roll/Pitch full 2×2 wrist.json) — together the full 7-DOF
        arm. Each group is emitted only if all its joints are active (so right-only contracts yield
        just the R_ groups). Order per hand = thumb, index, middle, ring, little, shoulder, elbow,
        wrist."""
        if self.control_mode != "motor":
            return []
        active = self.active_joints()

        def group(kind: str, params_key: str, joints: list[str], model_file, arm_slice=None):
            need = 2 if kind == "arm" else 3
            n = sum(j in active for j in joints)
            assert n in (0, need), (   # partial-active = misconfig (can't couple a fixed-DOF transmission)
                f"control_mode=motor: {params_key} has {n}/{need} joints active — need all or none")
            return MotorCouplingSpec(kind=kind, params_key=params_key, joints=joints,
                                     model_file=model_file, arm_slice=arm_slice) if n == need else None

        groups: list[Optional[MotorCouplingSpec]] = []
        for side, hand in _HANDS:
            groups.append(group("hand", f"thumb_{hand}",
                                 [f"{side}_Thumb_{r}_Joint" for r in _THUMB_ROLES], _THUMB_MODEL))
            for name in _FINGER_NAMES:
                groups.append(group("hand", f"{name.lower()}_{hand}",
                                    [f"{side}_{name}_{r}_Joint" for r in _FINGER_ROLES], None))
            groups.append(group("shoulder", f"arm_{hand}",
                                 [f"{side}_Shoulder_{r}_Joint" for r in _SHOULDER_ROLES], _SHOULDER_MODEL,
                                 _ARM_SHOULDER_SLICE))
            groups.append(group("arm", f"arm_{hand}",
                                 [f"{side}_Elbow_Joint", f"{side}_Wrist_Yaw_Joint"], _ELBOW_MODEL,
                                 _ARM_ELBOW_SLICE))
            groups.append(group("arm", f"arm_{hand}",
                                 [f"{side}_Wrist_{r}_Joint" for r in _WRIST_ROLES], _WRIST_MODEL, _ARM_WRIST_SLICE))
        return [g for g in groups if g is not None]

    @model_validator(mode="after")
    def _check(self) -> "RobotSpec":
        assert "mjcf" in self.asset, "robot.asset.mjcf required (MJCF is source of truth)"
        bad = {n: v for n, v in self.joints.items() if v not in (0, 1)}
        assert not bad, f"joints mask allows only 0/1 — {bad}"
        badc = {n: v for n, v in self.collision.items() if v not in (0, 1)}
        assert not badc, f"collision mask allows only 0/1 — {badc}"
        active = self.active_joints()
        for g, names in self.action_groups.items():
            unknown = set(names) - active
            assert not unknown, f"action_groups[{g}] has inactive/unknown joints: {sorted(unknown)}"
        unknown = set(self.joint_mode_param) - active
        assert not unknown, f"joint_mode_param has inactive/unknown joints: {sorted(unknown)}"
        if self.gravcomp is not None:
            gc_unknown = set(self.gravcomp.joints()) - active
            assert not gc_unknown, f"gravcomp has inactive/unknown joints (check robot mask): {sorted(gc_unknown)}"
            dup = set(self.gravcomp.actuator_joints) & set(self.gravcomp.passive_joints)
            assert not dup, f"joint listed in both actuator & passive gravcomp: {sorted(dup)}"
        self.coupled_groups()   # control_mode=motor: fail loud on any partial-active finger
        return self

    def motor_coupling_on(self) -> bool:
        """Is motor-space coupled PD actually active for THIS run? ``control_mode`` AND the build toggle.

        ONE place for the pair, because they are read by the loader's report, the genesis parser's
        force-range decision and the backends' owner construction — three call sites that must never
        disagree about whether the coupled kernel owns a joint."""
        return bool(self.coupled_groups()) and os.environ.get("METALAB_MOTOR_COUPLING", "1") != "0"

    def effort_ignored_joints(self) -> list[str]:
        """Joints whose ``joint_mode_param.effort`` this run will NOT apply, sorted (empty in joint mode).

        A MOTOR-COUPLED joint's only torque clamp is in motor space (τ_m to ``envelope(φ̇) ∩ ±rated`` inside
        the coupled-PD kernel). The joint-space bound that implies — ``Gᵀ·(envelope ∩ rated)`` — moves with
        pose and speed, so no scalar could stand in for it (it is a readout: standalone's "Joint Torque
        Limit"). So ``effort`` is inert there on BOTH engines: newton's coupled τ rides ``qfrc_applied``,
        which ``jnt_actfrcrange`` never sees, and genesis opens ``force_range`` to ±inf for exactly these
        joints (``genesis/parser._open_coupled_force_range``).

        A QUERY, not a check — the loader reports it once, because these values are not dead in general: under
        ``control_mode="joint"`` native PD makes them the real actuator limit, and there they are not
        redundant with the MJCF either, adding per-direction asymmetry its symmetric ``actuatorfrcrange``
        cannot express (R_Index_MCP: [-2, 5] here vs ±5 in the XML)."""
        if not self.motor_coupling_on():          # joint mode / toggle off → every effort is live
            return []
        coupled = {j for g in self.coupled_groups() for j in g.joints}
        return sorted(coupled & {j for j, ov in self.joint_mode_param.items() if ov.effort != "default"})


Shape = Literal["box", "cylinder", "sphere", "capsule"]


class ObjectPartSpec(_Data):
    """One procedural primitive of an object, posed RELATIVE to the object's own frame.

    An object is spawned either from an asset (``ObjectSpec.asset``) or from ``parts`` like this one — the
    procedural path exists so a contract can state a shape's dimensions directly (a table's size is a task
    knob; burying it in an MJCF would mean editing an asset to change the layout)."""

    shape: Shape
    size: Vec3                                      # box: full extents; cylinder/capsule: (radius, length, _)
    pos: Vec3 = (0.0, 0.0, 0.0)                     # relative to the object's init_pos
    quat: Quat = (1.0, 0.0, 0.0, 0.0)               # wxyz, relative to the object's init_quat


class ObjectSpec(_Data):
    """Object part. Spawned as either a procedural composite (parts) or an asset (asset)."""

    name: str
    mass: float = Field(gt=0.0)
    # No friction/restitution here on purpose: contact friction is MuJoCo's own default at build (nothing
    # authored in the object MJCFs either) and the reset DR event ``set_shape_friction`` owns it at runtime,
    # writing an absolute mu per env. A contract that sets "friction" now fails loud instead of being ignored.
    asset: Optional[dict[str, str | list[str]]] = None  # set for asset spawn (usd may be a variant list)
    parts: list[ObjectPartSpec] = Field(default_factory=list)  # set for procedural composite (see ObjectPartSpec)
    # Welded to the world: no free joint, never moves, and NOT the "object" the RL terms read (obs/reward and
    # the DR selector "object" mean the first NON-fixed object). This is how scenery with task-owned
    # dimensions — a table — lives in `objects` instead of a separate fixture list.
    fixed: bool = False
    variants: int = Field(default=1, ge=1)          # per-env round-robin variant count
    init_pos: Optional[Vec3] = None                 # spawn position (parser default if None)
    init_quat: Quat = (1.0, 0.0, 0.0, 0.0)          # wxyz — canonical orientation (parser reads this)
    init_rpy: Optional[Vec3] = None                 # (roll, pitch, yaw) DEGREES; if set → init_quat (either/or)
    randomize: dict[str, Any] = Field(default_factory=dict)    # per-reset randomization spec
    # False = GHOST spawn: the asset's class="collision" geoms are stripped, so the object still renders and
    # still falls under gravity but touches nothing. Contact-dependent DR (object friction, object scale) then
    # has no surface to write, so a contract that wants a ghost drops those events too.
    collision: bool = True

    @model_validator(mode="after")
    def _ghost(self) -> "ObjectSpec":
        assert self.collision or self.asset, \
            f"object '{self.name}': collision=False is an MJCF transform — procedural `parts` cannot be a ghost"
        return self

    @model_validator(mode="after")
    def _spawn_source(self) -> "ObjectSpec":
        assert bool(self.asset) != bool(self.parts), \
            f"object '{self.name}': set exactly one of asset / parts (got asset={bool(self.asset)}, parts={bool(self.parts)})"
        return self

    @model_validator(mode="after")
    def _orient(self) -> "ObjectSpec":
        if self.init_rpy is not None:               # rpy convenience → fold into the canonical init_quat
            assert self.init_quat == (1.0, 0.0, 0.0, 0.0), \
                f"object '{self.name}': set either init_quat or init_rpy, not both"
            self.init_quat = rpy_deg_to_quat(self.init_rpy)
            self.init_rpy = None                    # folded → clear so re-validation stays idempotent
        return self


class SceneSpec(_Data):
    """The stage the task is set on — what exists regardless of the robot, the objects, or the props.

    Today that is the ground: ``ground=True`` adds the engine's infinite ground plane (newton: a single
    world(-1) plane shared across envs; genesis: ``gs.morphs.Plane``). It is not a fixture — a fixture is a
    prop with a pose and a size that a task places, while the ground is the frame everything else is placed
    against, and every contract wants exactly one."""

    ground: bool = True                             # infinite ground plane at z=0
    ground_name: str = "plane"                      # selector name for contact/DR targeting


class FixtureSpec(_Data):
    """Fixed/kinematic structure (desk, floor, etc.)."""

    name: str
    kind: Literal["box"]                            # the ground is not a fixture — see SceneSpec.ground
    size: Optional[Vec3] = None                     # required for box
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)


class CameraSpec(_Data):
    """Off-screen camera pose for recording (engine- and physics-agnostic). Eval recording frames env-0
    from this pose. Resolution is set at record time, not in the contract (task owns only where/how to look)."""

    eye: Vec3                                       # camera position (world, env-0)
    lookat: Vec3                                    # point looked at (world)
    fov: float = Field(default=40.0, gt=0.0)        # vertical field of view [deg]


class GoalSpec(_Data):
    """**Fixed** goal pose + keypoint success criterion (no resampling). Referenced by reward
    (keypoint progress, reach bonus). keypoint_max_dist (max distance over the object/goal 4-point cage)
    fuses pos+rot error into a single distance."""

    pos: Vec3
    quat: Quat = (1.0, 0.0, 0.0, 0.0)                       # wxyz
    keypoint_half_extent: Vec3 = (0.05, 0.05, 0.08)         # keypoint cage half-size [m]
    # Near-goal keypoint_max_dist threshold [m] carried with the goal as scene metadata (the dashboard reads
    # it). What success is judged at is NOT this: the curriculum states the level's bar and GateSpec the
    # final one.
    goal_dist_tol: float = Field(default=0.05, gt=0.0)


class GateSpec(_Data):
    """The task's FINAL success bar — the definition ``val/SR`` is measured against.

    Authored in the contract as ``GATE`` (between EVENTS and TERMINATE) so "solved" has ONE explicit
    definition instead of being implied by whichever level the curriculum happens to have reached. The
    fields are hard requirements, ANDed, evaluated every policy step and latched for the episode::

        keypoint_max_dist(object, goal) <= goal_dist_tol
        AND ||palm_pos - grasp_point|| <= palm_distance          (position only, orientation ignored)
        for `hold_steps` steps, counted per `hold_mode` (consecutive = a hold; cumulative = a total)
        while >= `contact_count` fingertips press the object with > `force_threshold` [N] on their PAD
        AND every fingertip named in `contact_fingers` is pressing it that way

    The PREDICATE that evaluates all of that is named here too (``predicate = gate.object_at_goal``) rather
    than hardwired into the driver: the driver may not know what "solved" means, only how to count and latch
    it. A task whose success has a different shape declares a different one, and the reward term that pays for
    success calls the very same function at the curriculum's bar.

    ``lift_height`` is deliberately NOT one of those conditions: at the gate's own tolerance, being at the
    goal already implies being off the table (on hammer_lift by 13.5x), so testing it would add nothing. It
    still belongs here because it is the task's "off the table" definition — see the field.

    WHICH bodies count as fingertips is not stated here — it is a property of the hand, so it comes from
    ``RobotSpec.fingertips``. What the gate states is the task decision on top of that list: HOW MANY of them
    must be gripping (``contact_count``) and, when the grasp is built around particular fingers, WHICH ones
    (``contact_fingers``, named out of that same list).

    The curriculum moves the LEARNING bar (reward + terminate share ``goal_dist_tol``) but never
    touches this one, so the metric means the same thing at iteration 0 and at the end — and it matches
    what an eval run judges (``--curriculum_end``), so the training curve and the eval SR are the same
    quantity. Needs a ``goal`` for the keypoint reference; the loader fails loud if either is missing
    while the other is present, or if a grip is required and the robot declares no fingertips."""

    # [m] rise above the SPAWN height that counts as "lifted" — the task's PHASE BOUNDARY, not one of the
    # success conditions above. The driver latches it per episode and publishes it as the ``lifted`` gate that
    # the reward / termination / event / obs terms read; on the learning side it is load-bearing, because the
    # curriculum's loose early tolerance is reachable without leaving the table at all.
    # 0 = no lift phase at all: ``lifted`` then stays False for the whole episode, which SILENTLY disables
    # every term gated on it (approach/progress/reach-bonus/grasp-lost/pull-force). Set it, or drop those.
    lift_height: float = Field(default=0.0, ge=0.0)
    predicate: Callable                                       # sim.metalab.terms.gate — fn(env, **knobs) -> (N,) bool
    goal_dist_tol: float = Field(gt=0.0)                  # [m] keypoint_max_dist to the goal
    hold_steps: int = Field(default=1, ge=1)                  # steps inside the tolerance (see hold_mode)
    # HOW those steps are counted. Both the GATE and the reach reward obey it, so "solved" means one thing.
    #   "consecutive" — the counter resets to 0 the step the conditions break. A HOLD: the object has to
    #                   stay put, in grip, for hold_steps in a row.
    #   "cumulative"  — the counter only ever grows: hold_steps qualifying steps ANYWHERE in the episode,
    #                   in any number of pieces. Strictly easier (a consecutive run of N is also N
    #                   cumulative steps), so the same hold_steps is a LOWER bar.
    # The trade-off is what "cumulative" stops measuring: a grip that touches the goal for one step, drops,
    # and comes back 20 times over an episode scores the same as one that held for 20 straight. Contact
    # FLICKER is the failure mode this used to catch, so prefer "consecutive" for the final bar and reach
    # for "cumulative" when the sparse success signal is too rare for the policy to find at all.
    hold_mode: Literal["consecutive", "cumulative"] = "consecutive"
    contact_count: int = Field(default=0, ge=0)               # fingertips that must grip (0 = grip not required)
    # WHICH fingertips must EACH be gripping (``RobotSpec.fingertips`` names; empty = not required), where
    # contact_count says only how many. ORDERED: the curriculum demands a growing PREFIX of this list, so the
    # first entry is the tip required from the lowest level on.
    contact_fingers: list[str] = Field(default_factory=list)
    force_threshold: float = Field(default=0.1, gt=0.0)       # [N] per-fingertip pad-normal "is gripping" bar
    # [m] max distance from the palm to the object's grasp point, POSITION only (a delivered-but-released
    # object must not count as solved: the hand has to still be there). 0 = not required. WHICH body is the
    # palm comes from the robot's `frames.palm` (a hand property). WHERE on the object the hand belongs is
    # not stated at all: the asset pipeline puts the body origin ON the grasp point, so every measurement
    # here already refers to it (sim/metalab/assets/tools/usd_to_mjcf.py).
    palm_distance: float = Field(default=0.0, ge=0.0)
    # [rad] how exactly the robot must hold the target POSTURE — a JOINT-SPACE condition, unlike every
    # other field here. Bounds the WORST joint (max, not mean: a mean lets one finger sit wide open as long as
    # the others compensate, which is not the shape that was asked for). 0 = not tested.
    joint_pose_tolerance: float = Field(default=0.0, ge=0.0)
    # {joint: angle [rad]} — the posture itself, stated ONCE per contract and in EITHER place: on the reward
    # term that shapes toward it (its ``joint_pose`` knob), which the loader copies in so the gate and the
    # dense signal cannot ask for different postures — or here, when no reward term shapes toward it at all.
    # Declaring both fails loud. Empty unless ``joint_pose_tolerance`` is set. Any COMMANDED joint may appear,
    # not just the hand's — a wrist angle is as much part of the final posture as a finger's.
    joint_final_pose: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Logic parts (callable refs; not value-validated)
# ---------------------------------------------------------------------------
class ObsNoise(_Logic):
    """Sensor noise for ONE obs term, declared in that term's OWN physical units.

    Applied at group assembly, **before** ``ObsTerm.scale`` (so the number here is in the unit the term
    emits, not in post-scale units) and only for the obs groups listed in ``TaskSpec.obs_noise_groups``
    — a term shared by the actor and the critic is therefore corrupted on the actor side only.

    Exactly one of the two forms:

    * ``std`` — i.i.d. Gaussian over every dim of the term. Unit = the term's unit: rad for a joint-position
      term, N·m for a torque term, m for a plain position term, action units for ``prev_action_targets``.
    * ``pos`` / ``rot`` — POSE layout: the term must be a multiple of 7 wide, each block ``pos3 + quat4``.
      ``pos`` [m] is added to the translation. ``rot`` [rad] is the PER-AXIS angular std: a quaternion cannot
      take per-component additive noise (it would leave the unit sphere), so the quat gets isotropic noise of
      std ``rot/2`` in R^4 and is renormalized — for small angles that is a rotation whose 3 axis components
      are each ~N(0, rot²), and the radial part is projected back out by the renormalization. Both knobs are
      therefore per-axis, exactly like IsaacLab's ``GaussianNoiseCfg``, so the TOTAL magnitude has rms
      ``knob * sqrt(3)`` (measured: rot=0.02 → total angle rms 0.0342 rad vs 0.0346 predicted). The quat form
      is deliberately **order-agnostic** — isotropic noise on all 4 components does not care which slot holds
      w (every pose term emits wxyz, but this knob would hold either way).

    ``0.0`` is a valid value and is skipped at runtime (no RNG draw), so a term can carry a visible,
    tunable-but-currently-off knob.
    """

    std: Optional[float] = Field(default=None, ge=0.0)
    pos: Optional[float] = Field(default=None, ge=0.0)
    rot: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _one_form(self):
        assert self.std is not None or self.pos is not None or self.rot is not None, \
            "ObsNoise: set std=, or pos=/rot= for a pose-layout term"
        assert self.std is None or (self.pos is None and self.rot is None), \
            "ObsNoise: std= is for a single-unit term; pos=/rot= is the pos3+quat4 form — do not mix"
        return self


class ObsTerm(_Logic):
    """Observation term = a FLAT function plus its named knobs: the driver calls ``fn(env, **params)`` once
    per obs GROUP per policy step, over the state-adapter interface (engine-agnostic).

    Same shape as :class:`RewardTerm`, deliberately — ``params`` is the contract's ``kwargs={...}`` with
    ``"@refs"`` resolved, and it stays MUTABLE at runtime, so a knob can be retuned live exactly the way the
    curriculum retunes a reward term. (The predecessor called the term as a FACTORY at load and kept only the
    closure it returned, which froze every knob, erased the function's name from tracebacks, and left the
    loader guessing ``dim_labels`` from the positional args.) See sim/metalab/terms/obs/common.py."""

    name: str
    fn: Callable
    params: dict[str, Any] = Field(default_factory=dict)
    scale: float = 1.0
    dim_labels: list[str] = []   # per-dim names (a joint/body list knob) for the telemetry dashboard; [] = index fallback
    noise: Optional[ObsNoise] = None   # sensor noise, term units, applied pre-scale in noisy groups only
    # Dashboard display only (never physics) — the RL tab's plot tabs, matching drive/monitor.Channel.
    unit: str = ""               # shown next to the tab title, in POST-scale units (e.g. "deg", "N*m")
    digits: int = Field(default=3, ge=0)   # decimals the dashboard prints for this term


class RewardTerm(_Logic):
    """Reward term = a FLAT function plus its named knobs: the driver calls ``fn(env, **params)`` every
    policy step and pays ``weight * value``.

    ``params`` is the contract's ``kwargs={...}`` with ``"@refs"`` resolved (reward terms take no positional
    args — a flat signature is only readable if every knob is named). It stays MUTABLE at runtime: the
    curriculum retunes a term by writing into this dict, which the next step picks up. Per-env episode state
    goes through ``EnvDriver.buffer`` (driver-owned, auto-reset), so a term needs no init/reset hook and
    stays a plain function. See sim/metalab/terms/reward/common.py."""

    name: str
    fn: Callable
    weight: float
    params: dict[str, Any] = Field(default_factory=dict)


class ActionDelaySpec(_Data):
    """Sim2Real command latency [policy steps] — ONE per-env lag ~ U[min,max], resampled at reset.

    Deliberately NOT a per-group knob: the delay is the controller->robot LINK, so every action group is
    written with the same lag. Split per group it could put an env's arm 2 steps behind its live hand, a
    combination no real robot has. Declared once at the ACTION block level and hoisted out of the group dict
    by ``TaskSpec._hoist_action_delay``.

    Delays the **post-EMA** value only (legacy DelayedEMA: the EMA recursion and the limit clamp keep the
    undelayed true target, and ``prev_action_targets`` still reports it — only the value written to sim lags).
    Default 0 = disabled (the legacy teacher has no delay either — "teacher clean").
    """

    min_delay: int = Field(default=0, ge=0)
    max_delay: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _delay_range(self) -> "ActionDelaySpec":
        assert self.min_delay <= self.max_delay, \
            f"invalid delay range: min_delay={self.min_delay} > max_delay={self.max_delay}"
        return self


class ActionGroupSpec(_Logic):
    """Action group (e.g. arm/hand): joints controlled, control mode, scale, EMA coeff.

    No delay here — that is one link for the whole robot, see :class:`ActionDelaySpec`."""

    joints: list[str] = Field(min_length=1)
    mode: Literal["position_to_limits", "position", "velocity", "torque"] = "position_to_limits"
    scale: float = 1.0
    # Command low-pass TIME CONSTANT [s] (None = no EMA). SECONDS, not a per-step coefficient: the driver
    # derives the coefficient from the contract's own policy rate, ``alpha = 1 - exp(-step_dt/ema_tau)``, so
    # the filter keeps its time-domain shape when ``physics.hz``/``decimation`` change — and a real robot
    # running at its own command rate reproduces it from this same number.
    ema_tau: Optional[float] = Field(default=None, gt=0.0)


class EventTerm(_Logic):
    """Event/DR term = a FLAT function plus its named knobs: the driver calls ``fn(env, env_ids, **params)``
    — env=EnvDriver (delegates backend read/write), env_ids=(K,) long. mode="reset" runs on the envs being
    reset, "interval" on all envs every step.

    Same shape as :class:`RewardTerm`, and for the same two reasons. ``params`` stays MUTABLE at runtime, so
    the CURRICULUM retunes a term by writing into this dict (``t.params["mu_scale"] = …``) — that is what
    used to force a tunable term to be a class with a mutable attribute. And per-env episode state is
    driver-owned (``env.buffer(key)``) rather than held inside the term, which is what used to force
    ``.init``/``.reset`` hooks."""

    name: str
    fn: Callable
    mode: Literal["reset", "interval"] = "reset"
    train_only: bool = False
    requires: tuple[str, ...] = ()      # engine capabilities; the driver drops the term without them
    params: dict[str, Any] = Field(default_factory=dict)


class TerminateTerm(_Logic):
    """Termination term = a FLAT function plus its named knobs: the driver calls ``fn(env, **params)`` every
    policy step and ORs the returned ``(N,) bool`` into ``done``.

    Same shape as :class:`RewardTerm`, and for the same reasons — ``params`` is the contract's knobs with
    ``"@refs"`` resolved and stays MUTABLE at runtime, so a bar like ``min_height`` could be ramped live the
    way the curriculum ramps a reward knob. See sim/metalab/terms/terminate/common.py.

    name = key for per-term done tracking/logging (Termination/<name>) + curriculum
    termination_rate(name)."""

    name: str
    fn: Callable
    params: dict[str, Any] = Field(default_factory=dict)
    truncation: bool = False   # True → this done bootstraps (time_outs), not a no-bootstrap terminal (success = truncation)


class CurriculumTerm(_Logic):
    """Curriculum term (resolved). ``fn(env) -> dict[str, float]`` — env=EnvDriver
    (common_step_counter, termination_rate, reward/event term access). Returned dict is logged as
    ``Curriculum/<key>``. Called at each done-reset boundary (gate lives inside the term)."""

    name: str
    fn: Callable


# ---------------------------------------------------------------------------
# Top-level contract
# ---------------------------------------------------------------------------
class EnvSpec(_Logic):
    """One task contract. An engine-agnostic environment definition composed from parts.

    A per-engine parser builds the scene from data parts, and runtime.env_driver runs each step via
    action/obs/reward/terminate/events/curriculum. command (dynamic goal) is fleshed out when porting
    perceptive (extract, not predict)."""

    name: str
    num_envs: int = Field(default=4096, ge=1)
    env_spacing: float = Field(default=1.5, gt=0.0)
    episode_length_s: float = Field(default=10.0, gt=0.0)  # episode length [s]; max_step = round(s / (dt*decimation))

    physics: PhysicsSpec
    scene: SceneSpec = Field(default_factory=SceneSpec)   # the stage (ground); see SceneSpec
    robot: RobotSpec
    robot_friction: Optional[float] = None   # build-time robot contact μ override (absolute); None=keep MJCF
    objects: list[ObjectSpec] = Field(default_factory=list)
    fixtures: list[FixtureSpec] = Field(default_factory=list)

    # --- contact params: per-group contact softness, ENGINE-AGNOSTIC -------------------------------------
    # Group key: "robot" | a fixture name ("table") | an object name ("hammer"). Per group, MuJoCo semantics:
    #   solref = (timeconst, dampratio)   timeconst is the one that bites — 0.01 is the legacy-validated
    #                                     stiff contact, MuJoCo's own default 0.02 is ~4x softer
    #   solimp = (dmin, dmax, width, midpoint, power)
    #   solmix = constraint mixing weight  <- NEWTON ONLY (genesis averages the two geoms 0.5/0.5 and has no
    #                                       solmix input, so declaring it makes the engines diverge)
    # NOT an override: both engines implement the same solref/solimp math (genesis stores them as geom
    # ``sol_params`` = solref ++ solimp and consumes them in ``imp_aref`` inside its contact constraint), so
    # this is a shared knob, not a per-engine correction. Omitted group -> asset/engine default. OBJECT geoms
    # are authored here too, not in the object MJCF, so an object's contact softness lives in exactly one file.
    contact_params: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def movable_objects(self) -> list[ObjectSpec]:
        """Objects that are actually simulated bodies (free joint) — the RL "object" is the first of these.

        ``objects`` also carries welded scenery (``fixed=True``, e.g. a table), so a bare ``if spec.objects``
        no longer answers "is there an object to observe/randomize"; ask this instead."""
        return [o for o in self.objects if not o.fixed]


    action: dict[str, ActionGroupSpec] = Field(default_factory=dict)   # scene-only contracts may leave empty (env_driver requires it)
    action_delay: ActionDelaySpec = Field(default_factory=ActionDelaySpec)   # ONE command lag for the robot
    obs: dict[str, list[ObsTerm]] = Field(default_factory=dict)        # group -> terms; filled just before training
    # Per obs-group frame-stack length (group -> H; 1 = no stacking). Resolved: every obs group is present.
    # env_driver stacks the last H policy-step frames (oldest-first) into each group's obs vector.
    obs_history_length: dict[str, int] = Field(default_factory=dict)
    # Obs groups whose terms get their ``ObsTerm.noise`` applied (MuJoCo/IsaacLab's per-group
    # `enable_corruption`). Groups NOT listed read the same terms clean, which is what keeps an asymmetric
    # critic privileged while the actor is corrupted. Noise stays on in eval/play by design.
    obs_noise_groups: list[str] = Field(default_factory=list)
    reward: list[RewardTerm] = Field(default_factory=list)
    terminate: list[TerminateTerm] = Field(default_factory=list)       # named — per-term done tracking/logging
    events: list[EventTerm] = Field(default_factory=list)              # DR/randomization (reset/interval)
    curriculum: list[CurriculumTerm] = Field(default_factory=list)     # difficulty progression (called at reset boundary)

    # command (dynamic goal) fleshed out when porting perceptive (placeholder for now).
    command: Optional[Any] = None

    camera: Optional[CameraSpec] = None    # off-screen camera pose for eval recording (absent → cannot record, fail-loud)
    goal: Optional["GoalSpec"] = None      # fixed goal pose (keypoint reward ref; absent → no goal reward)
    gate: Optional["GateSpec"] = None      # FINAL success bar — what val/SR measures (see GateSpec)
    #: Per-engine corrections (non-mergeable physics knobs). E.g. {"genesis": {...}, "newton": {...}}.
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    #: The TaskSpec's tuning knobs as plain values (contract/recipe.py), carried for the W&B run config —
    #: a Curr entry's knobs vanish into its curriculum fn at load, so they are unreadable off this spec.
    recipe: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _init_pose_keys_known(self) -> "EnvSpec":
        # init_pose may name ANY joint the robot knows: ACTIVE entries are the runtime spawn pose;
        # MASK-0 entries are consumed by mjcf_prep as the weld angle (posed-then-welded fixed joint,
        # e.g. hold the waist pitch at 45° on a right-only robot without paying for the dof).
        unknown = set(self.robot.init_pose) - set(self.robot.joints)
        assert not unknown, f"init_pose has unknown joints (check robot mask): {sorted(unknown)}"
        return self


# ---------------------------------------------------------------------------
# TaskSpec — the source schema of a task contract (declarative: names, tunables, term refs).
# A task is a **Python module** ``sim/metalab/contract/tasks/<task>.py`` that builds one ``TASK = TaskSpec(...)``.
# loader.load_task imports that module and resolves parts (robot/object) + term entries into an EnvSpec.
# A term entry lists an **imported symbol + its knobs** (the symbol is what enables Go-to-Definition):
# obs/reward are FLAT functions the driver calls as ``fn(env, **params)``, terminate/events/curriculum are
# factories called once at load to bind ``fn(env)``; derived values are pointed at via
# closed-vocabulary refs ("@joints.ctrl", "@frames.palm") — the ROBOT-owned vocabulary. Task-owned
# literals are passed as plain python values (the contract IS python; no indirection table).
# ---------------------------------------------------------------------------
class ActionCfg(_Data):
    """Action-group tunables, and optionally WHICH of the group's joints the policy commands.

    ``joints`` omitted → the whole ``robot.action_groups[<group>]`` (the robot yaml owns what is actuatable).
    Given → that subset, in that order: the policy's action vector for this group, so the task can freeze a
    joint at its init pose without editing the robot (a frozen joint keeps its position target and is still
    PD-held; it just stops being an action dim). Must be a subset of the robot group — a name outside it, a
    duplicate, or an empty list fails loud, because the action dim is what the policy checkpoint is shaped by.
    """

    joints: Optional[list[str]] = None
    scale: float = 1.0
    ema_tau: Optional[float] = Field(default=None, gt=0.0)   # [s] command low-pass time constant
    mode: Literal["position_to_limits", "position", "velocity", "torque"] = "position_to_limits"


class _Ref(_Logic):
    """Base for a contract's term entries — ONE type per category, so every field on it applies.

    A contract writes ``<term name> = <Category>(<fn symbol>, ...)``: the fn is the imported symbol (never a
    name string), which is what makes Go-to-Definition work and turns a typo into an *import* error, and the
    ATTRIBUTE NAME is the term's name (logging key, obs-group member, curriculum reference). Same shape as
    isaaclab's ``RewardsCfg``/``TerminationsCfg`` blocks. Literals pass through; ``"@cat.key"`` is resolved by
    the loader over the closed vocabulary (joints/frames/bodies/const)."""

    fn: Callable
    name: Optional[str] = None       # filled from the attribute name by `terms()`; explicit only in a list


class Rew(_Ref):
    """Reward entry — ``Rew(fn, weight=…, **knobs)``.

    The knobs go straight into ``RewardTerm.params`` and reach the flat reward function as
    ``fn(env, **params)``, so what the signature names is what the contract writes. ``weight`` is NOT a knob:
    the driver applies it (``weight * value``) and the CURRICULUM mutates it live to
    decay a dense term to 0, so it belongs to the entry, not to the function. Reward entries take no
    positional term args — a flat signature is readable only when every knob is named. ``weight`` is
    keyword-only and required, so the one number that decides how much a term is worth is never an
    unlabelled positional in a row of knobs.

    ``weight`` is the PER-STEP payout of a term returning its maximum — the driver applies it as-is, so the
    number in the contract is the reward the trainer actually sees on that step (a 5-tip grip on a
    ``weight=1.0`` contact term pays 1.0, one tip pays 0.2). A term paid EVERY step therefore totals
    ``weight * max_episode_length`` over a full episode while a ONE-SHOT term (a success bonus) totals
    ``weight``: to weigh the two against each other, scale the one-shot weight by the step count that
    ``episode_length_s``, ``physics.hz`` and ``decimation`` imply, and re-derive it if the horizon moves.
    The wandb ``Reward/<term>`` series divides the episode sum by that step count, so the LOG stays in
    per-step units whatever the horizon is."""

    weight: float
    params: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, fn: Callable, *, weight: float, name: str | None = None, **knobs: Any):
        super().__init__(fn=fn, weight=weight, name=name, params=knobs)


class Obs(_Ref):
    """Obs entry — ``Obs(fn, scale=…, noise=…, **knobs)``.

    The knobs go straight into ``ObsTerm.params`` and reach the flat obs function as ``fn(env, **params)``,
    so what the signature names is what the contract writes — the same rule as :class:`Rew`. Obs entries take
    no positional term args: `Obs(obs.object_pose, "@frames.chest_origin", [0, 0, 0])` gave no clue that the
    second value was an offset.

    ``scale`` and ``noise`` are NOT knobs — the driver applies both (noise pre-scale, and only in the groups
    listed in ``TaskSpec.obs_noise_groups``, so the actor reads corrupted while the critic reads the very same
    term clean). See :class:`ObsNoise`.

    ``labels`` names the dashboard columns for a term whose layout no ``names``/``bodies`` list knob spells
    out — without it those columns plot as bare indices, which say nothing about what each curve is."""

    params: dict[str, Any] = Field(default_factory=dict)
    scale: float = 1.0
    noise: Optional[ObsNoise] = None
    labels: list[str] = []            # per-dim names; [] = derive from a names/bodies knob, else index
    unit: str = ""                    # dashboard display only — see ObsTerm
    digits: int = Field(default=3, ge=0)

    def __init__(self, fn: Callable, *, name: str | None = None, scale: float = 1.0,
                 noise: Optional[ObsNoise] = None, labels: list[str] | None = None,
                 unit: str = "", digits: int = 3, **knobs: Any):
        super().__init__(fn=fn, name=name, params=knobs, scale=scale, noise=noise,
                         labels=list(labels or []), unit=unit, digits=digits)


class Done(_Ref):
    """Termination entry — ``Done(fn, truncation=False, **knobs)``.

    The knobs go straight into ``TerminateTerm.params`` and reach the flat termination function as
    ``fn(env, **params)``, so what the signature names is what the contract writes — the same rule as
    :class:`Rew` and :class:`Event`, and no positional term args (a flat signature is readable only when every
    knob is named).

    ``truncation=True`` marks a time-limit-style end (success / horizon): the trainer BOOTSTRAPS the value
    instead of treating it as a true terminal. It belongs to the ENTRY, not to the function: it says how the
    trainer should value the done, not how the done is detected."""

    params: dict[str, Any] = Field(default_factory=dict)
    truncation: bool = False

    def __init__(self, fn: Callable, *, name: str | None = None, truncation: bool = False, **knobs: Any):
        super().__init__(fn=fn, name=name, truncation=truncation, params=knobs)


class Event(_Ref):
    """Event/DR entry — ``Event(fn, "reset"|"interval", **knobs)``.

    ``"reset"`` runs on the envs being reset; ``"interval"`` is called every policy step. The knobs go
    straight into ``EventTerm.params`` and reach the flat event function as ``fn(env, env_ids, **params)``,
    so what the signature names is what the contract writes — the same rule as :class:`Rew` and :class:`Obs`.
    Event entries take no positional term args: the term is a flat function, so there is no factory left to
    take them.

    ``mode`` is the only scheduling field here — how OFTEN an interval term actually fires is the TERM's own
    knob (e.g. ``apply_object_external_force``'s ``interval_range_s``), because only the term knows what
    its cadence means: it may want to count wall time, lifted time, or steps in a window. The driver's job
    stops at calling every interval term once per step.

    ``train_only=True`` drops the term when the env is prepared for eval (``apply_curriculum_end``, which
    every recording and eval path calls before its first reset). For a disturbance that exists to harden the
    policy rather than to define the task — a tug on the object, a push on the base — that keeps every
    recording measuring the task itself, and keeps one checkpoint's clip comparable to the next's."""

    mode: Literal["reset", "interval"] = "reset"
    train_only: bool = False
    requires: tuple[str, ...] = ()
    params: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, fn: Callable, mode: str = "reset", *, name: str | None = None,
                 train_only: bool = False, requires: str | tuple[str, ...] = (), **knobs: Any):
        """``requires`` names the engine CAPABILITIES this term needs (``runtime/backend.py``'s
        ``CAPABILITIES``); on a backend without them the driver drops the term and says so, instead of the
        term reaching a write surface the engine does not have. It is the contract's way of stating that a
        DR channel exists only where the engine can honour it — so ONE contract runs on both engines and the
        engine that can do more still does it. Every other missing knob stays a loud failure."""
        super().__init__(fn=fn, name=name, mode=mode, train_only=train_only,
                         requires=(requires,) if isinstance(requires, str) else tuple(requires),
                         params=knobs)


class Curr(_Ref):
    """Curriculum entry — ``Curr(fn, **knobs)``. The knobs are the curriculum class's constructor args."""

    kwargs: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, fn: Callable, *, name: str | None = None, **kwargs: Any):
        super().__init__(fn=fn, name=name, kwargs=kwargs)


def values(block: Any) -> Any:
    """A DATA block written as a class → plain dict, recursively. Dicts/lists/scalars pass through.

    Lets every section of a contract be a named class (``class PHYSICS: hz = 200``) instead of a dict literal,
    so the whole file reads one way. Nested classes become nested dicts (``class OVERRIDES: class newton: …``
    → ``{"newton": {…}}``), while genuinely map- or list-shaped data stays as written — a joint-name → angle
    table (``init_pose``) and the ``objects`` list are data, not named knobs, and forcing them into class
    bodies would only add noise. Dunder attributes (incl. the docstring) are skipped."""
    if isinstance(block, type):
        return {k: values(v) for k, v in vars(block).items() if not k.startswith("__")}
    if isinstance(block, dict):
        return {k: values(v) for k, v in block.items()}
    if isinstance(block, (list, tuple)):
        return [values(v) for v in block]
    return block


def terms(block: Any, kind: type[_Ref]) -> list[_Ref]:
    """A contract's category block → ordered list of entries, each carrying its name.

    Two accepted forms:

    * a **class** whose attributes are entries (the isaaclab shape) — the ATTRIBUTE NAME becomes the term
      name, so a name is never written twice and the same fn can appear under two names (arm/hand penalties).
      Definition order is preserved (class bodies keep insertion order).
    * a **list** of entries — then each carries its own ``name=`` (default: ``fn.__name__``). A list may also
      hold other BLOCKS, which are flattened in order (``[ACTOR_OBS, CRITIC_OBS]`` = the critic's input is
      the actor's set plus the privileged one, with nothing repeated as a name list).

    Fail-loud: a wrong-category entry (an ``Obs`` in the reward block) and, in the class form, an entry that
    also passes ``name=`` (the attribute already says the name — two sources would drift)."""
    if isinstance(block, (list, tuple)):
        out: list[_Ref] = []
        for e in block:
            out.extend([e] if isinstance(e, _Ref) else terms(e, kind))
    else:
        out = []
        for key, val in vars(block).items():
            if key.startswith("_"):
                continue
            assert isinstance(val, _Ref), (
                f"{block.__name__}.{key} is {type(val).__name__}, not a term entry — a category block holds "
                f"only {kind.__name__}(...) entries")
            assert val.name is None, (
                f"{block.__name__}.{key} passes name={val.name!r}, but in a class block the ATTRIBUTE is the "
                f"name — drop the name= argument")
            out.append(val.model_copy(update={"name": key}))
    for e in out:
        assert isinstance(e, kind), (
            f"term entry {e.name or e.fn.__name__!r} is {type(e).__name__}, but this block takes "
            f"{kind.__name__}(...) — each category has its own entry type")
    return out


class TaskSpec(_Data):
    """Task contract — built in ``sim/metalab/contract/tasks/<task>.py`` as ``TASK = TaskSpec(...)``. Loader
    imports the module and resolves it into an EnvSpec. No engine import."""

    name: str
    num_envs: int = Field(default=4096, ge=1)
    env_spacing: float = Field(default=1.5, gt=0.0)
    episode_length_s: float = Field(default=10.0, gt=0.0)
    physics: PhysicsSpec
    # THE WORLD, in one block: everything that describes the physical stage rather than the task logic.
    #   scene={"ground": True, "robot": {...}, "objects": [...], "contact_params": {...},
    #          "camera": {...}, "goal": {...}}
    # The task logic (action / obs / reward / terminate / events / curriculum) stays at the top level — the
    # split is "what exists" vs "what we do with it". The loader unpacks this into EnvSpec's flat fields.
    scene: dict[str, Any]
    fixtures: list[FixtureSpec] = Field(default_factory=list)

    # group -> tunables. OMIT IT (or leave it empty) to take every action group the chosen robot declares,
    # with default tunables — that is what a robot-agnostic scene/standalone contract wants, so swapping
    # robot={"name": ...} between allex and allex_right needs no edit here. A policy contract spells its
    # groups (and usually its `joints`) out, because the action dim is what the checkpoint is shaped by.
    action: dict[str, ActionCfg] = Field(default_factory=dict)
    # Declared on the ACTION BLOCK, not per group — `_hoist_action_delay` lifts it out of `action` so the
    # contract reads as one block while this stays a typed field. See ActionDelaySpec.
    action_delay: ActionDelaySpec = Field(default_factory=ActionDelaySpec)

    @model_validator(mode="before")
    @classmethod
    def _hoist_action_delay(cls, data):
        if not isinstance(data, dict) or data.get("action") is None:
            return data
        act = values(data["action"])                  # ACTION may be a class; values() passes dicts through
        if not isinstance(act, dict):
            return data
        act = dict(act)
        dly = {k: act.pop(k) for k in ("min_delay", "max_delay") if k in act}
        if dly:
            assert "action_delay" not in data, \
                "declare the command delay once — inside the ACTION block, not also as action_delay="
            data["action_delay"] = dly
        data["action"] = act
        return data
    # Every term block takes the SAME two forms (see `terms()`): a class whose attributes are entries — the
    # attribute name IS the term name, isaaclab's shape — or a plain list of entries carrying their own name.
    obs: Any = Field(default_factory=list)                          # Obs entries (optional: groups may declare inline)
    # group -> "all" | [term name...] | [Obs block...]. The block form declares the terms IN the group, so a
    # contract can name its blocks (ACTOR_OBS / CRITIC_OBS) and wire them here with nothing to keep in sync.
    obs_groups: dict[str, Any] = Field(default_factory=dict)
    obs_history_length: dict[str, int] = Field(default_factory=dict)   # group -> frame-stack length H (1/omitted = off)
    obs_noise_groups: list[str] = Field(default_factory=list)          # groups where ObsTerm.noise applies (see TaskSpec)

    # Data sections may be written as a class (see `values`) — normalized to dicts before pydantic validates.
    @field_validator("physics", "scene", "action", "gate", "overrides", "fixtures", mode="before")
    @classmethod
    def _class_blocks_to_dicts(cls, v):
        return values(v)
    reward: Any = Field(default_factory=list)          # Rew entries
    terminate: Any = Field(default_factory=list)       # Done entries
    events: Any = Field(default_factory=list)          # Event entries — DR/randomization (reset/interval)
    # The FINAL success bar (what val/SR measures) — authored as the ``GATE`` block between
    # EVENTS and TERMINATE, i.e. next to the termination that ends an episode on the CURRICULUM bar. A
    # goal-bearing contract must declare it (loader fails loud otherwise). See GateSpec.
    gate: Optional[GateSpec] = None
    curriculum: Any = Field(default_factory=list)      # Curr entries — difficulty progression
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)   # per-engine corrections (carried into EnvSpec)
