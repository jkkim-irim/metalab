"""Shared core for the standalone manipulation contracts — a robot at a desk with something on it.

Same parent/child split as ``tasks/hammer_lift_student/_base.py``: every value two contracts would state
identically lives here once, and a contract passes only what differs to :func:`build_task`. The defaults
below are the ALLEX-right desk scene; ``dumbbell_test`` is the outlier that replaces most of them.

Override shapes are not uniform, and the reason is behavioural rather than stylistic:

* ``physics`` / ``desk`` / ``contact`` MERGE onto the base block, so a contract names one field.
* ``newton`` and ``robot`` REPLACE it. The loader passes ``overrides.newton`` through verbatim — an
  unset key stays unset rather than falling back to a documented default — so merging would hand a
  contract keys it never declared (``njmax``, ``integrator``, ``solver``, …) and silently change its
  solver. A replacing block restates the four scalars every contract happens to share; that duplication
  is the price of not inventing settings for whoever inherits.
* ``camera`` defaults to the base block and is replaced by whatever is passed, so a contract opts out
  with ``camera=None``. ``action`` defaults to EMPTY — see :func:`build_task`.
"""
from __future__ import annotations

from sim.metalab.contract.spec import TaskSpec, values

DESK_TOP = 0.871
"""[m] top face of DESK — ``init_pos`` z for an asset authored resting on z=0 (every YCB scan is)."""


# --- physics / solver --------------------------------------------------------------------------------
class PHYSICS:
    hz = 120            # control rate — physics dt = 1/hz
    substeps = 4        # physics substeps per control step
    decimation = 2      # physics steps per CONTROL step → 100 Hz, step_dt = 0.01 s (the viewer's frame rate)


# --- per-engine corrections (NOT shared knobs) -------------------------------------------------------
class NEWTON:
    use_mujoco_contacts = True    # False = newton-native collide(); True = mjwarp broad/narrowphase
    enable_multiccd = True        # up to 4 contact points per geom pair (newton default: 1)
    integrator = "implicitfast"   # euler | rk4 | implicit | implicitfast (= implicit minus Coriolis)
    solver = "newton"         # constraint solver: "newton" | "cg". PGS is NOT supported by mjwarp.
    cone = "pyramidal"        # friction cone: pyramidal (cheap) | elliptic (isotropic, needs impratio 100)
    impratio = 1.0            # frictional-to-normal impedance ratio (MuJoCo default 1.0)
    iterations = 100          # solver iterations (MuJoCo default 100)
    ls_iterations = 50        # line-search iterations (MuJoCo default 50)
    tolerance = 1.0e-8        # solver early-exit tolerance (MuJoCo default 1e-8)
    ls_tolerance = 0.01     # line-search tolerance (MuJoCo default 0.01)
    jacobian = "auto"       # "dense" | "sparse" | "auto" (auto = sparse iff nv>32; allex nv=33)
    nconmax = 128             # contact slots PER WORLD in the mjwarp buffer
    njmax = 256               # constraint rows PER WORLD (a contact costs 3-4)
    ccd_iterations = 35       # EPA/CCD workspace (MuJoCo default 35; lower = less transient memory)
    eq_solref = [0.01, 1.1]
    eq_solimp = [0.9999, 0.9999, 0.001, 0.5, 1.0]


# --- robot: right arm + hand at the desk --------------------------------------------------------------
class ROBOT:
    name = "allex_right"
    base_pos = [0.0, 0.0, 0.6]
    base_quat = [1.0, 0.0, 0.0, 0.0]   # wxyz
    fixed_base = True
    # The hand starts next to the spawn spot on the desk. This is scene setup, not learning — the
    # standalone runner holds it (zero drive beyond the PD target). DEGREES (loader → rad). ACTIVE
    # (mask 1) = runtime spawn pose, MASK-0 = weld angle. The 12 equality FOLLOWERS (waist dummy/upper,
    # finger IP/DIP) are auto-computed by the loader.
    init_pose = {
        # waist + neck (4 — mask-0 on allex_right → welded at these angles)
        "Waist_Yaw_Joint": 0.0, "Waist_Lower_Pitch_Joint": 30.0,
        "Neck_Pitch_Joint": 40.0, "Neck_Yaw_Joint": 0.0,
        # right arm (7 — active)
        "R_Shoulder_Pitch_Joint": 38.28, "R_Shoulder_Roll_Joint": -25.0, "R_Shoulder_Yaw_Joint": -20.0,
        "R_Elbow_Joint": -125.5, "R_Wrist_Yaw_Joint": 180.0, "R_Wrist_Roll_Joint": 0.0,
        "R_Wrist_Pitch_Joint": 5.0,
        # right hand (15 — active; IP/DIP auto-follow)
        "R_Thumb_Yaw_Joint": 0.0, "R_Thumb_CMC_Joint": 10.0, "R_Thumb_MCP_Joint": 20.0,
        "R_Index_ABAD_Joint": 0.0, "R_Index_MCP_Joint": 10.0, "R_Index_PIP_Joint": 20.0,
        "R_Middle_ABAD_Joint": 0.0, "R_Middle_MCP_Joint": 10.0, "R_Middle_PIP_Joint": 20.0,
        "R_Ring_ABAD_Joint": 0.0, "R_Ring_MCP_Joint": 10.0, "R_Ring_PIP_Joint": 20.0,
        "R_Little_ABAD_Joint": 0.0, "R_Little_MCP_Joint": 10.0, "R_Little_PIP_Joint": 20.0,
        # left arm (7 — mask-0 → welded)
        "L_Shoulder_Pitch_Joint": 0.0, "L_Shoulder_Roll_Joint": 0.0, "L_Shoulder_Yaw_Joint": 0.0,
        "L_Elbow_Joint": 0.0, "L_Wrist_Yaw_Joint": 0.0, "L_Wrist_Roll_Joint": 0.0,
        "L_Wrist_Pitch_Joint": 0.0,
        # left hand (15 — mask-0 → welded; IP/DIP auto-follow)
        "L_Thumb_Yaw_Joint": 0.0, "L_Thumb_CMC_Joint": 0.0, "L_Thumb_MCP_Joint": 0.0,
        "L_Index_ABAD_Joint": 0.0, "L_Index_MCP_Joint": 0.0, "L_Index_PIP_Joint": 0.0,
        "L_Middle_ABAD_Joint": 0.0, "L_Middle_MCP_Joint": 0.0, "L_Middle_PIP_Joint": 0.0,
        "L_Ring_ABAD_Joint": 0.0, "L_Ring_MCP_Joint": 0.0, "L_Ring_PIP_Joint": 0.0,
        "L_Little_ABAD_Joint": 0.0, "L_Little_MCP_Joint": 0.0, "L_Little_PIP_Joint": 0.0,
    }


# --- the desk: scenery, welded; size stays a task knob (procedural, no asset) -------------------------
# Its top face is DESK_TOP = init_pos.z + size.z/2 (contract sizes are FULL extents).
class DESK:
    name = "table"
    fixed = True
    mass = 1.0
    parts = [{"shape": "box", "size": [0.4, 0.8, 0.1]}]
    init_pos = [0.7, 0.0, 0.8]


# --- contact softness per group -----------------------------------------------------------------------
# Shared knob — genesis implements the same solref/solimp math; solmix is newton-only (genesis averages
# the two geoms 0.5/0.5 and cannot weight them).
class CONTACT:
    robot = {"solref": [0.005, 1.0], "solimp": [0.9, 0.9999, 0.001, 0.5, 2.0], "solmix": 4.0}
    table = {"solref": [0.01, 1.0], "solimp": [0.9, 0.99, 0.001, 0.5, 2.0], "solmix": 1.0}


class CAMERA:
    eye = [1.5, -1.1, 1.5]
    lookat = [0.4, -0.05, 1.0]
    fov = 30.0


_UNSET = object()


def build_task(name: str, *, objects, contact=None, physics=None, newton=None, robot=None, desk=None,
               action=None, camera=_UNSET, robot_friction=None) -> TaskSpec:
    """Assemble a standalone contract: this desk scene plus the movable ``objects`` put on it.

    ``objects`` is the movable list in spawn order; the desk is appended after it. Every other argument
    takes this module's block when omitted — see the module docstring for which merge and which replace.

    ``action`` is EMPTY by default, which makes the loader take the groups the robot's YAML declares:
    standalone writes ``backend.set_joint_targets`` directly and never runs ``EnvDriver.step``, the only
    reader of a group's ``scale``/``ema_tau``/delay, so an authored block would pin nothing but the
    joint membership — and for ``allex_right`` that membership is what the YAML already gives (measured:
    same arm 7 / hand 11, same order). Pinning it here would only make this base robot-specific.
    """
    scene = {
        "ground": True,
        "robot": values(ROBOT if robot is None else robot),
        "objects": [*values(objects), {**values(DESK), **values(desk or {})}],
        "contact_params": {**values(CONTACT), **values(contact or {})},
    }
    cam = values(CAMERA) if camera is _UNSET else values(camera)
    if cam is not None:
        scene["camera"] = cam
    if robot_friction is not None:
        scene["robot_friction"] = robot_friction
    return TaskSpec(
        name=name,
        num_envs=1,                    # standalone (the runner forces 1 via build_env anyway)
        physics={**values(PHYSICS), **values(physics or {})},
        overrides={"newton": values(NEWTON if newton is None else newton)},
        scene=scene,
        action=values(action or {}),
    )
