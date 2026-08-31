"""MJCF preprocessing — write a /tmp copy honoring the joint-active mask (1/0) and
collision mask (1/0) (engine-agnostic).

Use the MJCF (source of truth) as-is, except:
- joints with value 0 in RobotSpec.joints are welded FIXED: remove the ``<joint>``,
  its ``<actuator>``, and any referencing ``<equality><joint>`` so the body is
  rigidly fixed to its parent. Default weld angle is 0°; a fixed joint listed in
  ``fixed_pose`` is welded AT THAT ANGLE instead — the joint rotation is baked into
  the body frame (posed-then-welded), so e.g. a right-only robot can hold the waist
  pitch at 45° without paying for the dof.
- bodies with value 0 in RobotSpec.collision have their ``<geom class="collision">``
  removed: excluded from self-collision (otherwise kept) and saves collision-mesh
  loading. Visual geoms (contype/conaffinity 0) are kept, so rendering is unchanged.

Active (1) joints, gains, armature and equality stay verbatim. Both engines
(genesis MJCF / newton add_mjcf) load the same copy, so model structure matches =
parity. Physics values are not changed (structure only). meshdir/texturedir are pinned absolute.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco

_CACHE = Path(tempfile.gettempdir()) / "allex_mjcf_prep"


def _floats(s: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split()) if s else default


def _quat_mul(a, b):
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def _quat_rotate(q, v):
    """Rotate vec3 v by wxyz quaternion q (q ⊗ v ⊗ q⁻¹)."""
    qv = (0.0, *v)
    w, x, y, z = _quat_mul(_quat_mul(q, qv), (q[0], -q[1], -q[2], -q[3]))
    return (x, y, z)


def _axis_angle_quat(axis, angle_rad: float):
    """Unit wxyz quaternion for a rotation of angle_rad about axis (normalized here)."""
    n = math.sqrt(sum(a * a for a in axis))
    assert n > 1e-9, f"degenerate joint axis {axis}"
    s = math.sin(angle_rad / 2.0) / n
    return (math.cos(angle_rad / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def _bake_hinge(body: ET.Element, joint: ET.Element, angle_rad: float) -> None:
    """Weld ``body`` at ``angle_rad`` of its hinge ``joint`` by folding the joint rotation into the
    body frame (the joint is removed by the caller). MJCF kinematics: X_parent→body(θ) =
    T(body.pos, body.quat) ∘ Rot(about joint.pos, joint.axis, θ), and a rotation about a point is
    T(p − R·p, R) — so the posed body frame is pos' = pos + Q_b·(p − R·p), quat' = Q_b ⊗ R."""
    jtype = joint.get("type", "hinge")
    assert jtype == "hinge", f"fixed_pose only supports hinge joints — {joint.get('name')!r} is {jtype!r}"
    assert joint.get("ref") is None, f"fixed_pose does not support joint ref offsets ({joint.get('name')!r})"
    assert body.get("euler") is None and body.get("axisangle") is None and body.get("xyaxes") is None, \
        f"body {body.get('name')!r} uses a non-quat orientation attr — unsupported by fixed_pose bake"
    p = _floats(joint.get("pos"), (0.0, 0.0, 0.0))
    axis = _floats(joint.get("axis"), (0.0, 0.0, 1.0))
    b_pos = _floats(body.get("pos"), (0.0, 0.0, 0.0))
    b_quat = _floats(body.get("quat"), (1.0, 0.0, 0.0, 0.0))
    r = _axis_angle_quat(axis, angle_rad)
    rp = _quat_rotate(r, p)
    anchor_shift = tuple(pi - rpi for pi, rpi in zip(p, rp))          # p − R·p (body-local)
    shift_w = _quat_rotate(b_quat, anchor_shift)                      # → parent frame via Q_b
    body.set("pos", " ".join(f"{b_pos[i] + shift_w[i]:.9g}" for i in range(3)))
    body.set("quat", " ".join(f"{c:.9g}" for c in _quat_mul(b_quat, r)))


def equality_follower_pose(mjcf_path: str, known: dict[str, float]) -> dict[str, float]:
    """Auto-compute the equality-FOLLOWER joint angles [rad] from the master angles in ``known``.

    Reads the MJCF ``<equality><joint joint1 joint2 polycoef>`` couplings (MuJoCo semantics:
    ``q1 = a0 + a1·q2 + a2·q2² + a3·q2³ + a4·q2⁴``, qpos0 = 0 on ALLEX) and returns the missing side
    of each pair, so a task's init_pose only needs the ACTUATED joints:
      - ``joint2`` known, ``joint1`` missing (fingers: DIP=poly(PIP), Thumb IP=poly(MCP)) → evaluate.
      - ``joint1`` known, ``joint2`` missing (waist: lower=±upper/dummy) → invert; LINEAR couplings only
        (a2..a4 = 0, a1 ≠ 0) — a quartic has no unique inverse, fail loud.
      - both known → respect the explicit values (manual override); neither known → fail loud (the
        actuated side must be listed — the 48-actuated init_pose convention).
    Angles are radians in and out (polycoef is authored in radians)."""
    src = Path(mjcf_path).resolve()
    assert src.is_file(), f"MJCF not found: {src}"
    eq = ET.parse(src).getroot().find("equality")
    out: dict[str, float] = {}
    for e in (eq if eq is not None else []):
        j1, j2 = e.get("joint1"), e.get("joint2")
        if e.tag != "joint" or not j1 or not j2:
            continue
        a = list(_floats(e.get("polycoef"), (0.0, 1.0, 0.0, 0.0, 0.0))) + [0.0] * 5
        if j1 in known and j2 in known:
            continue                                        # explicit override — trust the task
        if j2 in known and j1 not in known:                 # follower = poly(master)
            x = known[j2]
            out[j1] = a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3 + a[4] * x**4
        elif j1 in known and j2 not in known:               # invert (linear couplings only)
            assert abs(a[2]) + abs(a[3]) + abs(a[4]) < 1e-12 and abs(a[1]) > 1e-12, \
                f"equality {j1}~{j2}: cannot invert non-linear polycoef {a[:5]} — list {j2} explicitly"
            out[j2] = (known[j1] - a[0]) / a[1]
        else:
            raise AssertionError(
                f"equality pair ({j1}, {j2}): neither side in init_pose — list the actuated side")
    return out


def prepare_object_mjcf(mjcf_path: str, mass: float, collision: bool = True,
                        out_dir: Path | None = None) -> str:
    """Return the path of a copy of ``mjcf_path`` whose body carries the CONTRACT's mass.

    The object MJCFs author no ``<inertial>``: MuJoCo then derives mass and inertia from the collision
    geometry at its default density, which is self-consistent but arbitrary in magnitude (the hammer comes
    out 0.9918 kg). The task contract owns the real number (``ObjectSpec.mass``), so we compile the asset
    once here, read the derived mass/inertia, and write back an ``<inertial>`` with the requested mass and
    the inertia scaled by the SAME ratio — uniform-density rescaling, which keeps the tensor consistent with
    the shape instead of pairing a new mass with a stale tensor.

    Done as an MJCF transform rather than a post-build write because both engines then read the identical
    file: genesis' runtime mass API shifts mass only (inertia untouched), so patching after the fact would
    make the two engines disagree on rotational dynamics.

    ``collision=False`` (``ObjectSpec.collision``) additionally drops every ``class="collision"`` geom, the
    same transform the robot's collision mask applies: the visual geoms stay, so the object still renders,
    and the ``<inertial>`` written above already fixes mass/inertia, so its dynamics are untouched — only its
    contact surface goes away. An asset that authors no ``class="collision"`` geom fails loud here.
    """
    src = Path(mjcf_path).resolve()
    assert src.is_file(), f"MJCF not found: {src}"
    assert mass > 0.0, f"{src.name}: object mass must be > 0 (got {mass})"
    m = mujoco.MjModel.from_xml_path(str(src))
    bodies = [b for b in range(1, m.nbody) if m.body_mass[b] > 0.0]
    assert len(bodies) == 1, \
        f"{src.name}: expected exactly ONE massive body to take ObjectSpec.mass, found {len(bodies)}"
    b = bodies[0]
    m0 = float(m.body_mass[b])
    scale = mass / m0
    ipos, iquat, diag = m.body_ipos[b], m.body_iquat[b], m.body_inertia[b] * scale

    tree = ET.parse(src)
    root = tree.getroot()
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    body = next((e for e in root.iter("body") if e.get("name") == name), None)
    assert body is not None, f"{src.name}: body '{name}' not found in the XML tree"
    for old in body.findall("inertial"):          # authored inertial would defeat the point — none expected
        body.remove(old)
    ET.SubElement(body, "inertial", {
        "pos": " ".join(f"{v:.9g}" for v in ipos),
        "quat": " ".join(f"{v:.9g}" for v in iquat),
        "mass": f"{mass:.9g}",
        "diaginertia": " ".join(f"{v:.9g}" for v in diag),
    })
    if not collision:
        dropped = 0
        for b in root.iter("body"):
            for g in [e for e in b.findall("geom") if e.get("class") == "collision"]:
                b.remove(g)
                dropped += 1
        assert dropped, f'{src.name}: collision=False but the asset authors no class="collision" geom to drop'

    comp = root.find("compiler")          # pin asset dirs absolute so they resolve from the cache copy
    for _d in ("meshdir", "texturedir"):  # texturedir too: a textured asset else fails to open its png
        if comp is not None and comp.get(_d):
            comp.set(_d, str((src.parent / comp.get(_d)).resolve()))
    out_dir = out_dir or _CACHE
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}__m{mass:g}{'' if collision else '__ghost'}.xml"
    tree.write(out, encoding="unicode")
    return str(out)


def prepare_mjcf(mjcf_path: str, joint_mask: dict[str, int],
                 collision_mask: dict[str, int] | None = None, out_dir: Path | None = None,
                 fixed_pose: dict[str, float] | None = None) -> str:
    """Return the path of a copy of ``mjcf_path`` preprocessed per the masks. joint_mask must cover every real MJCF joint.

    joint_mask: {joint_name: 1(active)|0(fixed)}. Missing/unknown joints fail loud.
    collision_mask: {body_name | mesh_name: 1(keep)|0(remove collision geom)}. A BODY key covers every
        ``class="collision"`` geom of that body; a MESH key covers just the geoms referencing that mesh (the
        ALLEX geoms carry no ``name``, so the mesh name — 1:1 with the .stl file — is the per-geom handle).
        The mesh key is the more specific one and WINS over the body key, so a body can be kept while a few
        of its shells are dropped (``R_Palm_Link: 1`` + ``R_Palm_Back_Collision1: 0``) or dropped while one is
        kept. Unlisted = 1 (keep). A key that is neither a body nor a collision mesh fails loud.
        CAVEAT: a mesh key hits EVERY body using that mesh. 17 of the 69 ALLEX collision meshes are shared —
        ``Finger_*_Collision`` x8 (both hands, 4 fingers) and ``Upper_Arm_*``/``Forearm_*``/``Wrist_*``/
        ``Elbow_*``/``Shoulder_RollYaw_*`` x2 (left+right). Use the BODY key when you need one side only.
        The other 52 (all ``R_Palm_*``, ``R_Thumb_*``, torso, head) are 1:1 and safe to switch individually.
    fixed_pose: {joint_name: angle_rad} for MASK-0 joints — welded at this angle instead of 0° (the joint
        rotation is baked into the body frame before removal). Keys must be mask-0 hinge joints (fail loud).
        NOTE equality-coupled followers are NOT solved for you: welding e.g. Waist_Lower_Pitch at 45° means
        its linkage partners (Waist_Pitch_Dummy/Upper) must be listed with their own coupled angles, or they
        stay welded at 0° (kinematically inconsistent pose, silently wrong geometry).
    """
    collision_mask = collision_mask or {}
    fixed_pose = fixed_pose or {}
    src = Path(mjcf_path).resolve()
    assert src.is_file(), f"MJCF not found: {src}"
    tree = ET.parse(src)
    root = tree.getroot()
    wb = root.find("worldbody")
    assert wb is not None, f"{src}: no <worldbody>"

    real = {j.get("name") for j in wb.iter("joint")}
    mask_names = set(joint_mask)
    missing = real - mask_names           # real joints not listed in the mask
    unknown = mask_names - real           # in the mask but absent from the MJCF
    assert not missing, f"joints mask missing (must list all): {sorted(missing)}"
    assert not unknown, f"joints mask has joints absent from MJCF: {sorted(unknown)}"
    fixed = {n for n, v in joint_mask.items() if v == 0}
    bad_fp = set(fixed_pose) - fixed
    assert not bad_fp, f"fixed_pose keys must be mask-0 joints — not fixed: {sorted(bad_fp)}"

    all_bodies = {b.get("name") for b in wb.iter("body")}
    # Meshes actually referenced by a collision geom — the only mesh names the mask may key on (a visual-only
    # mesh would silently do nothing, so listing one is a contract bug, not a no-op).
    col_meshes = {g.get("mesh") for b in wb.iter("body") for g in b.findall("geom")
                  if g.get("class") == "collision" and g.get("mesh")}
    unknown_b = set(collision_mask) - all_bodies - col_meshes
    assert not unknown_b, (
        f"collision mask keys are neither a body nor a collision mesh: {sorted(unknown_b)}")
    col_off_body = {n for n, v in collision_mask.items() if v == 0 and n in all_bodies}
    col_off_mesh = {n for n, v in collision_mask.items() if v == 0 and n in col_meshes}
    col_on_mesh = {n for n, v in collision_mask.items() if v == 1 and n in col_meshes}
    col_off = col_off_body | col_off_mesh

    if not fixed and not col_off:
        return str(src)   # nothing to change → return original

    # (1) pin meshdir/texturedir absolute (so assets resolve even from the /tmp copy)
    comp = root.find("compiler")
    for _d in ("meshdir", "texturedir"):
        if comp is not None and comp.get(_d):
            comp.set(_d, str((src.parent / comp.get(_d)).resolve()))

    # (2) remove <joint> of fixed joints (walking parent bodies) → body welded to parent. A fixed_pose
    # entry first bakes the joint rotation into the body frame (posed-then-welded); no entry = weld at 0°.
    for body in wb.iter("body"):
        for j in list(body.findall("joint")):
            if j.get("name") in fixed:
                ang = fixed_pose.get(j.get("name"), 0.0)
                if abs(ang) > 1e-12:
                    _bake_hinge(body, j, float(ang))
                body.remove(j)
        # (2b) drop class="collision" geoms (visual kept = render unchanged). A mesh key overrides its body:
        # mesh 0 drops that shell even in a kept body, mesh 1 keeps it even in a dropped body.
        body_off = body.get("name") in col_off_body
        for g in list(body.findall("geom")):
            if g.get("class") != "collision":
                continue
            mesh = g.get("mesh")
            if mesh in col_off_mesh or (body_off and mesh not in col_on_mesh):
                body.remove(g)

    # (3) remove actuators of fixed joints (else MuJoCo/genesis fails referencing a nonexistent joint)
    act = root.find("actuator")
    if act is not None:
        for a in list(act):
            if a.get("joint") in fixed:
                act.remove(a)

    # (4) remove equality<joint> referencing a fixed joint
    eq = root.find("equality")
    if eq is not None:
        for e in list(eq):
            if e.get("joint1") in fixed or e.get("joint2") in fixed:
                eq.remove(e)

    out_root = out_dir or _CACHE
    out_root.mkdir(parents=True, exist_ok=True)
    key = (str(src) + str(src.stat().st_mtime_ns) + "|".join(sorted(fixed)) + "#" + "|".join(sorted(col_off))
           + "@" + "|".join(f"{n}={fixed_pose[n]:.9g}" for n in sorted(fixed_pose)))
    tag = hashlib.md5(key.encode()).hexdigest()[:12]
    dst = out_root / f"{src.stem}_{tag}.xml"
    tree.write(dst, encoding="unicode")
    return str(dst)
