from __future__ import annotations

import genesis as gs

from sim.metalab.backends.genesis import _patches
from sim.metalab.backends.genesis import friction as _friction
from sim.metalab.contract.asset_path import resolve_asset
from sim.metalab.contract.mjcf_prep import prepare_mjcf, prepare_object_mjcf
from sim.metalab.contract.spec import EnvSpec, RobotSpec


class ByBasename:
    def __init__(self, entity):
        object.__setattr__(self, "_entity", entity)

    def __getattr__(self, name):
        return getattr(self._entity, name)

    def _find(self, seq, name):
        matches = [x for x in seq if x.name.rsplit("/", 1)[-1] == name.rsplit("/", 1)[-1]]
        assert len(matches) == 1, f"'{name}' matched {len(matches)} — check name contract"
        return matches[0]

    def get_joint(self, name):
        return self._find(self._entity.joints, name)

    def get_link(self, name):
        return self._find(self._entity.links, name)


def _apply_overrides(robot: ByBasename, r: RobotSpec) -> None:
    coupled = _coupled_joints(r)
    for jname, ov in r.joint_mode_param.items():
        idx = list(robot.get_joint(jname).dofs_idx_local)
        if ov.kp != "default":           robot.set_dofs_kp([ov.kp], idx)
        if ov.kv != "default":           robot.set_dofs_kv([ov.kv], idx)
        if ov.armature != "default":     robot.set_dofs_armature([ov.armature], idx)
        if ov.effort != "default" and jname not in coupled:
            lo, hi = (-ov.effort, ov.effort) if isinstance(ov.effort, (int, float)) else ov.effort
            robot.set_dofs_force_range([lo], [hi], idx)
    _open_coupled_force_range(robot, coupled)


def _coupled_joints(r: RobotSpec) -> set[str]:
    return {j for g in r.coupled_groups() for j in g.joints} if r.motor_coupling_on() else set()


def _open_coupled_force_range(robot: ByBasename, coupled: set[str]) -> None:
    if not coupled:
        return
    inf = float("inf")
    for jname in sorted(coupled):
        idx = list(robot.get_joint(jname).dofs_idx_local)
        robot.set_dofs_force_range([-inf], [inf], idx)


def _apply_contact_params(spec: EnvSpec, handles: dict) -> None:
    cp = spec.contact_params or {}
    if not cp:
        return
    ents = {"robot": handles["robot"], **handles["fixtures"],
            **{o.name: e for o, e in zip(handles["object_specs"], handles["objects"])}}
    for key, p in cp.items():
        ent = ents.get(key)
        assert ent is not None, \
            f"contact_params key '{key}' matches no entity (have: {'|'.join(ents)})"
        if "solmix" in p:
            gs.logger.warning(f"contact_params['{key}'].solmix={p['solmix']} ignored — genesis mixes the two "
                              "geoms 0.5/0.5 and has no solmix input (newton honors it; this is a sim2sim gap)")
        if "solref" not in p and "solimp" not in p:
            continue
        for geom in ent.geoms:
            sp = [float(v) for v in geom.sol_params]
            assert len(sp) == 7, f"contact_params['{key}']: unexpected sol_params length {len(sp)}"
            if "solref" in p:
                sp[0:2] = [float(v) for v in p["solref"]]
            if "solimp" in p:
                sp[2:7] = [float(v) for v in p["solimp"]]
            geom.set_sol_params(sp)


def build_scene(spec: EnvSpec, num_envs: int | None = None, viz: bool = False, backend=None) -> dict:
    gs.init(backend=backend if backend is not None else gs.gpu, logging_level="warning")
    _patches.apply()
    _patches.apply_round_robin_variants()

    ph = spec.physics
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=ph.dt / ph.substeps, substeps=1),
        rigid_options=gs.options.RigidOptions(enable_self_collision=ph.self_collision, batch_dofs_info=True),
        vis_options=gs.options.VisOptions(),
        show_viewer=viz,
    )
    handles: dict = {"scene": scene, "fixtures": {}, "objects": [], "substeps": ph.substeps,
                     "object_specs": [o for o in spec.objects if not o.fixed]}

    if spec.scene.ground:
        handles["fixtures"][spec.scene.ground_name] = scene.add_entity(gs.morphs.Plane())

    for fx in spec.fixtures:
        if fx.kind == "box":
            assert fx.size is not None, f"box fixture '{fx.name}' requires size"
            handles["fixtures"][fx.name] = scene.add_entity(
                gs.morphs.Box(size=tuple(fx.size), pos=tuple(fx.pos), fixed=True)
            )
        else:  # pragma: no cover
            raise ValueError(f"unsupported fixture kind: {fx.kind}")

    for obj in spec.objects:
        if obj.fixed:
            assert obj.parts, f"fixed object '{obj.name}': procedural parts required"
            for part in obj.parts:
                assert part.shape == "box", f"object '{obj.name}': fixed parts support shape 'box' for now"
                pos = tuple(a + b for a, b in zip(obj.init_pos or (0.0, 0.0, 0.0), part.pos))
                ent = scene.add_entity(gs.morphs.Box(size=tuple(part.size), pos=pos, fixed=True))
            handles["fixtures"][obj.name] = ent
            continue
        mjcfs = obj.asset.get("mjcf") if obj.asset else None
        assert mjcfs, f"object '{obj.name}': MJCF asset required (asset.mjcf)"
        variants = mjcfs if isinstance(mjcfs, list) else [mjcfs]
        kw: dict = {"quat": tuple(obj.init_quat)}
        if obj.init_pos is not None:
            kw["pos"] = tuple(obj.init_pos)
        morphs = [gs.morphs.MJCF(file=prepare_object_mjcf(str(resolve_asset(v)), obj.mass, obj.collision),
                                 default_armature=None, **kw)
                  for v in variants]
        ent = scene.add_entity(morphs if len(morphs) > 1 else morphs[0])
        handles["objects"].append(ent)

    r = spec.robot
    assert "mjcf" in r.asset, "robot.asset.mjcf required (MJCF source of truth)"
    fixed_pose = {n: v for n, v in spec.robot.init_pose.items() if r.joints.get(n) == 0}
    mjcf = prepare_mjcf(str(resolve_asset(r.asset["mjcf"])), r.joints, r.collision, fixed_pose=fixed_pose)
    robot = ByBasename(scene.add_entity(gs.morphs.MJCF(
        file=mjcf, pos=tuple(r.base_pos), quat=tuple(r.base_quat), batch_fixed_verts=True,
        requires_jac_and_IK=True, default_armature=None,
    )))
    handles["robot"] = robot

    if viz:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(use_force=True, color=(0.1, 0.6, 0.8, 0.6))
        )

    scene.build(
        n_envs=num_envs if num_envs is not None else spec.num_envs,
        env_spacing=(spec.env_spacing, spec.env_spacing),
    )
    _apply_overrides(robot, r)
    _apply_contact_params(spec, handles)
    _friction.install(scene.sim.rigid_solver)

    return handles
