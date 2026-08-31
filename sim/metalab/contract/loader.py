"""Component/contract loader — robot/object YAML + task ``.py`` → validated EnvSpec.

The loader owns validation, defaults, and fail-loud: missing file/module / schema violation
(unknown field or value constraint) / non-callable or bad-signature term / unknown reference
all raise immediately.

``load_task`` is the core: import the task module (``TASK: TaskSpec``) and (1) load
components (robot/object YAML), (2) resolve closed-vocabulary refs in the entries' knobs
("@joints.ctrl", "@frames.palm"), (3) bind each term — obs/reward/terminate/events are FLAT
functions whose resolved knobs become ``term.params`` (checked against the signature here, so a
typo fails at LOAD), curriculum alone is still a factory called to bind ``fn(env)`` — turning the
declarative contract into an executable EnvSpec.
Parsers (genesis/newton) see only EnvSpec. This file imports no engine and no term
factory (the task module imports the factories it references — that is what enables
Go-to-Definition).
"""
from __future__ import annotations

import importlib
import inspect
import math
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
import yaml

from sim.metalab.contract.mjcf_prep import equality_follower_pose
from sim.metalab.contract.recipe import recipe as recipe_values
from sim.metalab.contract.spec import (
    ActionCfg,
    ActionGroupSpec,
    CameraSpec,
    Curr,
    CurriculumTerm,
    Done,
    EnvSpec,
    Event,
    EventTerm,
    GoalSpec,
    ObjectSpec,
    Obs,
    ObsTerm,
    Rew,
    RewardTerm,
    RobotSpec,
    SceneSpec,
    TaskSpec,
    TerminateTerm,
    terms,
)

_ENVS_DIR = Path(__file__).resolve().parent  # sim/metalab/contract
_REPO = _ENVS_DIR.parents[2]                 # <repo> (sim/metalab/contract → repo root)
T = TypeVar("T", bound=BaseModel)


def load_yaml(path: Path) -> dict[str, Any]:
    """YAML file → dict. Fail loud if the top level is not a mapping."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML top level must be a mapping (dict) — got {type(data).__name__}")
    return data


def load_component(category: str, name: str, schema: type[T], **overrides: Any) -> T:
    """Read ``sim/metalab/contract/<category>/<name>.yaml`` and validate against ``schema``. overrides = top-key overwrite."""
    path = _ENVS_DIR / category / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"component not found: {path}")
    data = load_yaml(path)
    data.update(overrides)
    return schema.model_validate(data)  # fail-loud


def load_robot(name: str, **overrides: Any) -> RobotSpec:
    return load_component("robot", name, RobotSpec, **overrides)


def _with_equality_followers(robot: RobotSpec, init_pose: dict[str, float]) -> dict[str, float]:
    """init_pose [rad] + the equality-FOLLOWER angles auto-computed from the robot MJCF couplings
    (see :func:`mjcf_prep.equality_follower_pose`). The task only authors the ACTUATED joints; followers
    (waist dummy/upper, finger IP/DIP) come from the MJCF <equality> polycoef — explicit task entries win."""
    followers = equality_follower_pose(str(_REPO / robot.asset["mjcf"]), init_pose)
    unknown = set(followers) - set(robot.joints)
    assert not unknown, f"equality followers absent from the robot joint mask: {sorted(unknown)}"
    return {**init_pose, **followers}


# ---------------------------------------------------------------------------
# task contract resolution (TaskSpec module → EnvSpec)
# ---------------------------------------------------------------------------
def _resolve_ref(v: Any, refs: dict[str, dict]) -> Any:
    """``"@cat.key"`` → refs[cat][key] (closed vocabulary). Recurse into list/dict; else literal. Unknown ref fails loud."""
    if isinstance(v, str) and v.startswith("@"):
        cat, _, key = v[1:].partition(".")
        table = refs.get(cat)
        if table is None or key not in table:
            valid = {c: sorted(t) for c, t in refs.items()}
            raise ValueError(f"unknown reference {v!r} — valid vocabulary: {valid}")
        return table[key]
    if isinstance(v, list):
        return [_resolve_ref(x, refs) for x in v]
    if isinstance(v, dict):
        return {k: _resolve_ref(x, refs) for k, x in v.items()}
    return v


def _check_params(fn, params: dict[str, Any], kind: str, fixed: int = 1) -> None:
    """Fail at LOAD on knobs a FLAT term's signature cannot take.

    A flat term is called ``fn(env, **params)`` deep inside the step loop, so without this a misspelled or
    missing knob would surface as a TypeError on the first step of a launched run instead of at contract
    load. (The factory form got this for free — it was called at load. Keeping the check is what makes the
    flat form no worse.) ``fixed`` = leading parameters the DRIVER passes positionally and a contract
    therefore never names: 1 (``env``) for obs/reward, 2 (``env, env_ids``) for an event term.

    ``**kwargs`` waives the UNKNOWN half only. A named parameter without a default stays required — python
    raises for ``fn(env, a, b, **kwargs)`` called without ``a`` — so ``fn(env, **kwargs)`` accepting anything
    must not be read as ``fn(env, a, b, **kwargs)`` needing nothing."""
    sig = inspect.signature(fn)
    ps = list(sig.parameters.values())
    assert len(ps) >= fixed, \
        f"{kind} term {fn.__name__!r} takes {len(ps)} argument(s) — a flat term starts with {ps[:fixed]!r}"
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps)
    named = [p for p in ps[fixed:]                        # drop the driver-passed positionals and the
             if p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)]  # *|** slots
    accepted = [p.name for p in named]
    unknown = [] if var_kw else [k for k in params if k not in accepted]
    assert not unknown, \
        f"{kind} term {fn.__name__!r}: unknown knob(s) {unknown} — its signature takes {accepted}"
    missing = [p.name for p in named if p.default is inspect.Parameter.empty and p.name not in params]
    assert not missing, \
        f"{kind} term {fn.__name__!r}: missing required knob(s) {missing} — its signature takes {accepted}" \
        + (" (**kwargs takes extras, but a named parameter with no default is still required)" if var_kw else "")


def _build_fn(ref, refs: dict[str, dict], kind: str):
    """Curr entry → resolve kwargs refs, CALL the factory at load → fn(env). Bad signature fails loud here.

    The last factory-shaped category: a curriculum term is a class holding progress state (``level``,
    ``last_iter``) across resets, which the driver reads (``t.fn.level``) — obs/reward/terminate/events are all
    flat ``fn(env, **params)`` and never come through here."""
    assert callable(ref.fn), f"{kind} term fn must be an imported factory callable — got {ref.fn!r}"
    kwargs = {k: _resolve_ref(x, refs) for k, x in ref.kwargs.items()}
    try:
        return ref.fn(**kwargs)                 # factory → fn(env)
    except TypeError as e:
        raise TypeError(f"{kind} term {ref.fn.__name__!r} arg mismatch: kwargs={kwargs} — {e}") from e


_reported_effort: set = set()      # robots already reported — one notice per process, not per load_task call


def _report_ignored_effort(robot) -> None:
    """Say ONCE that this run's ``effort`` knobs are inert, so a dead knob never looks effective.

    Reported here rather than from ``RobotSpec``'s validator: pydantic re-runs that on every construction
    (the loader builds the robot spec, then again with the task's overlays), so a print there fires twice per
    load. Rationale for why they are inert, and why they are still worth keeping in the YAML:
    :meth:`RobotSpec.effort_ignored_joints`."""
    dead = robot.effort_ignored_joints()
    key = (robot.asset.get("mjcf"), robot.control_mode, tuple(dead))
    if not dead or key in _reported_effort:
        return
    _reported_effort.add(key)
    print(f"[robot] control_mode={robot.control_mode}: joint_mode_param `effort` is IGNORED for "
          f"{len(dead)} motor-coupled joint(s) — a coupled joint's clamp lives in MOTOR space, and its "
          f"joint-space bound Gᵀ·(envelope ∩ rated) is pose/speed-dependent (standalone 'Joint Torque "
          f"Limit' tab). Live again under control_mode=joint. First: {', '.join(dead[:3])}…", flush=True)


def task_recipes(name: str) -> list[str]:
    """The recipe names of task family ``tasks/<name>/``, or [] when it is not a family folder.

    A recipe file is ``tasks/<name>/<name>_<recipe>.py``; ``_*.py`` (the shared ``_base``) is a library,
    not a recipe. The prefix is enforced, not just matched — a differently named file would otherwise
    drop out of every list silently and its contract would be unreachable."""
    d = Path(__file__).parent / "tasks" / name
    if not (d / "__init__.py").is_file():
        return []
    files = [f for f in sorted(d.glob("*.py")) if not f.stem.startswith("_")]
    for f in files:
        assert f.stem.startswith(f"{name}_"), \
            f"recipe {f} must be named {name}_<recipe>.py (the family name is the prefix)"
    return [f.stem[len(name) + 1:] for f in files]


def _standalone_module(name: str) -> str:
    """Standalone contract name → its dotted module path, found under ``tasks/standalone/<group>/``.

    Scene-only contracts are filed by group (``manipulation/``, ``physics_test/``) rather than sitting
    flat, but the group is a shelf, not part of the task's identity: ``--task hammer-lift`` names the
    file wherever it is shelved. Two groups holding the same stem would make that ambiguous, so it fails
    here instead of resolving to whichever sorted first."""
    d = Path(__file__).parent / "tasks" / "standalone"
    hits = [f for f in sorted(d.glob(f"{name}.py")) + sorted(d.glob(f"*/{name}.py"))
            if not f.stem.startswith("_")]
    assert hits, f"standalone contract '{name}' not found under {d}"
    assert len(hits) == 1, \
        f"standalone contract '{name}' is ambiguous — {', '.join(str(f.relative_to(d)) for f in hits)}"
    return "sim.metalab.contract.tasks.standalone." + ".".join(hits[0].relative_to(d).with_suffix("").parts)


def load_task(name: str, recipe: str | None = None, num_envs: int | None = None) -> EnvSpec:
    """(task, recipe) → resolved EnvSpec.

    Two axes. ``tasks/<name>/`` is a task FAMILY — a shared ``_base.py`` core plus one file per recipe,
    and it is NOT runnable by itself, so ``recipe`` is required and names ``<name>_<recipe>.py``. A
    single-file contract (``tasks/<name>.py``, or ``tasks/standalone/<name>.py`` for the scene-only ones)
    takes no recipe. Every miss — no recipe given, unknown recipe, a family with none — fails here with
    the candidates listed, rather than resolving to whatever the family last defaulted to.

    The chosen module exposes ``TASK: TaskSpec`` (the normal form — a contract is declarative, so there is
    nothing for a builder function to do), or a ``build_task() -> TaskSpec`` when one genuinely needs to
    compute it. Knob VALUES are SIM-OWNED and written inline in the contract — nothing is shipped over the
    wire and the trainer is never imported. num_envs, if given, overrides the contract value (training
    CLI)."""
    avail = task_recipes(name)
    if (Path(__file__).parent / "tasks" / name / "__init__.py").is_file():
        assert avail, f"task family '{name}' has no recipe — add " \
                      f"sim/metalab/contract/tasks/{name}/{name}_<recipe>.py"
        assert recipe, f"task '{name}' is a family: pick a recipe (--recipe) — {', '.join(avail)}"
        rec = recipe.replace("-", "_")
        assert rec in avail, f"task '{name}': unknown recipe {recipe!r} — have {', '.join(avail)}"
        mod = importlib.import_module(f"sim.metalab.contract.tasks.{name}.{name}_{rec}")
    else:
        # Single-file contract. The guard re-raises a REAL import error inside a task module (only
        # "module absent" falls through to tasks/standalone/, which is kept out of the train/eval list).
        assert recipe is None, f"task '{name}' is a single-file contract — it takes no recipe ({recipe!r})"
        modname = f"sim.metalab.contract.tasks.{name}"
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if e.name != modname:
                raise
            mod = importlib.import_module(_standalone_module(name))
    builder = getattr(mod, "build_task", None)
    ts = builder() if builder is not None else getattr(mod, "TASK", None)
    assert isinstance(ts, TaskSpec), \
        f"task module {mod.__name__} must define build_task()->TaskSpec or " \
        f"TASK: TaskSpec — got {type(ts).__name__}"
    # The task's robot dict = {"name": robot/<name>.yaml} + RobotSpec overlays (base placement etc.), so the
    # task file shows the full spawn in one place (load_component top-key overwrite).
    # `scene` is the authoring container for the world; validate it here and unpack into the flat pieces the
    # rest of this function (and EnvSpec) already speak. Unknown keys fail loud — a typo in a contract must
    # not silently drop a table or a camera.
    scene = dict(ts.scene)
    known = {"ground", "ground_name", "robot", "objects", "contact_params", "camera", "goal", "robot_friction"}
    unknown = set(scene) - known
    assert not unknown, f"task '{name}': scene has unknown keys {sorted(unknown)} — valid: {sorted(known)}"
    assert "robot" in scene, f"task '{name}': scene needs a 'robot' block"
    stage = SceneSpec(**{k: scene[k] for k in ("ground", "ground_name") if k in scene})
    objects = [ObjectSpec(**o) if isinstance(o, dict) else o for o in scene.get("objects", [])]
    contact_params = scene.get("contact_params", {}) or {}
    camera = CameraSpec(**scene["camera"]) if isinstance(scene.get("camera"), dict) else scene.get("camera")
    goal = GoalSpec(**scene["goal"]) if isinstance(scene.get("goal"), dict) else scene.get("goal")
    # GATE = the contract's FINAL success bar (val/SR). A goal-bearing task MUST state it: the
    # curriculum keeps moving the learning bar, so without GATE "solved" would only ever be implied by the
    # level the run happened to reach. The two are paired — neither is meaningful alone.
    assert (goal is None) == (ts.gate is None), (
        f"task '{name}': scene.goal and GATE must be declared together — "
        f"goal={'set' if goal is not None else 'missing'}, GATE={'set' if ts.gate is not None else 'missing'}. "
        f"GATE (the final success bar val/SR measures) goes between EVENTS and TERMINATE.")
    robot_friction = scene.get("robot_friction")

    rcfg = dict(scene["robot"])
    rname = rcfg.pop("name", None)
    assert rname, f"task '{name}': robot={{...}} needs a 'name' key (robot/<name>.yaml)"
    pose_deg = rcfg.pop("init_pose", {}) or {}       # authored in DEGREES for readability
    robot = load_robot(rname, **{k: v for k, v in rcfg.items() if v is not None})
    robot.init_pose = _with_equality_followers(robot, {j: math.radians(v) for j, v in pose_deg.items()})
    assert robot.base_pos is not None, \
        f"task '{name}': robot={{... 'base_pos': [...]}} is required (base placement is task-owned)"
    # GATE states HOW MANY fingertips must grip and MAY name which ones; both resolve against the hand's own
    # list (robot yaml `fingertips`).
    gate = ts.gate
    if gate is not None and gate.contact_count > 0:
        assert len(robot.fingertips) >= gate.contact_count, (
            f"task '{name}': GATE.contact_count={gate.contact_count} but robot({rname}) declares "
            f"{len(robot.fingertips)} fingertips — add them to robot/{rname}.yaml `fingertips:`")
    if gate is not None and gate.contact_fingers:
        unknown = [b for b in gate.contact_fingers if b not in robot.fingertips]
        assert not unknown, (
            f"task '{name}': GATE.contact_fingers names {unknown}, which robot({rname}) does not declare as "
            f"fingertips — it has {robot.fingertips}")

    # closed-vocabulary ref context: joints (action groups + ctrl = all control joints in order) and frames.
    # Both resolve against the ROBOT yaml, which the task does not own — that is why they are string refs.
    # Task-owned literals need no indirection: the contract is Python, so it passes its own variables in.
    # Action groups: omitted (`action={}`) = every group this robot declares, with default tunables. That is
    # what a robot-agnostic contract wants — a scene/standalone task should follow whichever robot the scene
    # picked (allex -> arm_r/hand_r/arm_l/hand_l, allex_right -> arm/hand) instead of naming the groups of one
    # robot and breaking on the other. A policy contract still spells its groups out: the action dim is what
    # the checkpoint is shaped by, so there a typo must fail rather than resolve to "everything".
    task_action = ts.action or {g: ActionCfg() for g in robot.action_groups}
    for g in task_action:
        assert g in robot.action_groups, \
            f"task '{name}': action group {g!r} not in robot({rname}).action_groups — valid: " \
            f"{sorted(robot.action_groups)}. Omit `action` entirely to take all of this robot's groups."
    ctrl = [j for g in task_action for j in robot.action_groups[g]]
    # Read-only joint groups (robot.joint_groups) share the "@joints.*" vocabulary but are NOT action groups
    # — they name joints a task may OBSERVE without commanding them (see RobotSpec.joint_groups). Shadowing
    # an action group or the derived `ctrl` would silently change what a contract's obs reads, so it fails loud.
    clash = [g for g in robot.joint_groups if g in robot.action_groups or g == "ctrl"]
    assert not clash, (
        f"robot({rname}).joint_groups {clash} collide with an action group / 'ctrl' — a read-only group "
        f"must have its own name (it is observable but not commandable)")
    refs = {
        "joints": {**robot.action_groups, **robot.joint_groups, "ctrl": ctrl},
        "frames": dict(robot.frames),
        # Named BODY GROUPS — the same idea as `frames`, for terms that take a list of bodies rather than one.
        # "which bodies are the fingertips" is a property of the hand, so a contract references it instead of
        # repeating the MJCF names: `args=["@bodies.fingertips"]`. Only present when the robot declares them,
        # so a contract that references fingertips on a robot without any fails loud (unknown reference).
        "bodies": {**({"fingertips": robot.fingertips} if robot.fingertips else {}),
                   **({"nail": robot.nail_friction.bodies} if robot.nail_friction else {})},
    }

    # Action dim per group: the task's explicit `joints` subset when given, else the whole robot group.
    def _act_joints(g: str, a) -> list[str]:
        full = robot.action_groups[g]
        if a.joints is None:
            return list(full)
        assert a.joints, f"action group {g!r}: joints=[] — omit the key to use the robot's whole group"
        dup = [j for j in set(a.joints) if a.joints.count(j) > 1]
        assert not dup, f"action group {g!r}: duplicate joints {sorted(dup)}"
        extra = [j for j in a.joints if j not in full]
        assert not extra, (f"action group {g!r}: {extra} not in robot({rname}).action_groups[{g!r}] — "
                           f"the policy can only command joints the robot declares actuatable: {full}")
        return list(a.joints)

    action = {
        g: ActionGroupSpec(joints=_act_joints(g, a), scale=a.scale,
                           ema_tau=a.ema_tau, mode=a.mode)
        for g, a in task_action.items()
    }


    # obs terms (flat fn + resolved knobs → ObsTerm, the reward-term shape). A term is declared EITHER in the
    # flat ``obs`` block (groups then select it by name) OR inline as a group's value — an Obs block, which
    # lets a contract read as named blocks (ACTOR_OBS / CRITIC_OBS) instead of one long pool plus a separate
    # name list to keep in sync.
    obs_terms: dict[str, ObsTerm] = {}
    obs_refs: dict[str, Obs] = {}
    obs_order: list[ObsTerm] = []

    def _obs_term(ref: Obs) -> ObsTerm:
        tname = ref.name or ref.fn.__name__
        if tname in obs_terms:
            # The SAME entry in several groups is normal (the actor block is usually part of the critic's
            # input), and `terms()` hands out a per-group copy, so compare by VALUE not identity. Two
            # DIFFERENT declarations under one name is a contract bug — that still fails loud.
            assert obs_refs[tname] == ref, (
                f"duplicate obs term name {tname!r} with different declarations:\n"
                f"  {obs_refs[tname]}\n  {ref}")
            return obs_terms[tname]
        assert callable(ref.fn), f"obs term fn must be an imported flat function — got {ref.fn!r}"
        params = {k: _resolve_ref(v, refs) for k, v in ref.params.items()}
        _check_params(ref.fn, params, "obs")
        # Dashboard dim labels = the knob that NAMES the columns (joints/bodies), taken by name rather than
        # guessed from an arg position: one column per entry, in that list's order. A contract that states
        # `labels` wins — the escape hatch for a term whose layout no such knob spells out.
        labels = ref.labels or next(
            (params[k] for k in ("names", "bodies") if isinstance(params.get(k), list)), [])
        term = ObsTerm(name=tname, fn=ref.fn, params=params, scale=ref.scale,
                       dim_labels=list(labels), noise=ref.noise, unit=ref.unit, digits=ref.digits)
        obs_terms[tname] = term
        obs_refs[tname] = ref
        obs_order.append(term)
        return term

    for ref in terms(ts.obs, Obs):
        _obs_term(ref)
    obs: dict[str, list[ObsTerm]] = {}
    for gname, sel in ts.obs_groups.items():
        if sel == "all":
            obs[gname] = None                        # resolved below, once every inline block is registered
            continue
        if isinstance(sel, list) and sel and all(isinstance(x, str) for x in sel):
            for n in sel:                            # name-list form: every name must already be declared
                assert n in obs_terms, f"obs_groups[{gname}] unknown term {n!r} — valid: {sorted(obs_terms)}"
            obs[gname] = [obs_terms[n] for n in sel]
        else:                                        # block form (class or list of Obs entries)
            group = terms(sel, Obs)
            assert group, f"obs_groups[{gname!r}] must be 'all', a non-empty name list, or an Obs block"
            obs[gname] = [_obs_term(r) for r in group]
    for gname, sel in obs.items():                   # "all" = every declared term, in declaration order
        if sel is None:
            obs[gname] = list(obs_order)

    # obs history (per-group frame stacking): keys must be obs groups, values >= 1; unlisted groups default to 1.
    for gname, h in ts.obs_history_length.items():
        assert gname in obs, f"obs_history_length[{gname!r}] not an obs group — valid: {sorted(obs)}"
        assert h >= 1, f"obs_history_length[{gname!r}]={h} must be >= 1"
    obs_history_length = {g: int(ts.obs_history_length.get(g, 1)) for g in obs}

    # obs noise groups: names must be real groups, and a declared noise knob must actually be reachable from
    # one of them — a term carrying `noise=` that no noisy group holds is a silently dead knob, not a default.
    for gname in ts.obs_noise_groups:
        assert gname in obs, f"obs_noise_groups: {gname!r} is not an obs group — valid: {sorted(obs)}"
    _noisy = {t.name for g in ts.obs_noise_groups for t in obs[g]}
    for tname, term in obs_terms.items():
        assert term.noise is None or tname in _noisy, (
            f"obs term {tname!r} declares noise= but is in no group of obs_noise_groups="
            f"{list(ts.obs_noise_groups)} — the knob would do nothing"
        )

    # Reward terms are FLAT functions — `fn(env, **params)`, called directly each step (no factory, no
    # closure). So a Rew entry's knobs are ref-resolved into `params` instead of being applied to a factory.
    reward = []
    for ref in terms(ts.reward, Rew):
        params = {k: _resolve_ref(v, refs) for k, v in ref.params.items()}
        _check_params(ref.fn, params, "reward")
        reward.append(RewardTerm(name=ref.name or ref.fn.__name__,
                                 fn=ref.fn, weight=ref.weight, params=params))

    # GATE.joint_pose_tolerance says HOW exact the posture must be; the posture itself is stated ONCE, in
    # whichever place owns it — the reward term that pays for approaching it (`joint_pose` knob), so the gate
    # cannot disagree with what the dense signal rewards, or `GATE.joint_final_pose` when no term shapes
    # toward it. Two sources would let them drift apart, so two fail the same way none does.
    if gate is not None and gate.joint_pose_tolerance > 0.0:
        poses = {t.name: t.params["joint_pose"] for t in reward if "joint_pose" in t.params}
        if gate.joint_final_pose:
            poses["GATE.joint_final_pose"] = dict(gate.joint_final_pose)
        assert len(poses) == 1, (
            f"task '{name}': GATE.joint_pose_tolerance={gate.joint_pose_tolerance} needs the posture stated "
            f"EXACTLY once — as a `joint_pose` knob on one reward term, or as `GATE.joint_final_pose`. "
            f"Found {sorted(poses) or 'none'}. With none the gate would test an empty pose, i.e. drop the "
            f"condition from val/SR with no sign; with two they could ask for different postures.")
        pose = next(iter(poses.values()))
        # The joints must be COMMANDABLE: a pose bound on a joint the policy cannot drive is unrecoverable
        # once contact pushes it out of tolerance, so success would be blocked with no way back.
        commandable = {j for g_ in action.values() for j in g_.joints}
        unknown = [j for j in pose if j not in commandable]
        assert not unknown, (
            f"task '{name}': joint_pose lists {unknown}, which no action group commands — "
            f"commandable joints are {sorted(commandable)}")
        gate = gate.model_copy(update={"joint_final_pose": dict(pose)})

    # terminate/events/curriculum — all named (key for logging / curriculum refs); duplicate name fails loud.
    def _named(refs, kind, build):
        seen: set[str] = set()
        out = []
        for ref in refs:
            tname = ref.name or ref.fn.__name__
            assert tname not in seen, f"duplicate {kind} term name: {tname!r}"
            seen.add(tname)
            out.append(build(tname, ref))
        return out

    # Terminate terms are FLAT functions too — `fn(env, **params)`, called directly each step (no factory, no
    # closure), so a Done entry's knobs are ref-resolved into `params` exactly like a Rew entry's. `truncation`
    # is the entry's, not the term's: it tells the TRAINER how to value the done, not how to detect it.
    def _terminate_term(n, r):
        params = {k: _resolve_ref(v, refs) for k, v in r.params.items()}
        _check_params(r.fn, params, "terminate")
        return TerminateTerm(name=n, fn=r.fn, params=params, truncation=r.truncation)

    terminate = _named(terms(ts.terminate, Done), "terminate", _terminate_term)
    # Event terms are FLAT functions too — `fn(env, env_ids, **params)`, called directly (no factory, no
    # closure), so an Event entry's knobs are ref-resolved into `params` exactly like a Rew entry's. That is
    # what lets the CURRICULUM retune a knob live (it writes into this dict) without the term having to be a
    # class holding a mutable attribute. env + env_ids are the driver's two positionals, hence fixed=2.
    def _event_term(n, r):
        params = {k: _resolve_ref(v, refs) for k, v in r.params.items()}
        _check_params(r.fn, params, "event", fixed=2)
        return EventTerm(name=n, fn=r.fn, mode=r.mode, train_only=r.train_only, requires=r.requires,
                         params=params)

    events = _named(terms(ts.events, Event), "event", _event_term)
    curriculum = _named(terms(ts.curriculum, Curr), "curriculum",
                        lambda n, r: CurriculumTerm(name=n, fn=_build_fn(r, refs, "curriculum")))

    _report_ignored_effort(robot)

    return EnvSpec(
        name=ts.name,
        num_envs=num_envs if num_envs is not None else ts.num_envs,
        env_spacing=ts.env_spacing, episode_length_s=ts.episode_length_s,
        physics=ts.physics, robot=robot, robot_friction=robot_friction, objects=objects, fixtures=ts.fixtures,
        contact_params=contact_params,
        scene=stage,
        # TaskSpec authors init_pose in DEGREES (user-facing, readable); everything downstream
        # (parsers, backends, mjcf_prep fixed_pose, standalone) consumes EnvSpec radians. The task lists
        # only the ACTUATED joints; the equality-follower angles (waist dummy/upper, finger IP/DIP) are
        # auto-computed here from the MJCF <equality> polycoef — active followers spawn consistent, and
        # mask-0 followers weld at the kinematically-coupled angle (e.g. waist pitch held at 45°).
        camera=camera, goal=goal, gate=gate, action=action, action_delay=ts.action_delay,
        obs=obs, obs_history_length=obs_history_length, obs_noise_groups=list(ts.obs_noise_groups),
        reward=reward, terminate=terminate, events=events, curriculum=curriculum,
        overrides=ts.overrides, recipe=recipe_values(ts),
    )
