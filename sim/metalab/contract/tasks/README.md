# Writing a task contract — `sim/metalab/contract/tasks/`

**For a coding agent (Claude / Gemini / GPT) or a human.** A task is one declarative Python module
that builds a single `TASK = TaskSpec(...)`: names, tunables, and term references — **no logic**
(no `if`/`for`/computation inside the term lists). Obey the rules here and it resolves into a valid
`EnvSpec` and trains on **both** engines (newton, genesis) unchanged. Anything the schema rejects
**fails loud at import/load time** — no silent fallbacks.

**Why Python and not YAML?** So that every `fn=` is a real imported symbol: **Go-to-Definition**,
autocomplete, and "a typo is an import error" all work. That only holds if you follow rule 2 below —
pass the term's *symbol*, never a name string.

**Fastest path:** copy the closest shipped task and edit it.

| Task | What it is |
|---|---|
| `hammer_lift_teacher.py` | privileged-state teacher — the reference contract, minimal comments |
| `hammer_lift_student/` | asymmetric actor/critic, obs noise, geometry/mass DR, the fullest `overrides` block |
| `standalone/*.py` | scene-only contracts (no learning) for the Launchpad's Standalone mode |

- File `tasks/<task>.py` → the module must define `TASK: TaskSpec`. The trainer arg is its kebab stem
  (`hammer_lift_teacher.py` → `--task hammer-lift-teacher`; the loader maps `-`→`_` and imports
  `sim.metalab.contract.tasks.<task>`, falling back to `sim.metalab.contract.tasks.standalone.<task>`).
  A module may expose `build_task() -> TaskSpec` instead of `TASK`; the loader prefers it. That is the
  escape hatch for a contract that must compute something — a normal contract does not need it.
- Schema source of truth: `sim/metalab/contract/spec.py` (`TaskSpec`). Loader: `sim/metalab/contract/loader.py`.
- Units: **SI + radians** (never degrees), quaternions **`wxyz`** (identity `[1,0,0,0]`), world is z-up.
  **One exception:** `scene.robot.init_pose` is authored in **DEGREES** — the loader converts it to radians
  (a 44-joint pose table is unreadable in radians). Everything downstream of the loader is radians.

---

## The seven rules

1. **One module → one `TASK = TaskSpec(...)`.** Declarative only. File stem = task name (snake_case).
   `scene` is required and holds the world (ground, robot placement + init pose, objects, contact params,
   camera, goal); everything else hangs off `TASK` directly.
2. **`fn=` is the imported symbol, never a name string** — `from sim.metalab.terms.reward import lifting_reward`
   → `Rew(lifting_reward, weight=…)`. **obs, reward, terminate and events** are flat functions the driver
   calls directly (`fn(env, **params)`, events `fn(env, env_ids, **params)`); **curriculum** alone is still a
   *factory* called once at load, which returns the per-step `fn`. Import each `fn` from the package that
   **matches its category**
   (obs→`sim.metalab.terms.obs`, reward→`sim.metalab.terms.reward`, terminate→`sim.metalab.terms.terminate`,
   events→`sim.metalab.terms.events`, curriculum→`sim.metalab.terms.curriculum`). A term `fn` from the wrong category has
   the wrong return shape and fails at runtime — there is no registry catching it for you anymore, so
   the import must be right.
3. **`@refs` are the only "magic" strings** (see the table below). Everything else is a literal.
4. **Units:** SI + radians, quaternions `wxyz`, z-up — except `init_pose` (degrees, above).
5. **Data parts may be plain dicts or class blocks** — pydantic coerces + validates them into their spec
   types. Only the term lists need imported `fn` symbols.
6. **Need a term that doesn't exist?** Write the `fn`, re-export it, import it (see "When the contract
   isn't enough").
7. **Commented-out terms** are ready-to-enable alternatives. To turn one on, uncomment it **and** add its
   `fn` to the imports at the top.

---

## `@refs` — the only "magic"

A string starting with `@` is resolved by the loader; anything else is a literal. Unknown ref → error
listing valid keys. All three resolve against the **robot YAML**, which the task does not own — that is
why they are strings.

| Ref | Resolves to |
|---|---|
| `@joints.<group>` | joints of that action group (`robot.action_groups[<group>]`) |
| `@joints.<read-only group>` | joints of a `robot.joint_groups` entry — observable but NOT commandable |
| `@joints.ctrl` | **all** controlled joints, in action order |
| `@frames.<key>` | one MJCF body name (`robot.frames`, e.g. `@frames.palm`, `@frames.chest_origin`) |
| `@bodies.<key>` | a named body GROUP for terms taking a list (`@bodies.fingertips`) |

`@refs` live inside a term's `kwargs={...}` as strings — the loader resolves them into `term.params` (into
the factory call, for curriculum). (Plain Python literals you own, like a grasp offset, can also just be a
module-level python variable referenced directly — the contract is python, so a task-owned literal
needs no ref table.)

---

## Fields (`TaskSpec`)

**R** = required; otherwise the default applies. Unknown keys are rejected. Data-part values shown as
`{...}` / `[...]` are passed as Python dicts/lists (or class blocks) and validated by pydantic.

**Top level** — `name` **(R)**, `num_envs` (4096), `env_spacing` (1.5 — visualization grid pitch on both
engines: worlds overlap in physics and only the viewer tiles them), `episode_length_s` (10.0).

**`physics`** **(R)** — `hz` **(R)**, `substeps` (1), `decimation` (1 — physics steps per POLICY step, so the
policy runs at `hz/decimation`), `gravity` (`[0,0,-9.81]`), `solver_iterations` (100), `friction` (1.0),
`restitution` (0.0), `self_collision` (**true** — keep it on for hands so fingers don't interpenetrate; to
save time drop *unused* bodies via the robot `collision` mask, don't disable globally). Engine-specific
solver/contact knobs go in `overrides`, not here.

**`scene`** **(R)** — the world. Known keys only (anything else fails loud):

- `ground` (true), `ground_name` (`"plane"`) — infinite ground plane at z=0.
- `robot` **(R)** — `{name (R, → robot/<name>.yaml), base_pos (R), base_quat ([1,0,0,0]), fixed_base,
  init_pose}`. `init_pose` is a joint-name → **degrees** table: ACTIVE joints get the runtime spawn pose,
  MASK-0 joints get the weld angle (posed-then-welded). Equality FOLLOWERS (waist dummy/upper, finger
  IP/DIP) are auto-computed from the MJCF `<equality>` polycoef — do not list them.
- `objects` (`[]`) — a **list** (order = variant round-robin), each
  `{name, mass (R), asset:{mjcf:[…]}, parts:[{shape,size,pos,quat}], fixed, variants, init_pos,
  init_rpy|init_quat, randomize}`. `asset.mjcf` may list several files: env *i* gets variant *i % variants*.
  `parts` builds procedural geometry instead (the table is a `fixed` box). Authored right here — there is no
  object YAML.
- `contact_params` (`{}`) — `{group: {solref, solimp, solmix}}` per group key (`robot`, an object name, a
  fixture name). Shared knob; genesis implements the same solref/solimp math (`solmix` is newton-only).
- `camera` (optional) — `{eye, lookat, fov (40)}`. Used by eval recording and the viewers' framing.
- `goal` (optional) — `pos` **(R)**, `quat` (`[1,0,0,0]`), `keypoint_half_extent` (`[0.05,0.05,0.08]`),
  `success_tolerance` (0.05). Referenced by keypoint reward/terminate.
- `robot_friction` (optional) — global override; per-group friction DR is the `set_shape_friction` event.

**`fixtures`** (`[]`, top level) — `{name, kind: "box" (R), size, pos, quat}`. Legacy sibling of a `fixed`
object with `parts`; new contracts use `scene.objects` (that is what carries mass and contact params).

**`action`** — `{group: {joints, scale (1.0), ema_tau (null), mode (position_to_limits)}}`. Group name must
exist in `robot.action_groups`; `joints` may narrow it to a subset (omit for the whole group, omit `action`
entirely for every group the robot declares). Group order sets `@joints.ctrl` and the action-vector layout.
**`min_delay`/`max_delay` (0)** are declared on the ACTION **block**, not per group — the command delay is one
controller→robot link, so every group is written with the same per-env lag (redrawn at reset, reported by
the `action_delay` obs term).

**`gate`** — required **iff** `scene.goal` is set (the loader asserts the pair). The FINAL success bar, i.e.
what `val/SR` measures, authored as a `GATE = {...}` block between `EVENTS` and `TERMINATE`:
`success_tolerance` **(R)** [m], `hold_steps` (1), `hold_mode` (`"consecutive"` | `"cumulative"`),
`contact_count` (0), `force_threshold` (1e-3) [N], `palm_distance` (0.0) [m], `grasp_offset` (`[0,0,0]`) [m],
`lift_height` (0.0) [m — the `lifted` PHASE boundary, *not* a success condition). The curriculum ramps its own
`*_start` values UP TO these, so the last level IS the gate; `contact_count` must not exceed the robot's
`fingertips` count.

**Terms** — `obs`, `reward`, `terminate`, `events`, `curriculum`: lists of term refs (below).
`obs_groups`: `{group: "all" | [names] | <Obs block>}` (what each group exposes; teacher = `{actor: "all",
privileged: "all"}`, student wires its `ACTOR_OBS`/`CRITIC_OBS` blocks in directly).
`obs_history_length`: `{group: H}` (frame-stack, default 1 = off). `obs_noise_groups`: `[group…]` — the
groups whose terms get their `noise=` applied (IsaacLab's per-group `enable_corruption`), so an asymmetric
setup lists only `["actor"]` and the critic reads the same terms clean. Noise stays ON in eval/play.

**`overrides`** — `{engine: {...}}` per-engine physics knobs (the other engine ignores its block). The fullest
annotated newton set — solver/cone/tolerances, `nconmax`/`njmax`, `use_mujoco_contacts`, hull reduction
(`hull_maxvert`, `object_hull_maxvert`, `hull_reduce_above`), `eq_solref`/`eq_solimp` — is in
`hammer_lift_student/_base.py`; rationale in `sim/metalab/docs/01_engine_parity.md`.

> Network routing (which group the actor reads), PPO, `max_iterations`, wandb → **not here**, they live
> in `learning/rl/dexblind/<task>/experiment.py`.

---

## Term entry format

ONE entry type per category (all from `sim.metalab.contract.spec`), so every field on it applies:

```python
Obs(fn, scale=1.0, noise=None, **knobs)      Done(fn, truncation=False, **knobs)
Rew(fn, weight=…,              **knobs)      Event(fn, "reset"|"interval", **knobs)
                                             Curr(fn, **ctor_knobs)
```

`fn` **(R)** is the imported **symbol** (not a string). `weight` is **required** on `Rew` and keyword-only.
`truncation=True` on a `Done` = bootstrapping done (success / time-limit) — it says how the trainer VALUES
the done, not how the done is detected, which is why it sits on the entry and not in the function's knobs.
The knobs are `@ref`-resolved and become `term.params`, handed to the flat function every step — for **obs,
reward, terminate and events**; `Curr` alone still passes them to a factory at load. Those four therefore
take **named knobs only**; a positional arg fails loud, because a flat signature is readable only when every
knob is named (`Obs(obs.object_pose, "@frames.palm", [0,0,0])` said nothing about what either value was, and
`Done(terminate.object_below_height, 0.8)` did not say the 0.8 was a height). The loader checks the knob
names against the function signature at LOAD, so a typo or a missing required knob fails at launch instead
of on the first step — including a required knob the contract simply omitted, which is the half a
`**kwargs`-taking term used to slip past. An obs entry may also carry `noise=ObsNoise(std=…)` or
`ObsNoise(pos=…, rot=…)` — see `obs_noise_groups`; `noise` and `scale` are the driver's, never the
function's.

The term NAME comes from WHERE the entry is written. A category **block** — a class whose attributes are
entries (isaaclab's shape) — uses the attribute name, so a name is never written twice and the same `fn` can
appear under two names (arm/hand penalties). Definition order is preserved:

```python
from sim.metalab.contract import events, obs, reward, terminate
from sim.metalab.contract.spec import Done, Event, Obs, Rew, TaskSpec

class OBS:
    joint_pos = Obs(obs.joint_positions, names="@joints.ctrl")

class REWARD:
    goal_progress = Rew(reward.object_goal_keypoint_progress, weight=200.0)

class TERMINATE:
    object_below_height = Done(terminate.object_below_height, min_height=0.8)
    object_reached_goal = Done(terminate.object_reached_goal, truncation=True)

class EVENTS:
    robot_friction = Event(events.set_shape_friction, "reset", target="robot", mu_range=[1.5, 1.5])

TASK = TaskSpec(..., obs=OBS, reward=REWARD, terminate=TERMINATE, events=EVENTS)
```

A plain **list** of entries works too — then each carries its own `name=` (default `fn.__name__`) — and a
list may also hold other blocks, which are flattened in order (`obs=[ACTOR_OBS, CRITIC_OBS]`). Passing
`name=` inside a class block fails loud: the attribute already says the name, and two sources would drift.

---

## Available `fn`s (import from the category package)

Read each term's docstring in `sim/metalab/contract/<category>/common.py` for its exact args/shape. Each is
re-exported from the category's `__init__.py`, so `from sim.metalab.contract.<category> import <fn>` works and
Go-to-Definition jumps straight to the definition.

- **obs** (`from sim.metalab.terms.obs import …`) → flat `fn(env, **params)->(N,d)`: `prev_action_targets`,
  `last_action`, `action_delay`, `joint_positions`, `joint_velocities`, `joint_accelerations`,
  `joint_torque_obs`, `joint_pd_torque_obs`, `joint_gravcomp_torque_obs`, `joint_state`,
  `palm_pose_in_chest`, `body_pose_in_chest`, `body_linear_velocity`, `body_angular_velocity`,
  `object_pose`, `object_state_world`, `object_linear_velocity`,
  `object_angular_velocity`, `object_keypoints`, `goal_keypoints`, `object_lifed`,
  `object_goal_keypoint_success`, `lifted_object`, `body_contact_flags`, `fingertip_contact_steps`,
  `hand_object_force_magnitude`, `hand_contact_force`, `fingertip_relative_pos`, `fingertip_relative_pose`,
  `fingertip_relative_vel`, `closest_keypoint_max_dist`, `closest_fingertip_dist`, `episode_step`,
  `instantaneous_reward`, `gate_success`, `curriculum_success`, `curriculum_state`.
- **reward** (`from sim.metalab.terms.reward import …`) → flat `fn(env, **params)->(N,)`: `lifting_reward`,
  `fingertip_object_contact`, `palm_object_proximity`, `fingertip_object_proximity`,
  `object_goal_keypoint_progress`, `object_goal_keypoint_tracking`, `object_goal_reach_bonus`,
  `joint_vel_l1`, `joint_torque_penalty`.
  A reward term **never scales itself** — it returns a physical quantity (metres of progress), a `[0,1]`
  fraction, or a `0/1` event, and `weight` is the only magnitude. So reward weights are large (100/200/1000
  is normal). Per-term formulas + number tables: `reward/common.py`.
- **terminate** (`from sim.metalab.terms.terminate import …`) → flat `fn(env, **params)->(N,)bool`:
  `object_below_height`, `object_far_from_body`, `object_velocity_exceeded`,
  `table_fingertip_contact_force_exceeded`, `object_reached_goal`, `grasp_lost_after_lift`.
  Every knob is NAMED, same rule as reward/obs/events; `truncation` is the ENTRY's, not the function's — it
  says how the trainer values the done, not how the done is detected.
- **events** (`from sim.metalab.terms.events import …`) → flat `fn(env, env_ids, **knobs)`: `reset_object_pose`,
  `reset_joints_by_offset`, `set_shape_friction`, `randomize_rigid_body_mass`, `randomize_object_scale`,
  `randomize_fixed_base_root_height`, `apply_object_external_force`, `apply_object_external_torque`,
  `apply_object_external_force_when_lifted`.
  Every knob is NAMED (no positional term args), and a knob a curriculum ramps (`mu_scale`, `mass_scale`,
  an external load's `z_range`) is an ordinary parameter it retunes through `EventTerm.params` — same rule
  as reward/obs. `randomize_object_scale` is **newton-only** (genesis bakes geometry scale into the morph at
  build time), so the contract writes it as `Event(..., requires="object_scale")` — an engine without that
  capability drops the term, out loud, and runs the rest of the contract; the term's docstring states what
  each backend honours.
- **curriculum** (`from sim.metalab.terms.curriculum import …`) → class, `fn(env)->dict`:
  `goal_tolerance_curriculum`, `hammer_lift_success_curriculum`, `task_success_difficulty`.

The authoritative list is each category's `__all__` (in its `__init__.py`) — this table mirrors it.

---

## Fail-loud (rejected at import/load time)

Unknown field · **unknown `fn` symbol → `ImportError`/`NameError`** (typo caught the moment the module
imports — earlier than the old string registry) · non-callable `fn` · unknown `@ref` · term
arg/signature mismatch · duplicate term name · reward without `weight` · unknown `scene` key ·
`scene.robot` without `name` or `base_pos` · `init_pose`/masks naming inactive/unknown joints · `action`
group not in the robot (or `joints` naming one the group does not actuate, or duplicated) · a read-only
`robot.joint_groups` name colliding with an action group or `ctrl` · `obs_groups`/`obs_history_length`
naming a missing term/group · value constraints (`hz>0`, masks ∈{0,1}, `min_delay≤max_delay`, …) · module
missing `TASK`/`build_task` or `TASK` not a `TaskSpec` · **a flat term given positional `args`**
(obs/reward/terminate/events take named params only) · **`scene.goal` without `gate`** (or the reverse) ·
`gate.contact_count` above the robot's `fingertips` count · a curriculum `*_start` that does not ramp toward
its `GATE` counterpart · an obs term carrying `noise=` that no group in `obs_noise_groups` holds (a knob that
would do nothing).

**Not caught statically:** importing a term `fn` from the *wrong category* (rule 2). It has the wrong
return shape and blows up at runtime, not import — so get the import package right.

---

## When the contract isn't enough

The contract only *composes* existing pieces. Add the missing piece, then import and reference it.

- **New reward or obs `fn`** — write a flat function `def <name>(env, <knob>=<default>, …) -> (N,)` /
  `-> (N,d)` in `reward/common.py` / `obs/common.py` using `sim/metalab/api` primitives + `env` reads
  (**no engine import**), re-export it in the category `__init__.py` (`from .common import …` **and**
  `__all__`), then `import` it in your task `.py`. Return a physical quantity / fraction / event — never
  pre-scaled and, for obs, never self-corrupted (`scale`/`noise` belong to the entry). Name a joint-list
  knob `names` and a body-list knob `bodies`: those are what the dashboard reads for per-dim labels.
  An obs term must be STATELESS — it is evaluated once per obs GROUP, so a counter kept inside it would
  tick twice per step for anything the actor and critic share; put such state in the driver and read it.
- **New terminate/event `fn`** — write a flat function `def <name>(env, <knob>=<default>, …) -> (N,) bool`
  (terminate) or `def <name>(env, env_ids, <knob>=…)` (events), re-export it, import it. Match an existing
  term's return shape. A termination reads `env.lifted` / `env.reached_goal` rather than re-deriving a
  predicate the driver or a reward term already publishes. An event that needs a backend capability calls a
  `SimBackend` method (`env.set_object_scale`, `env.set_shape_friction`, …) — implement it on BOTH backends
  or make the unsupported one raise, never no-op.
- **Stateful terms** — a reward term keeps per-episode state through `env.buffer(key, shape, fill, dtype)`:
  the driver allocates it on first use and restores `fill` for the envs it resets, so there is no
  `init`/`reset` hook to write and the term stays a plain function (mutate the buffer **in place**). A
  curriculum retunes a reward term by writing `env.reward_terms[name].params[knob]`. Curriculum terms are
  the only classes left — they carry `__call__` + `jump_to_end` + logging; obs/reward/terminate/events are
  all flat functions.
- **New robot** — add `robot/<name>.yaml` + the MJCF under `sim/metalab/assets/…` (MJCF is the physics
  source of truth). See `robot/allex_right.yaml`. **Objects have no YAML** — declare them inline in
  `scene.objects` and point `asset.mjcf` at the file(s).
- **Per-engine physics** — `overrides={"<engine>": {...}}`, starting from `hammer_lift_student/_base.py`.
- **PPO/network/wandb** — `learning/rl/dexblind/<task>/experiment.py`. Run/eval via
  `learning/scripts/local/metalab_train.sh` / `metalab_eval.sh` (`sim/metalab/docs/10_metalab_tutorial.md`).

**Smoke test:** `learning/scripts/local/metalab_train.sh --sim newton --task <task> --num_envs 4 --viz gl`
(`--viz` takes a value: `none` | `gl` | `rtx`; the envs are drawn on an `env_spacing` grid and the dashboard's
env tabs move the camera between them).
