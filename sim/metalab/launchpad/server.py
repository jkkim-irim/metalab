#!/usr/bin/env python3
"""MetaLab Launchpad — standalone launcher/monitor web console (stdlib-only, no conda).

Boots an HTTP server, auto-discovers engines + tasks from the repo, and serves the launcher page
(engine/task/mode/knobs + a live, accurate command preview). Launch shell-outs to the maintained
scripts (metalab_train.sh / metalab_eval.sh / standalone.sh) — the single
source of truth — with a run registry, live log tail, and per-run Stop.

Runs under any python3 (stdlib only) — the Launchpad never needs an engine env; the launched
training scripts activate their own env. Discovery reflects the CURRENT layout:
  engines = sim/metalab/<name>/ spokes that ship a build_env server.py     -> genesis, newton
  tasks   = sim/metalab/contract/tasks/rl/ contracts (single source of truth) -> hammer-lift-teacher
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

# repo layout anchor: <repo>/sim/metalab/launchpad/server.py
SIM = Path(__file__).resolve().parents[2]          # <repo>/sim
REPO = SIM.parent                                   # <repo>
TASKS_DIR = SIM / "metalab" / "contract" / "tasks"
RL_DIR = TASKS_DIR / "rl"                  # Train/Eval contracts
STANDALONE_DIR = TASKS_DIR / "standalone"  # scene-only contracts (no learning)
TRAJ_DIR = SIM / "metalab" / "assets" / "data" / "spline"      # via-point CSV groups ('*_group' dirs)


# Engine spokes live under sim/metalab/backends/ (genesis/newton — the dirs that ship a server.py).
BACKENDS_DIR = SIM / "metalab" / "backends"


def discover_engines() -> list[str]:
    """sim/metalab/backends/<name>/ spokes that ship a build_env server.py AND a backend.py (genesis, newton).
    backend.py is the engine signature — a dir without it (e.g. a future non-engine folder) must not match."""
    if not BACKENDS_DIR.is_dir():
        return []
    return sorted(
        d.name for d in BACKENDS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "server.py").is_file() and (d / "backend.py").is_file()
    )


def _task_families() -> list[Path]:
    """tasks/rl/<family>/ or rl/<group>/<family>/ dirs carrying an __init__.py — one per task family.

    A dir whose packages are subdirs is a GROUP shelf (rl/manipulation/): its families are listed,
    never itself — mirrors sim/metalab/contract/loader.py:_rl_family_dir."""
    if not RL_DIR.is_dir():
        return []
    out = []
    for d in sorted(RL_DIR.iterdir()):
        if not (d.is_dir() and (d / "__init__.py").is_file()):
            continue
        subs = [s for s in sorted(d.iterdir()) if s.is_dir() and (s / "__init__.py").is_file()]
        out += subs or [d]
    return out


def discover_tasks() -> list[str]:
    """The Train/Eval TASK list: everything under sim/metalab/contract/tasks/rl/ — one entry per
    task-FAMILY folder (its bare name = the alias task its __init__ builds) plus any single-file
    rl/*.py contract. 'hammer_lift_teacher' -> 'hammer-lift-teacher' (underscores become hyphens). A
    family's recipes are the SECOND axis — see discover_task_recipes(). tasks/standalone/ is a separate
    shelf and never appears here; '_*.py' is a shared library, not a contract."""
    names = {f.stem for f in RL_DIR.glob("*.py") if not f.stem.startswith("_")} if RL_DIR.is_dir() else set()
    names |= {d.name for d in _task_families()}
    return sorted(n.replace("_", "-") for n in names)


def discover_task_recipes() -> dict[str, list[str]]:
    """task -> its recipe names — the Launchpad's recipe combobox, and the value of ``--recipe``.

    A recipe is ``tasks/rl/<family>/<family>_<recipe>.py``; the entry is the SUFFIX ('only-ycb'), since
    the family prefix is already the task. '_*.py' (the shared _base) is a library, not a recipe. The
    prefix is enforced, not just matched — mirrors sim/metalab/contract/loader.py:task_recipes, and a
    differently named file would otherwise vanish from this list instead of failing."""
    out = {}
    for d in _task_families():
        names = []
        for f in sorted(d.glob("*.py")):
            if f.stem.startswith("_"):
                continue
            assert f.stem.startswith(f"{d.name}_"), \
                f"recipe {f} must be named {d.name}_<recipe>.py (the family name is the prefix)"
            names.append(f.stem[len(d.name) + 1:].replace("_", "-"))
        out[d.name.replace("_", "-")] = names
    return out


def _standalone_groups() -> list[Path]:
    """tasks/standalone/<group>/ dirs carrying an __init__.py — one per standalone contract group."""
    d = STANDALONE_DIR
    return [g for g in sorted(d.iterdir()) if g.is_dir() and (g / "__init__.py").is_file()] if d.is_dir() else []


def _contracts_in(d: Path) -> list[str]:
    """The runnable contracts directly in ``d`` — '_*.py' is a library, not a contract."""
    return sorted(f.stem.replace("_", "-") for f in d.glob("*.py") if not f.stem.startswith("_"))


def discover_standalone_tasks() -> list[str]:
    """The Standalone TASK list: one entry per contract GROUP under tasks/standalone/ (manipulation,
    physics_test), plus any contract still sitting flat in tasks/standalone/. Scene-only, so it is kept
    out of discover_tasks and Train/Eval never list them.

    Same two axes as Train/Eval: the group is the first, its contracts the second — see
    discover_standalone_recipes(). A flat contract has no second axis, exactly like a single-file task."""
    d = STANDALONE_DIR
    return sorted([g.name.replace("_", "-") for g in _standalone_groups()] + (_contracts_in(d) if d.is_dir() else []))


def discover_standalone_recipes() -> dict[str, list[str]]:
    """standalone task -> its contracts — the Standalone recipe combobox. A GROUP lists the contracts it
    holds; a flat contract lists none (it IS the contract).

    Unlike a task family, the file names carry no group prefix: the group is a shelf, and the loader
    resolves a standalone contract by STEM wherever it is shelved, so ``--task hammer-lift`` keeps
    working. That is why the launched command sends the chosen CONTRACT as --task and no --recipe."""
    out = {g.name.replace("_", "-"): _contracts_in(g) for g in _standalone_groups()}
    d = STANDALONE_DIR
    return {**out, **{c: [] for c in (_contracts_in(d) if d.is_dir() else [])}}


# The task-contract sections that are TUNED between runs — the recipe a run was trained with.
_RECIPE_CLASSES = ("PHYSICS", "ACTION", "REWARD", "EVENTS", "TERMINATE", "GATE", "CURRICULUM")


def _task_recipe(task: str, recipe: str, mode: str) -> dict:
    """{class name: source text} for the contract sections above, AS OF THIS LAUNCH.

    ``git_sha`` cannot recover them — a `-dirty` tree pins nothing — so the text is copied into the
    run's registry entry and stays readable after the contract file moves on. Declaration order is
    kept; for a task FAMILY the sections are split across ``_base.py`` (physics / action / events /
    termination) and the chosen recipe (reward / gate / curriculum), read in that order. A standalone
    GROUP splits the same way — its ``_base.py`` holds the scene, the contract holds what differs."""
    stem = task.replace("-", "_")
    if mode == "standalone":
        contract = (recipe or task).replace("-", "_")
        group = STANDALONE_DIR / stem
        paths = ([group / "_base.py", group / f"{contract}.py"] if group.is_dir()
                 else [STANDALONE_DIR / f"{contract}.py"])
    elif (fam := next((d for d in [RL_DIR / stem] + sorted(RL_DIR.glob(f"*/{stem}")) if d.is_dir()), None)):
        rec = recipe.replace("-", "_")
        paths = [fam / "_base.py", fam / f"{stem}_{rec}.py"]
    else:
        paths = [RL_DIR / f"{stem}.py"]
    out: dict = {}
    for path in paths:
        src = path.read_text()
        lines = src.splitlines()
        out.update({n.name: "\n".join(lines[n.lineno - 1:n.end_lineno])
                    for n in ast.parse(src).body
                    if isinstance(n, ast.ClassDef) and n.name in _RECIPE_CLASSES})
    return out


def discover_traj_groups() -> list[dict]:
    """Trajectory via-point groups = every '*_group' dir under sim/metalab/assets/data/spline (recursive).
    Each drives the Standalone trajectory player (Play → cubic-Hermite from that dir's CSVs). Returns
    ``[{label, path}]`` sorted by path; ``path`` is repo-relative POSIX (the value handed to the runner),
    ``label`` is ``<parent>/<name>`` for readability (e.g. ``dumbbell/demo2_dumbbell_group``)."""
    if not TRAJ_DIR.is_dir():
        return []
    return [{"label": f"{d.parent.name}/{d.name}", "path": d.relative_to(REPO).as_posix()}
            for d in sorted(TRAJ_DIR.rglob("*_group")) if d.is_dir()]


def discover() -> dict:
    return {"engines": discover_engines(), "tasks": discover_tasks(),
            "task_recipes": discover_task_recipes(),
            "standalone_tasks": discover_standalone_tasks(),
            "standalone_recipes": discover_standalone_recipes(),
            "traj_groups": discover_traj_groups(), "repo": str(REPO)}


# ---------------------------------------------------------------------------
# Standalone dashboard preview (/simui) — the tab shows its window even with no run
# ---------------------------------------------------------------------------
# The dashboard page is owned by the sim runtime (``drive/dashboard_page.py``, a zero-import
# module) and normally served by the runner's own control_server. The Launchpad serves the SAME string so
# the Standalone tab is never an empty placeholder: with no run you get the real page (layout, tabs from
# the last run's channel set) with no data — enough to see a page edit without booting a sim.
# Loaded BY FILE PATH, not imported: the Launchpad must stay stdlib-only (that module's package pulls in
# torch/pydantic), and re-executing it per request means an edit lands on reload with no restart.
_DASH_PAGE_FILE = SIM / "metalab" / "drive" / "dashboard_page.py"


def _dashboard_page_module():
    spec = importlib.util.spec_from_file_location("_metalab_dashboard_page", _DASH_PAGE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                      # zero-import module: nothing but constants
    return mod


def standalone_preview_describe() -> dict:
    """``/simui/describe`` — the LAST standalone run's describe payload (cached by the runner at
    ``dashboard_page.DESCRIBE_CACHE_REL``) tagged ``preview: True``, or just the tag when no run has
    cached one yet. The tag tells the page to render tabs/series but open no stream and disable the
    controls (there is no runner to command)."""
    cache = REPO / _dashboard_page_module().DESCRIBE_CACHE_REL
    d = json.loads(cache.read_text()) if cache.is_file() else {}
    return {**d, "preview": True}


def creds() -> dict:
    """Is wandb logged in? The UI locks the run to ``--no_wandb`` when it is not.

    The SAME test learning/scripts/local/metalab_train.sh makes before a run (WANDB_MODE /
    WANDB_API_KEY / the ``api.wandb.ai`` entry ``wandb login`` writes to ~/.netrc), so the page and the
    script never disagree about whether a run can log.

    Its own endpoint, not /api/discover, which the launch path also calls. The page reads it once on
    load, so log in first or reload after ``wandb login``.
    """
    netrc = Path.home() / ".netrc"
    wandb_ok = bool(os.environ.get("WANDB_MODE") or os.environ.get("WANDB_API_KEY")) or (
        netrc.is_file() and "api.wandb.ai" in netrc.read_text(errors="ignore"))
    return {"wandb": wandb_ok}



def list_gpus() -> dict:
    """Local GPU list (cuda:0..N-1) for the device combo — shells nvidia-smi (Launchpad is stdlib-only)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=8)
        rows = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()] if out.returncode == 0 else []
    except Exception:
        rows = []
    devs = []
    for i, r in enumerate(rows):
        name = r.split(",", 1)[1].strip() if "," in r else ""
        devs.append({"device": f"cuda:{i}", "name": name})
    return {"devices": devs}


def list_dir(rel: str) -> dict:
    """One directory's contents for the checkpoint tree browser: subdirs (to descend) + *.pt (to pick).
    Confined to logs/ (no escaping via ..). Dirs newest-first (timestamped run names), files newest (mtime)."""
    base = (REPO / "logs").resolve()
    try:
        target = (REPO / rel).resolve() if rel else base
    except Exception:
        target = base
    if not (target == base or base in target.parents) or not target.is_dir():
        target = base
    dirs, files = [], []
    for p in target.iterdir():
        try:
            r = str(p.relative_to(REPO))
        except ValueError:
            continue
        if p.is_dir():
            dirs.append((p.name, r))
        elif p.suffix == ".pt":
            try:
                files.append((p.stat().st_mtime, p.name, r))
            except OSError:
                pass
    dirs.sort(key=lambda x: x[0].lower(), reverse=True)   # newest run (timestamp-named) first
    files.sort(reverse=True)                               # newest mtime first
    return {"cwd": str(target.relative_to(REPO)),
            "parent": (str(target.parent.relative_to(REPO)) if target != base else ""),
            "dirs": [{"name": n, "path": r} for n, r in dirs],
            "files": [{"name": n, "path": r} for _, n, r in files]}


# ── run launch + registry (P0 Step 3) ────────────────────────────────────────
RUNS_DIR = REPO / "logs" / "launchpad"      # under /logs/ (gitignored) — runtime state only
RUNS_JSONL = RUNS_DIR / "runs.jsonl"        # append-only launch log (persists across Launchpad restarts)
LOGS_DIR = RUNS_DIR / "runs"                # per-run stdout+stderr log files
PIDFILE = RUNS_DIR / "launchpad.pid"        # written by launchpad.sh --bg; cleared on Exit
_SCRIPT = {
    "train":      "learning/scripts/local/metalab_train.sh",
    "eval":       "learning/scripts/local/metalab_eval.sh",
    "standalone": "sim/metalab/standalone.sh",
}   # mode → the maintained script it shells out to.

_runs: dict = {}                             # run_id -> {proc, logf, meta}  (this session's launches)
_runs_lock = threading.Lock()
_httpd = None                                # set by serve(); used by request_shutdown()
_poll_seq = 0                                # bumped on each /api/runs poll — a browser heartbeat


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=str(REPO)) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def _int(v) -> str:
    v = str(v).strip()
    if v and not re.fullmatch(r"-?\d+", v):
        raise ValueError(f"정수여야 합니다: {v!r}")
    return v


def _label(v) -> str:
    """run-name label segment → validated, or "" (the launcher then omits the segment entirely).

    It lands inside the run name, which is at once a directory name and a W&B run name — so anything but
    path-safe characters is rejected here rather than producing a broken name later."""
    v = str(v or "").strip()
    if v and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", v):
        raise ValueError(f"run label 은 영문/숫자/._- 만 (공백·/ 불가): {v!r}")
    return v


def _device(v) -> str:
    v = str(v).strip()
    if v and not re.fullmatch(r"cpu|cuda(:\d+)?", v):
        raise ValueError(f"device 형식 오류(cpu|cuda|cuda:N): {v!r}")
    return v


def _build(params: dict) -> tuple[str, list, dict]:
    """SERVER-authoritative build (client preview is only UX): validate engine/task against
    discovery + numeric knobs, then assemble the maintained script's flags. No shell is used, so
    args are passed literally. Returns (script_relpath, flag_args, env_vars)."""
    mode = params.get("mode", "train")
    engine, task = params.get("engine"), params.get("task")
    recipe = (params.get("recipe") or "").strip()
    d = discover()
    if engine not in d["engines"]:
        raise ValueError(f"알 수 없는 엔진: {engine!r}")
    valid_tasks = d["standalone_tasks"] if mode == "standalone" else d["tasks"]
    if task not in valid_tasks:
        raise ValueError(f"알 수 없는 태스크: {task!r} (mode={mode})")
    # Second axis: a task FAMILY (train/eval) or a contract GROUP (standalone) is a shared core, not a
    # runnable contract, so it needs a recipe; a single-file contract must not be given one. Same rule
    # as the loader.
    avail = (d["standalone_recipes"] if mode == "standalone" else d["task_recipes"]).get(task, [])
    if avail and recipe not in avail:
        raise ValueError(f"태스크 {task!r} 의 레시피를 골라야 합니다 — {', '.join(avail)} "
                         f"(받은 값: {recipe!r})")
    if not avail and recipe:
        raise ValueError(f"태스크 {task!r} 는 단일 파일 계약이라 레시피를 받지 않습니다 ({recipe!r})")
    if mode not in _SCRIPT:
        raise ValueError(f"지원 안 함: mode={mode!r}")
    knob = params.get("knobs") or {}
    adv = params.get("adv") or {}
    # Robot drive mode — motor-space coupled PD (the robot YAML's control_mode) vs native per-joint PD.
    # Read at BUILD time by both engines (newton parser, genesis backend), default on, so it is set
    # explicitly on every run: the preview/registry then records which drive the run actually used.
    ctrl = params.get("ctrl", "motor")
    if ctrl not in ("motor", "joint"):
        raise ValueError(f"알 수 없는 구동 방식: {ctrl!r} (motor|joint)")
    env: dict = {"METALAB_MOTOR_COUPLING": "1" if ctrl == "motor" else "0"}
    flags = ["--sim", engine, "--task", task] + (["--recipe", recipe] if recipe else [])

    if mode == "standalone":   # env-only GUI run: --sim/--task only. Runs on the default (display) GPU so
        # The group is a UI shelf, not part of the contract's name — standalone.sh and the loader both
        # take the CONTRACT stem — so the chosen recipe IS the --task value and no --recipe is sent.
        return _SCRIPT[mode], ["--sim", engine, "--task", recipe or task], env

    if mode == "train":
        algo = params.get("algo", "ppo")
        if algo == "sapg":
            # SAPG: user picks envs-per-policy(block) + num-policies(blocks). Total num_envs = per_block *
            # blocks; blocks -> SAPG_BLOCKS env var (read by hammer_lift_teacher/experiment.py). ALGO=sapg.
            epb, nb = _int(knob.get("envs_per_block", "")), _int(knob.get("num_blocks", ""))
            if epb and nb:
                flags += ["--num_envs", str(int(epb) * int(nb))]
                env["ALGO"] = "sapg"
                env["SAPG_BLOCKS"] = nb
                ed = str(knob.get("embed_dim", "")).strip()
                if ed:
                    int(ed)  # validate — fail loud on non-numeric
                    env["SAPG_EMBED_DIM"] = ed
                sc = str(knob.get("ir_coef_scale", "")).strip()
                if sc:
                    float(sc)  # validate — fail loud on non-numeric
                    env["SAPG_IR_COEF_SCALE"] = sc
            for k, fl in (("max_iterations", "--max_iterations"), ("seed", "--seed")):
                v = _int(knob.get(k, ""))
                if v:
                    flags += [fl, v]
        else:
            for k, fl in (("num_envs", "--num_envs"), ("max_iterations", "--max_iterations"), ("seed", "--seed")):
                v = _int(knob.get(k, ""))
                if v:
                    flags += [fl, v]
        lbl = _label(knob.get("run_label", ""))     # empty → the launcher leaves the label segment out
        if lbl:
            flags += ["--run_label", lbl]
        dv = _device(knob.get("device", ""))
        if dv:
            flags += ["--device", dv]
        if adv.get("viz"):
            flags += ["--viz", "gl"]   # "3D 뷰어 창" → GL viewer; --viz needs a value (gl|none, newton도 rtx)
        for k, fl in (("no_wandb", "--no_wandb"), ("record", "--record")):
            if adv.get(k):
                flags.append(fl)
    else:  # eval (local) — some knobs are env vars (SEED, EPISODES); flags mirror metalab_eval.sh
        v = _int(knob.get("num_envs", ""))
        if v:
            flags += ["--num_envs", v]
        ck = str(knob.get("checkpoint", "")).strip()
        if ck:
            flags += ["--checkpoint", ck]
        if adv.get("viz"):
            flags.append("--viz")
        for k, name in (("seed", "SEED"), ("episodes", "EPISODES"), ("gpu", "GPU")):
            v = _int(knob.get(k, ""))
            if v:
                env[name] = v
    return _SCRIPT[mode], flags, env


def _display(env: dict, script: str, flags: list) -> str:
    """Repo-root-relative command string for the UI (copy-paste friendly), env prefix first."""
    pre = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    body = " ".join(shlex.quote(a) for a in [script, *flags])
    return f"{pre} {body}" if pre else body


def _append_registry(meta: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_JSONL, "a") as f:
        f.write(json.dumps(meta) + "\n")


def launch(params: dict) -> dict:
    """Spawn the maintained local script as a detached subprocess (its own process group, so it
    can be stopped as a whole), stream stdout+stderr to a per-run log, and persist to the registry.
    Executed as `bash <abs script>` (robust to cwd/exec-bit); the displayed command stays relative."""
    script, flags, env = _build(params)
    now = datetime.now()
    run_id = now.strftime("%Y%m%d-%H%M%S") + f"-{params['engine']}-{params['task']}-{params['mode']}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logpath = LOGS_DIR / f"{run_id}.log"
    logf = open(logpath, "wb")
    full_env = dict(os.environ)
    full_env.update(env)
    exec_argv = ["bash", str(REPO / script), *flags]
    proc = subprocess.Popen(exec_argv, cwd=str(REPO), env=full_env, stdout=logf,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    meta = {"run_id": run_id, "mode": params["mode"],
            "engine": params["engine"], "task": params["task"],
            "task_recipe": (params.get("recipe") or ""),
            # the form state as clicked — lets a card click restore the whole launcher (algo/knobs/adv)
            # for one-click reproduce/re-launch, not just show the command.
            "algo": params.get("algo", "ppo"), "ctrl": params.get("ctrl", "motor"),
            "knobs": params.get("knobs") or {}, "adv": params.get("adv") or {},
            # the tuned contract sections VERBATIM — what this run was actually trained with, kept
            # readable after the file (and the -dirty sha) moves on.
            "recipe": _task_recipe(params["task"], params.get("recipe") or "", params["mode"]),
            "cmd": _display(env, script, flags), "argv": [script, *flags], "env": env, "pid": proc.pid,
            "git_sha": _git_sha(), "log": str(logpath),
            "started_at": now.astimezone(timezone.utc).isoformat(timespec="seconds"), "status": "running"}
    with _runs_lock:
        _runs[run_id] = {"proc": proc, "logf": logf, "meta": meta}
    _append_registry(meta)
    print(f"[launchpad] launch {run_id} pid={proc.pid}: {meta['cmd']}", flush=True)
    return meta


def _append_update(run_id: str, **kv) -> None:
    """Persist a status change for an existing card — an append-only ``{"run_id", "update": {...}}``
    line that ``list_runs`` folds over the original meta (the registry itself is never rewritten
    in place; compaction happens at hub start, see ``_reset_state``)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_JSONL, "a") as f:
        f.write(json.dumps({"run_id": run_id, "update": kv}) + "\n")


def _read_registry() -> tuple[dict, dict]:
    """(metas, updates) from the registry: original launch metas by run_id + folded update overlays."""
    metas: dict = {}
    updates: dict = {}
    if RUNS_JSONL.exists():
        for line in RUNS_JSONL.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                m = json.loads(line)
            except Exception:
                continue
            if "update" in m:
                updates.setdefault(m.get("run_id", ""), {}).update(m["update"])
            elif m.get("run_id"):
                metas[m["run_id"]] = m
    return metas, updates


def list_runs() -> list:
    """Merge the persisted registry (past sessions) + its status updates with this session's live
    process status, and drop dismissed cards. Newest first."""
    metas, updates = _read_registry()
    out: dict = {}
    for rid, m in metas.items():
        m = dict(m)
        m.update(updates.get(rid, {}))
        m["live"] = False
        out[rid] = m
    with _runs_lock:
        for rid, r in _runs.items():
            rc = r["proc"].poll()
            m = dict(r["meta"])
            m.update(updates.get(rid, {}))
            if rc is None:
                m["status"] = "running"
            else:
                # launcher finished — persist the terminal status ONCE so a later console session shows
                # done/failed (not the pid-gone "ended" fallback).
                new = "done" if rc == 0 else "failed"
                m["status"] = new
                if "status" not in updates.get(rid, {}):
                    _append_update(rid, status=new)
            m["returncode"] = rc
            m["live"] = rc is None
            out[rid] = m
    out = {rid: m for rid, m in out.items() if not m.get("dismissed")}
    # Non-live cards left "running" by a previous console session: the run is detached (the console that
    # spawned it is gone), so re-derive from the pid — alive & ours → still running, else → ended
    # (persist once). This is what makes "close the console, reopen it" show the truth.
    for rid, m in out.items():
        if m.get("live") or m.get("status") != "running":
            continue
        pid = int(m.get("pid") or 0)
        if pid > 0 and _pid_alive(pid) and _is_our_run_pid(pid):
            continue                                # genuinely still running (detached, reparented to init)
        m["status"] = "ended"
        _append_update(rid, status="ended", ended_by="process gone")
    return sorted(out.values(), key=lambda m: m.get("started_at", ""), reverse=True)



def dismiss_run(run_id: str) -> dict:
    """Remove a card from the list (persisted). Refused while the run is still ``running`` — Stop it
    first (or let the reconciler flip it) so a dismissed card can never hide a live training."""
    for m in list_runs():
        if m.get("run_id") == run_id:
            if m.get("status") == "running":
                return {"ok": False, "run_id": run_id, "error": "run is running — Stop it first"}
            _append_update(run_id, dismissed=True)
            return {"ok": True, "run_id": run_id}
    return {"ok": False, "run_id": run_id, "error": "run not found"}


def stop_all() -> dict:
    """Terminate EVERY running launched process (train / eval / any future sim task): SIGINT the whole
    process tree (Ctrl-C, clean → wandb marks the run 'killed'), SIGKILL survivors after the grace. The
    Launchpad itself stays up. Pids are captured up front (reparent-safe). Covers runs a previous Launchpad session
    launched too (via the registry, guarded by /proc cmdline)."""
    roots = _run_root_pids()
    threading.Thread(target=lambda: _kill_roots(roots), daemon=True).start()
    return {"stopped": len(roots)}




def stop_run(run_id: str) -> dict:
    """Stop ONLY the given run, leaving siblings alone: kills that run's process tree
    (SIGINT→SIGKILL, sweep=False)."""
    root, meta = None, None
    with _runs_lock:
        r = _runs.get(run_id)
        if r:
            root, meta = r["proc"].pid, r["meta"]
    if meta is None:
        for m in list_runs():
            if m.get("run_id") == run_id:
                meta = m
                if m.get("pid"):
                    root = int(m["pid"])
                break
    if meta is None and root is None:
        return {"ok": False, "stopped": 0, "run_id": run_id, "error": "run not found"}
    if root is not None:
        threading.Thread(target=lambda: _kill_roots([root], sweep=False), daemon=True).start()
    # persist "stopped" so the card stays stopped across console restarts (otherwise a restart
    # re-derives it from the now-dead pid as a generic "ended").
    if meta:
        _append_update(run_id, status="stopped")
    return {"ok": True, "stopped": 1, "run_id": run_id}


def reset_run(run_id: str) -> dict:
    """Reset a STANDALONE run's sim to its initial state ON DEMAND — SIGUSR1 the runner, whose handler
    calls backend.reset_idx (contract init pose, NO domain randomization). Only sim.metalab.runtime.standalone
    installs a SIGUSR1 handler; train/eval do not, so aiming this at them finds no runner (no-op). Root
    pid = this session's tracked Popen else the registry pid; the runner = the descendant whose argv is
    the standalone module."""
    root = None
    with _runs_lock:
        r = _runs.get(run_id)
        if r:
            root = r["proc"].pid
    if root is None:
        for m in list_runs():
            if m.get("run_id") == run_id and m.get("pid"):
                root = int(m["pid"]); break
    if root is None:
        return {"ok": False, "reset": 0, "run_id": run_id, "error": "run not found"}
    n = 0
    for pid in _descendants(root):
        try:
            cmd = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        except OSError:
            continue
        if b"sim.metalab.runtime.standalone" in cmd:
            _sig(pid, signal.SIGUSR1); n += 1
    return {"ok": n > 0, "reset": n, "run_id": run_id,
            **({} if n else {"error": "standalone runner not found (only Standalone runs support reset)"})}


def read_log(run_id: str, max_bytes: int = 200_000) -> str:
    logpath = None
    with _runs_lock:
        r = _runs.get(run_id)
        if r:
            logpath = r["meta"]["log"]
    if logpath is None:
        for m in list_runs():
            if m["run_id"] == run_id:
                logpath = m.get("log")
                break
    if not logpath or not Path(logpath).is_file():
        return ""
    return Path(logpath).read_bytes()[-max_bytes:].decode("utf-8", "replace")






def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _is_our_run_pid(pid: int) -> bool:
    """True if pid is a live process WE launched (cmdline is one of our local scripts) — guards
    against acting on a stale/reused pid from the registry."""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return any(m in cmd for m in _SCRIPT_MARKERS)


def _descendants(root: int) -> list:
    """root + every descendant pid (walked via /proc ppid). The trainer nests through conda + an
    inner setsid'd shell into its OWN session+group, so killpg on the launched pid's group misses the
    real trainer; setsid keeps the ppid link, so a ppid walk reaches the whole tree."""
    ppid_of = {}
    for e in os.listdir("/proc"):
        if not e.isdigit():
            continue
        try:
            data = (Path("/proc") / e / "stat").read_bytes()
        except OSError:
            continue
        rp = data.rfind(b")")                       # comm is parenthesized and may contain spaces
        parts = data[rp + 2:].split() if rp >= 0 else []
        if len(parts) >= 2:
            ppid_of[int(e)] = int(parts[1])
    kids: dict = {}
    for pid, pp in ppid_of.items():
        kids.setdefault(pp, []).append(pid)
    out, seen, stack = [], set(), [root]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def _sig(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _sigpg(pgid: int, sig: int) -> None:
    """Signal a whole process GROUP. launch() starts each run with start_new_session=True, so the run's
    pid is a session+group leader and every child that did NOT setsid into its own group shares that
    pgid — killpg reaches them ATOMICALLY and is reparent-proof (pgid survives the parent dying), which a
    ppid-based _descendants walk is not. Standalone's runner stays in this group (no inner setsid), so
    this is what reliably reaps it; the trainer setsid's into its own group and is handled by the walk."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


_SIM_MARKERS = (b"learning.train", b"learning.eval", b"sim.metalab.runtime.standalone")
# Launcher script basenames a run's argv can carry. FUNCTIONAL, not cosmetic: the pid matchers below
# identify our runs by these, so a script rename that misses this tuple silently breaks both run
# detection and the Stop button. One tuple, two call sites (_alive_pid / _our_run_procs).
_SCRIPT_MARKERS = (b"metalab_train.sh", b"metalab_eval.sh", b"standalone.sh")


def _our_run_procs() -> list:
    """Live pids whose argv is one of our launched sim commands — matched on EXACT argv fields (a
    NUL-separated token equal to 'learning.train'/'learning.eval'/'sim.metalab.runtime.standalone', or ending
    in one of _SCRIPT_MARKERS), so a shell that merely mentions the string is NOT
    matched. Catches the real trainer/runner even after it setsid'd into its own session and its
    launcher ancestors already exited."""
    out = []
    for e in os.listdir("/proc"):
        if not e.isdigit():
            continue
        try:
            fields = (Path("/proc") / e / "cmdline").read_bytes().split(b"\x00")
        except OSError:
            continue
        for f in fields:
            if f in _SIM_MARKERS or f.endswith(_SCRIPT_MARKERS):
                out.append(int(e))
                break
    return out


def _proc_is_trainer(pid: int) -> bool:
    """True if pid's cmdline is the TRAINER/RUNNER (learning.train/eval or the standalone runner) — the
    process that owns the wandb run AND its sim-server. Stop SIGINTs THIS (clean KeyboardInterrupt →
    wandb 'killed' + it tears down its own server), never the server directly (racing the server → the
    trainer's next RPC raises RuntimeError, not KeyboardInterrupt → run shows 'crashed')."""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return any(mk in cmd for mk in _SIM_MARKERS)


def _kill_roots(roots: list, grace: float = 15.0, sweep: bool = True) -> None:
    """Terminate a run's process tree, SIGINT-ing the TRAINER/RUNNER FIRST so it exits cleanly.

    Phase 1 (≤ grace): SIGINT ONLY trainer/runner procs (learning.train/eval/standalone). Each catches
      KeyboardInterrupt, finishes wandb as 'killed', and tears down its OWN sim-server. Re-walk so late
      spawns are caught; return early once the whole tree is gone.
    Phase 2 (after grace): SIGKILL everything left (a stuck server, the shells, an unresponsive trainer);
      reparent-proof via pid accumulation + process-group kill. The Launchpad is never in `roots`, so untouched.
    (Previously the WHOLE tree was SIGINT'd at once — the server died before the trainer's next RPC step,
    which then raised RuntimeError instead of KeyboardInterrupt, so the run showed 'crashed'.)"""
    sent_int: set = set()
    t = 0.0
    while t < grace:                               # Phase 1: interrupt the trainer/runner, let it wind down
        tree = set(_our_run_procs()) if sweep else set()
        for root in roots:
            tree.update(_descendants(root))
        alive = {p for p in (tree | sent_int) if _pid_alive(p)}
        if not alive:
            return                                 # wound down cleanly (trainer finished + reaped children)
        for p in alive:
            if p not in sent_int and _proc_is_trainer(p):
                _sig(p, signal.SIGINT)             # SIGINT the trainer/runner ONLY (once per pid)
                sent_int.add(p)
        time.sleep(0.5)
        t += 0.5
    for _ in range(8):                             # Phase 2: grace elapsed → hard-kill whatever remains
        tree = set(_our_run_procs()) if sweep else set()
        for root in roots:
            tree.update(_descendants(root))
        alive = {p for p in (tree | sent_int) if _pid_alive(p)}
        if not alive:
            return
        for p in alive:
            _sig(p, signal.SIGKILL)
        for root in roots:
            _sigpg(root, signal.SIGKILL)
        time.sleep(0.5)


def _run_root_pids() -> list:
    """Root pids of still-alive runs: this session's Popens plus any registry pids still alive and
    still one of our scripts (covers runs a previous Launchpad session launched)."""
    roots = []
    with _runs_lock:
        procs = [r["proc"] for r in _runs.values()]
    for p in procs:
        if p.poll() is None:
            roots.append(p.pid)
    if RUNS_JSONL.exists():
        for line in RUNS_JSONL.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(json.loads(line).get("pid") or 0)
            except Exception:
                continue
            if pid > 0 and _pid_alive(pid) and _is_our_run_pid(pid):
                roots.append(pid)
    return list(dict.fromkeys(roots))


def request_shutdown() -> None:
    """Stop the Launchpad SERVER ONLY — launched runs KEEP RUNNING. The launchpad is a launcher+monitor,
    NOT the owner of the training processes: each run is a detached session (start_new_session=True) that
    outlives the console, so closing/refreshing/exiting the console must NEVER kill a run. Runs are
    terminated only by an explicit per-run Stop. On restart the runs' cards + status are re-derived from
    their pids (list_runs), so a reconnected console shows exactly what is still alive."""
    try:
        if PIDFILE.is_file() and PIDFILE.read_text().strip() == str(os.getpid()):
            PIDFILE.unlink()
    except Exception:
        pass
    if _httpd is not None:
        threading.Thread(target=_httpd.shutdown, daemon=True).start()


def _bump_poll() -> None:
    global _poll_seq
    _poll_seq += 1


def _closing() -> None:
    """A tab/window sent the 'closing' beacon (pagehide). Shut the Launchpad down like Exit — UNLESS
    the page's polling resumes within the grace window (a refresh reconnects fast, a real close
    does not). This distinguishes closing the window (→ shut down) from a page reload (→ keep)."""
    seq0 = _poll_seq
    threading.Timer(3.5, lambda: request_shutdown() if _poll_seq == seq0 else None).start()


def _chrome_bin() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            return p
    return None


def _open_browser(url: str) -> None:
    """Open the Launchpad as a standalone, maximized Chrome *app window* (no tabs/address bar) in its
    dedicated, isolated profile — its own Chrome instance, decoupled from the user's normal Chrome.
    This is the terminal/foreground path; the app-icon path (launchpad.sh --bg, which passes
    --no-browser here) opens the same kind of window. Falls back to the OS default browser when
    Chrome is absent. Best-effort; never raises — the printed URL is the fallback (e.g. headless)."""
    chrome = _chrome_bin()
    if chrome:
        xdg = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
        profile = os.environ.get("METALAB_HUB_BROWSER_PROFILE") or os.path.join(xdg, "metalab-hub", "browser")
        try:
            subprocess.Popen(
                [chrome, f"--app={url}", f"--user-data-dir={profile}",
                 "--no-first-run", "--no-default-browser-check", "--start-maximized"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return
        except Exception:
            pass                          # fall through to the default browser
    try:
        webbrowser.open(url, new=1)
    except Exception:
        pass


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):        # silence per-request stderr logging
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(json.dumps(obj).encode(), "application/json", code)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return {}

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/" or u.path.startswith("/index"):
                self._send(_PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/simui":                # the page's endpoints are relative → needs the slash
                self.send_response(301)
                self.send_header("Location", "/simui/")
                self.end_headers()
            elif u.path == "/simui/":               # offline Standalone dashboard preview (no run needed)
                self._send(_dashboard_page_module().PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/simui/describe":
                self._json(standalone_preview_describe())
            elif u.path == "/api/discover":
                self._json(discover())
            elif u.path == "/api/creds":
                self._json(creds())
            elif u.path == "/api/gpus":
                self._json(list_gpus())
            elif u.path == "/api/ls":
                self._json(list_dir((parse_qs(u.query).get("dir") or [""])[0]))
            elif u.path == "/api/runs":
                _bump_poll()                        # browser heartbeat (see _closing)
                self._json({"runs": list_runs()})
            elif u.path == "/api/log":
                rid = (parse_qs(u.query).get("run_id") or [""])[0]
                self._json({"run_id": rid, "log": read_log(rid)})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            body = self._body()
            try:
                if u.path == "/api/launch":
                    self._json({"ok": True, "run": launch(body)})
                elif u.path == "/api/stop-all":
                    self._json({"ok": True, **stop_all()})
                elif u.path == "/api/stop":
                    self._json(stop_run(str(body.get("run_id", ""))))
                elif u.path == "/api/reset":
                    self._json(reset_run(str(body.get("run_id", ""))))
                elif u.path == "/api/dismiss":
                    self._json(dismiss_run(str(body.get("run_id", ""))))
                elif u.path == "/api/shutdown":
                    request_shutdown()
                    self._json({"ok": True, "bye": True})
                elif u.path == "/api/closing":       # tab/window closed → shut down (unless a refresh)
                    _closing()
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 400)

    return Handler


def _reset_state() -> None:
    """Compact the registry at start — do NOT kill anything. ALL runs are preserved: a launched run is a
    detached session that OUTLIVES the launchpad, so its card + log must survive a restart, and
    ``list_runs`` re-derives the true status from the pid. COMPACT: fold the append-only status updates
    into each meta (baking status) and drop dismissed cards + their logs, so the registry never grows
    unbounded. (Previously cards were dropped here because Exit had killed them — Exit no longer kills.)"""
    try:
        keep: list[str] = []
        kept_logs: set[str] = set()
        metas, updates = _read_registry()
        for rid, m in metas.items():
            m = dict(m)
            m.update(updates.get(rid, {}))
            if m.get("dismissed"):                    # dismissed → gone for good (+ its log below)
                continue
            keep.append(json.dumps(m))
            if m.get("log"):
                kept_logs.add(Path(m["log"]).name)
        RUNS_JSONL.write_text("".join(ln + "\n" for ln in keep))
        if LOGS_DIR.exists():
            for f in LOGS_DIR.glob("*.log"):
                if f.name not in kept_logs:
                    f.unlink(missing_ok=True)
    except OSError:
        pass


def serve(host: str = "127.0.0.1", port: int = 8780, open_browser: bool = True) -> None:
    global _httpd
    _reset_state()   # fresh server → clean terminal (no past-session logs)
    try:
        _httpd = ThreadingHTTPServer((host, port), _make_handler())
    except OSError:                         # requested port taken → OS-assigned free port
        _httpd = ThreadingHTTPServer((host, 0), _make_handler())
    _httpd.daemon_threads = True
    bound = _httpd.server_address[1]
    url = f"http://{host}:{bound}"
    d = discover()
    print(f"[launchpad] MetaLab Launchpad → {url}", flush=True)
    print(f"[launchpad] engines={d['engines']}  tasks={d['tasks']}", flush=True)
    print("[launchpad] Ctrl-C 또는 웹의 Exit 버튼으로 종료.", flush=True)
    if open_browser:
        _open_browser(url)
    try:
        _httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[launchpad] 종료.", flush=True)
    finally:
        _httpd.shutdown()


# Self-contained launcher page — no external fetches (inline CSS/JS). Palette mirrors
# sim/__docs + telemetry.py. Launch is not wired yet (shows a notice); command preview
# is live and accurate to the maintained scripts.
_PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MetaLab Launchpad</title>
<style>
:root{--ground:#eef1f4;--surface:#fff;--surface2:#e6ebef;--ink:#131a21;--soft:#4c5964;--faint:#7a8894;
--line:#d3dae0;--line2:#b7c1c9;--accent:#0f7d8c;--accent-ink:#0a5a66;--signal:#b06a2c;--signal-ink:#8f5115;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif}
@media(prefers-color-scheme:dark){:root{--ground:#0b1015;--surface:#121a21;--surface2:#18222b;--ink:#dde5eb;
--soft:#96a4af;--faint:#67757f;--line:#233039;--line2:#35434e;--accent:#34aebd;--accent-ink:#6fcdd9;--signal:#cf925f;--signal-ink:#dda574}}
*{box-sizing:border-box}html{font-size:130%}body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:19.5px;line-height:1.55}
.wrap{width:100%;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;border-bottom:2px solid var(--ink);padding:12px 20px;flex:none}
.main{flex:1;display:flex;min-height:0;overflow:hidden}
.left{flex:1 1 62%;min-width:23rem;overflow-y:auto;padding:16px 20px 44px}
.right{flex:1 1 38%;min-width:0;display:flex;flex-direction:column;background:#0b1015;border-left:1px solid var(--line)}
@media(max-width:860px){.wrap{height:auto;overflow:visible}.main{flex-direction:column;overflow:visible}.left{max-width:none;flex:none;overflow:visible}.right{flex:none;height:46vh}}
.tabs{display:flex;gap:2px;flex:none;padding:0 20px;border-bottom:1px solid var(--line)}
.tab{font:inherit;font-size:.9rem;padding:9px 16px;border:none;background:none;color:var(--soft);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab.on{color:var(--accent-ink);border-bottom-color:var(--accent);font-weight:700}
.tdot{display:none;width:7px;height:7px;border-radius:50%;background:#1a9a5f;margin-left:7px;vertical-align:middle}
.tdot.live{display:inline-block}
.pane[hidden]{display:none!important}   /* beat #pane-telem's display:flex (ID > class+attr specificity) */
#pane-telem,#pane-sim{flex:1;min-height:0;display:flex;flex-direction:column;padding:12px 20px 20px}
#telemempty{margin:auto;color:var(--faint);text-align:center;max-width:34rem;line-height:1.7}
#telemframe,#simframe{flex:1;width:100%;border:1px solid var(--line2);border-radius:8px}
#telemframe{background:#0b1015}#simframe{background:var(--ground)}
#telemframe[hidden],#simframe[hidden]{display:none}
/* run cards (click to tail that process) + checkpoint picker */
.runcards{display:flex;flex-direction:column;gap:6px;margin-top:14px}
.rcard{display:flex;align-items:center;gap:9px;padding:8px 11px;border:1px solid var(--line);border-radius:7px;background:var(--surface);cursor:pointer;font-size:.82rem}
.rcard:hover{border-color:var(--line2)}
.rcard.on{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
.rdot{width:8px;height:8px;border-radius:50%;flex:none;background:var(--faint)}
.rdot.running{background:#1a9a5f;box-shadow:0 0 0 3px rgba(26,154,95,.22)}
.rdot.failed{background:#c0392b}
.rlabel{flex:1;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rst{font-family:var(--mono);font-size:.68rem;color:var(--soft)}
.rx{flex:none;font-family:var(--mono);font-size:.72rem;color:var(--faint);padding:0 3px;cursor:pointer;border-radius:4px}
.rx:hover{color:#c0392b;background:var(--surface2)}
/* run-card filters (mode subtabs + status chips) + date group headers */
.rcfilter{margin-top:16px;display:flex;flex-direction:column;gap:8px}
.rcseg{display:inline-flex;border:1px solid var(--line2);border-radius:7px;overflow:hidden;align-self:flex-start;flex-wrap:wrap}
.rcseg button{font-family:var(--sans);font-size:.78rem;padding:5px 13px;border:none;background:var(--surface);color:var(--soft);cursor:pointer;border-right:1px solid var(--line)}
.rcseg button:last-child{border-right:none}
.rcseg button.on{background:var(--accent);color:#fff;font-weight:700}
.rcchips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-family:var(--mono);font-size:.68rem;padding:3px 10px;border:1px solid var(--line);border-radius:999px;background:var(--surface);color:var(--soft);cursor:pointer}
.chip.on{border-color:var(--accent);color:var(--accent-ink);font-weight:700}
.rcdate{font-family:var(--mono);font-size:.66rem;color:var(--faint);margin:13px 0 5px;letter-spacing:.04em;border-bottom:1px solid var(--line);padding-bottom:3px}
/* right pane = [터미널]/[실행 정보] tabs; only one pane shows at a time */
.rtab{font-family:var(--mono);font-size:.72rem;padding:5px 13px;border:none;background:none;color:#96a4af;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-8px}
.rtab.on{color:#6fcdd9;border-bottom-color:#34aebd;font-weight:700}
#runinfo{flex:1;min-height:0;overflow:auto;padding:12px 14px;font-size:.78rem;color:var(--ink);background:var(--surface)}
#runinfo[hidden],#logtext[hidden]{display:none}
.ri-empty{color:var(--faint)}
.ri-hd{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);margin-bottom:8px}
.ri-copy{font-family:var(--mono);font-size:.64rem;padding:3px 10px;border:1px solid var(--line2);border-radius:6px;background:var(--surface2);color:var(--ink);cursor:pointer}
.ri-cmd{font-family:var(--mono);font-size:.72rem;background:#0b1015;color:#dde5eb;border:1px solid var(--line);border-radius:6px;padding:9px 11px;white-space:pre-wrap;word-break:break-all;margin-bottom:8px}
.ri-grid{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-family:var(--mono);font-size:.72rem}
.ri-grid .k{color:var(--faint);white-space:nowrap}.ri-grid .v{color:var(--ink);word-break:break-all}
.ri-grid .v b{color:var(--accent-ink)}
/* recipe = the run's tuned contract sections, one collapsible block per class */
.ri-rc{margin-top:14px}
.ri-rc details{border:1px solid var(--line);border-radius:6px;margin-bottom:5px;background:var(--surface2)}
.ri-rc summary{font-family:var(--mono);font-size:.7rem;padding:5px 10px;cursor:pointer;color:var(--accent-ink);letter-spacing:.03em}
.ri-rc details[open] summary{border-bottom:1px solid var(--line)}
.ri-rc pre{font-family:var(--mono);font-size:.7rem;line-height:1.45;margin:0;padding:9px 11px;overflow-x:auto;color:#dde5eb;background:#0b1015;border-radius:0 0 5px 5px}
.rnone{color:var(--faint);font-size:.8rem;padding:8px 2px}
.ckwrap{flex:1 1 100%}
.ckrow{display:flex;gap:8px}.ckrow input{flex:1}
#ckbrowse{font:inherit;font-size:.82rem;padding:6px 12px;border:1px solid var(--line2);border-radius:7px;background:var(--surface);color:var(--ink);cursor:pointer;white-space:nowrap}
#cklist{max-height:58vh;overflow:auto;margin-top:12px;border:1px solid var(--line);border-radius:8px}
.ckitem{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer;font-size:.82rem}
.ckitem:hover{background:var(--surface2)}.ckfile{font-family:var(--mono)}
.ckcwd{padding:8px 12px;font-family:var(--mono);font-size:.72rem;color:var(--accent-ink);background:var(--surface2);border-bottom:1px solid var(--line);position:sticky;top:0}
h1{font-size:1.2rem;margin:0;font-weight:750;letter-spacing:-.01em}
h1 span{color:var(--accent-ink)}
.repo{margin-left:auto;font-family:var(--mono);font-size:.68rem;color:var(--faint)}
#exit{flex:none;background:#c0392b;color:#fff;border:none;border-radius:7px;font-family:var(--sans);font-size:.82rem;font-weight:700;padding:7px 16px;cursor:pointer}
#exit:hover{background:#a93226}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;z-index:100;padding:16px}
.modal[hidden]{display:none}   /* beats .modal's display:flex so the `hidden` attr actually hides it */
.modalbox{background:var(--surface);border:1px solid var(--line2);border-top:5px solid #c0392b;border-radius:12px;padding:24px 26px;max-width:32rem;width:100%;box-shadow:0 14px 44px rgba(0,0,0,.45)}
.modalwarn{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:.82rem;font-weight:700;letter-spacing:.12em;color:#c0392b}
.modalwarn .ic{font-size:1.3rem}
.modaltitle{font-size:1.2rem;font-weight:750;margin-top:10px;color:var(--ink)}
.modalmsg{font-size:.98rem;color:var(--soft);margin-top:10px;line-height:1.55}
.modalmsg b{color:#c0392b}
.modalbtns{display:flex;justify-content:flex-end;gap:10px;margin-top:22px}
.mbtn-cancel{background:var(--surface2);color:var(--ink);border:1px solid var(--line2);border-radius:8px;padding:10px 20px;font-size:.92rem;cursor:pointer;font-family:var(--sans)}
.mbtn-danger{background:#c0392b;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:.92rem;font-weight:700;cursor:pointer;font-family:var(--sans)}
.mbtn-danger:hover{background:#a93226}
.live{font-family:var(--mono);font-size:.7rem;color:var(--faint)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--faint);margin-right:5px;vertical-align:middle}
.dot.on{background:var(--accent)}
.field{margin-top:18px}
.field>label{display:block;font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin-bottom:7px}
.seg{display:inline-flex;border:1px solid var(--line2);border-radius:7px;overflow:hidden;flex-wrap:wrap}
.seg button{font-family:var(--sans);font-size:.88rem;padding:8px 18px;border:none;background:var(--surface);color:var(--soft);cursor:pointer;border-right:1px solid var(--line)}
.seg button:last-child{border-right:none}
.seg button.on{background:var(--accent);color:#fff;font-weight:700}
.seg button:disabled{opacity:.4;cursor:not-allowed}
.seg.mode button.on{background:var(--signal)}
select.selbox{font-family:var(--sans);font-size:.92rem;padding:8px 12px;border:1px solid var(--line2);border-radius:7px;background:var(--surface);color:var(--ink);min-width:15rem}
select.selbox:disabled{opacity:.45;cursor:not-allowed}
.selrow{display:flex;gap:10px;flex-wrap:wrap}
.knobs{display:flex;flex-wrap:wrap;gap:12px}
.knob{display:flex;flex-direction:column;gap:5px}
.knob label{font-family:var(--mono);font-size:.62rem;letter-spacing:.05em;text-transform:uppercase;color:var(--faint)}
.knob input{font-family:var(--mono);font-size:.9rem;padding:7px 10px;border:1px solid var(--line2);border-radius:6px;background:var(--surface);color:var(--ink);width:8rem}
.knob input.wide{width:22rem;max-width:100%}
details{margin-top:16px;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:0 14px}
summary{cursor:pointer;font-family:var(--mono);font-size:.76rem;color:var(--accent-ink);padding:11px 0;font-weight:600;list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸ ";color:var(--faint)}
details[open] summary::before{content:"▾ "}
.adv{padding:2px 0 14px}
.checks{display:flex;flex-wrap:wrap;gap:14px 18px;font-size:.86rem;color:var(--soft)}
.checks label{display:flex;align-items:center;gap:7px;cursor:pointer}
.cmdlabel{font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin:22px 0 7px}
.cmdprev{background:#0b1015;border:1px solid var(--line);border-radius:8px;padding:14px 15px;font-family:var(--mono);font-size:.8rem;line-height:1.7;color:#dde5eb;overflow-x:auto;white-space:pre-wrap;word-break:break-all}
.cmdprev .fl{color:#6fcdd9}.cmdprev .st{color:#dda574}.cmdprev .env{color:#7fd39a}
.launch{margin-top:16px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.btn-go{font-family:var(--sans);font-size:1rem;font-weight:750;padding:11px 28px;border:none;border-radius:8px;background:var(--accent);color:#fff;cursor:pointer}
.btn-stop{font-family:var(--sans);font-size:1rem;font-weight:750;padding:11px 22px;border:none;border-radius:8px;background:var(--signal);color:#fff;cursor:pointer}
.btn-stop:hover{background:var(--signal-ink)}
.btn-stop:disabled{opacity:.4;cursor:default}
.btn-reset{font-family:var(--sans);font-size:1rem;font-weight:750;padding:11px 22px;border:none;border-radius:8px;background:var(--accent);color:#fff;cursor:pointer}
.btn-reset:hover{background:var(--accent-ink)}
.btn-reset[hidden]{display:none}
.btn-ghost{font-family:var(--mono);font-size:.78rem;padding:10px 15px;border:1px solid var(--line2);border-radius:8px;background:var(--surface);color:var(--soft);cursor:pointer}
.toast{font-family:var(--mono);font-size:.74rem;color:var(--signal-ink)}
.hint{font-family:var(--mono);font-size:.7rem;color:var(--faint);margin-top:6px}
.loghd{display:flex;align-items:center;gap:10px;padding:9px 14px;background:#18222b;border-bottom:1px solid var(--line);flex:none}
.loghd .lt{flex:1;min-width:0;font-family:var(--mono);font-size:.7rem;color:#96a4af;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.loghd .close{cursor:pointer;color:#96a4af;font-family:var(--mono);font-size:.72rem;margin-left:6px}
.zoom{margin-left:auto;display:inline-flex;align-items:center;background:#0b1015;border:1px solid #35434e;border-radius:6px;overflow:hidden}
.zoom button{background:transparent;border:none;border-right:1px solid #233039;color:#96a4af;font-family:var(--mono);font-size:.72rem;padding:3px 10px;cursor:pointer;line-height:1.3}
.zoom button:last-child{border-right:none}
.zoom button:hover{background:#18222b;color:#dde5eb}
#zoomreset{min-width:3.6rem;text-align:center}
#logtext{margin:0;padding:12px 14px;flex:1;min-height:0;overflow:auto;font-family:var(--mono);font-size:.58rem;line-height:1.5;color:#dde5eb;white-space:pre-wrap;word-break:break-all}
</style></head><body><div class="wrap">
<header>
  <h1>MetaLab <span>Launchpad</span></h1>
  <span class="live"><span class="dot" id="dot"></span><span id="livetxt">connecting…</span></span>
  <span class="repo" id="repo"></span>
  <button id="exit" title="Launchpad 서버 종료">Exit</button>
</header>

<div class="tabs">
  <button class="tab on" data-tab="console">콘솔</button>
  <button class="tab" data-tab="sim">Standalone<span class="tdot" id="simdot"></span></button>
  <button class="tab" data-tab="telem">RL<span class="tdot" id="telemdot"></span></button>
</div>

<div class="main pane" id="pane-console">
<div class="left">
<div class="field"><label>1 · Backends</label><div class="seg" id="engines"></div></div>
<div class="field"><label>1b · 로봇 구동 방식 (Motor = 모터공간 coupled PD · Joint = 관절 native PD)</label>
  <div class="seg" id="ctrl">
    <button data-v="motor" class="on" title="robot_model.json 의 모터 게인·토크한계로 coupled PD (실기 펌웨어 미러)">모터 · Motor</button>
    <button data-v="joint" title="METALAB_MOTOR_COUPLING=0 — 로봇 YAML joint_mode_param 의 kp/kv 로 관절별 대각 PD">관절 · Joint</button>
  </div></div>
<div class="field"><label>2 · Mode</label>
  <div class="seg mode" id="mode">
    <button data-v="train" class="on">학습 · Train</button>
    <button data-v="eval">검증 · Eval</button>
    <button data-v="standalone">시뮬레이션 · Standalone</button>
  </div></div>
<div class="field"><label>3 · Task · Recipe (자동 탐색: sim/metalab/contract/tasks/rl/ 아래 family 폴더 = task, 그 안의 *.py = recipe)</label>
  <div class="selrow">
    <select class="selbox" id="task"></select>
    <select class="selbox" id="recipe"></select>
  </div></div>
<div class="field" id="algofield"><label>3a · 학습 알고리즘 (Algorithm)</label>
  <select class="selbox" id="algo">
    <option value="ppo">PPO</option>
    <option value="sapg">SAPG · Split-and-Aggregate</option>
  </select></div>
<div class="field"><label>4 · 노브 (기본값 프리필 — 그냥 두고 실행해도 됨)</label>
  <div class="knobs" id="knobs"></div></div>

<details id="advwrap"><summary>고급 설정</summary><div class="adv"><div class="checks" id="adv"></div>
  <div class="hint" id="advhint"></div></div></details>

<div class="cmdlabel">실행될 명령 (미리보기 · 그대로 복사 가능)</div>
<div class="cmdprev" id="cmd">…</div>

<div class="launch">
  <button class="btn-go" id="launch">▶ Launch</button>
  <button class="btn-stop" id="stop" disabled>■ Stop</button>
  <button class="btn-reset" id="reset" hidden>↺ Reset Simulator</button>
  <button class="btn-ghost" id="copy">명령 복사</button>
  <span class="toast" id="toast"></span>
</div>
<div class="hint">엔진·태스크·작업·노브를 바꾸면 위 명령이 실시간으로 갱신됩니다. Launch 하면 오른쪽에 라이브 로그가 뜨고, Stop 은 실행 중인 프로세스를 Ctrl-C 로 종료합니다.</div>
<div class="rcfilter">
  <div class="rcseg" id="rcmode">
    <button data-v="all" class="on">전체</button>
    <button data-v="local">학습·검증</button>
    <button data-v="standalone">Standalone</button>
  </div>
  <div class="rcchips" id="rcstatus"></div>
</div>
<div class="runcards" id="runcards"></div>
</div><!-- .left -->
<div class="right">
  <div class="loghd">
    <button class="rtab on" data-rt="term">터미널</button>
    <button class="rtab" data-rt="info">실행 정보</button>
    <span class="lt" id="logtitle"></span>
    <span class="zoom" id="zoomwrap"><button id="zoomout" title="축소">−</button><button id="zoomreset" title="100%로 리셋">100%</button><button id="zoomin" title="확대">+</button></span></div>
  <pre id="logtext">Launch 하면 현재 실행 중인 프로세스의 라이브 로그가 여기에 표시됩니다. (전체 로그는 wandb)</pre>
  <div id="runinfo" hidden><div class="ri-empty">run 카드를 선택하면 실행 정보(모드·파라미터·git sha)가 여기에 표시됩니다.</div></div>
</div>
</div><!-- .main / pane-console -->

<div class="pane" id="pane-sim" hidden>
  <iframe id="simframe"></iframe>
</div>

<div class="pane" id="pane-telem" hidden>
  <div id="telemempty">telemetry 는 <b>local 에서 GUI(--viz gl)</b> 를 활성화 할 경우에만 사용 가능합니다.<br>
    로컬 학습/검증을 "3D 뷰어 창" 체크(= --viz gl)로 실행하면 여기에 대시보드가 바로 표시됩니다.</div>
  <iframe id="telemframe" hidden></iframe>
</div>

<div id="ckmodal" class="modal" hidden>
  <div class="modalbox" style="max-width:46rem">
    <div class="modaltitle">체크포인트 선택 <span style="font-weight:400;color:var(--faint);font-size:.85rem">— 📁 폴더 클릭해 들어가고 · 📄 .pt 클릭해 선택</span></div>
    <div id="cklist">로딩…</div>
    <div class="launch" style="margin-top:14px"><button class="btn-ghost" id="ckcancel">닫기</button></div>
  </div>
</div>

<div id="exitmodal" class="modal" hidden>
  <div class="modalbox" style="border-top-color:var(--accent)">
    <div class="modaltitle">런치패드 콘솔 종료</div>
    <div class="modalmsg">콘솔(웹 서버)만 종료됩니다. <b style="color:var(--accent-ink)">실행 중인 학습·검증·시뮬레이션은 계속 돌아갑니다</b> — 각 run 은 콘솔과 분리된 프로세스입니다. 종료하려면 해당 run 카드를 눌러 <b style="color:var(--accent-ink)">Stop</b> 하세요. 다시 열면 카드가 현재 상태(실행중/종료)로 복원됩니다.</div>
    <div class="modalbtns">
      <button id="exitcancel" class="mbtn-cancel">Cancel</button>
      <button id="exitconfirm" class="mbtn-danger" style="background:var(--accent)">콘솔만 종료</button>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const ANSICOL={30:"#7a8894",31:"#d9795e",32:"#4fbf88",33:"#dda574",34:"#6fa8dc",35:"#c58fd6",36:"#6fcdd9",37:"#dde5eb",90:"#67757f",91:"#e39177",92:"#79d3a3",93:"#e6bb7a",94:"#8fb8e6",95:"#d3a3e0",96:"#8fd7e0",97:"#ffffff"};
function ansiToHtml(raw){                         // render terminal SGR (bold/color); strip other escapes
  let s=String(raw).replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g,"").replace(/\r(?!\n)/g,"\n").replace(/\r/g,"");
  const parts=s.split(/\x1b\[([0-9;]*)m/); let out="",bold=false,color=null;
  for(let i=0;i<parts.length;i++){
    if(i%2===0){let t=parts[i].replace(/\x1b\[[0-9;?]*[A-Za-z]/g,"").replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g,""); if(!t)continue;
      const sty=(bold?"font-weight:700;":"")+(color?"color:"+color+";":"");
      out+= sty?`<span style="${sty}">${esc(t)}</span>`:esc(t);
    }else{parts[i].split(";").forEach(cs=>{const c=parseInt(cs||"0",10);
      if(c===0){bold=false;color=null;}else if(c===1)bold=true;else if(c===22)bold=false;
      else if(c===39)color=null;else if(ANSICOL[c])color=ANSICOL[c];});}
  } return out;
}
let DESC=null;
// ctrl = robot drive mode: "motor" (motor-space coupled PD, the robot YAML's control_mode) or "joint"
// (native per-joint diagonal PD) — carried to every run as METALAB_MOTOR_COUPLING=1|0.
const state={engine:null,task:null,recipe:"",mode:"train",algo:"ppo",ctrl:"motor",knob:{},adv:{}};
// Is wandb logged in (/api/creds)? Optimistic until the probe answers — a failed probe must never lock a
// working setup out; the launched script still fails loudly if it was wrong.
let CREDS={wandb:true};

// mode-dependent knob + advanced specs (matched to the maintained local scripts)
const SPEC={
  train:{
    script:"learning/scripts/local/metalab_train.sh",
    knobs:[["num_envs","4096","--num_envs"],["max_iterations","5000","--max_iterations"],
           ["seed","42","--seed"],["device","cuda:0","--device"],["run_label","","--run_label"]],
    adv:[["viz","--viz","3D 뷰어 창 (엔진 GUI)"],["no_wandb","--no_wandb","wandb 끄기"],
         ["record","--record","체크포인트별 녹화"]],
    advhint:""
  },
  eval:{
    script:"learning/scripts/local/metalab_eval.sh",
    knobs:[["num_envs","1","--num_envs"],["checkpoint","","--checkpoint"]],
    adv:[["viz","--viz","3D 뷰어 관전 (녹화 자동 off)"]],
    envknobs:[["seed","42","SEED"],["episodes","","EPISODES"],["gpu","","GPU"]],
    advhint:"checkpoint 비우면 해당 task의 최신 로컬 체크포인트 자동 선택. seed/episodes 는 환경변수로 전달."
  },
  standalone:{
    script:"sim/metalab/standalone.sh",
    knobs:[],
    adv:[],
    advhint:"환경만 생성해 GL 뷰어로 실행 — 정책·학습 없음(초기 포즈 유지). num_envs=1·GL·디스플레이 GPU 고정(compute+뷰어 같은 GPU, 노브 없음). Stop/Ctrl-C 로 종료."
  }
};

function curSpec(){return SPEC[state.mode];}
function renderKnobs(){
  const sp=curSpec();
  let html=sp.knobs.map(([k,def])=>{
    if(k==="num_envs" && state.mode==="train" && state.algo==="sapg")
      return `<div class="knob"><label>envs / policy (블록당)</label><input id="k_envs_per_block" value="${esc(state.knob.envs_per_block??"512")}"></div>`
           + `<div class="knob"><label>num policies</label><input id="k_num_blocks" value="${esc(state.knob.num_blocks??"4")}"></div>`
           + `<div class="knob"><label>entropy scale</label><input id="k_ir_coef_scale" value="${esc(state.knob.ir_coef_scale??"0")}"></div>`
           + `<div class="knob"><label>embed dim</label><select id="k_embed_dim" class="selbox">${[32,16,8].map(d=>`<option value="${d}"${(state.knob.embed_dim??"32")===String(d)?" selected":""}>${d}</option>`).join("")}</select></div>`;
    if(k==="device") return `<div class="knob"><label>device (GPU)</label><select id="k_device" class="selbox"></select></div>`;
    if(k==="run_label") return `<div class="knob"><label>run label (선택)</label><input id="k_run_label" placeholder="(비우면 생략)" value="${esc(state.knob[k]??def)}"></div>`;
    if(k==="checkpoint") return `<div class="knob ckwrap"><label>checkpoint</label><div class="ckrow"><input id="k_checkpoint" placeholder="(비우면 최신 자동)" value="${esc(state.knob[k]??def)}"><button type="button" id="ckbrowse">📁 찾아보기</button></div></div>`;
    return `<div class="knob"><label>${k}</label><input id="k_${k}" value="${esc(state.knob[k]??def)}"></div>`;
  }).join("");
  if(sp.envknobs) html+=sp.envknobs.map(([k,def])=>
    k==="gpu"
      ? `<div class="knob"><label>GPU (device)</label><select id="k_gpu" class="selbox"></select></div>`
      : `<div class="knob"><label>${k}</label><input id="k_${k}" value="${esc(state.knob[k]??def)}"></div>`).join("");
  $("knobs").innerHTML=html;
  [...sp.knobs,...(sp.envknobs||[])].forEach(([k])=>{
    const el=$("k_"+k); if(!el) return;
    if(k==="device"){el.onchange=e=>{state.knob[k]=e.target.value;render();};loadGpus();}
    else if(k==="gpu"){el.onchange=e=>{state.knob[k]=e.target.value;render();};loadEvalGpus();}
    else el.oninput=e=>{state.knob[k]=e.target.value;render();};});
  ["envs_per_block","num_blocks","ir_coef_scale"].forEach(k=>{const el=$("k_"+k);if(el)el.oninput=e=>{state.knob[k]=e.target.value;render();};});
  {const ed=$("k_embed_dim");if(ed)ed.onchange=e=>{state.knob.embed_dim=e.target.value;render();};}
  const cb=$("ckbrowse"); if(cb) cb.onclick=openCkModal;
}
// Eval GPU combo: same GPU list, but the value is the physical index (0,1) for metalab_eval.sh's GPU env var.
function loadEvalGpus(){
  const sel=$("k_gpu"); if(!sel) return;
  fetch("/api/gpus").then(r=>r.json()).then(d=>{
    const list=(d.devices&&d.devices.length)?d.devices:[{device:"cuda:0",name:""}], cur=state.knob.gpu||"0";
    sel.innerHTML=list.map(g=>{const idx=g.device.replace("cuda:","");
      return `<option value="${esc(idx)}" ${idx===cur?"selected":""}>${esc(g.device)}${g.name?" · "+esc(g.name):""}</option>`;}).join("");
    state.knob.gpu=sel.value;render();
  }).catch(()=>{sel.innerHTML='<option value="0">cuda:0</option>';});
}
function loadGpus(){
  const sel=$("k_device"); if(!sel) return;
  fetch("/api/gpus").then(r=>r.json()).then(d=>{
    const list=(d.devices&&d.devices.length)?d.devices:[{device:"cuda:0",name:""}], cur=state.knob.device||"cuda:0";
    sel.innerHTML=list.map(g=>`<option value="${esc(g.device)}" ${g.device===cur?"selected":""}>${esc(g.device)}${g.name?" · "+esc(g.name):""}</option>`).join("");
    state.knob.device=sel.value;render();
  }).catch(()=>{sel.innerHTML='<option value="cuda:0">cuda:0</option>';});
}
function openCkModal(){ $("ckmodal").hidden=false; loadCkDir("logs"); }
// folder-tree browse under logs/: click a 📁 to descend, ⬆ to go up, 📄 .pt to select.
function loadCkDir(dir){
  const box=$("cklist"); box.innerHTML="로딩…";
  fetch("/api/ls?dir="+encodeURIComponent(dir)).then(r=>r.json()).then(d=>{
    let h=`<div class="ckcwd">📂 ${esc(d.cwd)}</div>`;
    if(d.cwd!=="logs") h+=`<div class="ckitem ckdir" data-d="${esc(d.parent)}">⬆ .. (상위 폴더)</div>`;
    h+=(d.dirs||[]).map(x=>`<div class="ckitem ckdir" data-d="${esc(x.path)}">📁 ${esc(x.name)}</div>`).join("");
    h+=(d.files||[]).map(x=>`<div class="ckitem ckfile" data-p="${esc(x.path)}">📄 ${esc(x.name)}</div>`).join("");
    if(!(d.dirs&&d.dirs.length)&&!(d.files&&d.files.length)) h+='<div class="rnone">(하위 폴더·.pt 없음)</div>';
    box.innerHTML=h;
    box.querySelectorAll(".ckdir").forEach(it=>it.onclick=()=>loadCkDir(it.dataset.d));
    box.querySelectorAll(".ckfile").forEach(it=>it.onclick=()=>{state.knob.checkpoint=it.dataset.p;
      const inp=$("k_checkpoint");if(inp)inp.value=it.dataset.p;$("ckmodal").hidden=true;render();});
  }).catch(()=>{box.innerHTML="조회 실패";});
}
function renderAdv(){
  const sp=curSpec();
  // wandb not connected: the run can only go with --no_wandb (the script refuses otherwise), so the
  // box is forced on and locked instead of letting the launch die at the script's login check.
  const wlock=!CREDS.wandb;
  if(wlock&&sp.adv.some(([k])=>k==="no_wandb")) state.adv.no_wandb=true;
  $("adv").innerHTML=sp.adv.map(([k,fl,lab])=>{
    const lock=wlock&&k==="no_wandb";
    return `<label${lock?' title="wandb 미연동 — wandb login 후 새로고침하면 해제됩니다"':""}>`+
           `<input type="checkbox" id="a_${k}" ${state.adv[k]?"checked":""}${lock?" disabled":""}>`+
           ` ${esc(lab)}${lock?" (wandb 미연동 — 고정)":""}</label>`;
  }).join("");
  sp.adv.forEach(([k])=>{$("a_"+k).onchange=e=>{state.adv[k]=e.target.checked;render();};});
  $("advhint").textContent=sp.advhint||"";
}
function buildCmd(){
  const sp=curSpec(), e=state.engine, t=state.task;
  if(!e||!t) return {env:[],parts:[]};
  const val=k=>{const el=$("k_"+k);return el?el.value.trim():"";};
  // Standalone: the group is a UI shelf, so the chosen contract IS --task and no --recipe is sent
  // (standalone.sh and the loader both take the contract stem). Mirrors _build on the server.
  let parts=[[sp.script],["--sim",e],["--task",state.mode==="standalone"?(state.recipe||t):t]];
  if(state.recipe&&state.mode!=="standalone") parts.push(["--recipe",state.recipe]);
  // Drive mode first (applies to every mode — the engines read it at build time, default 1).
  let env=[["METALAB_MOTOR_COUPLING",state.ctrl==="joint"?"0":"1"]];
  if(state.mode==="train"){
    if(state.algo==="sapg"){
      const epb=val("envs_per_block")||"512", nb=val("num_blocks")||"4", sc=val("ir_coef_scale"), ed=val("embed_dim")||"32";
      env.push(["ALGO","sapg"],["SAPG_BLOCKS",nb],["SAPG_EMBED_DIM",ed]); if(sc) env.push(["SAPG_IR_COEF_SCALE",sc]);
      parts.push(["--num_envs", String((parseInt(epb)||0)*(parseInt(nb)||0))]);
      sp.knobs.forEach(([k,,fl])=>{if(k!=="num_envs"){const v=val(k);if(v)parts.push([fl,v]);}});
    }else{
      sp.knobs.forEach(([k,,fl])=>{const v=val(k);if(v)parts.push([fl,v]);});
    }
    sp.adv.forEach(([k,fl])=>{if(state.adv[k])parts.push(k==="viz"?["--viz","gl"]:[fl]);});
  }else{
    (sp.envknobs||[]).forEach(([k,,name])=>{const v=val(k);if(v)env.push([name,v]);});
    sp.knobs.forEach(([k,,fl])=>{const v=val(k);if(v)parts.push([fl,v]);});
    sp.adv.forEach(([k,fl])=>{if(state.adv[k])parts.push([fl]);});
  }
  return {env,parts};
}
function render(){
  const {env,parts}=buildCmd();
  // colored HTML preview
  let html=env.map(([n,v])=>`<span class="env">${esc(n)}=${esc(v)}</span> `).join("");
  html+=parts.map((p,i)=>{
    if(i===0) return `<span>${esc(p[0])}</span>`;
    if(p.length===2) return ` <span class="fl">${esc(p[0])}</span> <span class="st">${esc(p[1])}</span>`;
    return ` <span class="fl">${esc(p[0])}</span>`;
  }).join("");
  $("cmd").innerHTML=html||"…";
  // plain text for copy
  const flat=(env.map(([n,v])=>n+"="+v).join(" ")+(env.length?" ":"")+
    parts.map(p=>p.join(" ")).join(" ")).trim();
  $("cmd").dataset.plain=flat;
}
function selectEngine(e){state.engine=e;
  [...$("engines").children].forEach(b=>b.classList.toggle("on",b.dataset.v===e));render();}
// motor|joint — mutually exclusive (exactly one 'on'), like the engine/mode/target segments.
function selectCtrl(c){state.ctrl=c;
  [...$("ctrl").children].forEach(b=>b.classList.toggle("on",b.dataset.v===c));render();}
// task combobox source: Standalone lists the tasks/standalone/<group>/ folders, Train/Eval the tasks/rl/ family
// folders. Both modes therefore have the same two axes — pick the group/family, then what is inside it.
function taskList(){return (state.mode==="standalone"?(DESC&&DESC.standalone_tasks):(DESC&&DESC.tasks))||[];}
// recipe combobox source: the *.py inside the selected group/family. Empty for a single-file contract.
function recipeList(){const m=(state.mode==="standalone"?(DESC&&DESC.standalone_recipes):(DESC&&DESC.task_recipes))||{};
  return m[state.task]||[];}

function populateTasks(){const ts=taskList();
  $("task").innerHTML=ts.map(t=>`<option value="${esc(t)}">${esc(t)}</option>`).join("")||'<option>(태스크 없음)</option>';
  state.task=ts[0]||null;populateRecipes();}
// Restore a run card's two axes. The recipe is only honoured if this task still has it — a contract
// can be renamed away, and silently launching a different one would be worse than falling back visibly.
function selectTask(task,recipe){if(!taskList().includes(task))return;
  state.task=task;$("task").value=task;populateRecipes();
  if(recipe&&recipeList().includes(recipe)){state.recipe=recipe;$("recipe").value=recipe;}}
// A task FAMILY (or a standalone GROUP) is not runnable by itself, so there is no '(기본)' entry — the
// first recipe is preselected and one is ALWAYS sent. A single-file contract has none.
function populateRecipes(){const rs=recipeList(),sel=$("recipe");
  sel.innerHTML=rs.length?rs.map(r=>`<option value="${esc(r)}">${esc(r)}</option>`).join("")
                         :'<option value="">(레시피 없음)</option>';
  sel.disabled=!rs.length;state.recipe=rs[0]||"";sel.value=state.recipe;}
function selectMode(m){state.mode=m;
  [...$("mode").children].forEach(b=>b.classList.toggle("on",b.dataset.v===m));
  const af=$("algofield"); if(af) af.style.display=(m==="train")?"":"none";   // algorithm selector: train mode only
  const rb=$("reset"); if(rb) rb.hidden=(m!=="standalone");   // 'Reset Simulator' button: standalone only
  populateTasks();renderKnobs();renderAdv();render();}   // switch the task combobox to this mode's list

fetch("/api/discover").then(r=>r.json()).then(d=>{DESC=d;
  $("dot").classList.add("on");$("livetxt").textContent="ready";
  $("repo").textContent=d.repo;
  $("engines").innerHTML=d.engines.map((e,i)=>
    `<button data-v="${esc(e)}" class="${i===0?"on":""}">${esc(e)}</button>`).join("")
    ||'<button disabled>(엔진 없음)</button>';
  d.engines.forEach(e=>{$("engines").querySelector(`[data-v="${e}"]`).onclick=()=>selectEngine(e);});
  state.engine=d.engines[0]||null;
  populateTasks();   // fills from the current mode (train by default → top-level tasks/)
  $("task").onchange=e=>{state.task=e.target.value;populateRecipes();render();};
  $("recipe").onchange=e=>{state.recipe=e.target.value;render();};
  $("algo").onchange=e=>{state.algo=e.target.value;renderKnobs();render();};
  [...$("mode").children].forEach(b=>b.onclick=()=>selectMode(b.dataset.v));
  [...$("ctrl").children].forEach(b=>b.onclick=()=>selectCtrl(b.dataset.v));
  renderKnobs();renderAdv();render();
}).catch(()=>{$("livetxt").textContent="discover 실패";});

// No wandb login -> the run is pinned to --no_wandb. Independent of /api/discover (order does not
// matter — both re-render).
fetch("/api/creds").then(r=>r.json()).then(d=>{CREDS=d;renderAdv();render();}).catch(()=>{});

$("copy").onclick=()=>{const t=$("cmd").dataset.plain||"";
  navigator.clipboard.writeText(t).then(()=>{$("toast").textContent="복사됨 ✓";
    setTimeout(()=>$("toast").textContent="",1500);}).catch(()=>{$("toast").textContent="복사 실패";});};
function gatherParams(){const sp=curSpec(),knobs={},adv={};
  [...sp.knobs,...(sp.envknobs||[])].forEach(([k])=>{const el=$("k_"+k);if(el)knobs[k]=el.value;});
  ["envs_per_block","num_blocks","ir_coef_scale","embed_dim"].forEach(k=>{const el=$("k_"+k);if(el)knobs[k]=el.value;});
  sp.adv.forEach(([k])=>{const el=$("a_"+k);if(el)adv[k]=el.checked;});
  return {engine:state.engine,task:state.task,recipe:state.recipe,mode:state.mode,
          algo:state.algo,ctrl:state.ctrl,knobs,adv};}
$("launch").onclick=()=>{const p=gatherParams();
  if(!p.engine||!p.task){$("toast").textContent="엔진/태스크를 먼저 고르세요";return;}
  $("launch").disabled=true;
  fetch("/api/launch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)})
    .then(r=>r.json()).then(j=>{$("launch").disabled=false;
      if(j.ok){CURRENT=j.run.run_id;$("logtitle").textContent=j.run.run_id;$("logtext").textContent="…";
        $("toast").textContent="실행 시작: "+j.run.run_id;refreshLog();heartbeat();}
      else $("toast").textContent="실행 실패: "+j.error;
      setTimeout(()=>$("toast").textContent="",4500);})
    .catch(()=>{$("launch").disabled=false;$("toast").textContent="실행 요청 실패";});};

let CURRENT=null;   // run_id whose live log the right pane tails (the most recent launch / running run)
// Stop = terminate ONLY the selected run (the card/log you're viewing). Not all runs.
$("stop").onclick=()=>{
  if(!CURRENT){$("toast").textContent="정지할 run을 카드에서 먼저 선택하세요";setTimeout(()=>$("toast").textContent="",3000);return;}
  $("stop").disabled=true;$("toast").textContent="정지(Ctrl-C) 요청… ("+CURRENT+")";
  fetch("/api/stop",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({run_id:CURRENT})})
    .then(r=>r.json()).then(j=>{
      $("toast").textContent=j.ok
        ?("정지: "+CURRENT+" Ctrl-C 전송")
        :("정지 실패: "+(j.error||""));
      setTimeout(()=>$("toast").textContent="",4000);})
    .catch(()=>{$("stop").disabled=false;$("toast").textContent="정지 요청 실패";});};
// Reset Simulator (standalone only): SIGUSR1 the runner → backend.reset_idx (init pose, no DR).
$("reset").onclick=()=>{
  if(!CURRENT){$("toast").textContent="먼저 Launch 하세요 (리셋할 실행 없음)";setTimeout(()=>$("toast").textContent="",3000);return;}
  $("toast").textContent="Reset 요청… ("+CURRENT+")";
  fetch("/api/reset",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({run_id:CURRENT})})
    .then(r=>r.json()).then(j=>{$("toast").textContent=j.ok?("초기 상태로 리셋됨: "+CURRENT):("리셋 실패: "+(j.error||""));
      setTimeout(()=>$("toast").textContent="",4000);})
    .catch(()=>{$("toast").textContent="리셋 요청 실패";});};
let LASTLOG="";
function logHasSelection(){   // user is drag-selecting / has text highlighted inside the log pane
  const s=window.getSelection&&window.getSelection();
  if(!s||s.isCollapsed) return false;
  const pre=$("logtext");
  return !!(pre&&(pre.contains(s.anchorNode)||pre.contains(s.focusNode)));
}
// Standalone tab frame: a LIVE run's own dashboard while that run runs, else the Launchpad's offline
// preview of the same page (/simui/ — real layout + last run's tabs, no data). Always mounted, so the tab
// is never an empty placeholder and a page edit is visible without booting a sim.
// The decision is driven by the RUN LIST (heartbeat), not by which card is selected: selecting some other
// run must not unmount a live dashboard, and a dead run's port must not stay mounted.
const SIMUI="/simui/";
let SIMRUN=null;                    // {run_id, url} — standalone dashboard last scraped from a run log
function setSimFrame(url){const f=$("simframe");
  if(f.dataset.url===url)return;
  f.dataset.url=url;f.src=url;$("simdot").classList.toggle("live",url!==SIMUI);}
function syncSimFrame(runs){
  const live=SIMRUN&&(runs||[]).some(m=>m.run_id===SIMRUN.run_id&&m.status==="running");
  setSimFrame(live?SIMRUN.url:SIMUI);}

function refreshLog(){if(!CURRENT)return;
  fetch("/api/log?run_id="+encodeURIComponent(CURRENT)).then(r=>r.json()).then(j=>{
    const raw=j.log||"(로그 없음)", pre=$("logtext");
    // Rebuild innerHTML ONLY when the text actually changed AND the user isn't selecting inside the pane —
    // rebuilding wipes the live text selection, so a drag would clear itself on the next 2s poll.
    if(raw!==LASTLOG && !logHasSelection()){
      const bot=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-24;
      pre.innerHTML=ansiToHtml(raw); LASTLOG=raw; if(bot)pre.scrollTop=pre.scrollHeight;
    }
    // --viz run prints "[telemetry] live dashboard → http://127.0.0.1:PORT" → embed it in the telemetry tab
    const m=raw.match(/live dashboard[^\n]*?(http:\/\/127\.0\.0\.1:\d+)/);
    if(m){const f=$("telemframe");if(f.dataset.url!==m[1]){f.dataset.url=m[1];f.src=m[1];f.hidden=false;
      $("telemempty").hidden=true;$("telemdot").classList.add("live");}}
    // standalone run prints "[traj] control dashboard → http://127.0.0.1:PORT" → remember it; heartbeat
    // mounts it in the Standalone tab while that run lives (else the tab shows the offline preview).
    const ms=raw.match(/\[traj\][^\n]*?(http:\/\/127\.0\.0\.1:\d+)/);
    if(ms){SIMRUN={run_id:CURRENT,url:ms[1]};syncSimFrame(RUNS_CACHE);}
  }).catch(()=>{});}
// heartbeat: bumps the server's close-detection poll AND tracks running state (enables Stop, and
// picks a run to tail if none is selected). No run-history list — full logs live in wandb.
function heartbeat(){if(EXITING)return;            // the server is on its way out; stop polling it
  fetch("/api/runs").then(r=>r.json()).then(j=>{
    const running=(j.runs||[]).filter(m=>m.status==="running");
    $("stop").disabled=running.length===0;
    if(!CURRENT && running.length){CURRENT=running[0].run_id;$("logtitle").textContent=CURRENT;}
    renderCards(j.runs);
    syncSimFrame(j.runs);              // Standalone tab: live dashboard while its run lives, else preview
  }).catch(()=>{});}
// run cards: grouped mode(학습·검증/Standalone) → status filter → date. Click a card → show its launch
// info (mode/params/sha for reproduce) + tail its log. Non-running cards get ✕.
let RCFILTER={mode:"all",status:"all"}, RUNS_CACHE=[];
function cardMode(m){ return m.mode==="standalone" ? "standalone" : "local"; }
function renderStatusChips(){
  const inMode=RUNS_CACHE.filter(m=>RCFILTER.mode==="all"||cardMode(m)===RCFILTER.mode);
  if(RCFILTER.status!=="all" && !inMode.some(m=>(m.status||"?")===RCFILTER.status)) RCFILTER.status="all";
  const sts=[...new Set(inMode.map(m=>m.status||"?"))].sort();
  const cnt=s=>s==="all"?inMode.length:inMode.filter(m=>(m.status||"?")===s).length;
  $("rcstatus").innerHTML=["all",...sts].map(s=>
    `<span class="chip${RCFILTER.status===s?" on":""}" data-s="${esc(s)}">${s==="all"?"전체":esc(s)} (${cnt(s)})</span>`).join("");
  $("rcstatus").querySelectorAll(".chip").forEach(c=>c.onclick=()=>{RCFILTER.status=c.dataset.s;renderCards(RUNS_CACHE);});
}
function renderCards(runs){
  RUNS_CACHE=runs||[];
  const el=$("runcards"); if(!el) return;
  renderStatusChips();
  let f=RUNS_CACHE.filter(m=>RCFILTER.mode==="all"||cardMode(m)===RCFILTER.mode)
                  .filter(m=>RCFILTER.status==="all"||(m.status||"?")===RCFILTER.status);
  if(!f.length){el.innerHTML='<div class="rnone">이 필터에 해당하는 run 이 없습니다. (실행하면 최근순·날짜별로 카드가 쌓입니다)</div>';return;}
  let html="",lastDate=null;                              // runs are newest-first → dates descend
  for(const m of f){
    const d=new Date(m.started_at), dk=isNaN(d)?"(날짜 미상)":d.toLocaleDateString("ko-KR");
    if(dk!==lastDate){html+=`<div class="rcdate">${esc(dk)}</div>`;lastDate=dk;}
    const st=m.status||"?";
    const x=(st!=="running")?`<span class="rx" data-x="${esc(m.run_id)}" title="카드 지우기">✕</span>`:"";
    html+=`<div class="rcard${m.run_id===CURRENT?" on":""}" data-rid="${esc(m.run_id)}"><span class="rdot ${esc(st)}"></span>`+
      `<span class="rlabel">${esc(m.engine||"")}·${esc(m.task||"")}${m.task_recipe?"/"+esc(m.task_recipe):""} <b>${esc(m.mode||"")}</b></span>`+
      `<span class="rst">${esc(st)}</span>`+x+`</div>`;
  }
  el.innerHTML=html;
  el.querySelectorAll(".rcard").forEach(c=>c.onclick=(e)=>{
    if(e.target.classList.contains("rx"))return;   // ✕ click is dismiss, not select
    CURRENT=c.dataset.rid;$("logtitle").textContent=CURRENT;
    const _m=RUNS_CACHE.find(m=>m.run_id===CURRENT);
    showRunInfo(_m); restoreForm(_m); refreshLog();   // restore the launcher form to this run's settings
    el.querySelectorAll(".rcard").forEach(x=>x.classList.toggle("on",x.dataset.rid===CURRENT));});
  el.querySelectorAll(".rx").forEach(b=>b.onclick=(e)=>{e.stopPropagation();dismissCard(b.dataset.x);});
}
// right-pane tabs: 터미널(log) / 실행 정보(runinfo). Only one shows; card click fills BOTH, tab stays put.
function showRTab(t){
  document.querySelectorAll(".rtab").forEach(b=>b.classList.toggle("on",b.dataset.rt===t));
  $("logtext").hidden=(t!=="term"); $("runinfo").hidden=(t!=="info");
  const zw=$("zoomwrap"); if(zw) zw.style.display=(t==="term")?"":"none";   // zoom is log-only
}
// launch-info tab content: EXACTLY how this card was run (any status) — cmd (copyable) + params + git sha.
// Fills content only; the tab controls visibility (does NOT force the info tab open on click).
function showRunInfo(m){
  const el=$("runinfo"); if(!el){return;}
  if(!m){el.innerHTML='<div class="ri-empty">run 카드를 선택하면 실행 정보(모드·파라미터·git sha)가 여기에 표시됩니다.</div>';return;}
  const argv=m.argv||[]; const params=[]; let i=1;   // argv[0]=script; pair up --flag value
  while(i<argv.length){const a=String(argv[i]);
    if(a.startsWith("--")){const nx=argv[i+1];
      if(nx!==undefined && !String(nx).startsWith("--")){params.push(a+" "+nx);i+=2;} else {params.push(a);i+=1;}}
    else i+=1;}
  const envp=Object.entries(m.env||{}).map(([k,v])=>k+"="+v);
  const when=m.started_at?new Date(m.started_at).toLocaleString("ko-KR"):"?";
  const row=(k,v)=>`<span class="k">${k}</span><span class="v">${v}</span>`;
  el.innerHTML=
    `<div class="ri-hd">실행 정보 · reproduce <button class="ri-copy" id="ricopy">cmd 복사</button></div>`+
    `<div class="ri-cmd">${esc(m.cmd||"")}</div>`+
    `<div class="ri-grid">`+
      row("engine·task·recipe",`<b>${esc(m.engine||"")}</b> · ${esc(m.task||"")}${m.task_recipe?" · "+esc(m.task_recipe):""}`)+
      row("mode",esc(m.mode||""))+
      row("params", params.length?esc(params.join("  ")):"(기본값)")+
      (envp.length?row("env",esc(envp.join("  "))):"")+
      row("git sha",`<b>${esc(m.git_sha||"?")}</b>`)+
      row("시작(KST)",esc(when))+
      row("status",`${esc(m.status||"?")}${m.ended_by?" ("+esc(m.ended_by)+")":""}`)+
    `</div>`+
    // recipe: the task-contract sections as they were AT LAUNCH. Cards launched before this feature
    // carry none — they get no section rather than an empty one.
    (Object.keys(m.recipe||{}).length
      ? `<div class="ri-rc"><div class="ri-hd">task contract · recipe (${esc(m.task||"")})</div>`+
        Object.entries(m.recipe).map(([k,v])=>
          `<details><summary>${esc(k)}</summary><pre>${esc(v)}</pre></details>`).join("")+
        `</div>`
      : "");
  const cb=$("ricopy"); if(cb)cb.onclick=()=>navigator.clipboard.writeText(m.cmd||"")
    .then(()=>{cb.textContent="복사됨 ✓";setTimeout(()=>cb.textContent="cmd 복사",1500);}).catch(()=>{});
}
// restore the LEFT launcher form to a card's settings (as clicked): mode/target/engine/task/algo buttons
// + knob/adv values. One-click reproduce — then just press Launch (or tweak first). Cards launched before
// this feature carry no knobs/adv, so only mode/target/engine/task restore for those.
function restoreForm(m){
  if(!m) return;
  if(m.engine) state.engine=m.engine;
  if(m.mode) state.mode=m.mode;
  // algo: stored field first; fall back to env ALGO / cmd (pre-feature cards recorded the SAPG choice
  // only in the env, not as a top-level field, so m.algo was undefined and the selector stayed on PPO).
  state.algo=((m.algo)||((m.env&&m.env.ALGO))||(/(^|[ =])ALGO=sapg/.test(m.cmd||"")?"sapg":"ppo")||"ppo").toLowerCase();
  // ctrl: stored field first, else the recorded env (pre-feature cards have neither → motor, the engine default).
  state.ctrl=(m.ctrl)||(((m.env&&m.env.METALAB_MOTOR_COUPLING)==="0")?"joint":"motor");
  state.knob={...(m.knobs||{})};
  state.adv={...(m.adv||{})};
  // reflect selections into the widgets (mirror selectMode's visibility rules)
  [...$("engines").children].forEach(b=>b.classList.toggle("on",b.dataset.v===state.engine));
  [...$("mode").children].forEach(b=>b.classList.toggle("on",b.dataset.v===state.mode));
  [...$("ctrl").children].forEach(b=>b.classList.toggle("on",b.dataset.v===state.ctrl));
  const af=$("algofield"); if(af) af.style.display=(state.mode==="train")?"":"none";
  const rb=$("reset"); if(rb) rb.hidden=(state.mode!=="standalone");
  const a=$("algo"); if(a) a.value=state.algo;
  populateTasks();                                  // task list for this mode
  renderKnobs(); renderAdv();                       // build inputs FROM state.knob/adv (prefills values)
  if(m.task) selectTask(m.task,m.task_recipe||"");   // the run's two axes
  render();
}
function dismissCard(rid){
  fetch("/api/dismiss",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({run_id:rid})})
    .then(r=>r.json()).then(j=>{
      if(j.ok){if(CURRENT===rid){CURRENT=null;$("logtitle").textContent="";showRunInfo(null);}heartbeat();}
      else{$("toast").textContent="지우기 실패: "+(j.error||"");setTimeout(()=>$("toast").textContent="",3000);}
    }).catch(()=>{});
}
document.getElementById("ckcancel").onclick=()=>{document.getElementById("ckmodal").hidden=true;};
// top tabs: 콘솔(런처+로그) · Standalone(궤적 재생 대시보드, standalone 런) · RL(--viz 텔레메트리, train/eval).
function showTab(t){document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("on",b.dataset.tab===t));
  $("pane-console").hidden=(t!=="console");$("pane-sim").hidden=(t!=="sim");$("pane-telem").hidden=(t!=="telem");}
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>showTab(b.dataset.tab));
document.querySelectorAll(".rtab").forEach(b=>b.onclick=()=>showRTab(b.dataset.rt));   // 터미널/실행정보
// run-card mode subtabs → re-filter the card list
[...$("rcmode").children].forEach(b=>b.onclick=()=>{RCFILTER.mode=b.dataset.v;
  [...$("rcmode").children].forEach(x=>x.classList.toggle("on",x.dataset.v===RCFILTER.mode));
  renderCards(RUNS_CACHE);});
let LOGZOOM=100;
function applyZoom(){$("logtext").style.fontSize=(13.6*LOGZOOM/100).toFixed(1)+"px";$("zoomreset").textContent=LOGZOOM+"%";}
$("zoomin").onclick=()=>{LOGZOOM=Math.min(300,LOGZOOM+10);applyZoom();};
$("zoomout").onclick=()=>{LOGZOOM=Math.max(60,LOGZOOM-10);applyZoom();};
$("zoomreset").onclick=()=>{LOGZOOM=100;applyZoom();};
applyZoom();
let EXITING=false;
function showExit(v){$("exitmodal").hidden=!v;}
$("exit").onclick=()=>showExit(true);
$("exitcancel").onclick=()=>showExit(false);
$("exitmodal").onclick=e=>{if(e.target===$("exitmodal"))showExit(false);};   // click backdrop = cancel
document.addEventListener("keydown",e=>{if(e.key==="Escape")showExit(false);});
function doExit(){EXITING=true;
  // CLOSE first, explain only if we could not. window.close() is refused for a window the script did not
  // open (a hand-typed tab, the default-browser fallback, or an app window that has navigated since), and
  // painting the farewell up front made every clean exit flash it for a frame — which is what made the
  // shutdown look different each time. Nothing is painted while the close may still land; if this timer
  // ever runs, the window survived and the message is the honest thing to show.
  const bye=()=>{
    document.title="MetaLab Launchpad — 콘솔 종료됨";
    document.body.innerHTML='<div style="font-family:ui-monospace,monospace;padding:48px;color:#96a4af;font-size:18px">MetaLab Launchpad 콘솔이 종료됐습니다. 실행 중이던 학습·검증은 계속 돌아갑니다(분리된 프로세스). 이 탭은 닫아도 됩니다.<br><br>다시 열기: 앱 아이콘 또는 <code>bash sim/metalab/launchpad.sh</code> — 카드가 현재 상태로 복원됩니다.</div>';
  };
  fetch("/api/shutdown",{method:"POST"}).catch(()=>{}).finally(()=>{
    try{window.close();}catch(e){}                       // best-effort: close the tab (browser may block)
    setTimeout(()=>{try{window.close();}catch(e){}},150);   // ...and once more, after the POST settles
    setTimeout(bye,350);                                 // still alive → the browser refused: say so
  });
}
$("exitconfirm").onclick=doExit;
// Safety: closing the tab/window (or refreshing) asks for confirmation first — the same guard as
// Exit, but via the browser's native prompt (custom modals can't run during unload; text is fixed
// by the browser). Skipped when leaving via our own Exit button (EXITING).
window.addEventListener("beforeunload",e=>{if(EXITING)return; e.preventDefault(); e.returnValue="";});
// Closing the tab/window == Exit: on unload, tell the server to shut down. A refresh fires this too,
// but the server cancels if the page's /api/runs polling resumes within the grace window.
window.addEventListener("pagehide",()=>{if(!EXITING)navigator.sendBeacon("/api/closing");});
setSimFrame(SIMUI);   // Standalone tab starts on the offline preview; a live run swaps it in (refreshLog)
heartbeat(); setInterval(()=>{heartbeat(); if(CURRENT)refreshLog();},2000);
</script></div></body></html>"""


def main():
    # STOP THE LEAK AT THE SOURCE. launchpad.sh starts us detached (`nohup … &`), and POSIX has a
    # non-interactive shell set SIGINT to SIG_IGN for anything it runs asynchronously. SIG_IGN
    # survives exec and is INHERITED by every process we launch — and CPython skips installing its
    # default_int_handler when it inherits SIG_IGN, so every trainer/runner started from the desktop
    # icon silently discarded SIGINT. Stop's phase-1 interrupt was a no-op and its phase-2 SIGKILL
    # did the killing, which is why those W&B runs showed "crashed" instead of "killed".
    # Re-arming SIGINT here fixes it for the whole tree at once: exec resets a CAUGHT signal to
    # SIG_DFL in the child, so launcher shells, trainers and sim servers all start interruptible.
    # (Each entrypoint also self-heals via restore_default_sigint.) Foreground runs are unaffected: SIGINT is
    # already default there, and serve() keeps catching KeyboardInterrupt either way.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    ap = argparse.ArgumentParser(description="MetaLab Launchpad (stdlib-only launcher/monitor web console)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)   # 878x = MetaLab (see launchpad.sh)
    ap.add_argument("--no-browser", action="store_true", help="do not auto-open the browser")
    args = ap.parse_args()
    serve(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
