"""ALLEX learning — Newton RL trainer.

Dispatched from learning/train.py. Runs IN-PROCESS in the learning
venv — like BCTrainer holds the BC loop, RLTrainer.run() holds the RL orchestration: it spawns the sim
server (`sim/isaaclab/server.py`) in the isaaclab env via `learning/rl/service.py`, builds the
sim-service client, and runs the (owned) PPO `OnPolicyRunner`. No Isaac Lab import on this side — the
server owns the Isaac env; the trainer owns the experiment (algorithm + task tunables, in
`learning/rl/dexblind/hammer_lift/experiment.py`) and ships the env tunables to the server.

The PPO trainer is owned in-repo under `learning/rl/` (no `rsl-rl-lib` dep); the RL extra is just
`tensordict` + `GitPython` (`pip install -e .[rl]`). Server node layout is env-overridable via
SIM_CONDA_SH / SIM_CONDA_ENV / ISAACLAB_DIR / SIM_SERVER_SCRIPT (see learning/rl/service.py); trainer
log root via RL_LOG_ROOT (default `$PWD/logs/rl`).
"""
from __future__ import annotations

import argparse
import copy
import glob
import importlib
import os
import shutil
import tempfile

from learning.eval.protocol import build_actor, eval_srv_args, rollout_first_episodes
from learning.rl.on_policy_runner import OnPolicyRunner
from learning.rl.service import ensure_transport_importable, sim_server
from learning.rl.utils.run_naming import build_run_name
from learning.rl.utils.video import slow_mp4

# The sim-service client imports the shared transport.py from the sim-service dir; put that dir on
# sys.path before importing the client. This module is imported lazily (only for `--trainer rl`, from
# learning/train.py), so this module-level setup runs when the trainer is loaded.
ensure_transport_importable()
from learning.rl.client import NaNSafeVecEnv, SimServiceVecEnv  # noqa: E402


def _publish_report(vdir: str, ckpt_path: str, it: int | None) -> str:
    """Put this checkpoint's rollout report where its CHECKPOINT lives; return the local path (or "").

    The sim-service writes the recording (``rollout.rrd`` + ``data.json`` + ``report.html``) into ``vdir``,
    which the caller deletes as soon as the recording is done, so it is copied to
    ``<log_dir>/report_<it>/`` — beside that run's ``model_*.pt``.

    ONE SUBDIR PER CHECKPOINT, because the page addresses its recording by BASENAME: a flat layout would
    collide across checkpoints (every one writes ``rollout.rrd``) and silently repoint an older page at a
    newer rollout.

    Synchronous by necessity — ``vdir`` is about to be deleted — and small (a few MB per checkpoint).
    """
    if not os.path.exists(os.path.join(vdir, "report.html")):
        print("[rl-trainer] WARN: recording has no report.html — nothing to publish "
              "(is the sim spoke's rollout log wired in?)", flush=True)
        return ""
    name = f"report_{it if it is not None else 'final'}"
    dest = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), name)
    shutil.copytree(vdir, dest, dirs_exist_ok=True)
    print(f"[rl-trainer] report -> {dest}", flush=True)
    return dest


def _make_record_callback(task: str, recipe: str, device: str, policy_cfg: dict, seed: int,
                          train_env=None, latest_metrics=None):
    """RECORD=1(env) → return a callback ``fn(ckpt_path)`` that, on each checkpoint, records a short eval
    rollout of that checkpoint and posts its report link to the CURRENT (training) W&B run under
    ``val/`` — else None. Sim-agnostic: the rollout runs through the ``EVAL_POLICY`` plugin
    (``learning/eval/policies/<name>``, set by the launch script — fail-loud if RECORD=1 without it),
    over a sim whose server supports server-side recording. Runs inside the trainer
    process (so ``wandb.run`` is the training run → everything lands in the SAME run, no sibling run),
    spinning up a short-lived RPC sim-service that records RECORD_ENVS envs server-side at the
    curriculum-END criteria. Runs on the training GPU (a few envs are tiny next to the training batch).

    WHAT GOES TO W&B IS A LINK. The sim writes a rerun ``.rrd`` plus the synced plot report into the recording
    dir (see sim/metalab/runtime/rollout_report.py), :func:`_publish_report` files that beside this run's
    checkpoints, and its path is logged on the checkpoint's own training step — so a checkpoint's replay, its
    per-step series and its ``.pt`` all live in one place. No ``wandb.Video``: the report's 3D pane replays the
    same rollout with a free camera, which the per-env MP4s (one fixed angle, ~32 s of blocking render per
    checkpoint measured) only ever approximated.

    ONE panel: ``val/report``, this checkpoint's link as ``wandb.Html`` — a per-step MEDIA value, which is
    what makes it LIVE. Media rides the history row and uploads as a RUN FILE, so the panel streams new
    checkpoints in with NO page reload (its step slider just extends) and older ones stay one drag away.
    A ``wandb.Table`` accumulating one row per checkpoint was tried instead (2026-08-04..05) and reverted:
    its cell value is a ``wandb-client-artifact://…:latest`` reference and the table panel resolves that
    from cache, so rows appeared only after a page RELOAD — measured against a link that was already
    committed to W&B within 30 s of the checkpoint.

    BLOCKING: records synchronously on the train thread — the loop pauses per checkpoint until the link is
    logged (``wandb.log(..., step=iter)``).

    Knobs (env): ``RECORD_ENVS`` (envs given a series + a report tab, default 4 = one per hammer variant,
    since env i gets variant i % N), ``RECORD_STEPS`` = policy steps per recording (0 = full episode;
    default 600). Best-effort — a recording failure must never kill training."""
    if os.environ.get("RECORD") != "1":
        return None
    # the eval-policy plugin that rolls the checkpoint out for recording — sim-owned choice, shipped by
    # the launch script (the sim's eval-policy plugin name). Resolved once, fail-loud (no silent skip).
    policy_name = os.environ.get("EVAL_POLICY")
    assert policy_name, "RECORD=1 needs EVAL_POLICY=<learning.eval.policies module> (set by the launch script)"
    _run_policy_eval = importlib.import_module(f"learning.eval.policies.{policy_name}")._run_eval
    record_envs = int(os.environ.get("RECORD_ENVS", "4"))
    record_steps = int(os.environ.get("RECORD_STEPS", "600"))   # policy steps per recording; <=0 = full episode

    def _record_and_upload(ckpt_path: str) -> None:
        import wandb  # lazy by design (same as the val-video hook): only needed when the hook fires
        if wandb.run is None:   # no W&B run to upload into (e.g. --no_wandb) → nothing to record for
            print(f"[rl-trainer] RECORD on but no W&B run → skip video for {os.path.basename(ckpt_path)}",
                  flush=True)
            return
        try:
            it = int(os.path.splitext(os.path.basename(ckpt_path))[0].split("_")[-1])   # model_<it>.pt
        except ValueError:
            it = None
        vdir = tempfile.mkdtemp(prefix="val_record_")   # OUTSIDE log_dir → no accidental double-upload
        # The recorder is a SEPARATE sim process and knows nothing about this run's curriculum progress, so
        # ask the TRAINING env which level it is at and reproduce that WORLD (gravity / object mass / dense
        # weights). Judgement still snaps to the END criteria, so val/SR stays the same absolute
        # bar for every checkpoint. Without this a level-3 checkpoint — trained near-weightless with a 50 g
        # hammer — was recorded under full gravity with a 550 g hammer, a task it had never seen.
        train_level = -1
        if train_env is not None:
            inner = getattr(train_env, "env", train_env)
            if hasattr(inner, "curriculum_level"):
                train_level = int(inner.curriculum_level())
        try:
            # A fresh short-lived sim-service records the rollout server-side; the EVAL_POLICY plugin loads
            # THIS checkpoint (a FRESH actor — runner.alg untouched) and rolls out one window on the training GPU.
            vargs = argparse.Namespace(
                checkpoint=ckpt_path, device=device, task=task, recipe=recipe,
                num_envs=record_envs, seed=seed, play=True, export=False, meta_out="",
                # video=False: the MetaLab spokes have no MP4 path any more, and the shared eval plugin
                # reads this flag (it still serves sims that do). --rrd IS the record path; data.json and
                # report.html land beside it in vdir, which _publish_report ships wholesale.
                video=False, record_envs=record_envs, rrd=os.path.join(vdir, "rollout.rrd"),
                viz="none", experiment=task, eval_sha="", train="", eval_date="",
                curriculum_end=True,             # the val rollout is JUDGED at the task's END criteria
                curriculum_train_level=train_level,   # ...but RUN in this checkpoint's own world (<0 = END)
            )
            with sim_server(vargs) as vport:
                venv = NaNSafeVecEnv(SimServiceVecEnv("127.0.0.1", vport, device=device))
                horizon = int(venv.max_episode_length)
                vargs.episodes = 0
                vargs.steps = horizon if record_steps <= 0 else min(record_steps, horizon)
                _run_policy_eval(venv, vargs, policy_cfg)
                venv.close()                                     # server finalizes the .rrd + data.json
            # SR = ``val/SR``, the metric this iteration ALREADY logged (env_driver: the GATE
            # verdict latched per episode, meaned over ALL training envs). Absolute and curriculum-
            # independent, so it stays comparable across checkpoints.
            #
            # Read from the LOGGER's own record of what it wrote, not from ``wandb.run.summary``: measured, a
            # key logged moments before this hook still comes back absent from the summary. Empty until the
            # training envs have finished their first episode (no completed episode, no gate verdict).
            #
            # NOT the recorded rollout's own SR: that one is a RECORD_ENVS-wide window that needs a COMPLETED
            # episode to be defined at all (`n_succ / n_ep`, else nan), so it was empty for most early
            # checkpoints — exactly when ranking them matters.
            sr = (latest_metrics() or {}).get("val/SR") if latest_metrics else None
            sr = None if sr is None else round(float(sr), 4)
            # Publish the .rrd + series + page beside this run's checkpoints, then log where it went.
            url = _publish_report(vdir, ckpt_path, it)
            if url:
                payload = {"val/report": wandb.Html(f"<code>{url}</code>")}
                # blocking: synchronous at checkpoint time → step=it is valid on the training step axis
                wandb.log(payload, step=it) if it is not None else wandb.log(payload)
                print(f"[rl-trainer] report @iter {it} (SR {sr}) → W&B (val/report) {url}", flush=True)
            else:
                print(f"[rl-trainer] WARN: no report published for {os.path.basename(ckpt_path)}", flush=True)
        finally:
            shutil.rmtree(vdir, ignore_errors=True)

    def _cb(ckpt_path: str) -> None:
        try:
            _record_and_upload(ckpt_path)
        except Exception as e:   # system boundary: a recorder/upload failure must never kill training
            print(f"[rl-trainer] WARN: video record/upload failed for {os.path.basename(ckpt_path)}: "
                  f"{e!r} (training continues)", flush=True)
    return _cb


class RLTrainer:
    """Newton RL trainer: runs the owned PPO trainer in-process against the Isaac Lab sim service."""

    def __init__(self, forward_args):
        # RL CLI args forwarded from learning/train.py
        # (--task / --num_envs / --max_iterations / --device / --logger / --wandb_project / --seed / ...).
        self.forward_args = list(forward_args)

    def run(self) -> int:
        """Spawn the sim server, wire the client into the PPO runner, and train. Returns the exit code."""
        p = argparse.ArgumentParser(description="Newton RL trainer via the Isaac Lab sim service.")
        p.add_argument("--num_envs", type=int, default=1024)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--max_iterations", type=int, default=None)
        p.add_argument("--logger", default=None, help="wandb / tensorboard / neptune")
        p.add_argument("--wandb_project", default=None)
        p.add_argument("--task", default="hammer-lift", help="run-name label only (one env: hammer_lift)")
        p.add_argument("--recipe", default="",
                       help="which recipe of --task the sim runs (a task FAMILY requires one). Selects the "
                            "CONTRACT only — the experiment package is chosen by --task alone.")
        p.add_argument("--viz", default="none",
                       help="open the sim server's viewer if it ships one "
                            "(gl|rtx; none = headless). Forwarded to the sim's launcher.")
        p.add_argument("--log_root",
                       default=os.environ.get("RL_LOG_ROOT", os.path.join(os.getcwd(), "logs", "rl")))
        p.add_argument("--experiment", default="dexblind",
                       help="which experiment package to train (resolved by convention: "
                            "learning.rl.<experiment>.<task>.experiment — see learning/rl/experiments.py; "
                            "a wrong name fails loudly at import)")
        p.add_argument("--wbt", action="store_true",
                       help="train the sim's whole-body-tracking (WBT) env variant (forwarded as --wbt)")
        p.add_argument("--reference_dir", default="sim_references",
                       help="dir of ref_*.npz reference motions (used with --wbt)")
        p.add_argument("--save_interval", type=int, default=None,
                       help="checkpoint every N iterations (overrides the experiment default; finer "
                            "intervals catch sharp eval peaks for checkpoint selection)")
        p.add_argument("--resume_ckpt", default="",
                       help="node path to a model_*.pt to resume from (actor/critic/optimizer + iter "
                            "counter); training continues to --max_iterations total")
        p.add_argument("--val_video_every", type=int, default=0,
                       help="every N iters, roll out --val_video_envs tracking envs with server-side "
                            "video and upload a sample to the training's W&B run (0 = off; tracking only)")
        p.add_argument("--val_eval_envs", type=int, default=64,
                       help="episodes in the in-run val eval (eval/SR sample; the val fire pauses "
                            "training either way, and env count is nearly free on the batched sim)")
        p.add_argument("--val_video_envs", type=int, default=4,
                       help="of those, envs to RENDER (per-env GL render is the real per-env cost)")
        p.add_argument("--val_video_n", type=int, default=4, help="per-env val MP4s to upload to W&B")
        args, _ = p.parse_known_args(self.forward_args)

        # Select the experiment by convention (learning.rl.<experiment>.<task>.experiment — see
        # learning/rl/experiments.py; no registry, wrong names fail loudly at import). Its EXP
        # configures the runner in-process. Task knobs are SIM-OWNED (written in the sim's
        # env/task files) — the experiment ships nothing to the server. (args.wbt /
        # args.reference_dir ride to the server via learning/rl/service.py._spawn.)
        from learning.rl.experiments import experiment_module
        # task axis selects the experiment package: --task hammer-lift -> hammer_lift (default),
        # --task hammer-lift-teacher -> hammer_lift_teacher. Each experiment pairs with the sim task
        # that consumes its knob values; the default (hammer-lift -> hammer_lift) is unchanged.
        _exp_mod = experiment_module(args.experiment, task=args.task.replace("-", "_"))
        EXP = _exp_mod.EXP
        exp = copy.deepcopy(EXP)
        # PRIVILEGE BOUNDARY (2nd chokepoint; the env validates term placement): the actor network may be
        # wired ONLY to the deployable 'actor' obs group — rewiring it to privileged/other groups would
        # silently train a policy that cannot run outside the simulator. ONE sanctioned exception:
        # the PRIVILEGED TEACHER (WBT_TEACHER_PRIV) reads ['actor', 'teacher_priv'] BY DESIGN — it is a
        # sim-only demonstrator for distillation, never a deployment artifact; the deployable student
        # distilled from it stays on ['actor'] (+ its own camera) and never sees the group.
        _ag = exp["obs_groups"].get("actor")
        if _ag == ["actor", "teacher_priv"] and exp.get("wbt", {}).get("teacher_priv"):
            print("[rl] PRIVILEGED TEACHER: actor reads ['actor', 'teacher_priv'] — SIM-ONLY "
                  "demonstrator (not deployable); distill to a camera student before any deploy.",
                  flush=True)
        elif _ag != ["actor"]:
            raise ValueError(f"PRIVILEGE BOUNDARY violation: obs_groups['actor'] must be ['actor'], "
                             f"got {_ag} — privileged groups are critic-only (sole exception: the "
                             f"WBT_TEACHER_PRIV teacher).")
        # Snapshot the actor + obs-group specs BEFORE OnPolicyRunner's construct_algorithm consumes them
        # from exp in-place (it pops "class_name" off exp["actor"]) — the val-video hook rebuilds a fresh
        # actor from these untouched specs.
        actor_spec = copy.deepcopy(exp["actor"])
        obs_groups_spec = copy.deepcopy(exp["obs_groups"])
        exp["seed"] = args.seed if args.seed is not None else exp["seed"]
        exp["device"] = args.device
        if args.max_iterations is not None:
            exp["max_iterations"] = args.max_iterations
        if args.logger is not None:
            exp["logger"] = args.logger
        if args.wandb_project is not None:
            exp["wandb_project"] = args.wandb_project
        if args.save_interval is not None:
            exp["save_interval"] = args.save_interval
        args.seed = exp["seed"]  # the server (env) gets the SAME resolved seed as the runner
        # A launcher may dictate the run name VERBATIM via RUN_NAME — the MetaLab launchers own their own
        # format (learning/scripts/local/metalab_train.sh: metalab_run_name), because only they know the
        # engine, the deploy location and the node's git SHA. Unset → the shared default format
        # (run_naming.build_run_name), which the isaaclab flow keeps using unchanged.
        run_name = os.environ.get("RUN_NAME", "").strip() or build_run_name(
            args.task, exp["max_iterations"], args.num_envs,
            run_name=exp.get("run_name", ""), repo_root=os.getcwd())
        # It becomes a directory name — reject anything that would escape it.
        assert run_name and "/" not in run_name and not run_name.startswith("."), \
            f"RUN_NAME must be a single safe path segment — got {run_name!r}"
        exp["run_name"] = run_name
        # Dataset/lineage provenance -> the runner cfg, so the W&B run config records WHICH reference
        # dataset this run tracked and any checkpoint it resumed from (the writer folds scalar cfg
        # entries into wandb.config).
        exp["reference_dir"] = args.reference_dir
        exp["resume_ckpt"] = args.resume_ckpt
        # The two contract axes, so a run says WHICH contract it trained on even when the run name does
        # not (the task has never been a name segment). `task_recipe`, not `recipe`: the writer already
        # publishes wandb.config["recipe"] = the contract's tuned KNOB VALUES (contract/recipe.py).
        exp["task"] = args.task
        exp["task_recipe"] = args.recipe
        # Run-shape knobs that only existed as CLI args (invisible on W&B): the training env count and
        # the in-run eval setup — grouped as eval_cfg, self-documenting down to the gate values the
        # eval scores against (the writer folds dicts into wandb.config as nested groups).
        exp["num_envs"] = int(args.num_envs)
        # eval_cfg shape is EXPERIMENT-OWNED (data-driven, not sim-identity): experiments re-exporting
        # the WBT tracking-eval gate (the 5 TASK_SUCCESS_*_GATE constants from the task-owned
        # envs/hammer_lift/gate.py, e.g. dexblind/tracking) get the WBT-shaped cfg below; every other
        # experiment gets the generic one — any *_GATE constants it re-exports, possibly none
        # (e.g. the teacher's gate lives inline in its sim task contract).
        wbt_gate = hasattr(_exp_mod, "TASK_SUCCESS_CONTACT_COUNT_GATE")
        if not wbt_gate:
            gate = {k: v for k, v in vars(_exp_mod).items() if k.endswith("_GATE")}
            exp["eval_cfg"] = {"episodes": int(args.val_eval_envs), "num_envs": int(args.num_envs), "gate": gate}
        else:
            exp["eval_cfg"] = {
                "episodes": int(args.val_eval_envs),
                "video_envs": int(args.val_video_envs),
                "every_iters": int(args.val_video_every),
                "fixed_refs": True, "t0_starts": True, "latched_claims": True,
                "hold_tail_frames": int(os.environ.get("WBT_EVAL_TAIL", "50")),
                "gate": {
                    "contact_count": _exp_mod.TASK_SUCCESS_CONTACT_COUNT_GATE,
                    "pos_threshold_m": _exp_mod.TASK_SUCCESS_POS_THRESHOLD_GATE,
                    "rot_threshold_rad": _exp_mod.TASK_SUCCESS_ROT_THRESHOLD_GATE,
                    "palm_threshold_m": _exp_mod.TASK_SUCCESS_PALM_THRESHOLD_GATE,
                    "hold_steps": _exp_mod.TASK_SUCCESS_HOLD_STEPS_GATE,
                },
            }
        # Training-process knobs (annotated into the run config; consumed by the per-iter hook below):
        # WBT_ENTROPY_SCHED="e0:e1:N" — linear entropy_coef anneal e0->e1 over the first N iterations.
        #   The measured arc behind it: flat 0.005 ignites grasp discovery but the action std spirals
        #   past the SR peak and the run collapses; flat 0.001 never spirals but plateaus ~2.5x lower
        #   — the anneal takes the ignition, then removes the fuel.
        # WBT_STD_MAX — hard ceiling on the learned action std, clamped after every update (log-space
        #   aware): caps the spiral mechanism itself so an entropy bonus can stay on without
        #   noise-death.
        ent_sched = None
        if os.environ.get("WBT_ENTROPY_SCHED"):
            _e0, _e1, _ni = os.environ["WBT_ENTROPY_SCHED"].split(":")
            ent_sched = (float(_e0), float(_e1), max(1, int(_ni)))
        std_max = float(os.environ.get("WBT_STD_MAX", "0")) or None
        # WBT_ACTOR_FREEZE=N — actor params frozen (requires_grad off) for the FIRST N iterations so
        #   the fresh critic fits values around the initialized actor before PPO may move it (the
        #   distill-then-continue warmup). Pin the LR alongside (WBT_LR): with a frozen actor the
        #   adaptive-KL controller sees kl~0 and would ramp the LR to its ceiling by unfreeze time.
        actor_freeze = int(os.environ.get("WBT_ACTOR_FREEZE", "0"))
        exp["wbt_sched"] = {"entropy_sched": os.environ.get("WBT_ENTROPY_SCHED", ""),
                            "std_max": std_max or 0.0,
                            "actor_freeze_iters": actor_freeze}

        with sim_server(args) as port:
            print(f"[rl-trainer] sim service on port {port}", flush=True)
            client = SimServiceVecEnv("127.0.0.1", port, device=args.device)
            # The rollout length is an RL hyperparameter, so it is single-sourced in EXP; a curriculum that
            # gates on ITERATIONS reads it back off the env instead of the contract restating the number.
            client.set_num_steps_per_env(exp["num_steps_per_env"])
            env = NaNSafeVecEnv(client)

            log_dir = os.path.join(args.log_root, exp["experiment_name"], run_name)
            os.makedirs(log_dir, exist_ok=True)
            print(f"[rl-trainer] log_dir={log_dir}", flush=True)

            # No monkeypatching: W&B logging is tensorboard-free (learning/rl/utils/wandb_writer.py),
            # and the RNN-actor/MLP-critic forward is folded into MLPModel.forward.
            # RECORD=1 (set by launch scripts that want it) records per-checkpoint eval videos into this
            # run's val/ (blocking; see _make_record_callback — self-gating: returns None unless RECORD=1).
            # The WBT flow keeps its own --val_video_every path below.
            # latest_metrics is late-bound on purpose: `runner` is built on the next line, and the hook it
            # ends up owning is only ever called from inside runner.learn().
            record_cb = _make_record_callback(args.task, args.recipe, args.device, _exp_mod.POLICY,
                                              args.seed,
                                              train_env=env,
                                              latest_metrics=lambda: runner.logger.last_metrics)
            runner = OnPolicyRunner(env, exp, log_dir=log_dir, device=args.device, on_checkpoint=record_cb)

            def _val_video(it: int) -> None:
                """Every --val_video_every iters: roll out --val_video_envs tracking envs on a short-lived
                video sim service (server renders one MP4/env of the first episode), then upload a sample
                to the SAME W&B run under val/ at step=it. Best-effort — a failure must not kill training."""
                import wandb  # lazy by design: only needed when the val hook fires under a wandb run
                try:
                    # THE EVAL PROTOCOL, shared with learning.eval.eval_service via learning/eval/protocol
                    # (single source of truth — the two paths drifted when this was hand-duplicated):
                    # deterministic refs, t=0 starts, hold-tail, latched success tags; fresh actor built
                    # from untouched specs + live weights (never reuse the mid-episode training policy).
                    vdir = os.path.join(log_dir, "val_videos", f"iter_{it:05d}")
                    # Fresh dir per fire: iteration-keyed dirs are shared ACROSS runs, and clips are
                    # named by outcome — a leftover envNN_<other-outcome>.mp4 from a previous run
                    # would ride this fire's glob and upload a stale (even different-era) clip.
                    shutil.rmtree(vdir, ignore_errors=True)
                    vargs = eval_srv_args(args.val_eval_envs, args.device, args.seed, play=False,
                                          wbt=args.wbt,
                                          reference_dir=args.reference_dir, video=True, video_dir=vdir,
                                          video_envs=args.val_video_envs)
                    # The in-train eval measures ROBUSTNESS under the eval tail — RSI'd starts
                    # perturbed inside the canonical tube (WBT_EVAL_RSI_NOISE=1.0), the per-
                    # trajectory form of start diversity. It replaced the task-init random-start
                    # protocol (2026-07-14): task-init starts only measured transfer within this
                    # task's accidentally-narrow init distribution — an artifact, not a general
                    # robustness axis. The exact-RSI number stays with the offline probes (the
                    # KPI): an RSI slice at this n flatlines near zero and charts as noise
                    # (measured — the eval/SR≡0 incident).
                    if wandb.run is not None and "in_train_eval_protocol" not in wandb.run.config:
                        wandb.run.config.update({"in_train_eval_protocol":
                            "RSI-tube eps=1.0 (WBT_EVAL_RSI_NOISE), eval tail, fixed refs, first episodes"})
                    # ε=0.25 for the in-train TREND (reads nonzero as soon as RSI competence
                    # ignites; v44 = 0.82 there vs 0.375 at ε=1.0 — full-σ robustness stays a
                    # verdict-sweep axis, not a per-200-iter instrument). The eval server reads
                    # WBT_EVAL_RSI_NOISE when it imports the experiment module — override it for
                    # this child spawn only, then restore the trainer's value.
                    _saved_eps = os.environ.get("WBT_EVAL_RSI_NOISE")
                    os.environ["WBT_EVAL_RSI_NOISE"] = "0.25"
                    try:
                        with sim_server(vargs) as vport:
                            venv = NaNSafeVecEnv(SimServiceVecEnv("127.0.0.1", vport, device=args.device))
                            policy = build_actor(venv.get_observations(), actor_spec, obs_groups_spec,
                                                 venv.num_actions, args.device,
                                                 runner.alg.save()["actor_state_dict"])
                            succ = rollout_first_episodes(venv, policy, args.device)
                            venv.close()
                    finally:
                        if _saved_eps is None:
                            del os.environ["WBT_EVAL_RSI_NOISE"]
                        else:
                            os.environ["WBT_EVAL_RSI_NOISE"] = _saved_eps
                    # eval/SR from the SAME protocol rollout that produced the videos (n = the val env
                    # count — a directional in-run curve; the 32-episode probes/evals stay the KPI).
                    if wandb.run is not None:
                        wandb.log({"eval/SR": float(succ.float().mean()),
                                   "eval/episodes": int(succ.numel())}, step=it)
                    mp4s = sorted(p for p in glob.glob(os.path.join(vdir, "env*.mp4")) if "_slow" not in p)
                    mp4s = [slow_mp4(m) for m in mp4s[: args.val_video_n]]  # W&B player has no speed control
                    if wandb.run is not None and mp4s:
                        wandb.log({f"val/{os.path.splitext(os.path.basename(m))[0]}": wandb.Video(m, format="mp4")
                                   for m in mp4s}, step=it)
                        print(f"[rl-trainer] val-video @iter {it}: uploaded {len(mp4s)} clips to W&B", flush=True)
                    else:
                        print(f"[rl-trainer] val-video @iter {it}: {len(mp4s)} MP4s "
                              f"(wandb {'on' if wandb.run else 'off'})", flush=True)
                except Exception as e:  # noqa: BLE001 — never let a val-video error kill training
                    print(f"[rl-trainer] WARN val-video hook failed @iter {it}: {e!r}", flush=True)

            num_iters = exp["max_iterations"]
            if args.resume_ckpt:
                runner.load(args.resume_ckpt, map_location=args.device)
                # Exploit-stage rescue: peak checkpoints from entropy-driven runs carry an exploded
                # learned std (sigma ~3) — resumed rollouts are then noise and PPO erodes the intact
                # mean policy (measured: 0.318 -> 0.065 in 150 iters, entropy already zero). The std
                # parameter is the poisoned part; reset it so sharpening can start from the good mean.
                rs = os.environ.get("WBT_RESUME_STD")
                if rs:
                    import math

                    import torch as _t
                    for _n, _p in runner.alg.actor.named_parameters():
                        if "log_std" in _n or _n.endswith("std_param"):
                            with _t.no_grad():
                                _p.fill_(math.log(float(rs)) if "log" in _n else float(rs))
                            print(f"[rl-trainer] resume std reset: {_n} -> {rs}", flush=True)
                num_iters = max(1, exp["max_iterations"] - runner.current_learning_iteration)
                print(f"[rl-trainer] resumed {args.resume_ckpt} at iter "
                      f"{runner.current_learning_iteration}; {num_iters} iterations to go", flush=True)
            freeze_handles = []
            if actor_freeze:
                # Freeze via GRADIENT HOOKS, not requires_grad: an all-requires_grad-False actor
                # crashes the recurrent PPO update (cuBLAS INVALID_VALUE in the batched forward),
                # and skipping the warmup is measurably fatal — one fresh-critic iteration took a
                # 0.25-SR clone to 0.00 (garbage advantages at init; v15c probe@0). Zeroed grads
                # leave the forward/backward graph untouched; Adam steps with zero grad are no-ops.
                import torch as _t
                freeze_handles = [p.register_hook(_t.zeros_like)
                                  for p in runner.alg.actor.parameters() if p.requires_grad]
                print(f"[rl-trainer] actor grad-FROZEN for the first {actor_freeze} iters "
                      f"(critic warmup; {len(freeze_handles)} params hooked)", flush=True)
            val_cb = _val_video if (args.wbt and args.val_video_every > 0) else None
            if ent_sched or std_max or actor_freeze:
                import math

                import torch

                def _iter_hook(it: int) -> None:
                    """Every-iteration schedule hook; the val-video cadence is gated inside."""
                    if actor_freeze and it == actor_freeze:
                        for _h in freeze_handles:
                            _h.remove()
                        freeze_handles.clear()
                        print(f"[rl-trainer] actor UNFROZEN at iter {it} (critic warmup over)", flush=True)
                    if ent_sched:
                        e0, e1, n_it = ent_sched
                        runner.alg.entropy_coef = e1 + (e0 - e1) * max(0.0, 1.0 - it / n_it)
                        if it % 50 == 0:
                            import wandb
                            if wandb.run is not None:
                                wandb.log({"sched/entropy_coef": runner.alg.entropy_coef}, step=it)
                    if std_max:
                        for _pn, _p in runner.alg.actor.named_parameters():
                            if "log_std" in _pn or _pn.endswith("std_param"):
                                with torch.no_grad():
                                    _p.clamp_(max=math.log(std_max) if "log" in _pn else std_max)
                    if val_cb is not None and it % args.val_video_every == 0:
                        val_cb(it)

            # Stop (SIGINT/Ctrl-C) during training: mark the W&B run "killed", not "crashed". The runner's
            # normal end calls stop_logging_writer(0)=finished; on interrupt it never reaches that, so the
            # process would just die and W&B would time out to "crashed". Catch it here and finish the run
            # with exit_code=255 (=killed) before propagating. (learn() itself is untouched — the loop's
            # deep indentation makes wrapping it in-place error-prone; the call site is the clean seam.)
            try:
                if ent_sched or std_max or actor_freeze:
                    runner.learn(num_learning_iterations=num_iters, init_at_random_ep_len=True,
                                 iter_callback=_iter_hook, callback_interval=1)
                else:
                    runner.learn(num_learning_iterations=num_iters, init_at_random_ep_len=True,
                                 iter_callback=val_cb, callback_interval=args.val_video_every)
            except KeyboardInterrupt:
                print("[rl-trainer] interrupted (Stop/SIGINT) — finishing W&B run as 'killed'", flush=True)
                if runner.logger.writer is not None:
                    runner.logger.stop_logging_writer(exit_code=255)
                env.close()
                return 130
            print("TRAIN_SERVICE_OK", flush=True)
            env.close()
        return 0
