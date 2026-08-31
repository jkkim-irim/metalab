"""RL actor policy for closed-loop sim-eval, over the sim-service RPC boundary (``--policy actor``).

Evals a trained actor checkpoint against ANY sim served over the sim-service (the spawn recipe is the
sim's own ``sim/<SIM>/launch.py``; select with the ``SIM`` env var): spawn the server ->
``SimServiceVecEnv`` over RPC -> load the checkpoint's actor -> step the env through the client,
reporting per-episode success rate + mean reward. Inference-only — it rebuilds just the actor and
loads its weights (NO PPO / rollout storage / optimizer / training loop). Sim-specific behaviors are
capability/flag-driven, not sim-identity-driven:

  * WBT tracking forensics (``--wbt``/``--replay``/``--rsi_play`` + the WBT_BREAKDOWN/GATE_DEBUG
    blocks) activate only when the server ships the matching extras (``wbt_metrics``/``gate_debug``).
  * ``--curriculum_end`` freezes the env's curriculum at its END criteria before the rollout (servers
    implementing the ``apply_curriculum_end`` ctl — pass it from launch scripts whose sim supports it).
  * ``--export`` bakes ``<ckpt_dir>/exported/{policy.pt,policy.onnx}`` (deployment-ready, raw-obs in).
  * ``--record``/``--record_envs`` = roll exactly one full episode window and let the sim record it
    (MetaLab: an .rrd + the synced report; ``--video`` is the separate, mp4-based path other sims use).
  * ``--episodes < 0`` = infinite watch mode (roll out until Ctrl-C / the viewer closes).

Isaac-free and engine-free (no sim import here). Driven reproducibly by
``learning/scripts/aws/rl_eval.sh`` and ``learning/scripts/local/rl_eval.sh``.

The eval entrypoints live under ``learning.eval`` (not ``learning.rl``) on purpose: the PPO trainer's
``resolve_callable`` scans the modules directly under ``learning.rl`` for a bare class name, so a
module there importing a config dict named like a class could shadow it.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os

import torch

from learning.eval.protocol import build_actor, eval_srv_args
from learning.rl.sapg import collapse_sapg_actor
from learning.rl.service import ensure_transport_importable, sim_server
from learning.rl.utils.video import slow_mp4

ensure_transport_importable()  # put the sim-service dir (transport.py) on sys.path before the client
from learning.rl.client import NaNSafeVecEnv, SimServiceVecEnv  # noqa: E402


def _load_policy(train_cfg: dict, obs, num_actions: int, checkpoint: str, device: str):
    """Rebuild the actor from POLICY and load its checkpoint weights (inference only) — the shared
    ``protocol.build_actor`` recipe, with the state dict sourced from a checkpoint file."""
    ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
    state = ckpt["actor_state_dict"]
    if "block_embed.weight" in state:      # SAPG checkpoint → fold the leader block into a plain MLP
        obs_dim = state["mlp.0.weight"].shape[1] - state["block_embed.weight"].shape[1]
        state = collapse_sapg_actor(state, obs_dim)
    return build_actor(obs, train_cfg["actor"], train_cfg["obs_groups"], num_actions, device, state)


def _export_policy(actor, checkpoint: str) -> str:
    """Bake the loaded actor into ``<ckpt_dir>/exported/{policy.pt, policy.onnx}`` and return that dir.

    The exported graphs are self-contained and deployment-ready (sim2sim / sim2real): the model's
    ``as_jit()`` / ``as_onnx()`` wrappers fold ``obs_normalizer`` + the deterministic (mean) output
    *into* forward, so they take **raw** obs and emit the action — no external normalization needed.
    Mirrors upstream rsl_rl ``play.py`` (export right after load) and reuses the exact same machinery
    as ``OnPolicyRunner.export_policy_to_jit`` / ``export_policy_to_onnx``.
    """
    export_dir = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "exported")
    os.makedirs(export_dir, exist_ok=True)

    # TorchScript (.pt) — deepcopies the submodules, so the live GPU actor is untouched for the rollout.
    # This is the primary deployable graph (sim2sim / sim2real).
    jit_model = actor.as_jit().to("cpu")
    torch.jit.script(jit_model).save(os.path.join(export_dir, "policy.pt"))

    # ONNX (.onnx) — optional. torch.onnx.export needs `onnxscript`, absent in some engine envs. Skip it
    # loudly (policy.pt already written) instead of aborting the eval — a capability check, not
    # error-swallowing.
    if importlib.util.find_spec("onnxscript") is None:
        print("[eval] WARN: onnxscript 없음 → ONNX export 건너뜀 (policy.pt 만 생성). "
              "ONNX 가 필요하면 이 venv 에 onnxscript 설치.", flush=True)
        return export_dir

    # ONNX (.onnx) — single "obs" input, same folded forward.
    onnx_model = actor.as_onnx(verbose=False).to("cpu")
    onnx_model.eval()
    torch.onnx.export(
        onnx_model,
        onnx_model.get_dummy_inputs(),
        os.path.join(export_dir, "policy.onnx"),
        export_params=True,
        opset_version=18,
        input_names=onnx_model.input_names,
        output_names=onnx_model.output_names,
    )
    return export_dir


def _upload_wandb_videos(args, sr: float, mean_step_reward: float, wbt_stats: dict | None = None) -> None:
    """Upload a sample of the per-env MP4s (+ metrics) to a W&B run so the videos show up in the wandb UI
    (visual validation). Runs on the node, where the MP4s and the wandb cred both live."""
    import wandb  # lazy by design: heavyweight and only needed under --wandb

    mp4s = sorted(p for p in glob.glob(os.path.join(args.video_dir, "env*.mp4")) if "_slow" not in p)
    mp4s = [slow_mp4(m) for m in mp4s[: max(0, args.wandb_video_n)]]  # W&B player has no speed control
    if not mp4s:
        print("[eval] WARN: no per-env MP4s found to upload to wandb", flush=True)
        return
    run = wandb.init(project=args.wandb_project, name=(args.wandb_run or None), job_type="eval",
                     config={"experiment": args.experiment, "train": args.train,
                             "wbt": bool(args.wbt), "replay": bool(args.replay),
                             "checkpoint": args.checkpoint_s3 or args.checkpoint, "num_envs": int(args.num_envs),
                             "reference_dir": args.reference_dir, "reference_s3": args.reference_s3})
    payload = {"eval/mean_step_reward": mean_step_reward}
    # eval/SR = the ORIGINAL lift task's success gate (goal pose + palm + contact-count, held). In WBT
    # builds the gate's state machine runs via the negligible-weight task_success reward term; sr is nan
    # when the env never shipped the gate flag (logging 0.0 then would be an artifact, not a result).
    if not math.isnan(sr):
        payload["eval/SR"] = sr
    if wbt_stats:
        payload.update({f"eval/{k}": v for k, v in wbt_stats.items()})
    for m in mp4s:
        payload[f"video/{os.path.splitext(os.path.basename(m))[0]}"] = wandb.Video(m, format="mp4")
    wandb.log(payload)
    run.finish()
    print(f"[eval] uploaded {len(mp4s)} sample videos to wandb: {args.wandb_project} run {run.id}", flush=True)


def _run_eval(env, args, POLICY, knobs: dict | None = None) -> dict:
    """Load the actor, (optionally) export it, and roll it out on ``env`` (over the sim-service),
    reporting the per-episode SR table + any forensic extras the server ships. Returns
    ``{"sr", "mean_step_reward", "wbt_stats"}`` for the caller's reporting (e.g. wandb upload).

    ``env`` is any VecEnv duck-type — only reset/num_actions/num_envs/step are used. Does NOT
    close ``env`` (the caller owns its lifetime). Also the in-training val-record hook's entrypoint
    (``learning/trainer/rl_trainer._make_record_callback``), which is why it is split out of ``run``."""
    # --curriculum_end: freeze the task's curriculum at its END world (full gravity, the contract's own
    # friction, no dense shaping) via the apply_curriculum_end ctl — pass it only for sims whose server
    # implements that ctl (flag-driven: the client class exposes the method for every sim, so hasattr can't
    # discriminate). What counts as SUCCESS is the GATE either way, independent of the curriculum.
    #
    # BEFORE reset(), not after: some curriculum knobs are consumed by the RESET EVENTS (friction
    # mu_scale, object mass_scale), so they only take effect at the NEXT reset. Freezing after the
    # reset left episode 0 running the level-0 world — on hammer_lift, friction_scale_start=3.0, i.e.
    # a 3x-grippier world than anything the checkpoint trained in — while every later episode ran at
    # the frozen level. That made the FIRST episode of every eval systematically fail (0/16 measured)
    # and is why a 300-step recording window, which only ever sees episode 0, never caught a success.
    if getattr(args, "curriculum_end", False):
        inner = getattr(env, "env", None)
        assert inner is not None and hasattr(inner, "apply_curriculum_end"), \
            "--curriculum_end needs a SimServiceVecEnv-style client exposing apply_curriculum_end"
        # --curriculum_train_level L: run the WORLD of level L. The trainer passes the
        # recorded checkpoint's own level here, because a curriculum that ramps gravity/object mass makes
        # the END world a DIFFERENT TASK from the one a mid-training checkpoint learned in. <0 = unset =
        # snap everything to END (a finished run's eval).
        # NOT `... or -1`: level 0 is falsy, so that spelling turned "record at the very first level" into
        # "unset" — the whole point of the flag, silently, for every checkpoint before the first level-up.
        _lvl = getattr(args, "curriculum_train_level", -1)
        lvl = -1 if _lvl is None else int(_lvl)
        inner.apply_curriculum_end(lvl if lvl >= 0 else None)
        print(f"EVAL_CURRICULUM_END applied (success judged at the GATE"
              f"{f'; world at training level {lvl}' if lvl >= 0 else '; world at END'})", flush=True)
    # reset(), NOT get_observations(): a fresh env is only spawn-posed, so episode 0 must be reset the
    # same way every later episode is, or the FIRST episode of every eval runs in a world the policy
    # never trained in. get_observations() skipped the driver's whole reset path, which cost two things:
    #   * the reset EVENTS never ran (the driver applies them in _post_reset) — no object-pose / joint-
    #     offset / friction / mass / root-height randomization, so e.g. the object sat at the contract's
    #     authored init_pos instead of the reset event's own (randomized) spawn envelope;
    #   * the action EMA was never seeded, so the first policy step applied its raw target in ONE jump
    #     instead of being rate-limited (`sim/metalab/runtime/env_driver.py` reset()/_apply_actions).
    # Both made episode 0 of a recorded eval diverge visibly from training (the arm lunged on step 1).
    obs, _ = env.reset()
    # --replay drives the env with the references' recorded actions server-side — no policy at all.
    replay = bool(getattr(args, "replay", False))
    policy = None if replay else _load_policy(POLICY, obs, env.num_actions, args.checkpoint, args.device)
    if policy is not None and getattr(args, "export", False):
        export_dir = _export_policy(policy, args.checkpoint)
        print(f"EXPORTED_POLICY dir={export_dir} files=policy.pt,policy.onnx", flush=True)
    # Per-EPISODE success/fail table, the authoritative record. SR = successes / episodes (a true
    # per-episode rate, num_envs-invariant); the dense reward is a poor success proxy (the task_success
    # reward term is tiny). The server rides a per-env task_success flag on the step extras; every env
    # that terminates this step contributes one episode. Run until --episodes complete (default 64 =
    # 16 envs x 4), so the sample is a fixed, comprehensible size — not a giant sweep.
    total = 0.0
    episodes: list[dict] = []  # [{"env": int, "success": bool, "step": int}], in completion order
    target = args.episodes
    # episodes<0 → infinite "watch" mode: roll out forever (upstream play.py-style) until Ctrl-C or the
    # viewer window is closed. episodes==0 → run exactly --steps. episodes>0 → until N episodes complete.
    watch = target < 0
    # safety cap scaled to the run: ~(episodes/envs) waves, generous per-episode budget (unused in watch mode)
    per_env = max(target, 0) // max(1, args.num_envs) + 2
    max_steps = args.steps if target <= 0 else max(args.steps, per_env * 400)
    step = 0
    # WBT tracking breakdown (server ships extras["wbt_metrics"] [N,10] on --wbt --video):
    # running MEAN over all live steps (tracking fidelity + which kernel earns) + per-episode
    # FINAL-frame stats (outcome). The done step's metrics reflect the post-auto-reset state, so the
    # final frame is taken from the PREVIOUS step's row (one control frame ≈ 20 ms early).
    wbt_sum, wbt_n, wbt_prev, wbt_finals, wbt_stats = None, 0, None, [], None
    saw_gate = False  # did the env ship the ORIGINAL task_success gate flag at all?
    # Original-gate per-condition diagnosis (server ships extras["gate_debug"] [N,6] on --wbt
    # --fixed_refs): live-step satisfaction rates per condition + per-episode MAX consecutive
    # all-conditions streak (a claim needs one streak >= the canonical play hold).
    hold = None  # resolved lazily from knobs the first time gate_debug arrives (WBT servers only)
    gd_sum, gd_n, gd_cur, gd_epmax, gd_streaks = None, 0, None, None, []
    gd_end_hold = 0  # episodes still holding ALL gate conditions through their final live frame
    # Termination forensics (server ships extras["term_reason"] index per done row +
    # ["term_names"]): which term ended each episode and at what length — early terminations
    # (an RSI pose ejecting the hammer -> wbt_bad_tracking within ~10 steps) are invisible
    # inside the aggregate SR without this.
    ep_lens = None
    term_counts: dict[str, int] = {}
    term_names: list[str] = []
    all_lens: list[int] = []
    try:
        while True:
            with torch.inference_mode():
                actions = (torch.zeros(env.num_envs, env.num_actions, device=args.device)
                           if policy is None else policy(obs))  # --replay: server ignores these, drives with recorded actions
                obs, rew, dones, extras = env.step(actions)
                if policy is not None:
                    policy.reset(dones)
            step += 1
            total += float(rew.mean())
            d = dones.bool()
            mt = extras.get("wbt_metrics") if isinstance(extras, dict) else None
            if mt is not None:
                live = ~d  # exclude rows already showing the NEXT episode's post-reset state
                if int(live.sum()):
                    wbt_sum = mt[live].sum(0) + (wbt_sum if wbt_sum is not None else 0.0)
                    wbt_n += int(live.sum())
                if int(d.sum()) and wbt_prev is not None:
                    wbt_finals += [wbt_prev[i] for i in d.nonzero(as_tuple=True)[0].tolist()]
                wbt_prev = mt.clone()
            gd = extras.get("gate_debug") if isinstance(extras, dict) else None
            if gd is not None:
                if hold is None:  # WBT servers ship gate_debug; the gate's hold length is experiment-owned
                    assert knobs is not None and "TASK_SUCCESS_HOLD_STEPS_GATE" in knobs, \
                        "server ships gate_debug (WBT) but the experiment module has no " \
                        "TASK_SUCCESS_HOLD_STEPS_GATE constant"
                    hold = int(knobs["TASK_SUCCESS_HOLD_STEPS_GATE"])
                live = ~d
                if gd_cur is None:
                    gd_cur = torch.zeros(gd.shape[0], device=gd.device)
                    gd_epmax = torch.zeros(gd.shape[0], device=gd.device)
                if int(live.sum()):
                    gd_sum = gd[live].sum(0) + (gd_sum if gd_sum is not None else 0.0)
                    gd_n += int(live.sum())
                gd_cur = torch.where(live & (gd[:, 5] > 0.5), gd_cur + 1, torch.zeros_like(gd_cur))
                gd_epmax = torch.maximum(gd_epmax, gd_cur)
                if int(d.sum()):
                    for i in d.nonzero(as_tuple=True)[0].tolist():
                        gd_streaks.append(float(gd_epmax[i]))
                        if float(gd_cur[i]) >= hold:  # streak alive through the final live frame
                            gd_end_hold += 1
                        gd_epmax[i] = 0.0
                        gd_cur[i] = 0.0
            if ep_lens is None:
                ep_lens = torch.zeros(int(env.num_envs), dtype=torch.long, device=d.device)
            ep_lens += 1
            if int(d.sum()):
                succ = extras.get("task_success") if isinstance(extras, dict) else None
                succ = succ.bool() if succ is not None else None
                if succ is not None:
                    saw_gate = True
                rc = extras.get("term_reason") if isinstance(extras, dict) else None
                if isinstance(extras, dict) and extras.get("term_names"):
                    term_names = list(extras["term_names"])
                for i in d.nonzero(as_tuple=True)[0].tolist():
                    episodes.append({"env": int(i), "success": bool(succ[i]) if succ is not None else False,
                                     "step": step})
                    all_lens.append(int(ep_lens[i]))
                    if rc is not None:
                        code = int(rc[i])
                        nm = term_names[code] if 0 <= code < len(term_names) else "unknown"
                        term_counts[nm] = term_counts.get(nm, 0) + 1
                ep_lens[d] = 0
            if watch:
                continue  # infinite: no episode/step cap — only Ctrl-C (or a closed viewer) stops us
            if target > 0 and len(episodes) >= target:
                break
            if target == 0 and step >= args.steps:
                break
            if step >= max_steps:
                break
    except KeyboardInterrupt:
        print("[eval] interrupted (Ctrl-C) — stopping rollout", flush=True)
    if target > 0:
        episodes = episodes[:target]  # a single step can close several episodes; trim to exactly N
    n_ep = len(episodes)
    n_succ = sum(1 for e in episodes if e["success"])
    # SR = the ORIGINAL task gate's success rate (nan if the env never shipped the gate flag —
    # then SR=0 would be a misleading artifact, not a result).
    sr = (n_succ / n_ep) if (n_ep and saw_gate) else float("nan")
    print(f"EVAL_OVER_SERVICE_OK play={args.play} num_envs={env.num_envs} steps={step} "
          f"SR={sr:.4f} episodes={n_ep} successes={n_succ} "
          f"mean_step_reward={total / step:.4f}", flush=True)
    # Per-reference outcome bitmap (fixed_refs: env i == ref i, first episode per env): which
    # references were EVER claimed. Cross-checkpoint overlap answers whether a plateau is
    # "a fixed solvable subset of references" (data-tier question) or spread thin.
    first = {}
    for e in episodes:
        first.setdefault(e["env"], bool(e["success"]))
    claimed = sorted(k for k, v in first.items() if v)
    if claimed:
        print(f"CLAIMED_REFS n={len(claimed)}: {','.join(map(str, claimed))}", flush=True)
    if all_lens:
        sl = sorted(all_lens)
        q = lambda p: sl[min(len(sl) - 1, int(p * len(sl)))]  # noqa: E731
        reasons = " ".join(f"{k}={v}" for k, v in sorted(term_counts.items(), key=lambda kv: -kv[1]))
        print(f"TERM_BREAKDOWN episodes={len(sl)} len_p10/p50/p90={q(.1)}/{q(.5)}/{q(.9)} "
              f"under15={sum(1 for x in sl if x < 15)} | {reasons}", flush=True)
    if wbt_n:
        m = (wbt_sum / wbt_n).tolist()  # [reward, k_obj_pos, k_obj_ori, k_contact, k_kp, k_joint, cm, deg, cm, deg]
        print(f"WBT_BREAKDOWN mean: reward={m[0]:.2f}/5 | kernels(0..1): obj_pos={m[1]:.2f} "
              f"obj_ori={m[2]:.2f} contact={m[3]:.2f} keypoint={m[4]:.2f} joint={m[5]:.2f} | "
              f"errors: hammer_pos={m[6]:.1f}cm hammer_ori={m[7]:.0f}deg keypoint={m[8]:.1f}cm "
              f"joint={m[9]:.0f}deg", flush=True)
        wbt_stats = {"wbt_mean_reward": m[0], "hammer_pos_err_cm": m[6], "hammer_ori_err_deg": m[7],
                     "keypoint_err_cm": m[8], "joint_err_deg": m[9]}
        if wbt_finals:
            f = torch.stack(wbt_finals).mean(0).tolist()
            fin_ok = sum(1 for row in wbt_finals if float(row[6]) < 5.0)
            print(f"WBT_BREAKDOWN final-frame (n={len(wbt_finals)} episodes): reward={f[0]:.2f}/5 "
                  f"hammer_pos={f[6]:.1f}cm hammer_ori={f[7]:.0f}deg | "
                  f"track_success(final hammer<5cm)={fin_ok}/{len(wbt_finals)}", flush=True)
            wbt_stats.update({"track_success_rate": fin_ok / len(wbt_finals),
                              "final_reward": f[0], "final_hammer_err_cm": f[6]})
    if gd_n:
        r = (gd_sum / gd_n).tolist()  # [contact, lifted, pos, rot, palm, all5] live-step rates
        st = sorted(gd_streaks)
        p90 = st[int(0.9 * (len(st) - 1))] if st else 0.0
        print(f"GATE_DEBUG rates: contact={r[0]:.2f} hammer_lifted={r[1]:.2f} hammer_pos={r[2]:.2f} "
              f"hammer_ori={r[3]:.2f} palm_dist={r[4]:.2f} all5={r[5]:.2f} | "
              f"max-hold-streak/episode: max={max(st) if st else 0:.0f} p90={p90:.0f} "
              f"mean={sum(st) / max(1, len(st)):.1f} (claim needs >= hold(gate)={hold})", flush=True)
        # SR self-verification: the streak counter is computed INDEPENDENTLY of the claim machinery,
        # so episodes_with_streak>=hold must equal the claim-based SR count (± a final-frame-claim
        # boundary miss) — a live consistency check every eval.
        c_hold = sum(1 for s in st if s >= hold)
        print(f"GATE_VERIFY episodes_with_streak>={hold}: {c_hold}/{len(st)} "
              f"(claim-based SR count: {n_succ})", flush=True)
    if args.meta_out:
        meta = {
            "experiment": args.experiment, "eval_date": args.eval_date, "eval_sha": args.eval_sha,
            "train": args.train, "checkpoint": args.checkpoint_s3 or args.checkpoint, "play": bool(args.play),
            "reference_s3": getattr(args, "reference_s3", ""),
            "num_envs": int(env.num_envs), "seed": args.seed, "steps": step,
            "SR": round(sr, 4), "episodes": n_ep, "successes": n_succ,
            "episode_detail": episodes,  # env + success + step for each episode
        }
        os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
        with open(args.meta_out, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"EVAL_META={args.meta_out}", flush=True)
    if args.video:
        print(f"EVAL_VIDEO_DIR={args.video_dir}", flush=True)
    return {"sr": sr, "mean_step_reward": total / max(1, step), "wbt_stats": wbt_stats}


def run(argv=None) -> int:
    p = argparse.ArgumentParser(description="Eval a checkpoint over the sim-service client boundary.")
    p.add_argument("--checkpoint", default="", help="path to a model_*.pt checkpoint; unused with --replay")
    p.add_argument("--checkpoint_s3", default="",
                   help="S3 source of the checkpoint — recorded in the meta json (the canonical, "
                        "reproducible location) instead of the node-local --checkpoint path")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--episodes", type=int, default=64,
                   help="run until N episodes complete (default 64 = 16 envs x 4); 0 = run a fixed --steps; "
                        "<0 = infinite watch (roll out until Ctrl-C / viewer closed)")
    p.add_argument("--steps", type=int, default=400, help="fixed step count when --episodes 0; else a safety cap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--play", action=argparse.BooleanOptionalAction, default=True,
                   help="eval with the play env variant (curriculum frozen at the play success criteria, "
                        "obs noise off) — the eval config; pass --no-play to eval against the raw training env")
    p.add_argument("--curriculum_end", action="store_true",
                   help="freeze the task's curriculum at its END world before the rollout (needs a server "
                        "implementing the apply_curriculum_end ctl — set by launch scripts whose sim supports it)")
    p.add_argument("--curriculum_train_level", type=int, default=-1,
                   help="with --curriculum_end: run the WORLD of this training level (gravity / object mass / "
                        "dense weights) instead of the END one. Set by the trainer's per-checkpoint recorder "
                        "to the level that checkpoint trained at; <0 = world at END. Success is the GATE's "
                        "either way.")
    p.add_argument("--export", action=argparse.BooleanOptionalAction, default=False,
                   help="bake <ckpt_dir>/exported/{policy.pt,policy.onnx} (self-normalizing, deployment-ready) "
                        "right after loading")
    p.add_argument("--video", action="store_true", help="server-side MP4 recording (forwarded to the sim server)")
    p.add_argument("--record", action="store_true",
                   help="roll exactly one full episode window for the sim to record; needs --num_envs >= "
                        "--record_envs. Does NOT imply --video (that is the separate mp4 path)")
    p.add_argument("--record_envs", type=int, default=1,
                   help="how many envs (0..N-1) the recording covers; needs --num_envs >= N")
    p.add_argument("--video_width", type=int, default=512)
    p.add_argument("--video_height", type=int, default=512)
    p.add_argument("--video_dir", default="/home/ubuntu/sim_eval_videos", help="where the server writes the MP4s")
    p.add_argument("--rrd", default="", help="record the rollout to this .rrd (rerun) for the report's 3D "
                                            "pane; data.json + report.html land beside it. Both MetaLab spokes")
    p.add_argument("--video_envs", type=int, default=0,
                   help="render only the first K envs (0 = all) — big eval batch, few clips")
    p.add_argument("--viz", default="none",
                   help="open the sim server's viewer if it ships one (gl|rtx; none = headless)")
    p.add_argument("--wbt", action="store_true",
                   help="eval the whole-body-tracking (WBT) variant (goal-conditioned; needs --reference_dir)")
    p.add_argument("--replay", action="store_true",
                   help="open-loop replay of the references' recorded actions (visualize stored references; needs --wbt --video, no policy)")
    p.add_argument("--rsi_play", action="store_true",
                   help="PLAY env with reference-RSI'd starts (fixed refs): score a plain task "
                        "policy under the tracker's exact eval start distribution (protocol control)")
    p.add_argument("--reference_dir", default="/home/ubuntu/sim_references",
                   help="dir of ref_*.npz reference motions on the node (used with --wbt)")
    p.add_argument("--reference_s3", default="",
                   help="S3 dir the reference dataset was built from (provenance -> wandb config + meta)")
    p.add_argument("--wandb", action="store_true", help="upload a sample of the per-env videos + metrics to a W&B run")
    p.add_argument("--wandb_project", default="chrisryu-simrl")
    p.add_argument("--wandb_run", default="", help="W&B run name (default: auto)")
    p.add_argument("--wandb_video_n", type=int, default=8, help="how many per-env sample videos to upload")
    p.add_argument("--meta_out", default="", help="write the run meta json (SR + per-episode S/F table) here (on the node)")
    p.add_argument("--task", default="hammer-lift",
                   help="task (dash or underscore) — selects the experiment module "
                        "(learning.rl.<pkg>.<task>.experiment) and is forwarded to sims whose server takes --task")
    p.add_argument("--recipe", default="",
                   help="which recipe of --task the sim runs (a task FAMILY requires one). Contract only — "
                        "the experiment module is chosen by --task alone.")
    p.add_argument("--experiment_pkg", default="",
                   help="experiment package namespace (learning.rl.<pkg>.<task>.experiment); "
                        "empty = derived from --experiment (legacy alias hammer-lift -> dexblind)")
    # provenance passthrough — recorded verbatim into the meta json so the metrics carry their lineage
    p.add_argument("--experiment", default="hammer-lift")
    p.add_argument("--eval_sha", default="")
    p.add_argument("--train", default="")
    p.add_argument("--eval_date", default="")
    args, _ = p.parse_known_args(argv)

    # The experiment module owns what an eval needs policy-side (POLICY + any re-exported *_GATE
    # constants for the WBT forensics): resolved by convention from (--experiment_pkg, --task) — no
    # registry; wrong names fail loudly at import (see learning/rl/experiments.py). Task knobs are
    # sim-owned; the server builds the env from its own files. History: a static dexblind import was
    # hardwired here — the actor spec silently rode dexblind's for EVERY experiment. Legacy
    # provenance alias: "hammer-lift" = the dexblind package.
    if args.experiment == "hammer-lift":
        args.experiment = "dexblind"
    pkg = args.experiment_pkg or args.experiment
    from learning.rl.experiments import experiment_module
    _exp_mod = experiment_module(pkg, args.task.replace("-", "_"))
    # Task knobs are SIM-OWNED; the experiment module carries only the policy spec (and, for WBT
    # experiments, the re-exported *_GATE constants the forensics below read).
    POLICY = _exp_mod.POLICY

    assert args.checkpoint or args.replay, "--checkpoint is required (unless --replay)"
    # --record = "roll one full window and record it"; it does NOT imply --video. The mp4 path is a
    # per-sim capability (see sim/<name>/launch.py) and MetaLab has none — it records an .rrd instead.
    recording = bool(args.record)
    if recording:
        assert args.num_envs >= args.record_envs, \
            f"--record_envs {args.record_envs} needs --num_envs >= {args.record_envs} (got {args.num_envs})"

    srv_args = eval_srv_args(args.num_envs, args.device, args.seed, play=args.play,
                             wbt=args.wbt, replay=args.replay, rsi=args.rsi_play,
                             reference_dir=args.reference_dir, video=args.video,
                             video_dir=args.video_dir, video_envs=args.video_envs)
    # fields consumed by sims whose launcher forwards them (task/viewer/per-env recording geometry)
    srv_args.task = args.task
    srv_args.recipe = args.recipe
    srv_args.viz = args.viz
    srv_args.record_envs = args.record_envs
    srv_args.video_width = args.video_width
    srv_args.video_height = args.video_height
    srv_args.rrd = args.rrd
    with sim_server(srv_args) as port:
        print(f"[eval] sim service on port {port}", flush=True)
        env = NaNSafeVecEnv(SimServiceVecEnv("127.0.0.1", port, device=args.device))
        if recording:   # roll a full window; the recorder captures the whole window (spans resets)
            args.steps, args.episodes = int(env.max_episode_length), 0
        out = _run_eval(env, args, POLICY, vars(_exp_mod))
        if args.video and args.wandb:
            _upload_wandb_videos(args, out["sr"], out["mean_step_reward"], out["wbt_stats"])
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
