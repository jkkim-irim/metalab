"""ALLEX learning — behavior-cloning trainer (ACT / VLA).

Step-driven training loop with step- or epoch-cadence validation, driven by learning/train.py.
Reuses the shared modules: data.dataset (loaders), model.act_policy (policy build), metrics.validation
(val loss + per-horizon + group/FK). RL gets its own trainer (learning/trainer/rl_trainer.py) later.

The loop counts OPTIMIZER STEPS, not epochs — "epoch" is just the data-reshuffle boundary, never a
control knob. Run length: ``--epochs N`` runs N full passes; otherwise the loop runs to ``cfg.steps``
total steps. Validation cadence: ``--val_every_n_steps N`` runs the full val suite every N steps;
``0`` validates at each epoch end.
"""
import copy
import json
import logging
from pathlib import Path
import shutil
import time

from accelerate.utils import gather_object
from termcolor import colored
import torch

from learning.configs.config import policy_type_name
from learning.data.dataset import make_dataloader
from learning.data.image_aug import apply_image_aug
from learning.metrics.validation import (
    ALLEX_ACTION_GROUPS,
    compute_cartesian_gripper_metrics,
    compute_prediction_metrics,
    compute_val_loss,
    log_fk_video,
)
from learning.model.act_policy import build as build_act
from learning.trainer.viz import log_first_batch_aug
from learning.utils.train_utils import save_checkpoint, update_last_checkpoint
from learning.utils.wandb_logger import WandBLogger


def _update_best(val_loss, best_val, no_improve, patience):
    """Best-val + early-stop bookkeeping (pure; call once right after each validation).

    Returns ``(best_val, no_improve, improved, should_stop)``. ``val_loss=None`` (no validation) is a
    no-op and never stops. ``patience`` counts consecutive non-improving VALIDATIONS (step- or
    epoch-cadence, whichever is in use), not epochs.
    """
    if val_loss is None:
        return best_val, no_improve, False, False
    improved = val_loss < best_val
    if improved:
        best_val, no_improve = val_loss, 0
    else:
        no_improve += 1
    should_stop = patience > 0 and no_improve >= patience
    return best_val, no_improve, improved, should_stop


def _step_of(step_dir):
    """Parse the integer step from a ``step_<N>`` checkpoint dir name."""
    return int(Path(step_dir).name.split("_")[1])


def _prune_step_dirs(step_dirs, keep_n, protect_step):
    """Pure: which ``step_<N>`` checkpoint dirs to DELETE. Keep the newest ``keep_n`` by step number,
    PLUS ``protect_step`` (the current best-select_metric step — never pruned, even if older than the
    window; this is what preserves the good model a rising surrogate loss would otherwise discard).
    ``protect_step=None`` protects nothing. Returns the dirs to remove (input order preserved)."""
    ordered = sorted(step_dirs, key=_step_of)                 # oldest -> newest
    keep = set(ordered[-keep_n:]) if keep_n > 0 else set()
    if protect_step is not None:
        keep |= {p for p in ordered if _step_of(p) == protect_step}
    return [p for p in step_dirs if p not in keep]


def _raw_frame_view(dataset):
    """Transform-free shallow view of ``dataset`` for the validation loader. Image augmentation is
    TRAIN-ONLY: ``copy.copy`` rebinds only ``image_transforms`` (-> None) on the new object, leaving
    the train dataset's transforms intact and sharing all heavy state (meta, lazy caches) by
    reference (``__getitem__`` only reads it). No-op when the dataset carries no transforms
    (image_aug off/gpu)."""
    view = copy.copy(dataset)
    view.image_transforms = None
    return view


def _build_policy(cfg, dataset):
    """Dispatch policy / processor / optimizer construction on ``cfg.policy.type`` (ACT vs GR00T).

    Both builders return the SAME ``(policy, preprocessor, postprocessor, optimizer, lr_scheduler)``
    tuple, so the rest of the trainer is policy-agnostic. GR00T is imported lazily — its heavy
    gr00t / torch backbone deps load only when ``--policy.type groot`` actually selects it (same
    optional-heavy-dep pattern as the lazy ``AllexFK`` import below).
    """
    if policy_type_name(cfg.policy) == "groot":
        from learning.model.groot_policy import build as build_groot
        return build_groot(cfg, dataset)
    return build_act(cfg, dataset)


def _reduce_metric_mean(accelerator, value):
    """Average a scalar val metric across DDP ranks so the logged number reflects the whole
    (sharded) val set, not one rank's shard. No-op on a single process."""
    if accelerator.num_processes <= 1:
        return value
    t = torch.tensor(float(value), device=accelerator.device)
    return float(accelerator.reduce(t, reduction="mean"))


class BCTrainer:
    """Behavior-cloning trainer (ACT / VLA): step-driven loop with step/epoch-cadence validation."""

    def __init__(self, cfg, extras, accelerator, is_main):
        self.cfg = cfg
        self.extras = extras
        self.accelerator = accelerator
        self.is_main = is_main

    def run(self, dataset, train_episodes, val_episodes):
        cfg, extras = self.cfg, self.extras
        accelerator, is_main = self.accelerator, self.is_main
        device = accelerator.device
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if is_main:
            logging.info(colored("\n=== training ===", "cyan", attrs=["bold"]))
            logging.info(
                f"train episodes ({len(train_episodes)}): {train_episodes[:10]}"
                f"{'...' if len(train_episodes) > 10 else ''}"
            )
            logging.info(
                f"val   episodes ({len(val_episodes)}): {val_episodes[:10]}"
                f"{'...' if len(val_episodes) > 10 else ''}"
            )

        # Build policy / processors / optimizer fresh for this run, dispatched on the policy type
        # (ACT today; GR00T when --policy.type groot — see _build_policy).
        policy, preprocessor, postprocessor, optimizer, lr_scheduler = _build_policy(cfg, dataset)

        # Dataloaders (learning/data/dataset.py)
        train_loader = make_dataloader(dataset, cfg, train_episodes, shuffle=True, device=device)
        # Validation must see RAW frames — image augmentation is TRAIN-ONLY. gpu_aug runs only in the
        # train loop; CPU image_transforms (image_aug=cpu) are baked into the dataset and applied in
        # __getitem__ on EVERY loader, so hand val a transform-free view. Augmenting val would measure
        # a distribution the policy never sees at deployment AND randomize the (otherwise fixed) val
        # estimator, corrupting the val curve + best-checkpoint selection. No-op for off/gpu (their
        # dataset carries no transforms).
        val_dataset = _raw_frame_view(dataset)
        # Val runs on a FIXED, seeded-random subset (val_max_batches*batch_size frames drawn across all
        # val episodes; 0 = full set) — representative + reproducible, reused identically every pass
        # (seeded by split_seed, so it's tied to the split, not the global RNG).
        val_loader = (
            make_dataloader(
                val_dataset, cfg, val_episodes, shuffle=False, device=device,
                max_samples=extras.val_max_batches * cfg.batch_size, sample_seed=cfg.split_seed)
            if len(val_episodes) > 0
            else None
        )

        # Prepare with accelerator
        policy, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            policy, optimizer, train_loader, lr_scheduler
        )
        if val_loader is not None:
            val_loader = accelerator.prepare(val_loader)
        # Normalization (Normalize/Unnormalize) lives outside the policy, so its stat buffers must be
        # moved to the accelerator's device too — the prepared loaders yield batches on that device.
        preprocessor = preprocessor.to(accelerator.device)
        postprocessor = postprocessor.to(accelerator.device)

        # Optionally torch.compile the INNER ACT model (the heavy ResNet+transformer). NOT the
        # ACTPolicy wrapper: its forward builds a loss dict via .item(), which graph-breaks dynamo
        # and blocks reduce-overhead's CUDA-graph capture; the inner model has no .item().
        if cfg.compile_mode is not None:
            inner = accelerator.unwrap_model(policy)
            inner.model = torch.compile(inner.model, mode=cfg.compile_mode)
            if is_main:
                logging.info(f"torch.compile on inner ACT model (mode={cfg.compile_mode})")

        # WandB (enabled by config; a failed init must crash loudly, not silently train unlogged)
        wandb_logger = None
        if cfg.wandb.enable and cfg.wandb.project and is_main:
            wandb_logger = WandBLogger(cfg, extras)

        # Forward-kinematics task-space validation (fingertip/wrist mm + per-group action L1).
        # Optional + robot-specific (ALLEX FK today). AllexFK is imported lazily so pytorch_kinematics
        # is needed only when enabled; a missing URDF must crash loudly, not silently skip the metric.
        fk = None
        if extras.fk_validation:
            from learning.metrics.allex_fk import DEFAULT_URDF, AllexFK
            urdf = extras.urdf_path or DEFAULT_URDF
            fk = AllexFK(urdf, device="cpu")
            if is_main:
                logging.info(f"FK validation ON (urdf={urdf}, {len(fk.pk_joints)} DOF, "
                             f"video_n={extras.fk_video_n})")

        # GPU image augmentation (train-only), selected by --dataset.image_aug gpu. The "cpu" option
        # runs the SAME apply_image_aug per-item in the dataloader instead; "off" → no aug. Here we
        # apply it on-device to each camera tensor of the assembled batch.
        gpu_aug = None
        if cfg.dataset.image_aug == "gpu":
            cam_keys = list(dataset.meta.camera_keys)
            aug_cfg = cfg.dataset.image_aug_config

            def gpu_aug(batch, _keys=cam_keys, _cfg=aug_cfg):
                for cam in _keys:
                    if cam in batch:
                        batch[cam] = apply_image_aug(batch[cam], _cfg)
                return batch

            if is_main:
                logging.info(f"GPU augmentation ON (train-only) for cameras {cam_keys}")

        # Run length + validation cadence (step-driven; epoch is only a reshuffle boundary).
        total_steps = cfg.steps
        max_epochs = extras.epochs or None    # None/0 -> run to total_steps; else run this many full passes
        val_every = extras.val_every_n_steps    # 0 -> validate at each epoch end
        world_size = accelerator.num_processes              # GPUs / DDP processes
        eff_batch_size = cfg.batch_size * world_size        # per-optimizer-step samples across all GPUs
        if is_main:
            run_len = f"{max_epochs} epochs" if max_epochs else f"{total_steps} steps"
            val_cad = f"every {val_every} steps" if val_every else "epoch end"
            logging.info(f"run: {run_len}    batch_size: {cfg.batch_size} x {world_size} gpu "
                         f"= {eff_batch_size} eff    iters/epoch ≈ {len(train_loader)}    val: {val_cad}")
            if val_loader is not None:
                vb = extras.val_max_batches
                val_set = (f"seeded-random {vb * cfg.batch_size}-frame subset "
                           f"(~{len(val_loader)} batches, seed={cfg.split_seed})"
                           if vb > 0 else f"full val set ({len(val_loader)} batches)")
                logging.info(f"val data: {val_set}")

        best_sel = float("inf")           # best value of the selection metric (--select_metric, lower better)
        best_sel_step = None              # step of the best selection metric — protected from pruning
        no_improve = 0
        last_fk_video_step = -(10 ** 9)   # rate-limit FK-video renders (slow matplotlib 3D)
        select_warned = False

        def _save_step(step):
            """Save ``checkpoints/step_<step>/``, point ``checkpoints/last`` at it, and prune to the
            newest ``--save_last_n`` step dirs plus the best-select_metric step (never pruned).
            Idempotent per step; only on the main process and when checkpointing is on."""
            if not (is_main and cfg.save_checkpoint and extras.save_last_n > 0):
                return
            ckpt_root = out_dir / "checkpoints"
            step_dir = ckpt_root / f"step_{step:08d}"
            if not step_dir.exists():
                save_checkpoint(
                    checkpoint_dir=step_dir, step=step, cfg=cfg,
                    policy=accelerator.unwrap_model(policy), optimizer=optimizer,
                    scheduler=lr_scheduler, preprocessor=preprocessor, postprocessor=postprocessor)
            update_last_checkpoint(step_dir)                 # checkpoints/last -> newest step
            for d in _prune_step_dirs(list(ckpt_root.glob("step_*")), extras.save_last_n, best_sel_step):
                shutil.rmtree(d, ignore_errors=True)

        def _select_value(val_loss, pm):
            """Pull the lower-is-better selection metric (--select_metric) from the merged val results;
            fall back to val_loss if the key isn't present (warn once)."""
            nonlocal select_warned
            key = extras.select_metric
            if key in pm:
                return pm[key], key
            if key in ("val_loss", "loss"):
                return val_loss, "val_loss"
            if not select_warned and is_main:
                logging.warning(f"select_metric '{key}' not in val metrics {sorted(pm)}; "
                                f"falling back to val_loss for selection/early-stop.")
                select_warned = True
            return val_loss, "val_loss(fallback)"

        def _validate_and_log(global_step, epoch, train_loss, iters_per_sec, epoch_time_secs=None):
            """Full val suite + wandb log (at the true global step) + best/rolling checkpoint + FK video
            + early-stop. Returns ``should_stop``. Used for BOTH step- and epoch-cadence validation."""
            nonlocal best_sel, best_sel_step, no_improve, last_fk_video_step
            # Validation/metrics use custom policy attrs (config, predict_action_chunk) that DDP does
            # NOT proxy — under multi-GPU `policy` is a DistributedDataParallel wrapper, so eval on the
            # unwrapped module (no-op on a single GPU / already-unwrapped policy).
            eval_policy = accelerator.unwrap_model(policy)
            val_loss = None
            pm: dict[str, float] = {}
            per_h_metrics: dict[str, float] = {}
            fk_metrics: dict[str, float] = {}
            is_allex_metrics = True   # False for EE-delta embodiments (LibERO) -> non-FK [val] summary
            if val_loader is not None:
                # val_loader is ALREADY the fixed representative val subset (seeded-random, built in
                # run() — see make_dataloader), so each pass runs the WHOLE loader. val_loss is its
                # own pass (train-mode forward, BN-protected + RNG-forked); all predict-based metrics
                # (per-horizon + group L1 + FK) share ONE eval pass instead of three that each re-ran
                # predict + re-decoded the val video.
                # Cap the epoch-level val passes (0 = full set). For slow-predict policies (GR00T's
                # denoising) a full pass is costly, so --val_eval_max_batches bounds it; ACT leaves
                # it 0 (full), unchanged.
                vcap = extras.val_eval_max_batches
                val_loss = compute_val_loss(eval_policy, val_loader, preprocessor, accelerator,
                                            max_batches=vcap)
                horizons = ([int(x) for x in extras.rollout_horizons.split(",") if x.strip()]
                            if extras.rollout_horizons else [])
                # group-L1 (l1_*_norm/_unnorm) is a pure action-space metric — always compute it,
                # independent of FK (which is opt-in and needs the URDF). Otherwise FK-off runs would
                # log no val L1 at all.
                # Embodiment-aware metric layout: ALLEX (ACT, or GR00T modality=allex) keeps the 44-D
                # group L1 + FK exactly. A non-ALLEX GR00T embodiment uses its spec's declared
                # val_metric_groups if present (e.g. LibERO's pos/rot/gripper L1) — else a plain
                # headline "all" L1 over ITS flat action width — and no ALLEX FK (the URDF doesn't apply).
                emb_spec = getattr(eval_policy, "spec", None)
                is_allex_metrics = emb_spec is None or getattr(emb_spec, "name", "allex") == "allex"
                if not is_allex_metrics:
                    val_groups = getattr(emb_spec, "val_metric_groups", None)
                    if val_groups:
                        metric_groups = dict(val_groups)
                    else:
                        flat_dim = max(b for _, (_, b) in emb_spec.action_groups.items())
                        metric_groups = {"all": (0, flat_dim)}
                    metric_fk = None
                else:
                    metric_groups, metric_fk = ALLEX_ACTION_GROUPS, fk
                # Flow policies: optionally override the sampler's Euler step count for val (inference-
                # only knob); restore after. no-op for ACT (no set_inference_steps). sample_seed fixes
                # the sampler's init noise so the sampled l1/fk are precise/repeatable across evals.
                vfms = extras.val_flow_matching_steps
                old_steps = (eval_policy.set_inference_steps(vfms)
                             if vfms > 0 and hasattr(eval_policy, "set_inference_steps") else None)
                try:
                    pm = compute_prediction_metrics(
                        eval_policy, val_loader, preprocessor, postprocessor, accelerator,
                        horizons=horizons, groups=metric_groups, fk=metric_fk, max_batches=vcap,
                        sample_seed=cfg.split_seed)
                finally:
                    if old_steps is not None:
                        eval_policy.set_inference_steps(old_steps)
                # EE-delta embodiments (e.g. LibERO OSC_POSE): the ALLEX URDF-FK doesn't apply, so ALSO
                # report physical-unit EE error (mm/deg) + gripper metrics from the spec's declarative
                # cartesian_metrics / gripper_dim — merged in under val/ alongside the l1_* group L1.
                if not is_allex_metrics:
                    cart = getattr(emb_spec, "cartesian_metrics", None)
                    grip = getattr(emb_spec, "gripper_dim", None)
                    if cart or grip is not None:
                        pm.update(compute_cartesian_gripper_metrics(
                            eval_policy, val_loader, preprocessor, accelerator,
                            cartesian_metrics=cart or (), gripper_dim=grip, max_batches=vcap))
                # DDP: the val_loader is sharded across ranks, so each rank computed metrics on its
                # own shard — mean-reduce so the logged values reflect the WHOLE val set (no-op 1 GPU).
                val_loss = _reduce_metric_mean(accelerator, val_loss)
                pm = {k: _reduce_metric_mean(accelerator, v) for k, v in pm.items()}
                # split only for stdout formatting; both groups log under val/ in wandb
                per_h_metrics = {k: v for k, v in pm.items()
                                 if k.startswith("loss_h0") or k.startswith("rollout_h")}
                fk_metrics = {k: v for k, v in pm.items() if k not in per_h_metrics}

            if is_main:
                msg = f"step {global_step:>7}/{total_steps} (ep {epoch}) | train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.4f}"
                for k, v in per_h_metrics.items():
                    msg += f" | {k}={v:.4f}"
                msg += f" | lr={optimizer.param_groups[0]['lr']:.2e} | {iters_per_sec:.1f} it/s"
                logging.info(msg)
                if fk_metrics and is_allex_metrics:
                    logging.info(
                        "  [val] "
                        f"all_l1(rad)={fk_metrics.get('l1_all_unnorm', float('nan')):.4f}  "
                        f"arm_l1(rad)={fk_metrics.get('l1_arm_unnorm', float('nan')):.4f}  "
                        f"finger_l1(rad)={fk_metrics.get('l1_finger_unnorm', float('nan')):.4f}  "
                        f"finger_mm={fk_metrics.get('fk_mm_finger_mean', float('nan')):.1f}  "
                        f"R_wrist_mm={fk_metrics.get('fk_mm_R_wrist', float('nan')):.1f}  "
                        f"L_wrist_mm={fk_metrics.get('fk_mm_L_wrist', float('nan')):.1f}  "
                        f"wrist_deg={fk_metrics.get('fk_deg_wrist_mean', float('nan')):.1f}")
                elif fk_metrics:
                    # EE-delta / generic embodiment (no ALLEX FK): report only the metrics THIS
                    # embodiment's spec actually declares — a plain "all" L1 headline for a generic
                    # 1-group embodiment (e.g. mikasa), plus pos/rot L1 + physical-unit EE (mm/deg) +
                    # gripper for one that declares them (LibERO). Absent keys are skipped, not printed
                    # as nan.
                    _val_fields = (
                        ("all_l1", "l1_all_unnorm", ".4f"), ("pos_l1", "l1_pos_unnorm", ".4f"),
                        ("rot_l1", "l1_rot_unnorm", ".4f"), ("mm_pos", "mm_pos", ".1f"),
                        ("deg_rot", "deg_rot", ".2f"), ("grip_l1", "gripper_l1", ".4f"),
                        ("grip_acc", "gripper_acc", ".3f"),
                    )
                    shown = "  ".join(f"{label}={format(fk_metrics[key], fmt)}"
                                      for label, key, fmt in _val_fields if key in fk_metrics)
                    logging.info(f"  [val] {shown}")
                extras_str = "".join(f" {k}={v:.4f}" for k, v in per_h_metrics.items())
                print(f"[PROGRESS] step {global_step}/{total_steps} train_loss={train_loss:.4f}"
                      + (f" val_loss={val_loss:.4f}" if val_loss is not None else "")
                      + extras_str, flush=True)

                # In-training CLOSED-LOOP sim eval (main process only): every --sim_eval_every_n_steps,
                # roll out the LIVE policy in the sim over the RPC boundary and log the real task-success
                # rate (sim_eval/SR) next to the offline metrics. Piggybacks on this validation point —
                # the policy is already unwrapped (eval_policy) and in eval mode; train mode is restored
                # at the end of this function. Imported lazily so a run without sim eval never pulls the
                # sim/RPC boundary onto the path. Fail loud: a broken eval must surface, not be swallowed.
                sim_metrics: dict[str, float] = {}
                sim_videos: list[tuple[str, str]] = []
                if (extras.sim_eval_every_n_steps > 0 and global_step > 0
                        and global_step % extras.sim_eval_every_n_steps == 0):
                    from learning.eval.intrain_sim_eval import run_sim_eval
                    sim_metrics, sim_videos = run_sim_eval(
                        eval_policy, suite=extras.sim_eval_suite, tasks=extras.sim_eval_tasks,
                        sim_python=extras.sim_eval_sim_python, chunk_size=cfg.policy.chunk_size,
                        device=str(accelerator.device), num_episodes=extras.sim_eval_episodes,
                        base_seed=extras.sim_eval_base_seed, resolution=extras.sim_eval_resolution,
                        max_episode_steps=extras.sim_eval_max_episode_steps or None,
                        replan_steps=extras.sim_eval_replan_steps,
                        num_inference_timesteps=extras.sim_eval_inference_timesteps or None,
                        video_episodes=extras.sim_eval_video_episodes,
                        video_fps=extras.sim_eval_video_fps,
                        video_dir=str(out_dir / "sim_eval_videos" / f"step_{global_step:08d}"),
                        server_script=extras.sim_eval_server_script or None)
                    logging.info("  [sim_eval] SR=%.4f  (%d/%d episodes over %d task(s))",
                                 sim_metrics["sim_eval/SR"], sim_metrics["sim_eval/n_success"],
                                 sim_metrics["sim_eval/n_episodes"], sim_metrics["sim_eval/n_tasks"])
                    print(f"[SIM_EVAL] step={global_step} SR={sim_metrics['sim_eval/SR']:.4f}", flush=True)

                if wandb_logger is not None:
                    log_dict = {
                        "train/loss": train_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "perf/iters_per_sec": iters_per_sec,            # end-to-end step rate
                        "perf/step_time_secs": 1.0 / max(iters_per_sec, 1e-9),   # per-step latency (s)
                        "perf/world_size": world_size,                 # GPUs (DDP processes)
                        "perf/eff_batch_size": eff_batch_size,         # batch_size x world_size
                        "perf/samples_per_sec": iters_per_sec * eff_batch_size,
                    }
                    if epoch_time_secs is not None:
                        log_dict["perf/epoch_time_secs"] = epoch_time_secs   # whole-epoch wall-time (s)
                    if val_loss is not None:
                        log_dict["val/loss"] = val_loss
                        log_dict["val/train_val_gap"] = train_loss - val_loss
                    for k, v in per_h_metrics.items():
                        log_dict[f"val/{k}"] = v
                    for k, v in fk_metrics.items():
                        log_dict[f"val/{k}"] = v
                    log_dict.update(sim_metrics)   # sim_eval/SR (+ per-task) — empty when not a sim-eval step
                    wandb_logger.log_dict(log_dict, global_step)
                    # Rollout MP4s go through the media path (log_dict is scalar-only and drops
                    # wandb.Video), same as log_fk_video — onto the SAME step as the SR scalars.
                    for key, path in sim_videos:
                        wandb_logger._wandb.log(
                            {key: wandb_logger._wandb.Video(path, fps=extras.sim_eval_video_fps,
                                                            format="mp4")}, step=global_step)

            # Selection metric (--select_metric, lower better) drives best-checkpoint retention AND
            # early-stop. Default l1_all_unnorm — the FK-free SAMPLED action L1: the right signal for a
            # generative head, whose denoising val_loss is a surrogate that overfits while the sampled
            # prediction keeps improving (val_loss would freeze the checkpoint on the wrong early step —
            # exactly what lost the converged GR00T model).
            sel_value, sel_key = _select_value(val_loss, pm)
            best_sel, no_improve, improved, should_stop = _update_best(
                sel_value, best_sel, no_improve, extras.early_stop_patience)
            if improved and sel_value is not None:
                best_sel_step = global_step                        # all ranks: keep the protected step consistent
                if is_main:
                    logging.info(f"new best {sel_key} = {best_sel:.4f} (step {global_step})")
            # FK skeleton videos (pred vs GT) on a new best, RATE-LIMITED: matplotlib-3D rendering is
            # slow (~min/clip), so render at most once per --fk_video_every_n_steps. Without this,
            # frequent validation (small --val_every_n_steps) would render on nearly every step early
            # in training (every val is a new best) and the run becomes rendering-bound.
            if (is_main and fk is not None and extras.fk_video_n > 0 and improved
                    and wandb_logger is not None and val_loader is not None
                    and global_step - last_fk_video_step >= extras.fk_video_every_n_steps):
                log_fk_video(
                    wandb_logger, eval_policy, val_loader, preprocessor, postprocessor, accelerator,
                    fk, global_step, out_dir / "fk_videos", n_videos=extras.fk_video_n)
                last_fk_video_step = global_step
            # Rolling step checkpoint at this validation's step — so the just-updated best-select step is
            # on disk and protected from pruning. Kill-safety between validations comes from the
            # decoupled periodic save in the training loop (--save_every_n_steps).
            _save_step(global_step)
            # Best-checkpoint POINTER (metadata, not a duplicate copy): the best-select step is one of
            # the retained step_<N>/ dirs (protected from pruning above). Record which one in
            # checkpoints/best.json, refreshed on every new best — so the best checkpoint is found
            # directly (local or on S3) without digging the training logs. We point, never duplicate.
            if improved and sel_value is not None and is_main and cfg.save_checkpoint and extras.save_last_n > 0:
                (out_dir / "checkpoints" / "best.json").write_text(json.dumps(
                    {"select_metric": sel_key, "value": best_sel, "step": best_sel_step,
                     "checkpoint": f"step_{best_sel_step:08d}"}, indent=2))
            if should_stop and is_main:
                logging.info(f"[EARLY STOP] no {sel_key} improvement for {no_improve} validations "
                             f"(patience={extras.early_stop_patience}); stop at step {global_step}, "
                             f"best={best_sel:.4f}.")
                print(f"[EARLY_STOP] step={global_step} best={best_sel:.4f}", flush=True)
            # Restore train mode LAST: the val helpers — including log_fk_video, which runs after the
            # metrics above — leave the policy in eval, and step-cadence training resumes right here
            # (no per-epoch policy.train() to mask it), so ACT's VAE path would otherwise see None.
            policy.train()
            return should_stop

        def _log_train(global_step, epoch, train_loss, iters_per_sec, epoch_time_secs=None):
            """Fine train/loss curve (stdout + wandb) — monitoring only, never validates. Carries the
            just-finished epoch's wall-time (perf/epoch_time_secs) when one ended since the last log."""
            if not is_main:
                return
            lr = optimizer.param_groups[0]["lr"]
            print(f"[STEP] ep{epoch} step {global_step}/{total_steps} "
                  f"train_loss={train_loss:.4f} lr={lr:.2e} {iters_per_sec:.1f} it/s", flush=True)
            if wandb_logger is not None:
                d = {"train/loss": train_loss, "train/lr": lr, "train/epoch": epoch,
                     "perf/iters_per_sec": iters_per_sec,
                     "perf/step_time_secs": 1.0 / max(iters_per_sec, 1e-9),
                     "perf/world_size": world_size,
                     "perf/eff_batch_size": eff_batch_size,
                     "perf/samples_per_sec": iters_per_sec * eff_batch_size}
                if epoch_time_secs is not None:
                    d["perf/epoch_time_secs"] = epoch_time_secs
                wandb_logger.log_dict(d, global_step)

        log_every = extras.log_every_n_steps   # 0 -> log at epoch end instead of every N steps
        # Fail fast on a misconfigured in-training sim eval BEFORE the run starts (not N steps in).
        if extras.sim_eval_every_n_steps > 0:
            assert extras.sim_eval_suite and extras.sim_eval_tasks and extras.sim_eval_sim_python, (
                "in-training sim-eval enabled (--sim_eval_every_n_steps > 0) but one of "
                "--sim_eval_suite / --sim_eval_tasks / --sim_eval_sim_python is unset")
            # Single-process only for now: the eval runs on rank 0, so under multi-GPU the other ranks
            # would stall at the next all-reduce (NCCL timeout). Fail loud rather than hang; rank-0 eval
            # + a barrier is a follow-up. (Sim eval is a monitoring signal — run its config on 1 GPU.)
            assert world_size == 1, (
                f"in-training sim-eval is single-process for now, but world_size={world_size}; "
                "run the sim-eval config on 1 GPU")
        global_step = 0
        epoch = 0
        stop = False
        pending_epoch_secs = None   # last finished epoch's wall-time; ridden out on the next log/val
        # running windows: 'log' for the fine train/loss curve, 'val' for the val point's train_loss
        log_loss, log_n, log_t0 = 0.0, 0, time.perf_counter()
        val_loss_acc, val_n_acc, val_t0 = 0.0, 0, time.perf_counter()

        while not stop:                       # epoch is 0-indexed (train/epoch starts at 0)
            if max_epochs is not None and epoch >= max_epochs:
                break
            policy.train()
            ep_t0 = time.perf_counter()
            ep_steps = 0
            for batch in train_loader:
                if gpu_aug is not None:
                    batch = gpu_aug(batch)    # GPU image aug on raw [0,1] frames, before normalization
                if global_step == 0:
                    # Show the augmented first batch from ALL ranks (each GPU augments its own shard
                    # independently), not just rank 0. Every rank must join each gather (collective);
                    # only main holds the wandb run and logs. gather is a no-op on a single GPU.
                    # (Already GPU-augmented when --dataset.image_aug=gpu, since aug runs above.)
                    # Gather the frames (for the grids) plus state/action (for the per-sample info
                    # table); task strings gather as objects (accelerator.gather is tensors-only).
                    gathered = {k: accelerator.gather(v) for k, v in batch.items()
                                if k.startswith("observation.images.")
                                or k in ("observation.state", "action")}
                    if "task" in batch:
                        gathered["task"] = gather_object(list(batch["task"]))
                    if wandb_logger is not None:
                        log_first_batch_aug(wandb_logger, gathered, cfg.dataset.image_aug_config,
                                            cfg.dataset.image_aug,
                                            policy=accelerator.unwrap_model(policy))
                batch = preprocessor(batch)
                with accelerator.autocast():
                    loss, _ = policy.forward(batch)
                accelerator.backward(loss)
                if cfg.optimizer.grad_clip_norm > 0:
                    accelerator.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), float("inf"), error_if_nonfinite=False)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                lv = loss.detach()    # keep on-GPU; no per-step .item() sync (it stalls the pipeline)
                global_step += 1
                ep_steps += 1
                log_loss += lv
                log_n += 1
                val_loss_acc += lv
                val_n_acc += 1

                # Step-cadence train log (default; monitoring only). log_every == 0 -> log at epoch end.
                if log_every and global_step % log_every == 0:
                    running = float(log_loss) / max(log_n, 1)   # float() syncs here, not per-step
                    ips = log_n / max(time.perf_counter() - log_t0, 1e-9)
                    _log_train(global_step, epoch, running, ips, pending_epoch_secs)
                    pending_epoch_secs = None
                    log_loss, log_n, log_t0 = 0.0, 0, time.perf_counter()

                # Step-cadence full validation. val_every == 0 -> validate at epoch end.
                if val_every and global_step % val_every == 0:
                    tl = float(val_loss_acc) / max(val_n_acc, 1)   # float() syncs here, not per-step
                    ips = val_n_acc / max(time.perf_counter() - val_t0, 1e-9)
                    stop = _validate_and_log(global_step, epoch, tl, ips, pending_epoch_secs)
                    pending_epoch_secs = None
                    val_loss_acc, val_n_acc, val_t0 = 0.0, 0, time.perf_counter()
                    if stop:
                        break

                # Decoupled kill-safety save: a rolling step checkpoint every --save_every_n_steps,
                # independent of the (possibly slow/infrequent) validation. Skip if this step already
                # saved via validation just above. _save_step is idempotent + main-process-gated.
                if (extras.save_every_n_steps and global_step % extras.save_every_n_steps == 0
                        and not (val_every and global_step % val_every == 0)):
                    _save_step(global_step)

                if max_epochs is None and global_step >= total_steps:
                    stop = True
                    break

            ep_secs = time.perf_counter() - ep_t0
            # Run the epoch-end reporting + increment ONLY if this epoch ran to completion. `stop` is
            # set exclusively inside the batch loop (step-budget reached or early-stop), so `stop` here
            # means the loop broke mid-epoch — in that case don't log it "complete" or count it.
            if not stop:
                if is_main:
                    logging.info(f"epoch {epoch} complete: {ep_steps} steps in {ep_secs:.1f}s")
                # The epoch boundary is only a reshuffle point + a reporting hook: stash this epoch's
                # wall-time so the NEXT step-cadence log/val event carries perf/epoch_time_secs (no
                # standalone per-epoch wandb write); when a cadence is in epoch mode (0), do it here.
                # (An epoch shorter than the log/val cadence may drop its epoch_time_secs sample.)
                pending_epoch_secs = ep_secs
                if not log_every:                          # log-at-epoch-end mode
                    running = float(log_loss) / max(log_n, 1)   # float() syncs here, not per-step
                    ips = log_n / max(time.perf_counter() - log_t0, 1e-9)
                    _log_train(global_step, epoch, running, ips, pending_epoch_secs)
                    pending_epoch_secs = None
                    log_loss, log_n, log_t0 = 0.0, 0, time.perf_counter()
                if not val_every:                          # validate-at-epoch-end mode
                    tl = float(val_loss_acc) / max(val_n_acc, 1)   # float() syncs here, not per-step
                    ips = val_n_acc / max(time.perf_counter() - val_t0, 1e-9)
                    stop = _validate_and_log(global_step, epoch, tl, ips, pending_epoch_secs)
                    pending_epoch_secs = None
                    val_loss_acc, val_n_acc, val_t0 = 0.0, 0, time.perf_counter()
                epoch += 1                     # 0-indexed: count only fully-completed epochs

            accelerator.wait_for_everyone()

        # Final checkpoint at the true last step (the rolling step_<N> saves + 'last' symlink already
        # ran during training + validation; this captures the exact final weights + re-points 'last').
        if is_main and cfg.save_checkpoint:
            _save_step(global_step)
            logging.info(f"saved final checkpoint: step_{global_step:08d}")

        if wandb_logger is not None:
            wandb_logger._wandb.run.finish()

        return {"steps": global_step, "epochs": epoch, "best": best_sel,
                "best_metric": extras.select_metric, "best_step": best_sel_step}
