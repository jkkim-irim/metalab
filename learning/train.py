#!/usr/bin/env python
"""ALLEX learning — training entrypoint (behavior cloning: ACT / VLA).

Parses config, sets up the accelerator + dataset + episode split, and dispatches to a trainer in
learning/trainer/ (BCTrainer today; RLTrainer later). Custom extras are stripped from argv before
the config parser (learning.configs.parser) loads.

Run as a module:  python -m learning.train --policy.type act --dataset.repo_id ... [extras]

Extras (stripped from argv before the config parser):
    --epochs / --early_stop_patience / --rollout_horizons / --log_every_n_steps /
    --val_every_n_steps / --val_max_batches / --cameras / --fk_validation / --urdf_path / --fk_video_n /
    --sim_eval_* (in-training closed-loop sim eval — off by default; see _parse_extras)
The train/val split (val_ratio / split_seed) is explicit config on TrainPipelineConfig, not an extra.
"""

import argparse
import logging
import os
from pprint import pformat
import sys

# Leaf module (stdlib `signal` only) — safe to import here, ahead of the staged imports below that
# must not run until our extras are stripped from argv.
from learning.utils.signals import restore_default_sigint


# ---------------------------------------------------------------------------
# Step 1. Strip our extras from sys.argv BEFORE the config parser sees it.
# (argparse only — importing the config parser here would let it see our extras.)
# ---------------------------------------------------------------------------
def _parse_extras():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Run length as N full data passes. Omit or 0 = run to --steps (cfg.steps) "
                         "instead. Epochs are otherwise only a reporting/reshuffle boundary.")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="Stop if val_loss has not improved for N consecutive validations (0 disables).")
    # NOTE: the train/val split (--val_ratio / --split_seed / --val_only_first_n_episodes) is now an
    # explicit config on TrainPipelineConfig (parsed by the config parser, logged to wandb config),
    # not a stripped extra. Read below as cfg.val_ratio etc.
    ap.add_argument(
        "--rollout_horizons", type=str, default="",
        help="Comma-separated horizon indices (h < chunk_size) for per-horizon val MSE.",
    )
    ap.add_argument(
        "--log_every_n_steps", type=int, default=100,
        help="Log running train_loss + lr (+ throughput) every N optimizer steps, to stdout and "
             "wandb (monitoring only; does NOT affect training). Default 100. "
             "0 = log at each epoch end instead.",
    )
    ap.add_argument(
        "--val_every_n_steps", type=int, default=1000,
        help="Run the FULL validation suite (val_loss + per-horizon + group/FK + wandb + best-ckpt) "
             "every N optimizer steps (default 1000). 0 = validate at each epoch end instead "
             "(natural for small datasets; step cadence suits large ones whose epoch is very long).",
    )
    ap.add_argument(
        "--val_max_batches", type=int, default=40,
        help="Validate on a FIXED seeded-random sample of N batches (N*batch_size frames) drawn "
             "across ALL val episodes — representative + reproducible, reused identically every "
             "pass (NOT the first-N prefix). 0 = the full val set. Default 40: bounded (a full-holdout "
             "predict is slow for a generative policy). In-training val is a monitoring signal, not the "
             "final number — for a precise/definitive metric re-evaluate a saved step_<N> checkpoint "
             "OFFLINE rather than enlarging every in-training val pass.",
    )
    ap.add_argument(
        "--val_eval_max_batches", type=int, default=0,
        help="Cap each EPOCH-level val metric pass (val_loss + per-horizon + group + FK) to N "
             "batches (0 = full val set). Bounds the cost for slow-predict policies (GR00T's "
             "denoising makes a full val pass expensive); harmless for ACT (leave 0).",
    )
    ap.add_argument(
        "--save_every_n_steps", type=int, default=1000,
        help="Save a rolling step checkpoint (checkpoints/step_<N>/) every N optimizer steps, "
             "DECOUPLED from validation — the kill-safety cadence, so a killed/preempted job (or a "
             "node terminated for cost) loses at most N steps. A checkpoint is also written at every "
             "validation and once at clean end-of-run. 0 disables the periodic cadence. For a big model "
             "(GR00T 3B) raise this — a full-state save is heavy I/O.",
    )
    ap.add_argument(
        "--save_last_n", type=int, default=5,
        help="Keep the newest N step_<N> checkpoints (rolling window) PLUS the best-select_metric step "
             "(never pruned). 0 disables checkpointing. `checkpoints/last` symlinks the newest step.",
    )
    ap.add_argument(
        "--select_metric", type=str, default="l1_all_unnorm",
        help="Lower-is-better val metric that drives best-checkpoint retention AND early-stop. Default "
             "l1_all_unnorm: the FK-free, sampled (predict_action_chunk) action L1 in radians — the "
             "right selection signal for a generative/flow policy (its denoising `val/loss` is a "
             "surrogate that overfits while the sampled prediction keeps improving), and URDF-free "
             "(unlike fk_mm). Any logged val key works (e.g. val_loss, fk_mm_finger_mean); falls back "
             "to val_loss if the key is absent.",
    )
    ap.add_argument(
        "--val_flow_matching_steps", type=int, default=0,
        help="Override the flow-matching sampler's Euler step count for the val prediction metrics "
             "(FLOW POLICIES ONLY; no-op for ACT). 0 = use the policy's own num_inference_timesteps "
             "(deployment-faithful). The step count is an inference-only choice (training learns a "
             "continuous velocity field), so it's safe to raise here for a lower-discretization "
             "'capability' reading — but the deployment number stays the default.",
    )
    ap.add_argument(
        "--cameras", type=str, default="",
        help="Comma-separated camera names (e.g. 'camera_1,camera_2') to feed the policy; any other "
             "observation.images.* in the dataset are dropped from the policy inputs. Empty = use "
             "all cameras present in the dataset.",
    )
    ap.add_argument(
        "--fk_validation", action="store_true",
        help="Enable task-space validation metrics each val epoch: per-body-part action L1 "
             "(val/l1_*) and Cartesian fingertip/wrist error in mm via forward kinematics "
             "(val/fk_mm_*). Requires pytorch_kinematics + the ALLEX URDF.",
    )
    ap.add_argument(
        "--urdf_path", type=str, default="",
        help="URDF for FK validation (default: the canonical allex_rl_dexblind/.../ALLEX.urdf).",
    )
    ap.add_argument(
        "--fk_video_n", type=int, default=3,
        help="With --fk_validation: render N pred-vs-GT 3D skeleton MP4s to wandb on a new-best "
             "validation (spread across the val set; 0 = off). Needs matplotlib + imageio-ffmpeg.",
    )
    ap.add_argument(
        "--fk_video_every_n_steps", type=int, default=10000,
        help="Rate-limit FK-video rendering to at most once per N steps (matplotlib 3D is slow, "
             "~min/clip). Renders on the first new best at/after each N-step window; 0 = render on "
             "every new best (only sane when validation is infrequent).",
    )
    ap.add_argument(
        "--trainer", choices=["bc", "rl"], default="bc",
        help="Which trainer to run. 'bc' (default) = ACT/VLA behavior cloning in this venv. 'rl' = Newton "
             "RL via the Isaac Lab sim service — delegated to learning/trainer/rl_trainer.py (it spawns "
             "the sim server over the boundary and does NOT use the BC dataset/accelerate stack below).",
    )
    # In-training CLOSED-LOOP sim eval (GR00T/BC): every N steps, roll out the LIVE policy in the sim
    # over the RPC boundary and log a real task-success rate (sim_eval/SR) alongside the offline val
    # metrics. Off by default (--sim_eval_every_n_steps 0) so ACT / non-sim recipes are unaffected; the
    # sim runs in its OWN venv (--sim_eval_sim_python) — see learning/eval/intrain_sim_eval.py.
    ap.add_argument(
        "--sim_eval_every_n_steps", type=int, default=0,
        help="Run in-training closed-loop sim eval every N optimizer steps (0 = off). Piggybacks on the "
             "validation cadence, so use a multiple of --val_every_n_steps. Main process only; each run "
             "spins up + tears down the sim server, so keep N coarse (sim eval is minutes, not seconds).",
    )
    ap.add_argument("--sim_eval_suite", type=str, default="",
                    help="Sim-eval suite (e.g. 'libero_90' or 'mikasa'); selects the sim server + obs layout.")
    ap.add_argument("--sim_eval_tasks", type=str, default="",
                    help="Sim-eval task(s), comma-separated (LIBERO int task-ids, e.g. '0' or '0,11').")
    ap.add_argument("--sim_eval_episodes", type=int, default=8,
                    help="Episodes PER task per sim-eval run (default 8; keep small for in-training).")
    ap.add_argument("--sim_eval_sim_python", type=str, default="",
                    help="Python of the SIM venv (LIBERO/ManiSkill) that hosts the sim server subprocess.")
    ap.add_argument("--sim_eval_resolution", type=int, default=256, help="LIBERO square camera render size.")
    ap.add_argument("--sim_eval_max_episode_steps", type=int, default=0,
                    help="Cap the per-episode step budget (0 = the server's default for the task).")
    ap.add_argument("--sim_eval_replan_steps", type=int, default=0,
                    help="Execute only the first N steps of each predicted chunk before re-observing "
                         "(0 = the full chunk; the reference LIBERO eval used 5).")
    ap.add_argument("--sim_eval_base_seed", type=int, default=0, help="First episode seed (reproducible).")
    ap.add_argument("--sim_eval_inference_timesteps", type=int, default=0,
                    help="Temporarily lower the flow-matching denoising steps for the in-training sim eval "
                         "only (0 = the policy's deployment default) — a faster reading, weights untouched.")
    ap.add_argument("--sim_eval_video_episodes", type=int, default=0,
                    help="Capture the first N rollouts per task as MP4s and log them to the wandb run "
                         "(sim_eval/videos/<env_id>/ep<idx>) for visual review. 0 = scalars only.")
    ap.add_argument("--sim_eval_video_fps", type=int, default=15, help="FPS for the sim-eval rollout MP4s.")
    ap.add_argument("--sim_eval_server_script", type=str, default="",
                    help="Override the sim-server script path (default: per --sim_eval_suite).")
    # NOTE: image augmentation is configured via --dataset.image_aug (off|cpu|gpu), not an extra.
    extras, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return extras


EXTRAS = _parse_extras()

# Before ANY trainer runs: a detached launcher (Launchpad `nohup … &`, AWS `setsid … &`) hands us an
# ignored SIGINT, which would make Stop a silent no-op followed by SIGKILL — and the W&B run
# "crashed" instead of "killed". Both dispatches below are covered by this one call.
if __name__ == "__main__":
    restore_default_sigint()


# ---------------------------------------------------------------------------
# --trainer rl: Newton RL runs the sim service + PPO stack, not the BC dataset/accelerate stack below.
# RLTrainer is a peer of BCTrainer but lives behind the sim-service boundary, so dispatch HERE — before
# importing the ACT / accelerate / dataset modules (which RL does not use). It re-parses its own args
# (--num_envs / --max_iterations / --seed / --logger / --wandb_project) from the remaining argv.
# ---------------------------------------------------------------------------
if EXTRAS.trainer == "rl" and __name__ == "__main__":
    from learning.trainer.rl_trainer import RLTrainer

    raise SystemExit(RLTrainer(sys.argv[1:]).run())


# ---------------------------------------------------------------------------
# Step 2. Now wire in our modules — the config parser only sees the stripped argv.
# Run as a module (`python -m learning.train`) or with the package installed (`pip install -e .`),
# so `learning` imports without sys.path hacks.
# ---------------------------------------------------------------------------
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import DistributedDataParallelKwargs  # noqa: E402
from termcolor import colored  # noqa: E402

from learning.configs import parser  # noqa: E402
from learning.configs.config import TrainPipelineConfig, policy_type_name  # noqa: E402
from learning.data.dataset import make_dataset, split_single  # noqa: E402
from learning.trainer.bc_trainer import BCTrainer  # noqa: E402
from learning.utils.import_utils import register_third_party_plugins  # noqa: E402
from learning.utils.logging_utils import format_big_number, init_logging  # noqa: E402
from learning.utils.seed import set_seed  # noqa: E402


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    cfg.validate()

    # gradient_as_bucket_view: .grad aliases the DDP reduction bucket instead of a separate copy,
    # reclaiming ~one gradient-set of VRAM (~6.5 GB for GR00T's 1.6B trainable params) — without it,
    # multi-GPU GR00T OOMs on a 40 GB A100 (the bucket pushes the 37.8 GB single-GPU footprint over).
    # DDP kwargs are incompatible with DeepSpeed (it manages parallelism itself), so drop them when
    # accelerate launch enabled DeepSpeed (--use_deepspeed; e.g. ZeRO for the full-backbone finetune).
    use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() == "true"
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True, gradient_as_bucket_view=True)
    force_cpu = cfg.policy.device == "cpu"
    # GR00T (3B VLA) trains in bf16 mixed precision: model weights stay fp32 and accelerator.autocast()
    # casts the forward to bf16 (FlashAttention needs bf16; the flow-matching time sampler stays fp32).
    # ACT keeps fp32 ("no"), preserving its current behavior.
    mixed_precision = "bf16" if policy_type_name(cfg.policy) == "groot" else "no"
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[] if use_deepspeed else [ddp_kwargs],
        cpu=force_cpu,
        mixed_precision=mixed_precision,
    )

    init_logging(accelerator=accelerator)
    is_main = accelerator.is_main_process

    if is_main:
        logging.info("ALLEX BC trainer (train.py) — config:")
        logging.info(pformat(cfg.to_dict()))
        logging.info(
            f"EXTRAS: epochs={EXTRAS.epochs} val_every_n_steps={EXTRAS.val_every_n_steps} "
            f"val_max_batches={EXTRAS.val_max_batches}"
        )
        logging.info(f"split (config): val_ratio={cfg.val_ratio} split_seed={cfg.split_seed}")

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Load dataset
    if is_main:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main:
        dataset = make_dataset(cfg)

    # Restrict the policy's cameras (e.g. match a 2-camera reference run on a 3-camera dataset):
    # drop the unlisted observation.images.* from the dataset metadata so the policy builds only the
    # requested camera encoders. The dataset frames/episodes are unchanged.
    if EXTRAS.cameras:
        keep = {f"observation.images.{c.strip()}" for c in EXTRAS.cameras.split(",") if c.strip()}
        feats = dataset.meta.features
        dropped = [k for k in list(feats)
                   if k.startswith("observation.images.") and k not in keep]
        for k in dropped:
            feats.pop(k, None)
            dataset.meta.stats.pop(k, None)
        if is_main:
            logging.info(f"cameras: keeping {sorted(keep)}; dropped {dropped}")

    num_episodes = dataset.num_episodes
    if is_main:
        logging.info(
            f"dataset: {dataset.num_frames} frames  ({format_big_number(dataset.num_frames)}) "
            f"across {num_episodes} episodes"
        )

    # Episode-wise train/val split (explicit config: cfg.val_ratio / cfg.split_seed)
    train_ep, val_ep = split_single(num_episodes, cfg.val_ratio, cfg.split_seed)
    if is_main:
        logging.info(
            f"Split: train={len(train_ep)} ep, val={len(val_ep)} ep "
            f"(val_ratio={cfg.val_ratio}, split_seed={cfg.split_seed})"
        )

    # Dispatch to the trainer (BC today; RLTrainer would branch here later)
    result = BCTrainer(cfg, EXTRAS, accelerator, is_main).run(dataset, train_ep, val_ep)

    if is_main:
        logging.info(colored("\n=== Summary ===", "green", attrs=["bold"]))
        logging.info(f"  epochs={result['epochs']}  "
                     f"best {result['best_metric']}={result['best']:.4f} @ step {result['best_step']}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
