# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import subprocess
import threading
import time

import torch

from learning.rl.models import MLPModel
from learning.rl.ppo import PPO
from learning.rl.utils import check_nan, resolve_callable
from learning.rl.utils.logger import Logger
from learning.rl.vec_env import VecEnv


class OnPolicyRunner:
    """On-policy runner for reinforcement learning algorithms."""

    alg: PPO
    """The actor-critic algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu",
                 on_checkpoint=None, ckpt_s3_root: str | None = None) -> None:
        """Construct the runner, algorithm, and logging stack.

        ``on_checkpoint``: optional engine-agnostic callback ``fn(path)`` fired right after each checkpoint
        is written (the trainer uses it to record eval videos of that checkpoint — see
        rl_trainer._make_record_callback).

        ``ckpt_s3_root``: ``s3://.../<user>/sim_rl/ckpts`` to upload each checkpoint to as it is written,
        or ``None`` for a local run (no upload). The caller (rl_trainer) requires a non-None value on
        managed runs and only passes None for an explicit ``--local`` run."""
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.on_checkpoint = on_checkpoint

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[PPO] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
            max_episode_length=int(self.env.max_episode_length),
        )

        self.current_learning_iteration = 0

        # Background S3 checkpoint uploads. Each save() fires a thread that pushes the just-written
        # model_<it>.pt to ckpt_s3_root the moment it exists (durable immediately, not batch-copied at
        # run end), tracked here so learn() can wait for them to drain before the process exits.
        self._ckpt_s3_root = ckpt_s3_root
        self._ckpt_upload_threads: list[threading.Thread] = []

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False,
              iter_callback=None, callback_interval: int = 0) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

            # Periodic user callback (e.g. a validation-video rollout). Runs OUTSIDE the rollout's
            # inference_mode; it may spawn its own sim service and switch the policy to eval, so restore
            # train mode afterwards. The caller guards it so a failure never raises into the loop.
            if iter_callback is not None and callback_interval > 0 and (it + 1) % callback_interval == 0:
                iter_callback(it + 1)
                self.alg.train_mode()

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

        # Don't exit until every background checkpoint upload has drained — otherwise the process can die
        # (and the S3 uploads with it) right after the final save, losing the last checkpoints from S3.
        self._join_ckpt_uploads()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # Publish to S3 as soon as the checkpoint exists (see _upload_ckpt_s3). Not uploaded to W&B —
        # duplicating every checkpoint into W&B run files added no durability and dominated storage cost.
        self._upload_ckpt_s3(path)
        # Engine-agnostic post-save hook — the trainer records eval videos of this checkpoint here
        # (event-driven; no polling). The hook loads a FRESH actor from the file, so self.alg is untouched.
        if self.on_checkpoint is not None:
            self.on_checkpoint(path)

    def _upload_ckpt_s3(self, path: str) -> None:
        """Push a just-written checkpoint to S3 in the background (no-op if ``ckpt_s3_root`` is None).

        Uploads to ``<ckpt_s3_root>/<run_name>/<file>`` where ``run_name`` is the log_dir basename, matching
        the key scheme the eval/launch scripts expect. Runs off-thread so the training loop never blocks on
        S3; the thread is tracked and joined in :meth:`_join_ckpt_uploads`. Only the rank-0 process (the one
        that owns the checkpoint) uploads. Failures are logged loudly, not raised — the local copy and the
        launch script's end-of-run backstop still cover a transient S3 error.
        """
        if not self._ckpt_s3_root or self.gpu_global_rank != 0:
            return
        run_name = os.path.basename(os.path.normpath(self.logger.log_dir))
        dest = f"{self._ckpt_s3_root.rstrip('/')}/{run_name}/{os.path.basename(path)}"

        def _upload() -> None:
            try:
                result = subprocess.run(["aws", "s3", "cp", path, dest], capture_output=True, text=True)
            except Exception as exc:  # e.g. aws CLI not on PATH — log loudly, don't kill training
                print(f"[rl-trainer] S3 checkpoint upload ERROR {dest}: {exc}", flush=True)
                return
            if result.returncode != 0:
                print(f"[rl-trainer] S3 checkpoint upload FAILED {dest}: {result.stderr.strip()}", flush=True)
            else:
                print(f"[rl-trainer] S3 checkpoint uploaded {dest}", flush=True)

        thread = threading.Thread(target=_upload, name=f"ckpt-s3-{os.path.basename(path)}")
        thread.start()
        self._ckpt_upload_threads.append(thread)

    def _join_ckpt_uploads(self) -> None:
        """Block until all in-flight background checkpoint uploads have finished."""
        pending = [t for t in self._ckpt_upload_threads if t.is_alive()]
        if pending:
            print(f"[rl-trainer] waiting for {len(pending)} checkpoint upload(s) to finish...", flush=True)
        for thread in self._ckpt_upload_threads:
            thread.join()
        self._ckpt_upload_threads.clear()

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
