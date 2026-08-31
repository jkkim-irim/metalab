"""Sim-service process boundary — the unified RPC + CUDA-IPC transport shared by EVERY ALLEX sim eval
service (isaaclab, metalab genesis/newton, libero, maniskill).

- :mod:`sim.service.transport` — the single wire protocol: socket endpoints (``RpcServer``/``RpcClient``)
  with the socket-RPC v1 path (``serve``/``call`` — libero/maniskill/eval_sim) + the CUDA-IPC
  shared-buffer hot path (handles exchanged once at connect; per-step payload stays on the GPU) +
  ``serve_vec_env`` (the handler-based vec_env server loop for the GPU RL sim-service).

Servers put this dir on ``sys.path`` and ``from transport import ...``; the MetaLab spokes adapt their
duck-typed VecEnv to the handler via :func:`sim.metalab.runtime.vec_env_handler.make_vec_env_handler`.
The client end is ``learning/rl/client.SimServiceVecEnv``.
"""
