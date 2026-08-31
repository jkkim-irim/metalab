"""ALLEX learning — Newton RL policy + sim-service client (learning venv; no Isaac Lab).

The client side of the Isaac Lab sim service, plus the policy it runs. Layout:
  - sim-service glue: ``client.py`` (the ``SimServiceVecEnv`` proxy) + ``service.py`` (spawn the
    server + wire the transport), over ``vec_env.py``.
  - the policy nets: ``models/`` (actor/critic) + ``nn/`` (NN blocks) + ``utils/`` — adapted from
    rsl_rl (rsl-rl-lib v5.0.1, BSD-3-Clause, © ETH Zürich & NVIDIA; per-file headers retained).
  - the experiment: ``dexblind/hammer_lift/experiment.py`` (the policy spec + reward/curriculum tunables).

Holds no Isaac Lab dependency — the Isaac env lives behind the server (``sim/isaaclab``), reached over
the RPC boundary. Eval a checkpoint over the boundary with ``learning/eval/eval_service.py`` (it builds
the actor and loads its weights — no training engine). The PPO trainer (ppo / rollout storage /
on-policy runner) is a follow-up, not in this PR.
"""
