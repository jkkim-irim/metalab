"""Eval entry points for learning/ — kept out of ``learning.rl`` so the PPO trainer's
``resolve_callable`` module scan can't pick up a config-dict named like a class (see
``policies/actor.py``)."""
