"""The SHARED eval protocol — single source of truth for every policy-evaluation context.

Used by BOTH the standalone eval (``learning.eval.eval_service``) and the in-training val-video hook
(``learning/trainer/rl_trainer._val_video``). These two paths previously duplicated this logic by hand
and drifted: val rollouts missed ``fixed_refs`` (so no deterministic references, no t=0 starts, no
hold-tail) and tagged successful episodes ``_fail``. Anything protocol-shaped belongs HERE, not inline.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from learning.rl.models import MLPModel, RNNModel
from learning.rl.utils import resolve_obs_groups


def eval_srv_args(num_envs: int, device: str, seed, *, play: bool = True,
                  wbt: bool = False, replay: bool = False, rsi: bool = False,
                  reference_dir: str = "/home/ubuntu/sim_references",
                  video: bool = False, video_dir: str = "",
                  video_envs: int = 0) -> SimpleNamespace:
    """Sim-service spawn args under the EVAL PROTOCOL. For WBT this pins ``fixed_refs`` — deterministic
    reference assignment (env i -> ref i), t=0 starts (no phase-RSI), the +hold-tail past the reference
    end (fair window for the original gate's consecutive hold), latched success claims, and gate
    metrics — so results are reproducible, row-comparable with the reference replays, and identical
    between standalone evals and in-training val rollouts."""
    return SimpleNamespace(
        num_envs=num_envs, device=device, seed=seed,
        play=(play and not wbt),  # the play env is the eval variant for the plain lift task
        wbt=wbt, replay=replay, rsi_play=rsi,
        fixed_refs=(wbt or rsi),  # deterministic env i -> ref i in both reference-conditioned modes
        reference_dir=reference_dir,
        video=video, video_length=0, video_dir=video_dir, video_envs=video_envs,
    )


def build_actor(obs, actor_spec: dict, obs_groups_spec: dict, num_actions: int, device: str, state_dict):
    """Build the inference actor from UNTOUCHED specs + weights. The one correct recipe for both paths:
    fresh module sized to this env's batch (never reuse a live training policy — its RNN state is sized
    to the training envs and mid-episode), specs deep-copied (construction pops keys in place), strict
    weight load (width drift fails loudly)."""
    og = resolve_obs_groups(obs, copy.deepcopy(obs_groups_spec), ["actor"])
    kwargs = copy.deepcopy(actor_spec)
    actor_cls = {"MLPModel": MLPModel, "RNNModel": RNNModel}[kwargs.pop("class_name")]
    actor = actor_cls(obs, og, "actor", num_actions, **kwargs).to(device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


def rollout_first_episodes(env, policy, device: str, tail_margin: int = 60,
                           stats: dict | None = None) -> torch.Tensor:
    """Step until every env has finished at least one episode (the per-env video recorder records each
    env's FIRST episode). Cap covers the max episode length plus the eval hold-tail.

    Returns each env's FIRST-episode success claim [num_envs] bool — the latched original-gate claim
    the server ships on done rows (False where the env never finished or the gate flag is absent), so
    callers can report an eval/SR from the same rollout that produced the videos.

    ``stats`` (optional dict, filled in place): first-episode forensics — ``lengths`` (steps),
    ``reasons`` (termination-term index per the server's ``term_names``), ``names``. Early
    terminations (e.g. an RSI pose ejecting the hammer → ``wbt_bad_tracking`` within ~10 steps) are
    invisible in the aggregate SR without this."""
    obs = env.get_observations()
    seen = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    succ = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    lens = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    cap = int(getattr(env, "max_episode_length", 250)) + int(tail_margin)
    for _ in range(cap):
        with torch.inference_mode():
            obs, _, d, extras = env.step(policy(obs))
            policy.reset(d)
        d = d.bool().to(seen.device)
        lens += 1
        flag = extras.get("task_success") if isinstance(extras, dict) else None
        if flag is not None:
            succ |= (d & ~seen) & flag.bool().to(seen.device)  # first episode only
        if stats is not None and isinstance(extras, dict):
            first_rows = d & ~seen
            if bool(first_rows.any()):
                stats.setdefault("lengths", []).extend(lens[first_rows].tolist())
                rc = extras.get("term_reason")
                if rc is not None:
                    stats.setdefault("reasons", []).extend(
                        rc.to(seen.device)[first_rows].tolist())
                if extras.get("term_names"):
                    stats["names"] = list(extras["term_names"])
        lens[d] = 0
        seen |= d
        if bool(seen.all()):
            break
    return succ
