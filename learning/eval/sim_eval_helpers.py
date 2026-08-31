"""Pure-logic helpers for the closed-loop sim eval (``vla_runner`` + the eval entrypoints).

Deliberately dependency-free (no torch / gr00t / imageio / wandb) so the replan-cap and
video-naming logic can be unit-tested locally, off the node, without the heavy sim/policy stack.
The sim-eval path (``vla_runner``, ``policies/groot.py``, ``intrain_sim_eval``) imports and calls these
exact functions — the tests exercise the production path.
"""
from __future__ import annotations


def replan_cap(chunk_len: int, replan_steps: int) -> int:
    """How many steps of a predicted action chunk to execute before re-observing / re-predicting.

    ``replan_steps == 0`` means "execute the full chunk" (the original behaviour); any positive
    value caps execution at that many steps (partial-chunk replan, tighter closed-loop feedback).
    Never exceeds ``chunk_len``.

    >>> replan_cap(8, 0)   # full chunk
    8
    >>> replan_cap(8, 5)   # partial replan
    5
    >>> replan_cap(8, 20)  # capped at the chunk length
    8
    """
    assert chunk_len > 0, f"chunk_len must be positive, got {chunk_len}"
    assert replan_steps >= 0, f"replan_steps must be >= 0, got {replan_steps}"
    return min(replan_steps or chunk_len, chunk_len)


def video_filename(idx: int, success: bool) -> str:
    """MP4 basename for a captured rollout episode: ``ep{idx}_{success|fail}.mp4``.

    >>> video_filename(0, True)
    'ep0_success.mp4'
    >>> video_filename(3, False)
    'ep3_fail.mp4'
    """
    return f"ep{idx}_{'success' if success else 'fail'}.mp4"


def resolve_suite(suite: str) -> tuple[str, str]:
    """Map an eval ``--suite`` to ``(modality, task_kind)`` — the suite is the single selector for the
    sim server + obs/state layout, and it fixes how ``--tasks`` entries are read.

    ``modality`` picks the server/obs layout (``mikasa`` -> ManiSkill, ``libero`` -> LIBERO);
    ``task_kind`` is ``"int"`` (LIBERO integer task-ids) or ``"envid"`` (MIKASA env-id strings).
    Fail loud on an unknown suite. (The server-script path itself lives in ``vla_runner.server_for``.)

    >>> resolve_suite("libero_90")
    ('libero', 'int')
    >>> resolve_suite("mikasa")
    ('mikasa', 'envid')
    """
    if suite == "mikasa":
        return "mikasa", "envid"
    if suite.startswith("libero"):
        return "libero", "int"
    raise ValueError(f"unknown --suite {suite!r} (expected 'mikasa' or a 'libero_*' suite)")


def parse_tasks(task_kind: str, tasks: str) -> list:
    """Split a ``--tasks`` string ("a,b,c") into the per-suite task list: ``int`` task-ids for LIBERO,
    env-id strings for MIKASA. Whitespace around entries is trimmed; fail loud on an empty list.

    >>> parse_tasks("int", "0, 11 ,42")
    [0, 11, 42]
    >>> parse_tasks("envid", "ShellGameTouch-VLA-v0")
    ['ShellGameTouch-VLA-v0']
    """
    items = [t.strip() for t in tasks.split(",") if t.strip()]
    if not items:
        raise ValueError("--tasks is empty")
    return [int(t) for t in items] if task_kind == "int" else items


def safe_name(name: str) -> str:
    """Filesystem-safe token from an env_id, for per-task video subdirs so a multi-task set never
    collides on ``ep{idx}`` filenames.

    >>> safe_name("libero_90:11")
    'libero_90_11'
    >>> safe_name("ShellGameTouch-VLA-v0")
    'ShellGameTouch-VLA-v0'
    """
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in name)


def aggregate_sim_metrics(results: list) -> dict:
    """Micro-average closed-loop SR over ALL episodes of ALL tasks, plus per-task SR, as a flat
    ``sim_eval/``-namespaced dict ready for wandb. ``results`` is one dict per evaluated task, as
    returned by the closed-loop runner (``vla_runner.run_task``): ``{env_id, sr, succ, total}``.

    Micro-average (total successes / total episodes over the whole set), NOT a mean of per-task SRs,
    so a set with uneven episode counts is weighted by episodes. Fail loud on an empty set / zero
    episodes — a sim-eval that evaluated nothing is a bug, not a 0.0 SR.

    >>> m = aggregate_sim_metrics([{"env_id": "a", "sr": 0.5, "succ": 2, "total": 4},
    ...                            {"env_id": "b", "sr": 1.0, "succ": 4, "total": 4}])
    >>> m["sim_eval/SR"], m["sim_eval/n_tasks"], m["sim_eval/a/SR"]
    (0.75, 2, 0.5)
    """
    assert results, "aggregate_sim_metrics: no task results"
    succ_total = sum(r["succ"] for r in results)
    ep_total = sum(r["total"] for r in results)
    assert ep_total > 0, "aggregate_sim_metrics: zero episodes evaluated"
    out = {
        "sim_eval/SR": succ_total / ep_total,          # micro-avg over ALL episodes of ALL tasks
        "sim_eval/n_success": succ_total,
        "sim_eval/n_episodes": ep_total,
        "sim_eval/n_tasks": len(results),
    }
    for r in results:
        out[f"sim_eval/{r['env_id']}/SR"] = r["sr"]     # per-task SR, env_id-namespaced (no collision)
    return out
