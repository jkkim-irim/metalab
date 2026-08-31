"""Parity harness — stands up the same contract on each engine backend and dumps/compares **what the engine actually read**.

Two modes:
- ``initdump`` — dump env-0 **initial state** (actual-read: joint·object·frame) after reset. Used by `verify_init_state.sh`.
- ``dump``/``compare`` — dump a **fixed target sequence** rollout trajectory, then compare per-field Δ.

**Drives the backend directly without the runtime (env_driver)** (no reward/done) → measures pure physics/solver
difference (this project's differentiator, sim2real error budget). Reads from the engine state buffer, not the input
(contract value), so engine interpretation shows through: equality coupling, quaternion normalization, limit clamp.

gs.init runs once per process → **one engine per process** (each in its own uv venv). The rollout target sequence
is contract init_pose + seed-fixed perturbation (CPU-generated, identical across engines).

  # initial-state dump (in each engine's uv venv) — verify_init_state.sh runs these in sequence
  python -m sim.metalab.runtime.parity_rollout initdump --engine genesis --task hammer-lift-teacher --out sim/metalab/parity/hammer_lift_teacher/genesis.json
  # rollout parity
  python -m sim.metalab.runtime.parity_rollout dump --engine genesis --task hammer-lift-teacher --seed 0 --out /tmp/g0.json
  python -m sim.metalab.runtime.parity_rollout compare /tmp/g0.json /tmp/g1.json --tol 1e-3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _build(engine: str, task: str, recipe: str | None, num_envs: int, device: str):
    """Per-engine build_env (returns EnvDriver; uses .backend·.spec). Lazy import — only in that engine's env."""
    if engine == "genesis":
        from sim.metalab.backends.genesis.server import build_env
    elif engine == "newton":
        from sim.metalab.backends.newton.server import build_env
    else:
        raise SystemExit(f"engine must be genesis|newton (got {engine!r})")
    return build_env(task=task.replace("-", "_"), recipe=recipe, num_envs=num_envs, device=device,
                     viz="none")


def _target_seq(spec, ctrl: list[str], n_steps: int, seed: int) -> torch.Tensor:
    """Contract init_pose (reference pose) + seed-fixed perturbation → (n_steps, k). CPU-generated, identical across engines."""
    # Explicit CPU generation (guarantees seed reproducibility + engine-agnostic identical values even if the engine defaults to cuda). dump does .to(device).
    base = torch.tensor([spec.robot.init_pose.get(j, 0.0) for j in ctrl], dtype=torch.float32, device="cpu")  # (k,)
    g = torch.Generator(device="cpu").manual_seed(seed)
    pert = torch.randn(n_steps, len(ctrl), generator=g, device="cpu") * 0.15
    return base.unsqueeze(0) + pert


def dump(engine: str, task: str, recipe: str | None, n_steps: int, num_envs: int, device: str,
         seed: int, out: str) -> None:
    env = _build(engine, task, recipe, num_envs, device)
    b, spec = env.backend, env.spec
    env.reset()
    ctrl = [j for gp in spec.action.values() for j in gp.joints]
    frames = list(spec.robot.frames.items())   # [(name, body_name), ...]
    seq = _target_seq(spec, ctrl, n_steps, seed).to(device)

    traj: dict[str, list] = {"object_pos": [], "object_quat": [], "object_lin_vel": [], "joint_pos": [],
                             # The APPLIED actuator torque, not just the state it produced. In motor mode this
                             # is where the two engines are most easily made to disagree while the motion still
                             # looks similar: the coupled τ is clamped in MOTOR space, and a joint-level clamp
                             # on top of it (which genesis applies to EVERY ctrl mode, newton to none) would
                             # cap τ silently. State-only parity hides that until it grows large enough to bend
                             # the trajectory — comparing τ directly catches it at the first saturating step.
                             "joint_torque": []}
    for fname, _ in frames:
        traj[f"frame_{fname}"] = []
    for t in range(n_steps):
        b.set_joint_targets(ctrl, seq[t].unsqueeze(0).expand(num_envs, -1))
        for _ in range(spec.physics.decimation):
            b.step()
        traj["object_pos"].append(b.object_pos()[0].tolist())
        traj["object_quat"].append(b.object_quat()[0].tolist())
        traj["object_lin_vel"].append(b.object_lin_vel()[0].tolist())
        traj["joint_pos"].append(b.joint_pos(ctrl)[0].tolist())
        traj["joint_torque"].append(b.joint_torque(ctrl)[0].tolist())
        for fname, body in frames:
            traj[f"frame_{fname}"].append(b.body_pos(body)[0].tolist())

    meta = {"engine": engine, "task": task, "n_steps": n_steps, "num_envs": num_envs, "seed": seed, "ctrl": ctrl}
    Path(out).write_text(json.dumps({"meta": meta, "traj": traj}))
    print(f"[parity-rollout] {engine} {task} {n_steps} steps (env0) → {out}", flush=True)


def initdump(engine: str, task: str, recipe: str | None, num_envs: int, device: str, out: str) -> None:
    """Stand up the engine spoke and dump env-0 initial state **as the backend actually read it** after reset, to JSON.

    Read from the engine's internal state buffer, not the input (contract init_pose) — the 'actual' initial state
    reflecting engine interpretation (equality coupling, quaternion normalization, limit clamp). For cross-engine comparison.
    """
    env = _build(engine, task, recipe, num_envs, device)
    b, spec = env.backend, env.spec
    env.reset()
    ctrl = [j for gp in spec.action.values() for j in gp.joints]
    # Use only methods the SimBackend Protocol guarantees (engine-agnostic). velocity is ≈0 right after reset, so exclude it.
    state: dict = {
        "joint_names": ctrl,
        "joint_pos": b.joint_pos(ctrl)[0].tolist(),
        "joint_vel": b.joint_vel(ctrl)[0].tolist(),
        "object_pos": b.object_pos()[0].tolist(),
        "object_quat": b.object_quat()[0].tolist(),
    }
    for fname, body in spec.robot.frames.items():
        state[f"frame_{fname}_pos"] = b.body_pos(body)[0].tolist()
        state[f"frame_{fname}_quat"] = b.body_quat(body)[0].tolist()
    meta = {"engine": engine, "task": task, "num_envs": num_envs}
    p = Path(out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": meta, "init_state": state}, indent=2))
    print(f"[init-dump] {engine} {task} (env0, actual-read) → {out}", flush=True)


def compare(path_a: str, path_b: str, tol: float) -> bool:
    A = json.loads(Path(path_a).read_text())
    B = json.loads(Path(path_b).read_text())
    ta, tb = A["traj"], B["traj"]
    print(f"=== rollout parity: {A['meta']['engine']} vs {B['meta']['engine']} "
          f"({A['meta']['task']}, {A['meta']['n_steps']} steps, tol={tol}) ===")
    worst, fail = 0.0, []
    for field in ta:
        a, c = torch.tensor(ta[field]), torch.tensor(tb.get(field, []))
        if a.shape != c.shape:
            print(f"  {field:22s} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(c.shape)}  [FAIL]")
            fail.append(field); continue
        d = (a - c).abs()
        mx, mn = float(d.max()), float(d.mean())
        per_step = d.flatten(1).max(dim=1).values if d.dim() > 1 else d
        first = int((per_step > tol).float().argmax()) if mx > tol else -1
        worst = max(worst, mx)
        print(f"  {field:22s} maxΔ={mx:.6f} meanΔ={mn:.6f} first>tol@{first}  [{'FAIL' if mx > tol else 'ok'}]")
        if mx > tol:
            fail.append(field)
    ok = not fail
    print(f"=== {'PASS' if ok else 'FAIL'}  worst Δ={worst:.6f}  divergent={fail} ===")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Rollout parity harness (engine-agnostic).")
    sub = ap.add_subparsers(dest="mode", required=True)
    i = sub.add_parser("initdump")
    i.add_argument("--engine", required=True)
    i.add_argument("--task", default="hammer-lift-teacher")
    i.add_argument("--recipe", default="privileged")
    i.add_argument("--num_envs", type=int, default=1)
    i.add_argument("--device", default="cuda:0")
    i.add_argument("--out", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--engine", required=True)
    d.add_argument("--task", default="hammer-lift-teacher")
    d.add_argument("--recipe", default="privileged")
    d.add_argument("--n_steps", type=int, default=100)
    d.add_argument("--num_envs", type=int, default=4)
    d.add_argument("--device", default="cuda:0")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a"); c.add_argument("b")
    c.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    if args.mode == "initdump":
        initdump(args.engine, args.task, args.recipe or None, args.num_envs, args.device, args.out)
    elif args.mode == "dump":
        dump(args.engine, args.task, args.recipe or None, args.n_steps, args.num_envs, args.device,
             args.seed, args.out)
    else:
        raise SystemExit(0 if compare(args.a, args.b, args.tol) else 1)


if __name__ == "__main__":
    main()
