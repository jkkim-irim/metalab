from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
_OUT_ROOT = _REPO / "_logs" / "parity"
_ENGINES = ("genesis", "newton")


def _build(engine: str, task: str):
    server = importlib.import_module(f"sim.metalab.backends.{engine}.server")
    return server.build_env(task=task.replace("-", "_"), num_envs=1, device="cuda:0", viz=None, telemetry=False)


def _git_rev() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO,
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=_REPO,
                           capture_output=True, text=True, check=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def sinusoid(t: float, center: list[float], amp: list[float], freq_hz: float, phase: list[float],
             ramp_s: float) -> list[float]:
    g = 1.0 if ramp_s <= 0.0 else min(1.0, t / ramp_s)
    return [c + a * g * math.sin(2.0 * math.pi * freq_hz * t + p) for c, a, p in zip(center, amp, phase)]


def record(engine: str, task: str, joints: list[str], bodies: list[str], amp_deg: float, freq_hz: float,
           seconds: float, ramp_s: float) -> Path:
    env = _build(engine, task)
    b, spec = env.backend, env.spec
    assert b.num_envs == 1, f"parity recording expects num_envs=1 (got {b.num_envs})"
    active = set(spec.robot.active_joints())
    names = [j for j in spec.robot.joints if j in active]
    drive = joints or names
    unknown = [j for j in drive if j not in names]
    assert not unknown, f"--joints names {unknown}, which are not active joints of this robot: {names}"

    decim = spec.physics.decimation
    dt = decim / spec.physics.hz
    steps = round(seconds / dt)
    center = [float(spec.robot.init_pose.get(j, 0.0)) for j in names]
    amp = [math.radians(amp_deg) if j in drive else 0.0 for j in names]
    phase = [2.0 * math.pi * drive.index(j) / len(drive) if j in drive else 0.0 for j in names]

    lo, hi = b.joint_limits(names)
    for j, c, a, low, high in zip(names, center, amp, lo.cpu().tolist(), hi.cpu().tolist()):
        assert low <= c - a and c + a <= high, (
            f"{j}: sinusoid [{c - a:.4f}, {c + a:.4f}] rad leaves the joint range [{low:.4f}, {high:.4f}] "
            f"(center = init pose {c:.4f}, amp = {a:.4f}); lower --amp-deg or drop it from --joints")

    reads = {
        "joint_pos": lambda: b.joint_pos(names),
        "joint_vel": lambda: b.joint_vel(names),
        "joint_torque": lambda: b.joint_torque(names),
        "joint_torque_pd": lambda: b.joint_torque_pd(names),
        "joint_torque_gravcomp": lambda: b.joint_torque_gravcomp(names),
    }
    for body in bodies:
        reads[f"body_pos.{body}"] = lambda body=body: b.body_pos(body)
        reads[f"body_quat.{body}"] = lambda body=body: b.body_quat(body)
        reads[f"body_lin_vel.{body}"] = lambda body=body: b.body_lin_vel(body)
        reads[f"body_ang_vel.{body}"] = lambda body=body: b.body_ang_vel(body)
    if bodies:
        reads["contact_force"] = lambda: b.contact_force(bodies)
    if spec.objects:
        reads["object_pos"] = b.object_pos
        reads["object_quat"] = b.object_quat
        reads["object_lin_vel"] = b.object_lin_vel
        reads["object_ang_vel"] = b.object_ang_vel
        if bodies:
            reads["contact_force_with.object"] = lambda: b.contact_force_with(bodies, "object")
            reads["contact_penetration.object"] = lambda: b.contact_penetration(bodies, "object")

    all_mask = torch.ones(b.num_envs, dtype=torch.bool, device=b.device)
    step_n = b.step_n if "batched_step" in env.capabilities else None
    b.reset_idx(all_mask)

    series: dict[str, list[torch.Tensor]] = {k: [] for k in reads}
    targets: list[list[float]] = []
    tgt = torch.zeros(b.num_envs, len(names), device=b.device)
    for i in range(steps):
        q = sinusoid(i * dt, center, amp, freq_hz, phase, ramp_s)
        tgt[:] = torch.tensor(q, dtype=tgt.dtype, device=b.device)
        b.set_joint_targets(names, tgt)
        if step_n is not None:
            step_n(decim, render=False)
        else:
            for _ in range(decim):
                b.step(render=False)
        targets.append(q)
        for k, fn in reads.items():
            series[k].append(fn()[0].detach().clone())

    arrays = {k: torch.stack(v).cpu().numpy() for k, v in series.items()}
    arrays["target"] = np.asarray(targets, dtype=np.float32)
    arrays["t"] = (np.arange(1, steps + 1) * dt).astype(np.float64)
    meta = {
        "mode": "backend", "engine": engine, "task": task, "git": _git_rev(),
        "joints": names, "driven": drive, "bodies": bodies,
        "amp_deg": amp_deg, "freq_hz": freq_hz, "ramp_s": ramp_s, "seconds": seconds,
        "hz": spec.physics.hz, "substeps": spec.physics.substeps, "decimation": decim, "dt": dt, "steps": steps,
    }
    return _save(engine, task, arrays, meta)


def record_mdp(engine: str, task: str, action_amp: float, freq_hz: float, seconds: float, ramp_s: float,
               seed: int) -> Path:
    env = _build(engine, task)
    spec = env.spec
    assert env.num_envs == 1, f"parity recording expects num_envs=1 (got {env.num_envs})"
    assert spec.obs and spec.reward, f"{task}: --mode mdp needs a contract with obs and reward terms"
    dt = env.step_dt
    steps = round(seconds / dt)
    dim = env.num_actions
    center = [0.0] * dim
    amp = [action_amp] * dim
    phase = [2.0 * math.pi * i / dim for i in range(dim)]

    env.seed(seed)
    env.reset()
    series: dict[str, list[torch.Tensor]] = {}

    def put(key: str, v: torch.Tensor) -> None:
        series.setdefault(key, []).append(v.detach().clone())

    actions: list[list[float]] = []
    act = torch.zeros(1, dim, device=env.device)
    for i in range(steps):
        a = sinusoid(i * dt, center, amp, freq_hz, phase, ramp_s)
        act[:] = torch.tensor(a, dtype=act.dtype, device=env.device)
        obs, rew, dones, extras = env.step(act)
        actions.append(a)
        for g in spec.obs:
            put(f"obs.{g}", obs[g][0])
        put("reward", rew[0])
        for j, t in enumerate(spec.reward):
            put(f"reward_term.{t.name}", env.last_reward_terms[0, j])
        put("done", dones[0].to(torch.uint8))
        put("time_out", extras["time_outs"][0].to(torch.uint8))

    arrays = {k: torch.stack(v).cpu().numpy() for k, v in series.items()}
    arrays["action"] = np.asarray(actions, dtype=np.float32)
    arrays["t"] = (np.arange(1, steps + 1) * dt).astype(np.float64)
    meta = {
        "mode": "mdp", "engine": engine, "task": task, "git": _git_rev(),
        "action_amp": action_amp, "freq_hz": freq_hz, "ramp_s": ramp_s, "seconds": seconds, "seed": seed,
        "action_dim": dim, "episode_length": env.max_episode_length,
        "obs": {g: [t.name for t in ts] for g, ts in spec.obs.items()},
        "reward": [t.name for t in spec.reward], "terminate": [t.name for t in spec.terminate],
        "hz": spec.physics.hz, "substeps": spec.physics.substeps, "decimation": spec.physics.decimation,
        "dt": dt, "steps": steps,
    }
    return _save(engine, task, arrays, meta)


def _save(engine: str, task: str, arrays: dict[str, np.ndarray], meta: dict) -> Path:
    out_dir = _OUT_ROOT / task.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{engine}_{meta['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(path, **arrays)
    meta["channels"] = {k: list(v.shape) for k, v in arrays.items()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record one engine under a sinusoidal command (headless, num_envs=1) to "
                    "_logs/parity/<task>/<engine>_<mode>_<stamp>.npz for parity_diff. "
                    "--mode backend drives joint targets and records every SimBackend read; "
                    "--mode mdp drives EnvDriver.step with a sinusoidal action and records obs/reward/done.")
    ap.add_argument("--engine", required=True, choices=_ENGINES)
    ap.add_argument("--task", required=True, help="standalone contract name (e.g. franka)")
    ap.add_argument("--mode", default="backend", choices=("backend", "mdp"))
    ap.add_argument("--joints", type=_csv, default=[],
                    help="[backend] comma-separated joints to drive (default: every active joint); undriven joints hold the init pose")
    ap.add_argument("--bodies", type=_csv, default=[],
                    help="[backend] comma-separated body names to record pose/velocity/contact for (default: none)")
    ap.add_argument("--amp-deg", type=float, default=10.0, help="[backend] sinusoid amplitude about the init pose [deg]")
    ap.add_argument("--action-amp", type=float, default=0.5, help="[mdp] sinusoid amplitude in normalized action units")
    ap.add_argument("--seed", type=int, default=0, help="[mdp] torch seed for the contract's reset events")
    ap.add_argument("--freq-hz", type=float, default=0.5, help="sinusoid frequency [Hz]")
    ap.add_argument("--seconds", type=float, default=8.0, help="recording length [s]")
    ap.add_argument("--ramp-s", type=float, default=1.0, help="amplitude ramps 0 to 1 over this time [s]; 0 = none")
    args = ap.parse_args()
    if args.mode == "mdp":
        assert not args.joints and not args.bodies, "--joints/--bodies apply to --mode backend only"
        path = record_mdp(args.engine, args.task, args.action_amp, args.freq_hz, args.seconds, args.ramp_s,
                          args.seed)
    else:
        path = record(args.engine, args.task, args.joints, args.bodies, args.amp_deg, args.freq_hz,
                      args.seconds, args.ramp_s)
    print(f"[parity] wrote {path.relative_to(_REPO)}", flush=True)


if __name__ == "__main__":
    main()
