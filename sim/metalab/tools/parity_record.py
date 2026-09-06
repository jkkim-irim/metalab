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

from sim.metalab.contract.loader import parity_module
from sim.metalab.contract.spec import values

_REPO = Path(__file__).resolve().parents[3]
_OUT_ROOT = _REPO / "_logs" / "parity"
_ENGINES = ("genesis", "newton")
_BACKEND_KEYS = {"mode", "joints", "bodies", "amp_deg", "freq_hz", "seconds", "ramp_s"}
_MDP_KEYS = {"mode", "action_amp", "seed", "freq_hz", "seconds", "ramp_s"}


def _build(engine: str, task: str, video: bool):
    server = importlib.import_module(f"sim.metalab.backends.{engine}.server")
    return server.build_env(task=task.replace("-", "_"), num_envs=1, device="cuda:0", viz=None, telemetry=False,
                            video=video)


def _recorder(engine: str, backend, task: str, mode: str, stamp: str, fps: float):
    mod = importlib.import_module(f"sim.metalab.backends.{engine}.video")
    return mod.VideoRecorder(backend, _out_dir(task) / "video" / f"{engine}_{mode}_{stamp}.mp4", fps)


def _out_dir(task: str) -> Path:
    return _OUT_ROOT / task.replace("-", "_")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def command(task: str) -> dict:
    mod = importlib.import_module(parity_module(task.replace("-", "_")))
    cmd = values(getattr(mod, "COMMAND", None))
    assert isinstance(cmd, dict), f"{mod.__name__}: a parity contract declares a COMMAND block next to TASK"
    want = {"backend": _BACKEND_KEYS, "mdp": _MDP_KEYS}.get(cmd.get("mode"))
    assert want is not None, f"{mod.__name__}.COMMAND.mode must be 'backend' or 'mdp' (got {cmd.get('mode')!r})"
    assert set(cmd) == want, f"{mod.__name__}.COMMAND for mode {cmd['mode']!r} needs exactly {sorted(want)}, got {sorted(cmd)}"
    return cmd


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


def record(engine: str, task: str, video: bool, *, joints: list[str], bodies: list[str], amp_deg: float,
           freq_hz: float, seconds: float, ramp_s: float) -> Path:
    stamp = _stamp()
    env = _build(engine, task, video)
    b, spec = env.backend, env.spec
    assert b.num_envs == 1, f"parity recording expects num_envs=1 (got {b.num_envs})"
    active = set(spec.robot.active_joints())
    names = [j for j in spec.robot.joints if j in active]
    drive = list(joints)
    assert drive, "COMMAND.joints must name at least one joint to drive"
    unknown = [j for j in drive if j not in names]
    assert not unknown, f"COMMAND.joints names {unknown}, which are not active joints of this robot: {names}"

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
            f"(center = init pose {c:.4f}, amp = {a:.4f}); lower COMMAND.amp_deg or drop it from COMMAND.joints")

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
    rec = _recorder(engine, b, task, "backend", stamp, 1.0 / dt) if video else None

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
        if rec is not None:
            rec.capture()
    if rec is not None:
        print(f"[parity] wrote {rec.close().relative_to(_REPO)}", flush=True)

    arrays = {k: torch.stack(v).cpu().numpy() for k, v in series.items()}
    arrays["target"] = np.asarray(targets, dtype=np.float32)
    arrays["t"] = (np.arange(1, steps + 1) * dt).astype(np.float64)
    meta = {
        "mode": "backend", "engine": engine, "task": task, "git": _git_rev(),
        "joints": names, "driven": drive, "bodies": bodies,
        "amp_deg": amp_deg, "freq_hz": freq_hz, "ramp_s": ramp_s, "seconds": seconds,
        "hz": spec.physics.hz, "substeps": spec.physics.substeps, "decimation": decim, "dt": dt, "steps": steps,
    }
    return _save(engine, task, stamp, arrays, meta)


def record_mdp(engine: str, task: str, video: bool, *, action_amp: float, seed: int, freq_hz: float,
               seconds: float, ramp_s: float) -> Path:
    stamp = _stamp()
    env = _build(engine, task, video)
    spec = env.spec
    assert env.num_envs == 1, f"parity recording expects num_envs=1 (got {env.num_envs})"
    assert spec.obs and spec.reward, f"{task}: COMMAND.mode='mdp' needs a contract with obs and reward terms"
    dt = env.step_dt
    steps = round(seconds / dt)
    dim = env.num_actions
    center = [0.0] * dim
    amp = [action_amp] * dim
    phase = [2.0 * math.pi * i / dim for i in range(dim)]

    env.seed(seed)
    env.reset()
    rec = _recorder(engine, env.backend, task, "mdp", stamp, 1.0 / dt) if video else None
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
        if rec is not None:
            rec.capture()
    if rec is not None:
        print(f"[parity] wrote {rec.close().relative_to(_REPO)}", flush=True)

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
    return _save(engine, task, stamp, arrays, meta)


def _save(engine: str, task: str, stamp: str, arrays: dict[str, np.ndarray], meta: dict) -> Path:
    out_dir = _out_dir(task)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{engine}_{meta['mode']}_{stamp}.npz"
    np.savez(path, **arrays)
    meta["channels"] = {k: list(v.shape) for k, v in arrays.items()}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record one engine under the contract's COMMAND (headless, num_envs=1) to "
                    "_logs/parity/<task>/<engine>_<mode>_<stamp>.npz for parity_diff/parity_plot. "
                    "COMMAND.mode='backend' drives joint targets and records every SimBackend read; "
                    "COMMAND.mode='mdp' drives EnvDriver.step with a sinusoidal action and records obs/reward/done.")
    ap.add_argument("--engine", required=True, choices=_ENGINES)
    ap.add_argument("--task", required=True, help="tasks/parity contract name (e.g. parity-joint-torque)")
    ap.add_argument("--video", action="store_true",
                    help="also record an offscreen mp4 from the contract's scene.camera to _logs/parity/<task>/video/")
    args = ap.parse_args()
    cmd = command(args.task)
    mode = cmd.pop("mode")
    fn = record_mdp if mode == "mdp" else record
    path = fn(args.engine, args.task, args.video, **cmd)
    print(f"[parity] wrote {path.relative_to(_REPO)}", flush=True)


if __name__ == "__main__":
    main()
