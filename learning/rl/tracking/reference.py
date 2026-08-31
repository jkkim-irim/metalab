# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Reference-motion extraction for WBT tracking.

The WBT collector (``collect_trajectories.py``) records a dedicated ``tracking_state`` obs group per step
— the flat ``[T, D]`` WBT tracked state (object pose+vel, palm pose+vel, fingertip positions, joints; the
layout is defined in ``sim/isaaclab/envs/hammer_lift/mdp/wbt.py``). Unlike an object-pose-only reference
(which slices the 125-d privileged obs), the WBT tracked state is recorded DIRECTLY — including the
fingertip positions that the privileged obs never carried — so this module just repackages
``tracking_state`` (plus the per-episode env ``setup`` for exact RSI) into ``ref_*.npz`` the WBT env
consumes: ``{state[T, D], setup[Ds]}``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def build_reference_dataset(traj_dir: str, out_dir: str, trim_head: int = 2, min_lift_cm: float = 0.0,
                            max_post_claim: int = 0) -> dict:
    """Convert a dir of WBT ``traj_*.npz`` (obs dumps carrying ``tracking_state``) into ``ref_*.npz``.

    - ``trim_head``: drop the first N frames — the recorded obs is STALE for ~2 frames after reset (the
      randomized hammer pose hasn't propagated into the obs), which would otherwise be garbage targets at
      phase 0-1 (and a visible pose jump in replays). The RSI ``setup`` reads the sim directly (not the
      obs), so it already matches the post-trim first frame.
    - ``min_lift_cm``: keep only decisive lifts — the source policy's success gate is lenient (broken
      auto-curriculum), so a few-cm nudge can count as "success"; filter those out of the reference set.
    - ``contact``: the fingertip-hammer contact pattern, sliced out of the recorded ``privileged`` obs
      (the WBT collector records every obs group) — target for the grasp/contact-match reward term.
    - ``max_post_claim``: cut each reference N frames after its first gate-crossing hold (claim-only
      collections record to time-out, and the post-claim tail is the SOURCE POLICY DEGRADING —
      out-of-distribution past its own success, it sags and half-drops; measured: fresh-n2000
      trajs end 3/5-finger near-table. 0 = no cut (already-terminating collections).
    """
    # Privileged-obs layout of the recorded trajectories (fixed by the collect env's obs group):
    # full vector is 125-d; fingertip-hammer contact flags sit at [117:122] (Index..Thumb).
    PRIVILEGED_DIM = 125
    FINGERTIP_CONTACT = slice(117, 122)

    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(traj_dir, "traj_*.npz")))
    lengths: list[int] = []
    dims: set[int] = set()
    n_setup = n_skipped = 0
    z_lift: list[float] = []
    j = 0
    for p in paths:
        d = np.load(p)
        if "tracking_state" not in d.files:
            raise KeyError(f"{p} has no 'tracking_state' array — collect with the WBT collector "
                           "(server built with --wbt --collect, so the tracking_state obs group is recorded)")
        state = d["tracking_state"].astype(np.float32)[trim_head:]  # [T, D]
        contact = None
        if "privileged" in d.files and d["privileged"].shape[-1] == PRIVILEGED_DIM:
            contact = d["privileged"][trim_head:, FINGERTIP_CONTACT].astype(np.float32)  # [T, 5]
        cut = state.shape[0]
        if max_post_claim > 0 and contact is not None:
            from learning.rl.service import ensure_transport_importable
            ensure_transport_importable()
            from envs.hammer_lift.gate import TASK_SUCCESS_HOLD_STEPS_GATE as _HG
            full = (contact.sum(1) >= 4.5).astype(np.int64)
            run = 0
            claim_at = -1
            for i, v in enumerate(full[: state.shape[0]]):
                run = run + 1 if v else 0
                if run >= _HG:
                    claim_at = i
                    break
            if claim_at >= 0:
                cut = min(state.shape[0], claim_at + 1 + max_post_claim)
        state = state[:cut]
        contact = contact[:cut] if contact is not None else None
        lift = float(state[:, 2].max() - state[0, 2])  # hammer z rise (chest-relative; state[:,2])
        if lift * 100.0 < min_lift_cm:
            n_skipped += 1
            continue
        ref = {"state": state}
        if contact is not None:
            ref["contact"] = contact
        if "setup" in d.files:  # per-episode env setup → exact RSI at tracking time
            ref["setup"] = d["setup"].astype(np.float32)
            n_setup += 1
        if "variant" in d.files:  # hammer shape the episode was demonstrated on (variant-matched assignment)
            ref["variant"] = d["variant"].astype(np.int64)
        if "action" in d.files:  # recorded actions → open-loop replay of the reference for visualization
            ref["action"] = d["action"].astype(np.float32)[trim_head:trim_head + cut]
        np.savez_compressed(os.path.join(out_dir, f"ref_{j:04d}.npz"), **ref)
        j += 1
        lengths.append(int(state.shape[0]))
        dims.add(int(state.shape[1]))
        z_lift.append(lift)
    meta = {
        "num_references": j, "source_traj_dir": traj_dir, "with_setup": n_setup,
        "trim_head": trim_head, "min_lift_cm": min_lift_cm, "max_post_claim": max_post_claim,
        "skipped_below_min_lift": n_skipped,
        "state_dim": sorted(dims),
        "keys": ["state[T,D] WBT tracked state (see mdp/wbt.py layout)", "contact[T,5] fingertip-hammer",
                 "setup[D] env init for RSI", "action[T,A] recorded actions (replay viz)"],
        "traj_len": {"min": min(lengths), "max": max(lengths)} if lengths else {},
        "z_lift": {"min": round(min(z_lift), 4), "mean": round(sum(z_lift) / len(z_lift), 4),
                   "max": round(max(z_lift), 4)} if z_lift else {},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"REFERENCE_OK references={j} skipped={n_skipped} (lift<{min_lift_cm}cm) state_dim={sorted(dims)} "
          f"with_setup={n_setup} z_lift_mean={meta.get('z_lift', {}).get('mean')} out={out_dir}", flush=True)
    return meta


def validate_reference_dataset(out_dir: str, min_lift_cm: float = 0.0) -> None:
    """Quality gate on the BUILT artifact (re-reads the written files, not intermediates): every
    ref must carry setup + variant and demonstrate the task gate — lift >= min_lift_cm and a
    full-grasp streak that crosses the gate's hold length in-recording OR reaches the final frame
    (the frozen-target tail extends terminal grasps, see wbt.py). Violations exit nonzero with a
    per-check table — a defective set must fail the pipeline, not surface as a mystery mid-run
    (the 20260709 n2000 set shipped with min_lift unapplied + 92% clipped hold windows)."""
    import glob as _glob

    from learning.rl.service import ensure_transport_importable
    ensure_transport_importable()
    from envs.hammer_lift.gate import TASK_SUCCESS_HOLD_STEPS_GATE as _HOLD_GATE

    files = sorted(_glob.glob(os.path.join(out_dir, "ref_*.npz")))
    bad = {"setup": [], "variant": [], "lift": [], "hold": []}
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "setup" not in z:
            bad["setup"].append(f)
        if "variant" not in z:
            bad["variant"].append(f)
        st, ct = z["state"], z["contact"]
        if min_lift_cm > 0 and float(st[:, 2].max() - st[:, 2].min()) < min_lift_cm / 100.0:
            bad["lift"].append(f)
        full = (ct.sum(1) == 5).astype(int)
        run = best = 0
        for v in full:
            run = run + 1 if v else 0
            best = max(best, run)
        if best < _HOLD_GATE and not (run > 0):  # run>0 == streak reaches the final frame
            bad["hold"].append(f)
    n_bad = sum(len(v) for v in bad.values())
    print(f"[reference-validate] n={len(files)} gate_hold={_HOLD_GATE} min_lift_cm={min_lift_cm} | "
          + " ".join(f"{k}_bad={len(v)}" for k, v in bad.items()), flush=True)
    if n_bad:
        for k, v in bad.items():
            for f in v[:5]:
                print(f"[reference-validate]   BAD {k}: {os.path.basename(f)}", flush=True)
        raise SystemExit(f"reference set FAILED validation: {n_bad} violations across "
                         f"{sum(1 for v in bad.values() if v)} checks")
    print("[reference-validate] OK - set is training-grade", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Build WBT reference motions from collected trajectory dumps.")
    p.add_argument("--traj_dir", required=True, help="dir of WBT traj_*.npz (obs dumps with tracking_state)")
    p.add_argument("--out_dir", required=True, help="dir to write ref_*.npz + meta.json")
    p.add_argument("--trim_head", type=int, default=2, help="drop the first N frames (stale post-reset obs)")
    p.add_argument("--min_lift_cm", type=float, default=0.0, help="keep only refs whose hammer rises >= this")
    p.add_argument("--max_post_claim", type=int, default=0,
                   help="cut each ref N frames after its first gate-crossing hold (trims the "
                        "post-claim source-degradation tail of claim-only collections; 0 = off)")
    p.add_argument("--validate", action="store_true",
                   help="after building, re-read every written ref and FAIL LOUDLY unless the set is "
                        "training-grade: setup + variant present, lift >= --min_lift_cm, and a full-grasp "
                        "streak that crosses the gate's hold length (in-recording, or reaching the final "
                        "frame — the frozen-target tail extends those)")
    args = p.parse_args()
    build_reference_dataset(args.traj_dir, args.out_dir, trim_head=args.trim_head, min_lift_cm=args.min_lift_cm,
                            max_post_claim=args.max_post_claim)
    if args.validate:
        validate_reference_dataset(args.out_dir, min_lift_cm=args.min_lift_cm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
