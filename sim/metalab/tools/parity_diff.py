from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_MUST_MATCH = ("mode", "task", "joints", "driven", "bodies", "amp_deg", "action_amp", "seed", "freq_hz",
               "ramp_s", "seconds", "hz", "substeps", "decimation", "dt", "steps")


def load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    arrays = dict(np.load(path))
    meta = json.loads(path.with_suffix(".json").read_text())
    return arrays, meta


def diff(a_path: Path, b_path: Path) -> str:
    a, ma = load(a_path)
    b, mb = load(b_path)
    mismatch = [k for k in _MUST_MATCH if ma.get(k) != mb.get(k)]
    assert not mismatch, (
        f"recordings are not comparable, meta differs in {mismatch}: "
        + "; ".join(f"{k}: {ma.get(k)!r} vs {mb.get(k)!r}" for k in mismatch))
    dt = float(ma["dt"])

    lines = [
        f"# parity diff: {ma['task']}",
        "",
        f"- A: `{a_path.name}` engine={ma['engine']} git={ma['git']}",
        f"- B: `{b_path.name}` engine={mb['engine']} git={mb['git']}",
        (f"- command: action amp={ma['action_amp']} seed={ma['seed']}" if ma.get("mode") == "mdp"
         else f"- command: joints={ma['driven']} amp={ma['amp_deg']} deg")
        + f" freq={ma['freq_hz']} Hz ramp={ma['ramp_s']} s; {ma['steps']} steps at dt={dt:.6f} s",
        "",
        "| channel | shape | max abs diff | rms diff | max abs A | first diff [s] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            lines.append(f"| {k} | | | | | absent in {'A' if k not in a else 'B'} |")
            continue
        x, y = a[k], b[k]
        assert x.shape == y.shape, f"{k}: shape {x.shape} vs {y.shape}"
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        per_step = d.reshape(d.shape[0], -1).max(axis=1)
        nz = np.flatnonzero(per_step > 0.0)
        first = f"{nz[0] * dt:.3f}" if nz.size else "never"
        lines.append(f"| {k} | {'x'.join(map(str, x.shape))} | {d.max():.3e} | "
                     f"{np.sqrt(np.mean(d * d)):.3e} | {np.abs(x).max():.3e} | {first} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-channel numeric diff of two parity_record .npz files (numpy only).")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="also write the markdown report here")
    args = ap.parse_args()
    report = diff(args.a, args.b)
    print(report, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)


if __name__ == "__main__":
    main()
