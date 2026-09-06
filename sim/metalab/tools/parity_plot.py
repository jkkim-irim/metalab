from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sim.metalab.tools.parity_diff import load_pair  # noqa: E402

_A = "#2a78d6"
_B = "#eb6834"
_DIFF = "#4a3aa7"
_INK = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e6e5e1"


def _style(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=8)


_PER_PAGE = 2
_ROW_IN = 3.6
_HEAD_IN = 1.4


def _page(a, ma, b, mb, t, channels: list[str], a_name: str, b_name: str, out: Path) -> None:
    height = _ROW_IN * len(channels) + _HEAD_IN
    fig, axes = plt.subplots(len(channels), 2, figsize=(16, height), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 2], "hspace": 0.55, "wspace": 0.2})
    fig.subplots_adjust(top=1.0 - _HEAD_IN / height, bottom=0.6 / height, left=0.06, right=0.98)
    for row, k in zip(axes, channels):
        x = a[k].reshape(len(t), -1).astype(np.float64)
        y = b[k].reshape(len(t), -1).astype(np.float64)
        assert x.shape == y.shape, f"{k}: shape {x.shape} vs {y.shape}"
        d = np.abs(x - y)
        alpha = 0.9 if x.shape[1] <= 3 else max(0.15, 3.0 / x.shape[1])

        ax = row[0]
        ax.plot(t, x, color=_A, linewidth=1.6, alpha=alpha)
        ax.plot(t, y, color=_B, linewidth=1.6, alpha=alpha, linestyle="--")
        ax.set_title(f"{k}   [{'x'.join(map(str, a[k].shape))}]", loc="left", fontsize=11, color=_INK)
        _style(ax)

        ax = row[1]
        ax.plot(t, d, color=_MUTED, linewidth=0.6, alpha=alpha)
        ax.plot(t, d.max(axis=1), color=_DIFF, linewidth=2.0)
        ax.set_title(f"|A - B|   max {d.max():.3e}   rms {np.sqrt(np.mean(d * d)):.3e}",
                     loc="left", fontsize=11, color=_INK)
        ax.set_ylim(bottom=0.0)
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("t [s]", color=_MUTED, fontsize=9)

    fig.legend(handles=[plt.Line2D([], [], color=_A, linewidth=2, label=f"A: {ma['engine']} ({a_name})"),
                        plt.Line2D([], [], color=_B, linewidth=2, linestyle="--",
                                   label=f"B: {mb['engine']} ({b_name})"),
                        plt.Line2D([], [], color=_DIFF, linewidth=2, label="max over components of |A - B|")],
               loc="upper left", bbox_to_anchor=(0.06, 1.0 - 0.35 / height), ncol=1, fontsize=9, frameon=False)
    fig.suptitle(f"parity: {ma['task']}   git A={ma['git']}  B={mb['git']}   ({out.name})", x=0.06, ha="left",
                 y=1.0 - 0.15 / height, fontsize=12, color=_INK)
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def plot(a_path: Path, b_path: Path, out: Path) -> list[Path]:
    a, ma, b, mb = load_pair(a_path, b_path)
    t = a["t"]
    channels = sorted(k for k in a if k != "t")
    missing = [k for k in channels if k not in b]
    assert not missing, f"channels absent in B: {missing}"

    out.parent.mkdir(parents=True, exist_ok=True)
    pages = [channels[i:i + _PER_PAGE] for i in range(0, len(channels), _PER_PAGE)]
    paths = []
    for n, chunk in enumerate(pages, start=1):
        path = out.with_name(f"{out.stem}_{n:02d}{out.suffix}")
        _page(a, ma, b, mb, t, chunk, a_path.name, b_path.name, path)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay two parity_record .npz files channel by channel and plot |A - B|, 2 channels per PNG page.")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True,
                    help="output stem: X.png writes X_01.png, X_02.png, ... (2 channels per page)")
    args = ap.parse_args()
    for path in plot(args.a, args.b, args.out):
        print(f"[parity] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
