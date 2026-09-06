from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sim.metalab.contract.loader import parity_module  # noqa: E402
from sim.metalab.tools.parity_diff import load_pair  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_FONT = "Noto Sans CJK KR"
_A = "#2a78d6"
_B = "#eb6834"
_INK = "#0b0b0b"
_MUTED = "#52514e"
_GRID = "#e6e5e1"
_PER_PAGE = 2
_ROW_IN = 3.4
_HEAD_IN = 2.4

_LABELS = {
    "joint_pos": "관절 위치 [rad]",
    "joint_vel": "관절 속도 [rad/s]",
    "joint_torque": "관절 토크 [N·m]",
    "joint_torque_pd": "PD 토크 [N·m]",
    "joint_torque_gravcomp": "중력 보상 토크 [N·m]",
    "target": "목표 관절 위치 (입력) [rad]",
    "body_pos": "바디 위치 [m]",
    "body_quat": "바디 자세 (쿼터니언)",
    "body_lin_vel": "바디 선속도 [m/s]",
    "body_ang_vel": "바디 각속도 [rad/s]",
    "contact_force": "바디 접촉력 [N]",
    "contact_force_with": "물체와의 접촉력 [N]",
    "contact_penetration": "물체 관통 깊이 [m]",
    "object_pos": "물체 위치 [m]",
    "object_quat": "물체 자세 (쿼터니언)",
    "object_lin_vel": "물체 선속도 [m/s]",
    "object_ang_vel": "물체 각속도 [rad/s]",
    "action": "행동 (입력, 정규화)",
    "obs": "관측",
    "reward": "보상 합",
    "reward_term": "보상 항",
    "done": "에피소드 종료",
    "time_out": "시간 초과",
}


def _label(key: str) -> str:
    head, _, tail = key.partition(".")
    assert head in _LABELS, f"no Korean label for channel {key!r}; add {head!r} to _LABELS"
    return f"{_LABELS[head]} {tail}".rstrip()


def _command_lines(ma: dict) -> list[str]:
    contract = _REPO / (parity_module(ma["task"].replace("-", "_")).replace(".", "/") + ".py")
    lines = [f"궤적: {contract.relative_to(_REPO)} 의 COMMAND"]
    if ma["mode"] == "mdp":
        lines.append(f"  EnvDriver.step 에 정규화 행동 사인파 — 진폭 {ma['action_amp']}, {ma['freq_hz']} Hz, "
                     f"ramp {ma['ramp_s']} s, seed {ma['seed']}, {ma['seconds']} s")
    else:
        lines.append(f"  관절 목표 사인파 — {', '.join(ma['driven'])}: 진폭 {ma['amp_deg']} deg, {ma['freq_hz']} Hz, "
                     f"ramp {ma['ramp_s']} s, {ma['seconds']} s")
    lines.append(f"  {ma['steps']} steps, dt {ma['dt'] * 1e3:.3f} ms (physics {ma['hz']:g} Hz × substeps "
                 f"{ma['substeps']}, decimation {ma['decimation']})")
    return lines


def _style(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=9)


def _page(a, ma, b, mb, t, channels: list[str], a_path: Path, b_path: Path, out: Path) -> None:
    height = _ROW_IN * len(channels) + _HEAD_IN
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, height), squeeze=False,
                             gridspec_kw={"hspace": 0.5})
    fig.subplots_adjust(top=1.0 - _HEAD_IN / height, bottom=0.6 / height, left=0.07, right=0.98)
    for (ax,), k in zip(axes, channels):
        x = a[k].reshape(len(t), -1).astype(np.float64)
        y = b[k].reshape(len(t), -1).astype(np.float64)
        assert x.shape == y.shape, f"{k}: shape {x.shape} vs {y.shape}"
        d = np.abs(x - y)
        alpha = 0.9 if x.shape[1] <= 3 else max(0.15, 3.0 / x.shape[1])
        ax.plot(t, x, color=_A, linewidth=1.6, alpha=alpha)
        ax.plot(t, y, color=_B, linewidth=1.6, alpha=alpha, linestyle="--")
        ax.set_title(f"{_label(k)}   {k} [{'x'.join(map(str, a[k].shape))}]        "
                     f"오차 |A−B|  최대 {d.max():.3e}   rms {np.sqrt(np.mean(d * d)):.3e}   "
                     f"(A 최대 절대값 {np.abs(x).max():.3e})",
                     loc="left", fontsize=11, color=_INK)
        _style(ax)
    axes[-1][0].set_xlabel("t [s]", color=_MUTED, fontsize=9)

    fig.suptitle(f"{ma['task']} — genesis 와 newton 을 같은 입력으로 구동해 비교   ({out.name})",
                 x=0.07, ha="left", y=1.0 - 0.15 / height, fontsize=13, color=_INK)
    fig.text(0.07, 1.0 - 0.45 / height, "\n".join(_command_lines(ma)), va="top", fontsize=9.5, color=_MUTED,
             linespacing=1.4)
    fig.legend(handles=[plt.Line2D([], [], color=_A, linewidth=2,
                                   label=f"A: {ma['engine']}  {a_path.name}  git {ma['git']}"),
                        plt.Line2D([], [], color=_B, linewidth=2, linestyle="--",
                                   label=f"B: {mb['engine']}  {b_path.name}  git {mb['git']}")],
               loc="upper left", bbox_to_anchor=(0.07, 1.0 - 1.3 / height), fontsize=9.5, frameon=False)
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def plot(a_path: Path, b_path: Path, out: Path) -> list[Path]:
    assert any(f.name == _FONT for f in fm.fontManager.ttflist), f"font {_FONT!r} not found; install fonts-noto-cjk"
    plt.rcParams["font.family"] = [_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
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
        _page(a, ma, b, mb, t, chunk, a_path, b_path, path)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay two parity_record .npz files channel by channel (2 channels per PNG page); "
                    "the error |A-B| is printed in each panel title.")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, required=True,
                    help="output stem: X.png writes X_01.png, X_02.png, ...")
    args = ap.parse_args()
    for path in plot(args.a, args.b, args.out):
        print(f"[parity] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
