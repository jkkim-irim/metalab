"""Forward kinematics for the ALLEX bimanual robot: map the 44-D LeRobot action/state
vector -> Cartesian link poses (fingertips, wrists, arm skeleton).

Used by the validation metrics (learning/metrics/validation.py) to report task-space
(fingertip/wrist) prediction error in millimeters and to render pred-vs-GT skeleton videos —
a physically meaningful complement to the joint-space L1 val_loss.

44-D layout (verified against data + allex_rl_dexblind/joint_constants.py):
    [0:7]   r_arm   Shoulder[Pitch,Roll,Yaw], Elbow, Wrist[Yaw,Roll,Pitch]
    [7:14]  l_arm   (same; left side may use ABAD instead of Roll -> resolved)
    [14:29] r_hand  Thumb[Yaw,CMC,MCP], then Index/Middle/Ring/Little [ABAD,MCP,PIP]
    [29:44] l_hand  (same, L_ prefix)

Units: ACTION is absolute joint position in RADIANS (verified: action≈deg2rad(state)).
       STATE is the same but in DEGREES -> use deg=True.

The URDF has 5 extra coupled joints per hand (4x *_DIP, Thumb_IP) handled via <mimic>:
DIP = mult * PIP, Thumb_IP = mult * Thumb_MCP. We re-inflate 44 -> full DOF by computing those
from their parents; waist/neck/base stay at 0 (init pose). pytorch_kinematics needs only the
URDF's kinematic tree for FK — the package:// mesh references are never resolved.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytorch_kinematics as pk
import torch

# FK task-space validation is opt-in (--fk_validation, off by default). When enabled without an
# explicit --urdf_path, the ALLEX description URDF is resolved from ALLEX_DESCRIPTION_DIR — the
# allex_rl/allex_assets SSOT env (the allex_description tree is being reorganized, so resolve via the
# env rather than hardcode a path). Empty when unset -> AllexFK fails loud on open, prompting an
# explicit --urdf_path. pytorch_kinematics needs only the tree; package:// mesh refs aren't resolved.
_DESC_DIR = os.environ.get("ALLEX_DESCRIPTION_DIR", "")
# Fallback: the pinned URDF vendored in the repo (learning/assets/, what the recipe passes via
# --urdf_path). So FK validation and its tests resolve a URDF on any checkout even without
# ALLEX_DESCRIPTION_DIR set (the description tree is being reorganized; allex_rl is going away).
_VENDORED_URDF = Path(__file__).resolve().parent.parent / "assets" / "ALLEX_0.1.3.urdf"
DEFAULT_URDF = (
    str(Path(_DESC_DIR) / "urdf" / "ALLEX.urdf") if _DESC_DIR
    else (str(_VENDORED_URDF) if _VENDORED_URDF.exists() else "")
)

FINGERS = ["Index", "Middle", "Ring", "Little"]


def _arm(side: str) -> list[str]:
    p = side + "_"
    # Roll vs ABAD resolved later against the actual chain joints.
    return [p + "Shoulder_Pitch_Joint", p + "Shoulder_Roll_Joint", p + "Shoulder_Yaw_Joint",
            p + "Elbow_Joint", p + "Wrist_Yaw_Joint", p + "Wrist_Roll_Joint", p + "Wrist_Pitch_Joint"]


def _hand(side: str) -> list[str]:
    p = side + "_"
    names = [p + "Thumb_Yaw_Joint", p + "Thumb_CMC_Joint", p + "Thumb_MCP_Joint"]
    for f in FINGERS:
        names += [f"{p}{f}_ABAD_Joint", f"{p}{f}_MCP_Joint", f"{p}{f}_PIP_Joint"]
    return names


class AllexFK:
    def __init__(self, urdf_path: str = DEFAULT_URDF, device: str = "cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        with open(urdf_path, "rb") as f:
            self.chain = pk.build_chain_from_urdf(f.read())
        self.chain = self.chain.to(dtype=dtype, device=self.device)
        self.pk_joints: list[str] = self.chain.get_joint_parameter_names()  # pk DOF order
        self.pk_set = set(self.pk_joints)
        self.link_names: list[str] = self.chain.get_link_names()
        self.link_set = set(self.link_names)

        # Resolve the 44 logical joints -> exact URDF names (Roll/ABAD fallback).
        self.vec_joints: list[str] = []
        for nm in _arm("R") + _arm("L") + _hand("R") + _hand("L"):
            self.vec_joints.append(self._resolve_joint(nm))
        assert len(self.vec_joints) == 44, len(self.vec_joints)

        # Mimic (coupled) joints: child -> (parent, multiplier). Parents are in the 44.
        self.mimic: dict[str, tuple[str, float]] = {}
        for side in ("R", "L"):
            self._add_mimic(f"{side}_Thumb_IP_Joint", f"{side}_Thumb_MCP_Joint", 0.7319)
            for f in FINGERS:
                self._add_mimic(f"{side}_{f}_DIP_Joint", f"{side}_{f}_PIP_Joint", 0.6361)

        # Index of each pk joint: where in the 44-vec it comes from, or a mimic, or fixed 0.
        self.vec_idx = {j: i for i, j in enumerate(self.vec_joints)}

        # Links of interest.
        self.tips: dict[str, str] = {}
        self.wrists: dict[str, str] = {}
        self.palms: dict[str, str] = {}
        for side in ("R", "L"):
            for n, f in enumerate(["Thumb"] + FINGERS, start=1):   # finger_1=thumb .. finger_5=little
                self.tips[f"{side}_finger_{n}"] = self._resolve_link(
                    [f"{side}_{f}_Distal_Link", f"{side}_{f}_DIP_Link", f"{side}_{f}_Tip_Link"])
            self.wrists[side] = self._resolve_link([f"{side}_Wrist_Pitch_Link"])
            self.palms[side] = self._resolve_link([f"{side}_Palm_Link"])

    # --- name resolution ------------------------------------------------------
    def _resolve_joint(self, name: str) -> str:
        if name in self.pk_set:
            return name
        alt = name.replace("_Roll_", "_ABAD_")
        if alt in self.pk_set:
            return alt
        raise KeyError(f"joint not in URDF DOF: {name} (alt {alt}); have e.g. "
                       f"{[j for j in self.pk_joints if name.split('_')[1] in j][:6]}")

    def _resolve_link(self, candidates: list[str]) -> str:
        for c in candidates:
            if c in self.link_set:
                return c
        raise KeyError(f"none of {candidates} in links")

    def _add_mimic(self, child: str, parent: str, mult: float):
        if child in self.pk_set:  # only matters if pk treats it as a DOF
            self.mimic[child] = (parent, mult)

    # --- forward kinematics ---------------------------------------------------
    def _full_th(self, vec44: torch.Tensor) -> torch.Tensor:
        """(N,44) radians -> (N, n_pk_dof) in pk joint order, with mimic + fixed 0."""
        n = vec44.shape[0]
        th = torch.zeros(n, len(self.pk_joints), dtype=self.dtype, device=self.device)
        for k, j in enumerate(self.pk_joints):
            if j in self.vec_idx:
                th[:, k] = vec44[:, self.vec_idx[j]]
            elif j in self.mimic:
                parent, mult = self.mimic[j]
                th[:, k] = vec44[:, self.vec_idx[parent]] * mult
        return th

    def forward(self, vec44: torch.Tensor, deg: bool = False) -> dict[str, torch.Tensor]:
        """(N,44) -> {link_name: (N,3) world positions in meters}."""
        if vec44.dim() == 1:
            vec44 = vec44[None]
        vec44 = vec44.to(self.dtype).to(self.device)
        if deg:
            vec44 = vec44 * (torch.pi / 180.0)
        th = self._full_th(vec44)
        fk = self.chain.forward_kinematics(th)
        return {ln: tf.get_matrix()[:, :3, 3] for ln, tf in fk.items()}

    def tip_wrist_positions(self, vec44: torch.Tensor, deg: bool = False):
        """Return (tips dict, wrists dict) of (N,3) positions."""
        pos = self.forward(vec44, deg=deg)
        tips = {k: pos[v] for k, v in self.tips.items()}
        wrists = {k: pos[v] for k, v in self.wrists.items()}
        return tips, wrists

    def tip_wrist_poses(self, vec44: torch.Tensor, deg: bool = False):
        """One FK pass -> (tips_pos, wrists_pos, wrists_rot).

        tips_pos / wrists_pos: ``{key: (N,3)}`` world positions (m); wrists_rot: ``{side: (N,3,3)}``
        world rotation matrices (the wrist == EE orientation). Single forward_kinematics call so the
        position + orientation metrics don't double the FK cost.
        """
        if vec44.dim() == 1:
            vec44 = vec44[None]
        vec44 = vec44.to(self.dtype).to(self.device)
        if deg:
            vec44 = vec44 * (torch.pi / 180.0)
        th = self._full_th(vec44)
        fk = self.chain.forward_kinematics(th)
        mats = {ln: tf.get_matrix() for ln, tf in fk.items()}      # {link: (N,4,4)}
        tips = {k: mats[v][:, :3, 3] for k, v in self.tips.items()}
        wrists = {k: mats[v][:, :3, 3] for k, v in self.wrists.items()}
        wrists_rot = {s: mats[self.wrists[s]][:, :3, :3] for s in ("R", "L")}
        return tips, wrists, wrists_rot

    # --- 3D skeleton rendering ------------------------------------------------
    def _skeleton_chains(self) -> list[list[str]]:
        """Ordered link-name polylines forming a recognizable bimanual skeleton."""
        def L(name):
            return name if name in self.link_set else None
        chains = [[p for p in (L("Base_Link"), L("Chest_Origin_Link")) if p]]
        for s in ("R", "L"):
            arm = [L("Chest_Origin_Link"), L(f"{s}_Shoulder_Pitch_Link"),
                   L(f"{s}_Elbow_Link"), L(f"{s}_Wrist_Pitch_Link"), self.palms[s]]
            chains.append([p for p in arm if p])
            for n in range(1, 6):   # finger_1..5
                chains.append([self.palms[s], self.tips[f"{s}_finger_{n}"]])
        return chains

    def render_motion(self, gt44, pred44, out_path: str, fps: int = 8,
                      deg: bool = False, elev: int = 16, azim: int = -68,
                      title: str = "") -> str:
        """Render a blue(GT)/red(pred) skeleton animation over an action chunk -> MP4.

        matplotlib + imageio-ffmpeg are imported lazily: they are needed ONLY for this optional
        video path, not for the FK metrics, so importing AllexFK never requires the plotting stack.
        """
        import matplotlib
        matplotlib.use("Agg")  # headless backend; must be set before importing pyplot
        # FFMpegWriter needs an ffmpeg binary; the system one isn't guaranteed (apt install can
        # fail), so use imageio-ffmpeg's bundled binary (pip-installed).
        import imageio_ffmpeg
        from matplotlib.animation import FFMpegWriter, FuncAnimation
        import matplotlib.pyplot as plt
        import numpy as np
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

        gt44 = torch.as_tensor(gt44, dtype=self.dtype)
        pred44 = torch.as_tensor(pred44, dtype=self.dtype)
        if gt44.dim() == 1:
            gt44 = gt44[None]
        if pred44.dim() == 1:
            pred44 = pred44[None]
        chains = self._skeleton_chains()
        links = sorted({ln for ch in chains for ln in ch})
        pg = self.forward(gt44, deg=deg)
        pp = self.forward(pred44, deg=deg)
        G = {ln: pg[ln].cpu().numpy() for ln in links}
        P = {ln: pp[ln].cpu().numpy() for ln in links}
        N = gt44.shape[0]

        stacked = np.concatenate(
            [np.stack([G[ln] for ln in links], 1).reshape(-1, 3),
             np.stack([P[ln] for ln in links], 1).reshape(-1, 3)], 0)
        ctr = (stacked.min(0) + stacked.max(0)) / 2
        rng = max((stacked.max(0) - stacked.min(0)).max() / 2 * 1.1, 1e-3)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        def draw(i):
            ax.clear()
            ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
            ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
            ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{title}  frame {i + 1}/{N}   blue=GT  red=pred")
            ax.set_xlabel("x")
            ax.set_ylabel("y (+=left)")
            ax.set_zlabel("z")
            for D, col in ((G, "tab:blue"), (P, "tab:red")):
                for ch in chains:
                    if len(ch) < 1:
                        continue
                    pts = np.stack([D[ln][i] for ln in ch], 0)
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, lw=1.8,
                            marker="o", ms=2.5)
            # Label the robot's right/left wrist so the side is unambiguous in 3D.
            for s in ("R", "L"):
                wl = self.wrists[s]
                if wl in G:
                    x, y, z = G[wl][i]
                    ax.text(x, y, z, f"  {s}", color="black", fontsize=12, fontweight="bold")
            return []

        anim = FuncAnimation(fig, draw, frames=N, interval=1000.0 / fps)
        anim.save(out_path, writer=FFMpegWriter(fps=fps), dpi=90)
        plt.close(fig)
        return out_path
