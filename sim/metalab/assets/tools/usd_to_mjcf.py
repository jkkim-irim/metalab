"""USD → MJCF object converter: point it at a USD, get an object asset folder.

    <out>/<name>/<name>.xml
    <out>/<name>/meshes/<name>_hull*.obj

That is `sim/metalab/assets/objects/`'s layout, so the result drops in as-is. Authored coordinates are baked in, including the frame knobs, so nothing has to
be fixed up at load time. Needs pxr + coacd (the newton env); `--help` documents the knobs.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import coacd
import numpy as np
from pxr import Gf, Usd, UsdGeom

_COACD = dict(preprocess_mode="auto", decimate=True, max_ch_vertex=256, seed=0)


def _extract_meshes(usd_path: Path):
    """USD → ([(prim name, verts, tris)] in default-prim-local coords, grasp_position or None)."""
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    meshes, grasp = [], None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == "grasp_position":
            rel, _ = cache.ComputeRelativeTransform(prim, root)
            grasp = tuple(float(x) for x in rel.ExtractTranslation())
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        rel, _ = cache.ComputeRelativeTransform(prim, root)
        pts = np.array([rel.Transform(Gf.Vec3d(*map(float, v))) for v in mesh.GetPointsAttr().Get()], dtype=np.float64)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        tris, off = [], 0
        for c in counts:                                      # polygon → triangle fan
            for k in range(1, c - 1):
                tris.append((idx[off], idx[off + k], idx[off + k + 1]))
            off += c
        meshes.append((prim.GetName(), pts, np.array(tris)))
    assert meshes, f"{usd_path}: no Mesh prims under the default prim"
    return meshes, grasp


def _rot_matrix(axis: str, deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def _coacd_split(verts: np.ndarray, tris: np.ndarray, parts: int, threshold: float):
    """``parts`` is a CAP — a mesh in many disconnected shells may not come down to it."""
    out = coacd.run_coacd(coacd.Mesh(verts, np.asarray(tris, dtype=np.int32)),
                          max_convex_hull=parts, threshold=threshold, **_COACD)
    assert out, "CoACD returned no hulls"
    return [(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)) for v, f in out]


def _write_obj(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {i + 1} {j + 1} {k + 1}" for i, j, k in tris]
    path.write_text("\n".join(lines) + "\n")


def convert(usd: Path, out_root: Path, name: str, rgba, coacd_parts: int = 0, threshold: float = 0.05,
            rot: tuple[str, float] | None = None, fit: tuple[str, float] | None = None,
            translate: tuple[float, float, float] | None = None) -> Path:
    """One USD → ``<out_root>/<name>/``. Returns the MJCF path.

    ``rot`` → ``fit`` → ``translate`` are applied in that order, all about the object frame origin, which
    is how the origin is moved onto a chosen point (a grasp point, say) and a mesh brought into a
    sibling's size class."""
    meshes, grasp = _extract_meshes(usd)
    mesh_dir = out_root / name / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if rot is not None:
        meshes = [(n, v @ _rot_matrix(*rot).T, t) for n, v, t in meshes]
    scale = 1.0
    if fit is not None:
        axis, size = fit
        a = np.vstack([v for _, v, _ in meshes])[:, {"x": 0, "y": 1, "z": 2}[axis]]
        scale = size / float(a.max() - a.min())
        meshes = [(n, v * scale, t) for n, v, t in meshes]
    if translate is not None:
        meshes = [(n, v + np.asarray(translate, dtype=np.float64), t) for n, v, t in meshes]

    if coacd_parts:
        assert len(meshes) == 1, f"{usd.name}: --coacd expects ONE authored mesh, found {len(meshes)}"
        parts = _coacd_split(meshes[0][1], meshes[0][2], coacd_parts, threshold)
    else:
        parts = [(v, t) for _, v, t in meshes]
    geoms = [(f"{name}_hull{i}", v, t) for i, (v, t) in enumerate(parts)]

    asset = []
    for mname, verts, tris in geoms:
        _write_obj(mesh_dir / f"{mname}.obj", verts, tris)
        asset.append(f'    <mesh name="{mname}" file="{mname}.obj"/>')
    # Each hull is emitted TWICE. genesis draws any geom with contype/conaffinity != 0 as a grey collision
    # shape, so the coloured copy has to be the non-colliding one.
    vis = [f'      <geom type="mesh" mesh="{m}" class="visual"/>' for m, _, _ in geoms]
    col = [f'      <geom type="mesh" mesh="{m}" class="collision"/>' for m, _, _ in geoms]

    src = (f"{usd.name} ({', '.join(m[0] for m in meshes)})"
           + (f", rotated {rot[1]:g} deg about {rot[0]}" if rot is not None else "")
           + (f", scaled x{scale:.4f} to {fit[0]}={fit[1] * 1000:g}mm" if fit is not None else "")
           + (f", translated {tuple(round(c * 1000, 1) for c in translate)}mm" if translate is not None else ""))
    prov = (f"CoACD (<= {coacd_parts} parts, threshold {threshold}, seed {_COACD['seed']}) → {len(parts)} hulls"
            if coacd_parts else "the USD's own convex decomposition")
    mjcf = out_root / name / f"{name}.xml"
    mjcf.write_text(f"""<mujoco model="{name}">
  <compiler meshdir="meshes" angle="radian"/>
  <!-- GENERATED by sim/metalab/assets/tools/usd_to_mjcf.py from {src}
       Collision: {prov}; the same hulls carry the visual.
       Mass and contact params are NOT authored here — the task contract owns them. -->
  <default>
    <default class="visual"><geom contype="0" conaffinity="0" group="2" rgba="{' '.join(f'{c:g}' for c in rgba)}"/></default>
    <default class="collision"><geom group="3"/></default>
  </default>
  <asset>
{chr(10).join(asset)}
  </asset>
  <worldbody>
    <body name="object" pos="0 0 0">
      <freejoint name="object_free"/>
{chr(10).join(vis)}
{chr(10).join(col)}
    </body>
  </worldbody>
</mujoco>
""")
    bbox = np.vstack([v for _, v, _ in geoms])
    print(f"[{name}] hulls={len(geoms)} verts={[len(v) for _, v, _ in geoms]} "
          f"bbox_min={np.round(bbox.min(0), 3).tolist()} max={np.round(bbox.max(0), 3).tolist()} "
          f"grasp={grasp} scale={scale:.4f} → {mjcf}")
    return mjcf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--usd", required=True, type=Path, help="source USD")
    ap.add_argument("--out", required=True, type=Path, help="output root; the asset lands in <out>/<name>/")
    ap.add_argument("--name", default=None, help="asset name (default: the USD stem, lowercased)")
    ap.add_argument("--rgba", nargs=4, type=float, default=[0.7, 0.7, 0.7, 1.0], metavar=("R", "G", "B", "A"),
                    help="visual colour, baked in so both engines render it the same")
    ap.add_argument("--coacd", type=int, default=0, metavar="N",
                    help="split ONE authored mesh into up to N convex parts for collision (0 = the USD "
                         "already ships a decomposition). A mesh geom collides as its convex hull, so a "
                         "non-convex object needs this or its openings fill in")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="CoACD concavity threshold; raise it when a fragmented mesh will not reach --coacd")
    ap.add_argument("--rot", nargs=2, default=None, metavar=("AXIS", "DEG"), help="rotate about x|y|z, degrees")
    ap.add_argument("--fit", nargs=2, default=None, metavar=("AXIS", "SIZE"),
                    help="uniform rescale until AXIS spans SIZE metres")
    ap.add_argument("--translate", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                    help="shift the geometry [m] — this is what moves the ORIGIN to a chosen point")
    a = ap.parse_args()
    convert(a.usd, a.out, a.name or a.usd.stem.lower(), a.rgba, coacd_parts=a.coacd, threshold=a.threshold,
            rot=(a.rot[0], float(a.rot[1])) if a.rot else None,
            fit=(a.fit[0], float(a.fit[1])) if a.fit else None,
            translate=tuple(a.translate) if a.translate else None)


if __name__ == "__main__":
    main()
