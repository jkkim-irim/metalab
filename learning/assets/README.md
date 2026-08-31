# Vendored FK reference URDF

`ALLEX_0.1.3.urdf` — the forward-kinematics reference for the S1 ACT recipe
(`learning/scripts/train_act_s1.sh`), used to compute `fk_mm_*` task-space validation metrics.

## Why vendored (not resolved via `ALLEX_DESCRIPTION_DIR`)
The ALLEX robot description is still in flux and can't be pegged to a stable
`ALLEX_DESCRIPTION_DIR` version yet, so we vendor a **pinned snapshot** for reproducible `fk_mm`.
Bump the filename when the robot generation changes.

## Version / provenance (verified 2026-07-03)
- **Generation: v0.1.3.** Kinematically matches `wirobotics-rih/allex_description` tag **`v0.1.3`**
  (71 joints / 72 links, 5-finger hand). Tag `v0.2.0` is a *different* model (81 joints) — do not use.
- **NOT byte-identical to the `v0.1.3` tag file.** This is the *FK-compatible* export the FK code +
  the bolt-drilling dataset were built against — recovered from the allex monorepo at `3ed429a^`
  (pre-`allex_rl`-restructure). The allex_description-repo URDFs use an incompatible hand model and a
  different link-naming convention (here `L_Middle_DIP_Link`; the tag has `L_Middle_Distal_Link`), and
  `learning/metrics/allex_fk.py` looks links up **by name** — so swapping in the raw tag file would
  break FK. Keep this file.
- **md5**: `f481b4933b575d9eedfefcfa9785b290`
