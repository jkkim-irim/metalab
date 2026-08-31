# Dataset conversion — raw → LeRobot v3.0 (132-D)

How the bolt-drilling training dataset is built from raw captures, and the exact recipe for the
**canonical baseline** the ACT reproduction uses.

## Files
- `convert_raw_to_v3_state_variant.py` — the ALLEX GR00T converter. Turns raw per-episode arrays
  + JPEG frames into a LeRobot **v3.0** dataset. State layout is selectable via flags:
  - default → **44-D** (`pos` only)
  - `--include-torque` → **88-D** (`pos | torque`)
  - `--include-velocity --include-torque` → **132-D** (`pos(44) | velocity(44) | torque(44)`)

  The 44 = `r_arm(7) | l_arm(7) | r_hand(15) | l_hand(15)` (also the action dim). The 132-D
  `pos|vel|torque` layout is the one hansol's `v3_132d_*` datasets use.
- `build_v3_132d_canonical.sh` — end-to-end recipe: download raw tarballs → extract → combine into
  one sequentially-numbered raw dir → convert (132-D) → verify counts → upload to S3.

## Raw input format
```
raw/episode_XXXXXX/
  state.npy  action.npy  velocity.npy  pos_target.npy  vel_target.npy  trq_target.npy
  camera_1/frame_XXXXXX.jpg   camera_2/frame_XXXXXX.jpg
  episode_meta.json
```

## Canonical dataset
`build_v3_132d_canonical.sh` produces **`v3_132d_20260510_12_382ep`** — 382 episodes / 416,936
frames, LeRobot v3.0, 132-D state — at
`s3://wirobotics-internal/chrisryu/datasets/allex/lerobot/bolt_drilling/v3_132d_20260510_12_382ep`.

This **rebuilds hansol's combined baseline dataset on S3.** His original `v3_132d_20260513_382ep`
lived only on his workstation (`/home/hansol/pippo/workspace/Dataset/...`) and was **deleted/lost**,
so the equivalence run could not use it directly. The rebuild uses the same source captures
(20260510 nice+middle, 20260512 nice+middle = 382 eps, old 2-camera setup) and reproduces the
reference run `5m3dzdwe` within run-to-run noise (≤ 0.003 val_loss). Dataset/run provenance:
hansol's data map — `s3://wirobotics-internal/hansol/docs/phase1_bolt_drilling_data_map.html`
(= https://d1iitptfxhu64e.cloudfront.net/hansol/docs/phase1_bolt_drilling_data_map.html, Slack-authed).

## Gotcha — image stats
The converter writes `state`/`action` stats but **not** image stats. Before training, patch
`meta/stats.json` with ImageNet `mean [0.485,0.456,0.406]` / `std [0.229,0.224,0.225]` (shape
`(3,1,1)`) for each `observation.images.*` key — otherwise lerobot's `make_dataset` raises
`KeyError` under visual mean/std normalization.
