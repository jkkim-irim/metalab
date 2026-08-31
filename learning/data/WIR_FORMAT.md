# wir_v1 — the wirobotics dataset contract

`wir_v1` is the dataset **interface contract** every wirobotics model reads through
`learning/data/` — ACT today, our GR00T integration and others next. Owning one reader + one
contract means **a dataset built once is consumable by every model with no per-consumer
conversion**.

`wir_v1` is **based on the LeRobot dataset format v3.0** (codebase_version `v3.0`). It does not
invent a new on-disk layout — it *names and pins the subset of v3.0 our pipeline depends on*, so we
can evolve that contract on our own schedule.

## Two version axes — keep them separate

Conflating these is exactly what made "v3" confusing (and why `learning/data/` is no longer a
`v3/` package):

| axis | example | meaning | who owns it |
|---|---|---|---|
| on-disk `codebase_version` | `v3.0` | how the parquet/video bytes are laid out | LeRobot (upstream) |
| **`wir_data_version`** | `wir_v1` | the stable *interface* models code against | wirobotics |

The on-disk format is an implementation detail of `lerobot_dataset.py`. `wir_v1` is the thing a
model breaks against. They version independently: we could move to a different on-disk format later
and keep serving `wir_v1`, or change the contract (→ `wir_v2`) while the bytes stay v3.0.

## On-disk layout (LeRobot v3.0)

```
<root>/
├── meta/
│   ├── info.json                       # codebase_version, fps, features{}, total_*, data/video_path templates
│   ├── stats.json                      # per-feature normalization stats (mean/std/min/max/q…)
│   ├── tasks.parquet                   # task strings
│   ├── episodes/chunk-*/file-*.parquet # per-episode rows incl. dataset_from_index / dataset_to_index
│   └── (subtasks.parquet)              # optional
├── data/chunk-{c:03d}/file-{f:03d}.parquet     # frames (many episodes per file)
└── videos/{video_key}/chunk-*/file-*.mp4       # consolidated video
```

## Schema contract (what `wir_v1` guarantees)

Embodiment-agnostic — it pins **structure**, not dims, so ALLEX / LIBERO / future robots all
satisfy one contract:

| feature | requirement |
|---|---|
| `observation.state` | 1-D **float** vector + normalization stats in `stats.json` |
| `action` | 1-D **float** vector + normalization stats |
| `observation.images.*` | ≥1 RGB camera, each shape `(3, H, W)`, dtype `video` or `image` |
| `timestamp`, `index`, `episode_index`, `frame_index` | v3.0 bookkeeping columns (reader + sampler rely on them) |
| `fps` | positive |

Concrete ALLEX example (`v3_phase2_132d_3cam`): `robot_type: allex`, `fps: 30`,
`observation.state` f32 `(132,)` (`pos|vel|torque`, 44 each), `action` f32 `(44,)`
(`r_arm 7 | l_arm 7 | r_hand 15 | l_hand 15`), `observation.images.camera_{1,2,3}` `(3,224,224)`.

## Field-split / modality — a separate, per-embodiment layer

`wir_v1` guarantees state/action are float vectors; it does **not** fix what the slices *mean* —
that's the **modality map** (e.g. ALLEX action = `r_arm[0:7] | l_arm[7:14] | r_hand[14:29] |
l_hand[29:44]`). It lives outside the structural contract:

- `learning/metrics/validation.py` `ALLEX_ACTION_GROUPS` (per-body-part metrics) is one consumer;
- a GR00T `meta/modality.json` is a **projection** of the same map — so our GR00T reads our reader
  directly, and only NVIDIA's *vanilla* GR00T loader needs the v3.0→v2.1 + `modality.json` bridge.

A different embodiment supplies its own modality map without touching `wir_v1`.

## Versioning policy

- Bump to `wir_v2` only when the **interface** changes in a way models must adapt to (e.g. state
  layout/semantics, a new required feature, a normalization convention). Adding an *optional* camera
  or a new robot's modality map does **not** bump it.
- New datasets **should stamp** `"wir_data_version": "wir_v1"` into `meta/info.json` (the converter
  should write it). `validate_wir_contract` asserts it matches when present; it does not *require*
  the field yet, so existing datasets remain valid on their structure alone.

## Enforcement

`learning/data/contract.py::validate_wir_contract(meta)` runs inside `make_dataset` — every dataset
entering the pipeline is checked at the boundary and **fails loud** (`WirContractError`) listing all
violations. `WIR_DATA_VERSION` / `WIR_BASED_ON` are the canonical constants.
