"""wir_v1 — the wirobotics dataset contract.

Defines and enforces the schema every wirobotics model reads through ``learning/data/`` (ACT today;
our GR00T integration + others next). The point of owning one reader + one contract is that a
dataset built once is consumable by every model with no per-consumer conversion.

Two version axes, kept deliberately separate (conflating them is what made "v3" confusing):

  * the on-disk LeRobot ``codebase_version`` (``v3.0``) — how the parquet/video bytes are laid out,
    an implementation detail the reader (``lerobot_dataset.py``) parses;
  * **wir_v1** — the stable *interface* consumers code against (a float state/action vector with
    normalization stats, >=1 RGB camera, the v3.0 bookkeeping columns, a positive fps). This is what
    breaks a model when it changes, so it carries an explicit wirobotics version, decoupled from
    lerobot's. wir_v1 is **based on LeRobot dataset format v3.0**.

The contract is embodiment-agnostic: it checks *structure*, not specific dims, so ALLEX (132-D
state / 44-D action), LIBERO, etc. all satisfy one contract. The per-robot field split (the
"modality" map: which slices of the action are r_arm/l_arm/hand/gripper) is a *separate* layer —
see ``WIR_FORMAT.md`` and ``learning/metrics/validation.py`` (group metrics) — and is what a GR00T
``modality.json`` projects from.

See ``learning/data/WIR_FORMAT.md`` for the full written spec.
"""
from typing import Any

# The wirobotics dataset-contract version. Bump (wir_v2, …) when the *interface* below changes.
WIR_DATA_VERSION = "wir_v1"

# The on-disk LeRobot format wir_v1 is built on (major version checked below).
WIR_BASED_ON = "lerobot dataset format v3.0"
_BASED_ON_MAJOR = "3"

_FLOAT_DTYPES = {"float32", "float64"}
_IMAGE_DTYPES = {"video", "image"}
# Bookkeeping columns every v3.0 frame table carries; the reader + EpisodeAwareSampler rely on them.
_REQUIRED_BOOKKEEPING = ("timestamp", "index", "episode_index", "frame_index")
# The two model-facing tensors wir_v1 guarantees.
_REQUIRED_VECTORS = ("observation.state", "action")


class WirContractError(ValueError):
    """Raised when a dataset does not satisfy the wir_v1 contract."""


def validate_wir_contract(meta: Any) -> None:
    """Assert a dataset's metadata satisfies wir_v1 (fail loud), collecting all violations.

    ``meta`` is a ``LeRobotDatasetMetadata`` (duck-typed: ``.info`` dict, ``.features``, ``.fps``,
    ``.stats``, ``.repo_id``). Called from ``make_dataset`` so every dataset entering the pipeline
    is checked at the boundary.
    """
    info = getattr(meta, "info", {}) or {}
    features = getattr(meta, "features", {}) or {}
    stats = getattr(meta, "stats", {}) or {}
    problems: list[str] = []

    # On-disk format wir_v1 is based on: LeRobot v3.x.
    cv = str(info.get("codebase_version", ""))
    if cv.lstrip("v").split(".")[0] != _BASED_ON_MAJOR:
        problems.append(f"codebase_version {cv!r} is not {WIR_BASED_ON} (wir_v1 is based on it)")

    # If a dataset declares its wir version, it must match (new datasets should stamp this).
    declared = info.get("wir_data_version")
    if declared is not None and declared != WIR_DATA_VERSION:
        problems.append(f"meta wir_data_version {declared!r} != {WIR_DATA_VERSION!r}")

    # observation.state / action: 1-D float vectors with normalization stats.
    for key in _REQUIRED_VECTORS:
        ft = features.get(key)
        if ft is None:
            problems.append(f"missing required feature {key!r}")
            continue
        if ft.get("dtype") not in _FLOAT_DTYPES:
            problems.append(f"{key!r} dtype {ft.get('dtype')!r} is not a float type")
        if len(tuple(ft.get("shape", ()))) != 1:
            problems.append(f"{key!r} must be a 1-D vector, got shape {tuple(ft.get('shape', ()))}")
        if key not in stats:
            problems.append(f"{key!r} has no normalization stats in meta/stats.json")

    # At least one RGB camera, each (C=3, H, W).
    cams = [k for k, ft in features.items() if ft.get("dtype") in _IMAGE_DTYPES]
    if not cams:
        problems.append("no camera feature (observation.images.*) found")
    for k in cams:
        shape = tuple(features[k].get("shape", ()))
        if len(shape) != 3 or shape[0] != 3:
            problems.append(f"camera {k!r} shape {shape} is not (3, H, W)")

    # Bookkeeping columns the reader / sampler depend on.
    for key in _REQUIRED_BOOKKEEPING:
        if key not in features:
            problems.append(f"missing bookkeeping feature {key!r}")

    fps = getattr(meta, "fps", None)
    if not fps or fps <= 0:
        problems.append(f"fps must be a positive number, got {fps!r}")

    if problems:
        raise WirContractError(
            f"dataset {getattr(meta, 'repo_id', '?')!r} violates the {WIR_DATA_VERSION} contract:\n  - "
            + "\n  - ".join(problems)
        )
