"""Self-contained CLI parser (de-lerobot'd, de-draccus'd).

Reproduces the ``@parser.wrap()`` decorator and the ``argv -> TrainPipelineConfig`` parse that
``learning/train.py`` relies on, for the ALLEX argv patterns (see ``learning/scripts/train.sh``),
WITHOUT depending on draccus.

Supported argv (matching draccus's behaviour for our recipe):
  * top-level scalars:   ``--batch_size 8``  ``--seed 1000``  ``--tolerance_s 0.1``
  * also the ``--key=value`` form
  * nested dataclasses:  ``--dataset.repo_id X``  ``--dataset.video_backend torchcodec``
  * deeply nested:       ``--dataset.image_aug_config.brightness 0.9,1.1``
  * the wandb block:     ``--wandb.enable true``  ``--wandb.project P``  ``--wandb.entity E``
  * policy selection:    ``--policy.type act`` selects ``ACTConfig`` (the only registered choice),
                         then ``--policy.<field>`` sets its fields.

Coercion is driven by the destination dataclass field's type annotation:
  * ``bool``         -> ``true``/``false`` (case-insensitive; also ``1``/``0``, ``yes``/``no``)
  * ``int``          -> ``int(value)``
  * ``float``        -> ``float(value)``  (so ``1e-05`` works)
  * ``str``          -> the raw string
  * ``X | None``     -> coerce to ``X`` unless the value is the literal ``null``/``none`` -> None
  * ``list[int]``    -> comma-separated or JSON list

This is a small, explicit parser (no draccus, no plugins, no ``.path`` / hub loading). It is
fail-loud: an unknown ``--key`` or a value that won't coerce raises.
"""

import dataclasses
from functools import wraps
import inspect
import json
import sys
import types
import typing
from typing import Any, Union, get_args, get_origin

from learning.configs.config import TrainPipelineConfig
from learning.model.act.configuration import ACTConfig
from learning.model.groot.configuration import GrootConfig

# Registry of policy choices selectable via ``--policy.type``. LeRobot registers many policy
# configs as draccus ChoiceRegistry subclasses; ALLEX only ever uses ACT, so this is the single
# entry our parser needs. Adding another policy = add it here.
POLICY_CHOICES: dict[str, type] = {"act": ACTConfig, "groot": GrootConfig}

# Policy CLI fields that LeRobot's ``PreTrainedConfig`` accepts but the de-lerobot'd ``ACTConfig``
# intentionally dropped (HF-hub / PEFT / AMP machinery the ALLEX trainer never reads). train.sh
# passes ``--policy.push_to_hub false``; draccus would set it on the lerobot config, but for us it
# is inert, so we accept-and-ignore these rather than fail on an "unknown field". Keeping this
# explicit (vs. silently swallowing any unknown ``--policy.X``) preserves fail-loud on real typos.
DROPPED_POLICY_FIELDS: frozenset[str] = frozenset(
    {
        "push_to_hub",
        "repo_id",
        "private",
        "tags",
        "license",
        "pretrained_path",
        "use_amp",
        "use_peft",
    }
)

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}
_NULL = {"null", "none", ""}


def _strip_optional(tp: Any) -> tuple[Any, bool]:
    """If ``tp`` is ``X | None`` / ``Optional[X]``, return ``(X, True)`` else ``(tp, False)``."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        is_optional = len(args) != len(get_args(tp))
        if len(args) == 1:
            return args[0], is_optional
        # Multi-type union (rare here) — return the union of non-None args; coercion falls back to
        # str. Keep is_optional so a `null` value maps to None.
        return Union[tuple(args)], is_optional
    return tp, False


def _coerce_scalar(value: str, tp: Any, field_path: str) -> Any:
    """Coerce a raw CLI string to the destination type ``tp``."""
    base, is_optional = _strip_optional(tp)
    if is_optional and value.lower() in _NULL:
        return None

    if base is bool:
        low = value.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"--{field_path}: cannot parse boolean from {value!r}")
    if base is int:
        return int(value)
    if base is float:
        return float(value)
    if base is str or base is Any:
        return value

    origin = get_origin(base)
    if origin in (list, typing.List):  # noqa: UP006 — typing.List for older annotations
        return _coerce_list(value, base, field_path)
    if origin is tuple:  # e.g. ImageAugConfig ranges: tuple[float, float] from "0.9,1.1"
        return _coerce_tuple(value, base, field_path)
    if dataclasses.is_dataclass(base):
        raise ValueError(
            f"--{field_path}: a whole dataclass cannot be set from a single value; "
            "use nested dot-args (e.g. --dataset.repo_id ...)."
        )
    # Fallback: pass the raw string through (covers exotic types we don't special-case).
    return value


def _coerce_list(value: str, tp: Any, field_path: str) -> list:
    """Coerce a comma-separated or JSON list string into ``list[<elem>]``."""
    elem_args = get_args(tp)
    elem_tp = elem_args[0] if elem_args else str
    stripped = value.strip()
    if stripped.startswith("["):
        items = json.loads(stripped)
    else:
        items = [v for v in stripped.split(",") if v != ""]
    return [_coerce_scalar(str(item), elem_tp, field_path) for item in items]


def _coerce_tuple(value: str, tp: Any, field_path: str) -> tuple:
    """Coerce a comma-separated / JSON string into ``tuple[...]`` (fixed-length, per-elem types).

    Supports ``tuple[X, Y]`` (each coerced to its own type) and ``tuple[X, ...]`` (all to ``X``)."""
    args = get_args(tp)
    stripped = value.strip()
    items = json.loads(stripped) if stripped.startswith("[") else \
        [v for v in stripped.split(",") if v != ""]
    if len(args) == 2 and args[1] is Ellipsis:      # tuple[X, ...]
        return tuple(_coerce_scalar(str(it), args[0], field_path) for it in items)
    if args and len(items) != len(args):
        raise ValueError(f"--{field_path}: expected {len(args)} values, got {len(items)} ({value!r}).")
    elem_types = args if args else [str] * len(items)
    return tuple(_coerce_scalar(str(it), et, field_path) for it, et in zip(items, elem_types))


def _field_type_map(dc_type: type) -> dict[str, Any]:
    """Map ``field_name -> annotation`` for a dataclass, resolving string annotations."""
    hints = typing.get_type_hints(dc_type)
    out: dict[str, Any] = {}
    for f in dataclasses.fields(dc_type):
        out[f.name] = hints.get(f.name, f.type)
    return out


def _normalise_argv(args: list[str]) -> list[tuple[str, str]]:
    """Turn raw argv into ``(dotted_key, value)`` pairs, handling both ``--k v`` and ``--k=v``."""
    pairs: list[tuple[str, str]] = []
    i = 0
    n = len(args)
    while i < n:
        tok = args[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected CLI token {tok!r}; expected a --key.")
        body = tok[2:]
        if "=" in body:
            key, value = body.split("=", 1)
            pairs.append((key, value))
            i += 1
        else:
            if i + 1 >= n:
                raise ValueError(f"--{body} expects a value.")
            pairs.append((body, args[i + 1]))
            i += 2
    return pairs


def _resolve_policy_type(pairs: list[tuple[str, str]]) -> type:
    """Find ``--policy.type`` and return the selected policy dataclass (default: act)."""
    chosen = None
    for key, value in pairs:
        if key == "policy.type":
            chosen = value
    if chosen is None:
        # LeRobot/draccus require an explicit choice for a ChoiceRegistry with no default; ALLEX
        # always passes --policy.type act. Be explicit rather than silently defaulting.
        raise ValueError(
            "No policy selected. Pass `--policy.type act` (the only registered policy)."
        )
    if chosen not in POLICY_CHOICES:
        raise ValueError(
            f"Unknown policy type {chosen!r}. Known: {sorted(POLICY_CHOICES)}."
        )
    return POLICY_CHOICES[chosen]


def _set_nested(root: Any, instances: dict[str, Any], dotted_key: str, value: str) -> None:
    """Set ``dotted_key`` (e.g. ``dataset.image_aug_config.brightness``) on the config tree.

    ``instances`` maps a dotted prefix to the (already-created) dataclass instance for that prefix,
    so nested sub-dataclasses are mutated in place. The leaf field is coerced using its annotation.
    """
    parts = dotted_key.split(".")
    # Walk to the owning instance, creating sub-dataclass instances on the way if missing.
    owner = root
    prefix = ""
    for part in parts[:-1]:
        prefix = f"{prefix}.{part}" if prefix else part
        type_map = _field_type_map(type(owner))
        if part not in type_map:
            raise ValueError(f"Unknown config field --{dotted_key} (no field {part!r}).")
        sub = getattr(owner, part, None)
        if sub is None or not dataclasses.is_dataclass(sub):
            sub_tp, _ = _strip_optional(type_map[part])
            if not dataclasses.is_dataclass(sub_tp):
                raise ValueError(
                    f"--{dotted_key}: {part!r} is not a nested config; cannot descend into it."
                )
            sub = sub_tp()
            setattr(owner, part, sub)
        instances[prefix] = sub
        owner = sub

    leaf = parts[-1]
    type_map = _field_type_map(type(owner))
    if leaf not in type_map:
        raise ValueError(f"Unknown config field --{dotted_key} (no field {leaf!r} on {type(owner).__name__}).")
    coerced = _coerce_scalar(value, type_map[leaf], dotted_key)
    setattr(owner, leaf, coerced)


def parse(args: list[str] | None = None) -> TrainPipelineConfig:
    """Parse ``args`` (default ``sys.argv[1:]``) into a ``TrainPipelineConfig``.

    Matches the field VALUES draccus would produce for the ALLEX recipe. Builds the dataset and
    policy sub-configs first (they have required fields / a choice type), then applies every
    remaining ``--key value`` in order.
    """
    if args is None:
        args = sys.argv[1:]
    pairs = _normalise_argv(list(args))

    # 1) repo_id is required to construct DatasetConfig; pull it (and root) up front.
    overrides = {key: value for key, value in pairs}  # last-wins, like draccus
    repo_id = overrides.get("dataset.repo_id")
    if repo_id is None:
        raise ValueError("--dataset.repo_id is required.")
    dataset = DatasetConfigFactory(overrides)

    # 2) policy: select the subclass via --policy.type, then construct an empty instance.
    policy_cls = _resolve_policy_type(pairs)
    policy = policy_cls()

    cfg = TrainPipelineConfig(dataset=dataset, policy=policy)

    # 3) apply every override (incl. nested) in argv order. policy.type / dataset.repo_id are
    #    already consumed but re-applying them is a harmless no-op (repo_id) / skip (policy.type).
    instances: dict[str, Any] = {"dataset": dataset, "policy": policy}
    for key, value in pairs:
        if key == "policy.type":
            continue  # already used to select the subclass
        if key.startswith("policy.") and key.split(".", 1)[1] in DROPPED_POLICY_FIELDS:
            continue  # hub/PEFT/AMP field dropped from ACTConfig; inert for training
        _set_nested(cfg, instances, key, value)

    # Re-run policy __post_init__ validation now that fields are set (mirrors draccus building the
    # dataclass with all fields, which triggers __post_init__ once at the end).
    if dataclasses.is_dataclass(cfg.policy):
        cfg.policy.__post_init__()
    return cfg


def DatasetConfigFactory(overrides: dict[str, str]) -> Any:  # noqa: N802 — factory, not a class
    """Construct a ``DatasetConfig`` with its required ``repo_id`` + ``s3_uri`` (other fields set
    later). Datasets are always sourced from S3, so ``--dataset.s3_uri`` is mandatory."""
    from learning.configs.config import DatasetConfig

    for req in ("dataset.repo_id", "dataset.s3_uri"):
        if req not in overrides:
            raise ValueError(f"--{req} is required (datasets are always sourced from S3).")
    return DatasetConfig(
        repo_id=overrides["dataset.repo_id"], s3_uri=overrides["dataset.s3_uri"]
    )


def wrap() -> Any:
    """``@parser.wrap()`` — decorate ``main(cfg: TrainPipelineConfig)``.

    Mirrors ``lerobot.configs.parser.wrap``: if the wrapped fn is called with a config instance,
    use it; otherwise parse ``sys.argv[1:]`` into the annotated config type and pass it in. We only
    support ``TrainPipelineConfig`` as the annotation (the sole call site).
    """

    def wrapper_outer(fn):
        argspec = inspect.getfullargspec(fn)
        argtype = argspec.annotations.get(argspec.args[0]) if argspec.args else None

        @wraps(fn)
        def wrapper_inner(*args: Any, **kwargs: Any) -> Any:
            if len(args) > 0 and argtype is not None and type(args[0]) is argtype:
                cfg = args[0]
                rest = args[1:]
            else:
                cfg = parse(sys.argv[1:])
                rest = args
            return fn(cfg, *rest, **kwargs)

        return wrapper_inner

    return wrapper_outer
