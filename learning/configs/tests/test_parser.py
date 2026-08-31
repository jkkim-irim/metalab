"""Unit tests for the self-contained config parser (learning/configs/parser.py).

Exercises the real ``parse`` on the argv patterns ``learning/scripts/train.sh`` (+ EXTRA_FLAGS)
produce: scalar/​bool/​float/​list coercion, ``--key=value``, deeply-nested dot-args, last-wins
override (the load-bearing behaviour the base recipe + EXTRA_FLAGS overrides rely on), the
accept-and-ignore set, and the fail-loud paths. No lerobot/draccus needed.

Run in the node env from the repo root:  python -m pytest learning -q
"""
import pytest

from learning.configs.config import policy_type_name
from learning.configs.parser import parse


def _minimal(*extra: str) -> list[str]:
    """The flags every recipe carries (datasets are always sourced from S3), plus test overrides."""
    return ["--policy.type", "act", "--dataset.repo_id", "u/ds",
            "--dataset.s3_uri", "s3://bkt/ds", *extra]


def test_minimal_recipe_parses():
    cfg = parse(_minimal())
    assert cfg.dataset.repo_id == "u/ds"
    assert policy_type_name(cfg.policy) == "act"


def test_scalar_coercion_types():
    cfg = parse(_minimal(
        "--batch_size", "32", "--seed", "7", "--tolerance_s", "0.1",
        "--policy.optimizer_lr", "3e-5",
    ))
    assert cfg.batch_size == 32 and isinstance(cfg.batch_size, int)
    assert cfg.seed == 7
    assert cfg.tolerance_s == 0.1
    assert cfg.policy.optimizer_lr == 3e-5  # scientific-notation float


def test_last_wins_override():
    # train.sh emits the base recipe then EXTRA_FLAGS; a repeated flag must take the LAST value.
    cfg = parse(_minimal(
        "--batch_size", "8", "--policy.latent_dim", "32",
        "--batch_size", "32", "--policy.latent_dim", "64",
    ))
    assert cfg.batch_size == 32
    assert cfg.policy.latent_dim == 64


def test_key_equals_value_form():
    cfg = parse(_minimal("--batch_size=16"))
    assert cfg.batch_size == 16


def test_bool_coercion():
    cfg = parse(_minimal("--policy.use_vae", "false", "--wandb.enable", "true"))
    assert cfg.policy.use_vae is False
    assert cfg.wandb.enable is True
    for raw, expected in [("1", True), ("0", False), ("yes", True), ("no", False)]:
        assert parse(_minimal("--wandb.enable", raw)).wandb.enable is expected


def test_deeply_nested_dot_arg():
    # 3-level nested arg + tuple[float,float] coercion (ImageAugConfig ranges).
    cfg = parse(_minimal("--dataset.image_aug_config.brightness", "0.9,1.1"))
    assert cfg.dataset.image_aug_config.brightness == (0.9, 1.1)


def test_optional_null_becomes_none():
    cfg = parse(_minimal("--dataset.root", "null"))
    assert cfg.dataset.root is None


def test_list_coercion():
    cfg = parse(_minimal("--dataset.episodes", "1,2,3"))
    assert cfg.dataset.episodes == [1, 2, 3]


def test_dropped_policy_fields_accepted_and_ignored():
    # train.sh passes --policy.push_to_hub false; it is inert for the de-lerobot'd config.
    cfg = parse(_minimal("--policy.push_to_hub", "false"))
    assert not hasattr(cfg.policy, "push_to_hub")


def test_validate_fills_optimizer_preset():
    cfg = parse(_minimal("--policy.optimizer_lr", "3e-5"))
    cfg.validate()
    assert cfg.optimizer.type == "adamw"
    assert cfg.optimizer.lr == 3e-5
    assert cfg.optimizer.grad_clip_norm == 10.0
    assert cfg.scheduler is None
    assert cfg.job_name == "act"


@pytest.mark.parametrize("argv", [
    ["--dataset.repo_id", "u/ds"],            # no --policy.type
    ["--policy.type", "act"],                 # no --dataset.repo_id
    ["--policy.type", "ppo", "--dataset.repo_id", "u/ds"],  # unknown policy
    ["--policy.type", "act", "--dataset.repo_id", "u/ds"],  # no --dataset.s3_uri (always-S3)
    # a non-S3 dataset source is rejected (datasets are always sourced from S3)
    ["--policy.type", "act", "--dataset.repo_id", "u/ds", "--dataset.s3_uri", "/local/ds"],
])
def test_required_or_unknown_selection_fails(argv):
    with pytest.raises(ValueError):
        parse(argv)


@pytest.mark.parametrize("extra", [
    ["--not_a_field", "1"],          # unknown top-level field
    ["--policy.bogus", "1"],         # unknown policy field
    ["--dataset.nope", "1"],         # unknown nested field
])
def test_unknown_field_is_fail_loud(extra):
    with pytest.raises(ValueError):
        parse(_minimal(*extra))


@pytest.mark.parametrize("argv", [
    ["positional"],                  # not a --key
    ["--policy.type", "act", "--dataset.repo_id"],  # flag missing its value
])
def test_malformed_argv_is_fail_loud(argv):
    with pytest.raises(ValueError):
        parse(argv)
