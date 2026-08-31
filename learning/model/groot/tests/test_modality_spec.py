"""Unit tests for the GR00T per-embodiment modality spec + slice/reassembly (torch/gr00t-free).

Exercises the PRODUCTION functions ``groot_policy`` calls — ``modality_spec.get_modality_spec`` and
the ``slicing`` helpers — not a re-implementation, so a green run here pins the real slicing math.
Two things are asserted: (1) the ALLEX 132-D/44-D layout is byte-for-byte the legacy hardcoded one
(the guarantee the existing path is unchanged), and (2) the mikasa 7-D single-group layout slices +
reassembles a fake ``(B,K,7)`` action and ``(B,7)`` state correctly (and pads to 132, not 7).

Runs anywhere with numpy (no torch / no gr00t):  python -m pytest learning/model/groot/tests -q
"""
import numpy as np
import pytest

from learning.model.groot.configuration import GrootConfig
from learning.model.groot.modality_spec import (
    BASE_MODEL_MAX_DIM,
    MODALITY_SPECS,
    get_modality_spec,
)
from learning.model.groot.slicing import (
    concat_action_groups,
    split_action_chunk,
    split_state_batch,
    split_state_row,
)

# The legacy hardcoded ALLEX layout the existing path used (pinned here as the oracle).
_LEGACY_STATE_GROUPS = {"q": (0, 44), "dq": (44, 88), "tau": (88, 132)}
_LEGACY_ACTION_GROUPS = {"r_arm_cmd": (0, 7), "l_arm_cmd": (7, 14), "r_hand_cmd": (14, 29), "l_hand_cmd": (29, 44)}
_LEGACY_ACTION_KEYS = ("r_arm_cmd", "l_arm_cmd", "r_hand_cmd", "l_hand_cmd")


# ---- ALLEX default: byte-for-byte unchanged ----------------------------------------------------
def test_allex_spec_matches_legacy_layout():
    spec = get_modality_spec("allex")
    assert spec.state_groups == _LEGACY_STATE_GROUPS
    assert spec.action_groups == _LEGACY_ACTION_GROUPS
    assert spec.action_keys == _LEGACY_ACTION_KEYS
    assert spec.state_keys == ("q", "dq", "tau")
    assert spec.max_state_dim == 132 and spec.max_action_dim == 132


def test_allex_is_the_default_modality():
    assert GrootConfig().modality == "allex"


def test_default_config_reproduces_legacy_fields():
    """A default GrootConfig() must still carry the exact legacy dims/state_keys (unchanged path)."""
    cfg = GrootConfig()
    assert cfg.state_keys == ["q", "dq", "tau"]
    assert cfg.max_state_dim == 132
    assert cfg.max_action_dim == 132
    assert cfg.max_action_horizon == 40
    assert cfg.chunk_size == 16


def test_allex_action_reassembly_preserves_group_order():
    """concat must stitch groups in action_keys order -> the flat 44-D vector (r_arm|l_arm|r_hand|l_hand)."""
    spec = get_modality_spec("allex")
    B, K = 2, 5
    per_key = {}
    flat = np.zeros((B, K, 44), dtype=np.float64)
    for name, (a, b) in _LEGACY_ACTION_GROUPS.items():
        block = np.full((B, K, b - a), fill_value=float(a), dtype=np.float64)  # marker = start index
        per_key[f"action.{name}"] = block
        flat[..., a:b] = block
    out = concat_action_groups(spec, per_key)
    assert out.shape == (B, K, 44)
    np.testing.assert_array_equal(out, flat)


def test_allex_state_row_slices_q_dq_tau():
    spec = get_modality_spec("allex")
    st = np.arange(132, dtype=np.float64)
    groups = split_state_row(spec, st)
    assert set(groups) == {"q", "dq", "tau"}
    assert groups["q"].shape == (1, 44) and groups["dq"].shape == (1, 44) and groups["tau"].shape == (1, 44)
    np.testing.assert_array_equal(groups["tau"][0], np.arange(88, 132))


# ---- mikasa: 7-D single-group embodiment -------------------------------------------------------
def test_mikasa_spec_single_group_pads_to_base_dim():
    spec = get_modality_spec("mikasa")
    assert spec.state_groups == {"state": (0, 7)}
    assert spec.action_groups == {"eef_cmd": (0, 7)}
    assert spec.action_keys == ("eef_cmd",)
    assert spec.state_keys == ("state",)
    # MUST pad up to the pretrained head width (132), NOT shrink to 7.
    assert spec.max_state_dim == BASE_MODEL_MAX_DIM == 132
    assert spec.max_action_dim == BASE_MODEL_MAX_DIM == 132


def test_mikasa_config_derives_layout_from_modality():
    cfg = GrootConfig(modality="mikasa")
    assert cfg.state_keys == ["state"]
    assert cfg.max_state_dim == 132 and cfg.max_action_dim == 132


def test_mikasa_state_slicing_row_and_batch():
    spec = get_modality_spec("mikasa")
    # training per-sample row (7,) -> {"state": (1,7)}
    st_row = np.arange(7, dtype=np.float64)
    row = split_state_row(spec, st_row)
    assert set(row) == {"state"} and row["state"].shape == (1, 7)
    np.testing.assert_array_equal(row["state"][0], st_row)
    # inference batch (B,7) -> {"state.state": (B,1,7)}
    B = 3
    st_batch = np.arange(B * 7, dtype=np.float64).reshape(B, 7)
    batch = split_state_batch(spec, st_batch)
    assert set(batch) == {"state.state"} and batch["state.state"].shape == (B, 1, 7)
    np.testing.assert_array_equal(batch["state.state"][:, 0, :], st_batch)


def test_mikasa_action_split_and_roundtrip():
    spec = get_modality_spec("mikasa")
    B, K = 4, 8
    # training: (K,7) chunk -> {"eef_cmd": (K,7)}
    chunk = np.random.default_rng(0).standard_normal((K, 7))
    split = split_action_chunk(spec, chunk)
    assert set(split) == {"eef_cmd"} and split["eef_cmd"].shape == (K, 7)
    np.testing.assert_array_equal(split["eef_cmd"], chunk)
    # inference reassembly: single group -> flat (B,K,7) is exactly that group
    unapplied = {"action.eef_cmd": np.random.default_rng(1).standard_normal((B, K, 7))}
    out = concat_action_groups(spec, unapplied)
    assert out.shape == (B, K, 7)
    np.testing.assert_array_equal(out, unapplied["action.eef_cmd"])


# ---- libero: 8-D state / 7-D action single-group embodiment ------------------------------------
def test_libero_spec_single_group_pads_to_base_dim():
    spec = get_modality_spec("libero")
    assert spec.state_groups == {"state": (0, 8)}
    assert spec.action_groups == {"eef_cmd": (0, 7)}
    assert spec.action_keys == ("eef_cmd",)
    assert spec.state_keys == ("state",)
    # 8-D state / 7-D action MUST still pad up to the pretrained head width (132), NOT shrink.
    assert spec.max_state_dim == BASE_MODEL_MAX_DIM == 132
    assert spec.max_action_dim == BASE_MODEL_MAX_DIM == 132


def test_libero_config_derives_layout_from_modality():
    cfg = GrootConfig(modality="libero")
    assert cfg.state_keys == ["state"]
    assert cfg.max_state_dim == 132 and cfg.max_action_dim == 132


def test_libero_state_slicing_row_and_batch():
    spec = get_modality_spec("libero")
    # training per-sample row (8,) -> {"state": (1,8)}
    st_row = np.arange(8, dtype=np.float64)
    row = split_state_row(spec, st_row)
    assert set(row) == {"state"} and row["state"].shape == (1, 8)
    np.testing.assert_array_equal(row["state"][0], st_row)
    # inference batch (B,8) -> {"state.state": (B,1,8)}
    B = 3
    st_batch = np.arange(B * 8, dtype=np.float64).reshape(B, 8)
    batch = split_state_batch(spec, st_batch)
    assert set(batch) == {"state.state"} and batch["state.state"].shape == (B, 1, 8)
    np.testing.assert_array_equal(batch["state.state"][:, 0, :], st_batch)


def test_libero_action_split_and_roundtrip():
    spec = get_modality_spec("libero")
    B, K = 4, 8
    # training: (K,7) chunk -> {"eef_cmd": (K,7)}
    chunk = np.random.default_rng(2).standard_normal((K, 7))
    split = split_action_chunk(spec, chunk)
    assert set(split) == {"eef_cmd"} and split["eef_cmd"].shape == (K, 7)
    np.testing.assert_array_equal(split["eef_cmd"], chunk)
    # inference reassembly: single group -> flat (B,K,7) is exactly that group
    unapplied = {"action.eef_cmd": np.random.default_rng(3).standard_normal((B, K, 7))}
    out = concat_action_groups(spec, unapplied)
    assert out.shape == (B, K, 7)
    np.testing.assert_array_equal(out, unapplied["action.eef_cmd"])


# ---- optional validation-metric fields (EE-delta embodiments) ----------------------------------
def test_libero_spec_carries_validation_metric_fields():
    # LibERO declares embodiment-aware val metrics: pos/rot/gripper group L1, physical-unit EE
    # (mm/deg) via the OSC_POSE output_max scales, and the gripper action index.
    spec = get_modality_spec("libero")
    assert spec.val_metric_groups == {"all": (0, 7), "pos": (0, 3), "rot": (3, 6), "gripper": (6, 7)}
    assert spec.gripper_dim == 6
    by_key = {e["key"]: e for e in spec.cartesian_metrics}
    assert set(by_key) == {"mm_pos", "deg_rot"}
    assert by_key["mm_pos"]["slice"] == (0, 3) and by_key["mm_pos"]["scale"] == 50.0
    assert by_key["mm_pos"]["unit"] == "mm"
    assert by_key["deg_rot"]["slice"] == (3, 6)
    assert by_key["deg_rot"]["scale"] == pytest.approx(28.6479)
    assert by_key["deg_rot"]["unit"] == "deg"


def test_allex_and_mikasa_have_no_validation_metric_fields():
    # These stay None so ALLEX keeps its URDF-FK path and mikasa is untouched (fillable later).
    for name in ("allex", "mikasa"):
        spec = get_modality_spec(name)
        assert spec.val_metric_groups is None
        assert spec.cartesian_metrics is None
        assert spec.gripper_dim is None


# ---- fail-loud + registry ----------------------------------------------------------------------
def test_unknown_modality_fails_loud():
    with pytest.raises(ValueError, match="Unknown GR00T modality"):
        get_modality_spec("nope")
    with pytest.raises(ValueError, match="Unknown GR00T modality"):
        GrootConfig(modality="nope")


def test_registry_has_allex_mikasa_and_libero():
    assert {"allex", "mikasa", "libero"} <= set(MODALITY_SPECS)


def test_spec_rejects_inconsistent_keys():
    from learning.model.groot.modality_spec import ModalitySpec

    with pytest.raises(ValueError, match="action_keys"):
        ModalitySpec(
            name="bad", state_groups={"s": (0, 7)}, action_groups={"a": (0, 7)},
            action_keys=("wrong",), state_keys=("s",),
        )
