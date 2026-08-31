"""Unit tests for the wir_v1 dataset contract (learning/data/contract.py).

Drives validate_wir_contract with synthetic metadata (a duck-typed stand-in for
LeRobotDatasetMetadata) — no dataset, no deps. Asserts a conforming dataset passes and each
structural violation fails loud.

Run in the node env from the repo root:  python -m pytest learning -q
"""
import pytest

from learning.data.contract import WIR_DATA_VERSION, WirContractError, validate_wir_contract


class _Meta:
    def __init__(self, info, features, fps, stats, repo_id="r/x"):
        self.info = info
        self.features = features
        self.fps = fps
        self.stats = stats
        self.repo_id = repo_id


def _valid_meta() -> _Meta:
    features = {
        "timestamp": {"dtype": "float32", "shape": (1,)},
        "index": {"dtype": "int64", "shape": (1,)},
        "episode_index": {"dtype": "int64", "shape": (1,)},
        "frame_index": {"dtype": "int64", "shape": (1,)},
        "observation.state": {"dtype": "float32", "shape": (132,)},
        "action": {"dtype": "float32", "shape": (44,)},
        "observation.images.camera_1": {"dtype": "video", "shape": (3, 224, 224)},
        "observation.images.camera_2": {"dtype": "video", "shape": (3, 224, 224)},
    }
    info = {"codebase_version": "v3.0", "features": features, "fps": 30}
    stats = {"observation.state": {"mean": [0.0] * 132}, "action": {"mean": [0.0] * 44}}
    return _Meta(info=info, features=features, fps=30, stats=stats)


def test_conforming_dataset_passes():
    validate_wir_contract(_valid_meta())  # no raise


def test_marker_matches_passes():
    m = _valid_meta()
    m.info["wir_data_version"] = WIR_DATA_VERSION
    validate_wir_contract(m)  # no raise


def test_missing_action_fails():
    m = _valid_meta()
    del m.features["action"]
    with pytest.raises(WirContractError, match="action"):
        validate_wir_contract(m)


def test_state_must_be_float_vector():
    m = _valid_meta()
    m.features["observation.state"] = {"dtype": "int64", "shape": (132,)}
    with pytest.raises(WirContractError, match="not a float"):
        validate_wir_contract(m)

    m = _valid_meta()
    m.features["observation.state"] = {"dtype": "float32", "shape": (132, 2)}
    with pytest.raises(WirContractError, match="1-D vector"):
        validate_wir_contract(m)


def test_missing_stats_fails():
    m = _valid_meta()
    del m.stats["action"]
    with pytest.raises(WirContractError, match="normalization stats"):
        validate_wir_contract(m)


def test_requires_at_least_one_rgb_camera():
    m = _valid_meta()
    for k in [k for k in list(m.features) if k.startswith("observation.images.")]:
        del m.features[k]
    with pytest.raises(WirContractError, match="no camera"):
        validate_wir_contract(m)


def test_camera_must_be_chw_rgb():
    m = _valid_meta()
    m.features["observation.images.camera_1"] = {"dtype": "video", "shape": (224, 224, 3)}
    with pytest.raises(WirContractError, match=r"is not \(3, H, W\)"):
        validate_wir_contract(m)


def test_bad_codebase_version_fails():
    m = _valid_meta()
    m.info["codebase_version"] = "v2.1"
    with pytest.raises(WirContractError, match="codebase_version"):
        validate_wir_contract(m)


def test_wir_marker_mismatch_fails():
    m = _valid_meta()
    m.info["wir_data_version"] = "wir_v2"
    with pytest.raises(WirContractError, match="wir_data_version"):
        validate_wir_contract(m)


def test_bad_fps_fails():
    m = _valid_meta()
    m.fps = 0
    with pytest.raises(WirContractError, match="fps"):
        validate_wir_contract(m)


def test_error_lists_all_violations():
    m = _valid_meta()
    del m.features["action"]
    m.fps = 0
    m.info["codebase_version"] = "v2.0"
    with pytest.raises(WirContractError) as ei:
        validate_wir_contract(m)
    msg = str(ei.value)
    assert "action" in msg and "fps" in msg and "codebase_version" in msg
