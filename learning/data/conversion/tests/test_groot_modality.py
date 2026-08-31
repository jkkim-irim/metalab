"""Unit tests for the GR00T modality.json generator (torch-free; runs anywhere).

Pins the canonical ALLEX layout = allex_groot/scripts/preprocess/convert_groot_drill_raw_to_lerobot_v2.py.
"""
import json

from learning.data.conversion.groot_modality import (
    ACTION_GROUPS,
    STATE_GROUPS,
    build_allex_modality,
    write_modality_json,
)


def test_state_is_132d_q_dq_tau():
    m = build_allex_modality(["camera_1", "camera_2"])
    assert m["state"] == {
        "q": {"start": 0, "end": 44},
        "dq": {"start": 44, "end": 88},
        "tau": {"start": 88, "end": 132},
    }


def test_target_and_residual_groups():
    m = build_allex_modality(["camera_1"])
    assert m["target"] == {
        "pos_target": {"start": 0, "end": 44},
        "vel_target": {"start": 0, "end": 44},
        "trq_target": {"start": 0, "end": 44},
    }
    assert m["residual"] == {"tau_residual": {"start": 0, "end": 44}}


def test_action_groups_cover_44d_contiguously():
    m = build_allex_modality(["camera_1"])
    assert set(m["action"]) == {"r_arm_cmd", "l_arm_cmd", "r_hand_cmd", "l_hand_cmd"}
    spans = sorted((v["start"], v["end"]) for v in m["action"].values())
    assert spans[0][0] == 0 and spans[-1][1] == 44
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))


def test_video_and_annotation_mapping():
    m = build_allex_modality(["camera_1", "camera_2"])
    assert m["video"] == {
        "camera_1": {"original_key": "observation.images.camera_1"},
        "camera_2": {"original_key": "observation.images.camera_2"},
    }
    assert m["annotation"] == {"human.action.task_description": {}}


def test_layout_matches_repo_converter_contract():
    assert STATE_GROUPS == {"q": (0, 44), "dq": (44, 88), "tau": (88, 132)}
    assert ACTION_GROUPS == {
        "r_arm_cmd": (0, 7), "l_arm_cmd": (7, 14), "r_hand_cmd": (14, 29), "l_hand_cmd": (29, 44),
    }


def test_write_modality_json(tmp_path):
    out = write_modality_json(tmp_path, ["camera_1"])
    assert out.endswith("modality.json")
    loaded = json.loads((tmp_path / "modality.json").read_text())
    assert loaded["state"]["tau"]["end"] == 132
    assert loaded["video"]["camera_1"]["original_key"] == "observation.images.camera_1"
