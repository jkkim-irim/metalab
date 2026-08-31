"""Unit tests for the direct MCAP -> WIR_v1 converter's embodiment mapping + resampling.

Exercises the PRODUCTION helpers that define the ALLEX bilateral layout
(``_pos_topics`` / ``_trq_topics`` / ``_cmd_topics`` / ``_block_spec`` -> the 44-D q / tau / action
vectors) and the zero-order-hold resampler (``_Series``) — the pieces that would silently corrupt
training data if a dim, joint order, or topic name drifted. Pure logic, no MCAP fixture / ffmpeg.
"""
import numpy as np

from learning.data.conversion.mcap_teleop_to_wir import (
    ACTION_DIM,
    ARM_DIM,
    CMD_SPEC,
    HAND_DIM,
    HAND_FINGER_DIM,
    NUM_FINGERS,
    POS_SPEC,
    Q_DIM,
    STATE_DIM,
    TRQ_SPEC,
    _Series,
    _block_spec,
    _cmd_topics,
    _pos_topics,
)


def test_embodiment_dims_are_the_wir_v1_allex_layout():
    # bilateral: r/l arm (7 each) + r/l hand (5 fingers x 3 = 15 each) = 44
    assert (ARM_DIM, HAND_FINGER_DIM, NUM_FINGERS, HAND_DIM) == (7, 3, 5, 15)
    assert Q_DIM == 44 and ACTION_DIM == 44
    assert STATE_DIM == 132  # observation.state = [q(44) | dq(44) | tau(44)]


def test_each_block_spec_sums_to_44_dims():
    for spec in (POS_SPEC, TRQ_SPEC, CMD_SPEC):
        assert sum(take for _, take in spec) == 44
        assert len(spec) == 12  # (1 arm + 5 fingers) x 2 sides


def test_pos_spec_topic_order_and_takes():
    spec = _block_spec(_pos_topics)
    # right side first (arm then 5 fingers), then left side likewise
    assert spec[0] == ("/allex/p001/robot_outbound_data/right_arm/joint_positions_deg", ARM_DIM)
    assert [t.split("/")[-2] for t, _ in spec] == [
        "right_arm", "right_thumb", "right_index", "right_middle", "right_ring", "right_little",
        "left_arm", "left_thumb", "left_index", "left_middle", "left_ring", "left_little",
    ]
    assert [take for _, take in spec] == [7, 3, 3, 3, 3, 3, 7, 3, 3, 3, 3, 3]


def test_topic_families_are_correct_for_state_torque_and_command():
    # state pos = outbound joint_positions_deg (deg); torque = outbound joint_torque;
    # action = inbound joint_command (rad). A mixup here would train on the wrong signal/units.
    assert all("robot_outbound_data" in t and t.endswith("/joint_positions_deg") for t, _ in POS_SPEC)
    assert all("robot_outbound_data" in t and t.endswith("/joint_torque") for t, _ in TRQ_SPEC)
    assert all("robot_inbound" in t and t.endswith("/joint_command") for t, _ in CMD_SPEC)


def test_series_zero_order_hold_holds_latest_and_clamps():
    s = _Series()
    # added out of order; finalize() must sort by time
    s.add(30, np.array([3.0], dtype=np.float32))
    s.add(10, np.array([1.0], dtype=np.float32))
    s.add(20, np.array([2.0], dtype=np.float32))
    s.finalize()
    assert s.first_t == 10
    out = s.zoh(np.array([5, 10, 15, 20, 25, 35], dtype=np.int64))
    # t < first -> clamp to first sample; otherwise the latest sample at-or-before t
    assert [float(v[0]) for v in out] == [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]


def test_series_zoh_is_multidim_and_row_aligned():
    s = _Series()
    s.add(0, np.array([0.0, 10.0], dtype=np.float32))
    s.add(100, np.array([1.0, 11.0], dtype=np.float32))
    s.finalize()
    out = s.zoh(np.array([50, 100, 150], dtype=np.int64))
    assert out.shape == (3, 2)
    np.testing.assert_array_equal(out, np.array([[0, 10], [1, 11], [1, 11]], dtype=np.float32))
