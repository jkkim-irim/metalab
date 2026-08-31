"""Unit tests for the canonical ALLEX state/action modality map
(``learning/data/allex_modality.py``) and the consumers refactored onto it. Pins the layout so it
cannot drift, and checks the refactor kept the derived consumer dicts value-identical.
"""
import pytest

from learning.data import allex_modality as am


def test_state_and_action_spans():
    assert am.STATE_DIM == 132
    assert am.STATE_GROUPS == {"q": (0, 44), "dq": (44, 88), "tau": (88, 132)}
    assert am.STATE_Q == (0, 44)
    assert am.ACTION_DIM == 44
    assert am.ACTION_GROUPS == {
        "r_arm": (0, 7), "l_arm": (7, 14), "r_hand": (14, 29), "l_hand": (29, 44),
    }
    assert am.ARMS == (0, 14)
    assert am.HANDS == (14, 44)
    assert am.ALL == (0, 44)


def test_metric_groups():
    assert am.ACTION_METRIC_GROUPS == {
        "all": (0, 44),
        "r_arm": (0, 7), "l_arm": (7, 14), "r_hand": (14, 29), "l_hand": (29, 44),
        "arm": (0, 14), "finger": (14, 44),
    }


def test_span_and_slice_helpers():
    assert am.span("q") == (0, 44)
    assert am.span("r_hand") == (14, 29)
    assert am.slice_(am.ARMS) == slice(0, 14)
    with pytest.raises(KeyError):
        am.span("nope")


def test_state_prefix_dim():
    assert am.state_prefix_dim(["q"]) == 44
    assert am.state_prefix_dim(["q", "dq"]) == 88
    assert am.state_prefix_dim(["q", "dq", "tau"]) == 132
    assert am.state_prefix_dim(["dq", "q"]) == 88          # order-independent


def test_state_prefix_dim_fail_loud():
    with pytest.raises(ValueError):
        am.state_prefix_dim(["q", "tau"])   # skips dq -> non-contiguous
    with pytest.raises(ValueError):
        am.state_prefix_dim(["dq"])         # doesn't start at q (dim 0)
    with pytest.raises(ValueError):
        am.state_prefix_dim([])             # empty
    with pytest.raises(KeyError):
        am.state_prefix_dim(["bogus"])      # unknown name


def test_validate_span_fail_loud():
    am.validate_span((0, 44), 44)      # exact fit
    am.validate_span((0, 44), 132)     # fits within a larger dim
    with pytest.raises(ValueError):
        am.validate_span((0, 45), 44)  # end past dim
    with pytest.raises(ValueError):
        am.validate_span((10, 5), 44)  # start >= end


def test_groot_modality_consumer_value_identical():
    from learning.data.conversion.groot_modality import ACTION_GROUPS, STATE_GROUPS
    assert STATE_GROUPS == {"q": (0, 44), "dq": (44, 88), "tau": (88, 132)}
    assert ACTION_GROUPS == {
        "r_arm_cmd": (0, 7), "l_arm_cmd": (7, 14), "r_hand_cmd": (14, 29), "l_hand_cmd": (29, 44),
    }


def test_validation_consumer_value_identical():
    from learning.metrics.validation import ALLEX_ACTION_GROUPS
    assert ALLEX_ACTION_GROUPS == {
        "all": (0, 44),
        "r_arm": (0, 7), "l_arm": (7, 14), "r_hand": (14, 29), "l_hand": (29, 44),
        "arm": (0, 14), "finger": (14, 44),
    }
