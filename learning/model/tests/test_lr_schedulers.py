"""Unit tests for the shared pure-torch LR schedulers (learning/model/lr_schedulers.py).

These test the builders directly; test_act_policy exercises the same functions through the ACT
build() dispatch (and the GR00T path uses cosine_with_warmup via groot_policy.build).
"""
import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

from learning.model.lr_schedulers import cosine_with_warmup, wsd_with_warmup


def _opt(lr=1.0):
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def test_cosine_warms_up_then_decays_to_zero():
    opt = _opt(1.0)
    sched = cosine_with_warmup(opt, warmup_steps=10, total_steps=100)
    assert isinstance(sched, LambdaLR)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)   # step 0
    sched.step()                                                        # step 1/10 warmup
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1, rel=1e-6)
    for _ in range(9):                                                  # -> step 10: cosine start (peak)
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, rel=1e-6)
    for _ in range(90):                                                 # -> step 100: end of cosine
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_wsd_holds_peak_then_anneals():
    # warmup=10, decay_steps=20 -> stable body [10, 80), cosine anneal [80, 100].
    opt = _opt(1.0)
    sched = wsd_with_warmup(opt, warmup_steps=10, total_steps=100, decay_steps=20)
    for _ in range(10):                                                 # -> step 10: end warmup
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, rel=1e-6)    # peak
    for _ in range(40):                                                 # -> step 50: still constant body
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0, rel=1e-6)    # decoupled from run length
    for _ in range(50):                                                 # -> step 100: end anneal
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)
