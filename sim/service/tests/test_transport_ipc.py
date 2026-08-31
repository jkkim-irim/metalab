# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Transport tests — the REAL `transport.py` both service ends import, driven end-to-end.

* socket RPC (v1): server thread + client in-process, CPU-only — guards the LIBERO/ManiSkill path.
* CUDA-IPC hot path (v2): `serve_vec_env` in a SPAWNED server process, client in this process —
  a true cross-process CUDA-IPC exchange (same-process handle opens are not real IPC). Skipped
  without a CUDA GPU. The handler is a stub at the env boundary (the system boundary); the loop,
  buffers, framing, and error paths under test are production code.
"""
from __future__ import annotations

import os
import sys
import threading

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # sim/service

from transport import (  # noqa: E402
    K_ACTION,
    K_DONES,
    K_REW,
    K_TASK_SUCCESS,
    K_TIME_OUTS,
    RpcClient,
    RpcServer,
    obs_key,
    serve_vec_env,
)

_N, _A = 4, 3  # envs, actions


def _make_handler(device: str):
    """Deterministic stub handler with the sim-service method contract (the env is the stub;
    everything else — loop, buffers, framing — is the production transport)."""
    dev = torch.device(device)

    def handler(method, payload):
        if method == "attrs":
            return {"num_envs": _N, "num_actions": _A, "device": device,
                    "max_episode_length": 100, "cfg_dict": {"k": 1}}
        if method == "get_observations":
            return {"policy": torch.full((_N, 6), 7.0, device=dev),
                    "critic": torch.full((_N, 2), -3.0, device=dev)}
        if method == "reset":
            return {"obs": {"policy": torch.ones(_N, 6, device=dev),
                            "critic": torch.zeros(_N, 2, device=dev)},
                    "extras": {"setup": torch.arange(_N, device=dev)}}
        if method == "step":
            a = payload  # the shared action buffer
            return {"obs": {"policy": torch.cat([a * 2.0, -a], dim=1),
                            "critic": a[:, :2] + 1.0},
                    "rew": a.sum(dim=1),
                    "dones": torch.tensor([True, False, True, False], device=dev),
                    "extras": {"time_outs": torch.tensor([False, False, True, False], device=dev),
                               "task_success": torch.tensor([True, False, False, False], device=dev),
                               "term_reason": torch.arange(_N, dtype=torch.int8, device=dev),
                               "log": {"x": 1.0}}}
        if method == "get_ep_len":
            return torch.arange(_N, device=dev)
        raise ValueError(f"unknown method: {method!r}")

    return handler


# --- socket RPC (v1) ----------------------------------------------------------------------------------
def test_socket_rpc_roundtrip():
    srv = RpcServer("127.0.0.1", 0)
    t = threading.Thread(target=lambda: (srv.accept(), srv.serve(_make_handler("cpu"))), daemon=True)
    t.start()
    client = RpcClient("127.0.0.1", srv.port)
    client._sock.settimeout(60)

    assert client.call("attrs")["num_envs"] == _N
    r = client.call("step", torch.ones(_N, _A))
    assert torch.equal(r["rew"], torch.full((_N,), 3.0))
    assert torch.equal(r["extras"]["time_outs"], torch.tensor([False, False, True, False]))
    with pytest.raises(RuntimeError, match="unknown method"):
        client.call("bogus")
    assert client.call("get_ep_len").tolist() == list(range(_N))  # still serving after the error
    client.close()
    t.join(timeout=30)
    assert not t.is_alive()


# --- CUDA-IPC hot path (v2) ---------------------------------------------------------------------------
def _serve_ipc(q):
    """Server-process entry (spawned): serve the stub handler over the CUDA-IPC loop."""
    srv = RpcServer("127.0.0.1", 0, map_location="cuda:0")
    q.put(srv.port)
    srv.accept()
    serve_vec_env(srv, _make_handler("cuda:0"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-IPC needs a CUDA GPU")
def test_cuda_ipc_serve_vec_env():
    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_serve_ipc, args=(q,), daemon=True)
    proc.start()
    try:
        client = RpcClient("127.0.0.1", q.get(timeout=120), map_location="cuda:0")
        client._sock.settimeout(300)  # first CUDA init in the child can be slow
        buffers = client.recv_shared()
        assert set(buffers) == {K_ACTION, K_REW, K_DONES, K_TIME_OUTS, K_TASK_SUCCESS,
                                obs_key("policy"), obs_key("critic")}

        def ctl(method, payload=None):
            client.send_ctl((method, payload))
            r = client.recv_ctl()
            if not r.get("ok"):
                raise RuntimeError(f"sim service error: {r.get('error')}\n{r.get('tb', '')}")
            return r.get("result")

        # cold path rides the control channel
        attrs = ctl("attrs")
        assert attrs["num_envs"] == _N and attrs["cfg_dict"] == {"k": 1}
        assert ctl("get_ep_len").tolist() == list(range(_N))

        # get_observations / reset: obs land in the shared buffers (zero-copy), extras over ctl
        assert ctl("get_observations") == {}
        assert torch.equal(buffers[obs_key("policy")], torch.full((_N, 6), 7.0, device="cuda:0"))
        r = ctl("reset")
        assert torch.equal(r["extras"]["setup"], torch.arange(_N, device="cuda:0"))
        assert torch.equal(buffers[obs_key("policy")], torch.ones(_N, 6, device="cuda:0"))

        # step: action goes out through the shared buffer, results come back through it
        actions = torch.arange(_N * _A, dtype=torch.float32, device="cuda:0").view(_N, _A)
        buffers[K_ACTION].copy_(actions)
        torch.cuda.synchronize()
        r = ctl("step")
        assert torch.equal(buffers[obs_key("policy")], torch.cat([actions * 2.0, -actions], dim=1))
        assert torch.equal(buffers[obs_key("critic")], actions[:, :2] + 1.0)
        assert torch.equal(buffers[K_REW], actions.sum(dim=1))
        assert buffers[K_DONES].tolist() == [True, False, True, False]
        assert buffers[K_TIME_OUTS].tolist() == [False, False, True, False]
        assert buffers[K_TASK_SUCCESS].tolist() == [True, False, False, False]
        # buffered keys are STRIPPED from the ctl extras; the rest passes through
        assert set(r["extras"]) == {"term_reason", "log"}
        assert r["extras"]["log"] == {"x": 1.0}
        assert torch.equal(r["extras"]["term_reason"],
                           torch.arange(_N, dtype=torch.int8, device="cuda:0"))

        # a server-side error reports back and the loop keeps serving
        client.send_ctl(("bogus", None))
        assert client.recv_ctl()["ok"] is False
        assert ctl("attrs")["num_envs"] == _N

        del buffers  # release the client's IPC handles before closing
        client.close()
        proc.join(timeout=60)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
