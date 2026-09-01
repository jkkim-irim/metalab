# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Unified sim-service transport — length-prefixed, torch-serialized RPC over a localhost socket, with a
CUDA-IPC hot path. THE single wire protocol of the sim service, imported by BOTH ends — the client
(``learning/rl/client``) and each engine spoke's ``server.py`` — so the two cannot drift. Package
import only: ``from sim.metalab.transport import ...``.

Two transports share this file (and one socket):

* **Socket RPC** (v1) — ``RpcServer.serve`` / ``RpcClient.call``: `torch.save`/`torch.load` the whole
  payload — `TensorDict` obs, nested `extras`, tensors, and scalars — in one shot, and serializing a
  CUDA tensor does the GPU→CPU→(bytes) round-trip automatically (map_location moves it back on
  receive). Simple, cross-process, cross-venv — the LIBERO / ManiSkill eval services + the
  cold-path control channel uses this.

* **CUDA-IPC hot path** (v2) — ``serve_vec_env`` / the shared-buffer methods: for the GPU RL
  sim-service the per-step serialize is pure overhead that grows with num_envs (measured ~6 ms/step
  ≈ +7.8 % wall-clock/iter at 1024 envs on an L40S; +149 % at 4096 envs on the MetaLab A/B).
  ``RpcServer.share()`` allocates fixed-shape GPU buffers and ships their CUDA-IPC handles ONCE at
  connect; per step the obs/action/rew/dones payload stays on the GPU (server writes / client reads
  the SAME physical memory) and only a tiny control message crosses the socket. Everything else in
  ``extras`` (term_reason, wbt_metrics, setup, episode logs, ...) still rides the control channel —
  small next to the obs. GPU-only, same-GPU (the deploy scripts pin trainer + server to one physical
  GPU via ``CUDA_VISIBLE_DEVICES`` inheritance); ``share()`` fails loudly if IPC is impossible.
  The client end is ``learning/rl/client.SimServiceVecEnv``.
"""
from __future__ import annotations

import io
import pickle
import socket
import struct
import traceback
from typing import Sequence

import torch
from torch.multiprocessing.reductions import reduce_tensor

_HDR = struct.Struct("!Q")  # 8-byte big-endian payload length prefix
_CHUNK = 1 << 20

# --- shared-buffer key convention (both ends agree on these names) -----------------------------------
OBS_PREFIX = "obs::"             # one buffer per obs group, keyed "obs::<group>"
K_ACTION = "action"              # client -> server (N, num_actions)
K_REW = "rew"                    # server -> client (N,)
K_DONES = "dones"                # server -> client (N,) bool
K_TIME_OUTS = "time_outs"        # server -> client (N,) bool  (extras["time_outs"])
K_TASK_SUCCESS = "task_success"  # server -> client (N,) bool  (extras["task_success"])


def obs_key(group: str) -> str:
    return OBS_PREFIX + group


def obs_group(key: str) -> str:
    return key[len(OBS_PREFIX):]


def _send(sock: socket.socket, obj) -> None:
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getbuffer()
    sock.sendall(_HDR.pack(len(data)))
    sock.sendall(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        chunk = sock.recv(min(n - got, _CHUNK))
        if not chunk:
            raise ConnectionError("sim-service socket closed mid-message")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _recv(sock: socket.socket, map_location=None):
    (n,) = _HDR.unpack(_recv_exact(sock, _HDR.size))
    return torch.load(io.BytesIO(_recv_exact(sock, n)), map_location=map_location, weights_only=False)


def _send_pickle(sock: socket.socket, obj) -> None:
    """Plain pickle — used only for the one-shot CUDA-IPC handle bundle (`(rebuild_fn, args)` tuples)."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HDR.pack(len(data)))
    sock.sendall(data)


def _recv_pickle(sock: socket.socket):
    (n,) = _HDR.unpack(_recv_exact(sock, _HDR.size))
    return pickle.loads(_recv_exact(sock, n))


class RpcServer:
    """Serves one client: recv ``(method, payload)`` → ``handler`` → send response.

    ``serve`` is the plain socket loop (v1). For the CUDA-IPC hot path, ``share()`` allocates the
    shared GPU buffers + ships their handles, and ``serve_vec_env`` (below) runs the loop with the
    hot-path payload in those buffers instead of serialized (``recv_ctl``/``send_ctl`` are the
    control-channel halves it is built from).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0, map_location=None):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.host, self.port = self._srv.getsockname()
        self._map = map_location
        self._conn: socket.socket | None = None
        self._buffers: dict[str, torch.Tensor] = {}

    def accept(self) -> None:
        self._conn, _ = self._srv.accept()
        self._conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def serve(self, handler) -> None:
        assert self._conn is not None, "call accept() first"
        while True:
            method, payload = _recv(self._conn, map_location=self._map)
            if method == "close":
                _send(self._conn, {"ok": True})
                return
            try:
                _send(self._conn, {"ok": True, "result": handler(method, payload)})
            except Exception as e:  # boundary — report the error back, keep serving
                _send(self._conn, {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})

    def share(self, layout: dict[str, tuple[Sequence[int], torch.dtype]], device: str) -> dict[str, torch.Tensor]:
        """Allocate each named buffer on ``device`` and ship its CUDA-IPC handle to the client.

        Returns the dict of server-side tensors (write into these). The client's ``recv_shared`` rebuilds
        tensors pointing at the SAME physical GPU memory. Fail-loud if the allocator can't produce a handle.
        """
        assert self._conn is not None, "call accept() first"
        bundle = {}
        for name, (shape, dtype) in layout.items():
            t = torch.zeros(tuple(shape), dtype=dtype, device=device)
            rebuild_fn, args = reduce_tensor(t)
            # args[7] is the cudaIpcMemHandle bytes (None under expandable_segments → IPC impossible).
            assert len(args) > 7 and args[7] is not None, (
                f"buffer {name!r}: no CUDA-IPC handle (device={device}). The caching allocator is likely "
                f"in expandable_segments mode — set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False.")
            self._buffers[name] = t
            bundle[name] = (rebuild_fn, args)
        _send_pickle(self._conn, bundle)
        return self._buffers

    def recv_ctl(self):
        assert self._conn is not None, "call accept() first"
        return _recv(self._conn, map_location=self._map)

    def send_ctl(self, obj) -> None:
        assert self._conn is not None, "call accept() first"
        _send(self._conn, obj)


class RpcClient:
    """Client end: ``call(method, payload)`` → result (raises on server-side error).

    ``call`` is the plain socket RPC (v1). Against a ``serve_vec_env`` server, use ``recv_shared()``
    once right after connect (opens the server's CUDA-IPC buffer handles), then drive the hot path by
    reading/writing those buffers around ``send_ctl``/``recv_ctl`` control messages — see
    ``learning/rl/client.SimServiceVecEnv``.
    """

    def __init__(self, host: str, port: int, map_location=None):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._map = map_location

    def call(self, method: str, payload=None):
        _send(self._sock, (method, payload))
        resp = _recv(self._sock, map_location=self._map)
        if not resp.get("ok"):
            raise RuntimeError(f"sim service error: {resp.get('error')}\n{resp.get('tb', '')}")
        return resp.get("result")

    def recv_shared(self) -> dict[str, torch.Tensor]:
        """Open the server's CUDA-IPC handles → tensors on the SAME physical GPU (zero-copy)."""
        bundle = _recv_pickle(self._sock)
        return {name: rebuild_fn(*args) for name, (rebuild_fn, args) in bundle.items()}

    def recv_ctl(self):
        return _recv(self._sock, map_location=self._map)

    def send_ctl(self, obj) -> None:
        _send(self._sock, obj)

    def close(self) -> None:
        try:
            _send(self._sock, ("close", None))
            _recv(self._sock)
        finally:
            self._sock.close()


# --- CUDA-IPC server loop (drop-in for ``RpcServer.serve``, same handler) ----------------------------
def build_layout(attrs: dict, obs_sample) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Fixed-shape shared-buffer layout from the service's ``attrs`` + one obs sample (TensorDict)."""
    n, a = int(attrs["num_envs"]), int(attrs["num_actions"])
    layout: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        K_ACTION: ((n, a), torch.float32),
        K_REW: ((n,), torch.float32),
        K_DONES: ((n,), torch.bool),
        K_TIME_OUTS: ((n,), torch.bool),
        K_TASK_SUCCESS: ((n,), torch.bool),
    }
    for group, v in obs_sample.items():
        layout[obs_key(group)] = (tuple(v.shape), v.dtype)
    return layout


def _write_obs(buffers: dict[str, torch.Tensor], obs) -> None:
    for group in obs.keys():
        buffers[obs_key(group)].copy_(obs[group])


def _copy_or_zero(buf: torch.Tensor, t) -> None:
    # modes that ship neither (e.g. replay's empty extras) read as all-False on the client
    if t is not None:
        buf.copy_(t)
    else:
        buf.zero_()


def serve_vec_env(server: RpcServer, handler) -> None:
    """Serve ``handler`` with the hot-path payload on shared GPU buffers instead of serialized.

    ``handler`` is the SAME ``(method, payload)`` handler ``RpcServer.serve`` takes (its "step" gets the
    shared action buffer as payload and must return ``{"obs", "rew", "dones", "extras"}``; "reset" →
    ``{"obs", "extras"}``; "get_observations" → the obs TensorDict). The loop writes obs/rew/dones plus
    ``extras``'s time_outs/task_success into the buffers, ``torch.cuda.synchronize()``s (so the writes
    are visible to the client before the ack), and replies with the REMAINING extras over the control
    channel. The client synchronizes after writing the action before signaling, so the read is safe.
    Cold methods (attrs/seed/ep_len/...) pass through to ``handler`` unchanged. Call after ``accept()``.
    """
    attrs = handler("attrs", None)
    obs0 = handler("get_observations", None)
    buffers = server.share(build_layout(attrs, obs0), attrs["device"])
    action = buffers[K_ACTION]

    while True:
        method, payload = server.recv_ctl()
        try:
            if method == "close":
                server.send_ctl({"ok": True})
                return
            if method == "step":
                r = handler("step", action)
                extras = r["extras"]
                _write_obs(buffers, r["obs"])
                buffers[K_REW].copy_(r["rew"])
                buffers[K_DONES].copy_(r["dones"])
                _copy_or_zero(buffers[K_TIME_OUTS], extras.get(K_TIME_OUTS))
                _copy_or_zero(buffers[K_TASK_SUCCESS], extras.get(K_TASK_SUCCESS))
                torch.cuda.synchronize()
                rest = {k: v for k, v in extras.items() if k not in (K_TIME_OUTS, K_TASK_SUCCESS)}
                server.send_ctl({"ok": True, "result": {"extras": rest}})
            elif method == "get_observations":
                _write_obs(buffers, handler("get_observations", None))
                torch.cuda.synchronize()
                server.send_ctl({"ok": True, "result": {}})
            elif method == "reset":
                r = handler("reset", None)
                _write_obs(buffers, r["obs"])
                torch.cuda.synchronize()
                server.send_ctl({"ok": True, "result": {"extras": r["extras"]}})
            else:
                server.send_ctl({"ok": True, "result": handler(method, payload)})
        except Exception as e:  # boundary — report the error back, keep serving
            server.send_ctl({"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()})
