"""Sim-service process plumbing (learning-venv side) — sim-agnostic.

Spawns the selected sim's server, waits for it to publish its port, and tears it down. Task knobs
are SIM-OWNED (written in the sim's env/task files), so nothing experiment-related crosses this
boundary — only run args. learning/ knows NOTHING sim-specific here: the spawn recipe
(interpreter/env activation/module/args) is owned by the sim itself as ``sim/<SIM>/launch.py``
exposing ``spawn(args, port_file) -> subprocess.Popen`` (pure stdlib, no engine imports). The sim
is selected with the ``SIM`` env var (the launch scripts set ``SIM=metalab``).
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
import time

# repo root (learning/rl/service.py -> learning/rl -> learning -> repo)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sim() -> str:
    """The sim package serving this run — a ``sim/<name>/`` dir shipping ``launch.py`` + ``server``."""
    return os.environ.get("SIM", "metalab").lower()


def _spawn(args, port_file: str):
    """Dispatch to the sim's own launcher (``sim.<SIM>.launch.spawn``)."""
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    launcher = importlib.import_module(f"sim.{_sim()}.launch")   # fail-loud on an unknown sim
    return launcher.spawn(args, port_file)


@contextlib.contextmanager
def sim_server(args):
    """Spawn the sim server, yield the port it binds, and terminate it on exit."""
    # Unique per spawn: concurrent sim services on one node (trainer + its val-video hook + a replay/eval
    # from another shell) must not race on a shared port file — each client must read ITS server's port.
    port_file = f"/tmp/sim_service_{os.getpid()}_{time.monotonic_ns()}.port"
    if os.path.exists(port_file):
        os.remove(port_file)
    srv = _spawn(args, port_file)
    try:
        port = None
        for attempt in (1, 2, 3):
            early_rc = None
            for _ in range(1200):  # sim build + CUDA-graph capture can take minutes
                if srv.poll() is not None:
                    early_rc = srv.returncode
                    break
                if os.path.exists(port_file) and open(port_file).read().strip():
                    port = int(open(port_file).read().strip())
                    break
                time.sleep(1)
            if port is not None or early_rc is None:
                break
            # Sim boot occasionally dies before binding the port (e.g. Isaac Sim's glibc malloc abort
            # "malloc(): unaligned tcache chunk detected", SIGABRT -> rc 250/-6) — a boot flake, not a
            # config error (v28 2026-07-12 lost a whole launch to one). Relaunch the server instead
            # of killing the run; an rc that repeats 3x is a real error and still raises.
            if attempt == 3:
                raise RuntimeError(f"sim service exited early rc={early_rc} (3 boot attempts)")
            print(f"[sim-service] boot attempt {attempt} exited early rc={early_rc} — retrying",
                  flush=True)
            srv = _spawn(args, port_file)
        if port is None:
            raise RuntimeError("sim service never wrote a port")
        yield port
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()
        if os.path.exists(port_file):
            os.remove(port_file)
