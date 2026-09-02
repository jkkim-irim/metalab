"""Live env telemetry dashboard — engine-agnostic, dependency-free (stdlib only).

env_driver (above both backends) publishes a throttled per-env snapshot (obs / per-term reward / action) to
this server; a browser renders it as **live plots** over Server-Sent Events, and can drive the 3D viewer back
(env focus) via a small command channel.

The page is **standalone's** (``dashboard/page.PAGE``) — this module used to carry its own copy, which
drifted from the polished one. ``dashboard/rl_monitor.py`` re-shapes env_driver's snapshots into the schema that
page consumes, so both dashboards share one plotter; the RL side only adds the env selector. A **--viz-only dev inspector** for checking a training env is wired
correctly (and, later, that an eval policy's obs/reward converge) — NOT part of a headless run. Cheap: the sim
main loop only builds a small env-0..2 snapshot every few steps and hands the CPU dict to a background thread;
plotting/buffering happens in the browser (separate process), so there is no GL contention and ~zero sim cost.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import time

# The plot viewer is standalone's, verbatim: one page, one plotter, so the two dashboards cannot drift
# again. It is a zero-import module, so telemetry stays stdlib-only.
from sim.metalab.dashboard import rl_monitor
from sim.metalab.dashboard.page import PAGE


class TelemetryServer:
    """Serves a self-contained dashboard (`/`), the static env description (`/describe`), a live per-env
    snapshot stream (`/stream`, SSE), and a web→sim command sink (`/cmd`, POST). Start once (on --viz);
    call :meth:`publish` with each throttled snapshot and :meth:`drain_commands` from the sim main loop.

    Never opens a browser: the Launchpad embeds this URL in its telemetry tab (it scrapes the printed
    "live dashboard →" line), and that is the ONLY viewer — no process here pops a window of its own."""

    def __init__(self, describe: dict, host: str = "127.0.0.1", port: int = 8782):
        self.describe = describe
        self.latest: dict | None = None
        self.clients = 0                       # active /stream connections (env_driver skips work when 0)
        self.commands: queue.Queue = queue.Queue()   # web → sim commands, drained by the sim main loop
        self._stop = threading.Event()
        self._lock = threading.Lock()

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):        # silence per-request stderr logging
                pass

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/describe":
                    self._send(json.dumps(server.describe).encode(), "application/json")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    with server._lock:
                        server.clients += 1
                    last = None
                    try:
                        while not server._stop.is_set():
                            snap = server.latest
                            if snap is not None and snap is not last:
                                last = snap
                                self.wfile.write(b"data: " + json.dumps(snap).encode() + b"\n\n")
                                self.wfile.flush()
                            time.sleep(0.01)    # poll >publish rate so ~60 Hz snapshots reach the browser
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass                    # client closed the tab — normal
                    finally:
                        with server._lock:
                            server.clients -= 1
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == "/cmd":        # web → sim command (e.g. focus viewer on env i); main loop drains
                    try:
                        n = int(self.headers.get("Content-Length", 0) or 0)
                        server.commands.put(json.loads(self.rfile.read(n) or b"{}"))
                        self.send_response(204)
                        self.end_headers()
                    except Exception:
                        self.send_error(400)
                else:
                    self.send_error(404)

        # Bind the preferred port; fall back to an OS-assigned free port if it is taken.
        try:
            self._httpd = ThreadingHTTPServer((host, port), Handler)
        except OSError:
            self._httpd = ThreadingHTTPServer((host, 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        url = f"http://{host}:{self.port}"
        # The Launchpad scrapes this exact marker from the run log to embed the URL in its telemetry tab.
        print(f"[telemetry] live dashboard → {url}   (--viz inspector: obs / reward / action plots)", flush=True)

    def has_clients(self) -> bool:
        return self.clients > 0

    def drain_commands(self) -> list:
        """Pop all pending web→sim commands (called by the sim main loop each step)."""
        out = []
        try:
            while True:
                out.append(self.commands.get_nowait())
        except queue.Empty:
            pass
        return out

    def publish(self, snapshot: dict) -> None:
        self.latest = snapshot                 # atomic ref swap (GIL) — the SSE threads read it

    def close(self) -> None:
        self._stop.set()
        self._httpd.shutdown()


class LiveDashboard:
    """The DRIVER side of the --viz dashboard: owns the server, the pause flag and the publish cadence.

    ``EnvDriver`` holds one of these and calls three methods — ``drain``, ``note_pause``, ``publish`` — so the
    step loop reads as RL order with three hooks rather than as RL order interleaved with SSE plumbing. A
    headless run holds :data:`NO_DASHBOARD` instead, whose methods are empty, so there is no ``if telemetry``
    in the loop at all.

    A dev-aux inspector: any failure here disables the dashboard rather than killing the run, which is why
    this module (not the driver) is where exceptions get swallowed.
    """

    ENVS = 3          # envs 0..ENVS-1 are sampled per publish
    paused = False

    @classmethod
    def start(cls, driver):
        """Build one, or return :data:`NO_DASHBOARD` if the server cannot come up."""
        try:
            return cls(driver)
        except Exception as e:
            print(f"[telemetry] disabled (server start failed): {e!r}", flush=True)
            return NO_DASHBOARD

    def __init__(self, driver):
        self._rl_monitor = rl_monitor
        desc = driver.snapshot_describe()
        self._overlays = rl_monitor.overlay_plan(desc)
        self._server = TelemetryServer(rl_monitor.describe(desc))
        self.paused = False
        self.every = 1
        self._run({"focus": 0}, driver)

    def drain(self, driver) -> None:
        """Apply every pending web→sim command. Called on the main loop so viewer/CUDA access is safe."""
        for cmd in self._server.drain_commands():
            self._run(cmd, driver)

    def _run(self, cmd: dict, driver) -> None:
        try:
            if "focus" in cmd:
                driver.backend.focus_env(int(cmd["focus"]))
            if "pause" in cmd:
                self.paused = bool(cmd["pause"])
        except Exception as e:
            print(f"[telemetry] command {cmd!r} failed: {e!r}", flush=True)

    def note_pause(self, driver, paused: bool) -> None:
        """Push a state-only snapshot so the page learns about a hold it cannot otherwise see.

        A fresh dict every call: the SSE loop ships ``latest`` only when the OBJECT changes (identity), so
        reusing one would be silently dropped. Carries no ``envs``, which the page already tolerates."""
        self._server.publish({"paused": bool(paused), "t": driver.common_step_counter * driver.step_dt})

    def publish(self, driver, obs) -> None:
        """One live snapshot of envs 0..ENVS-1 plus the val/SR counter, at the publish cadence."""
        if not self._server.has_clients() or driver.common_step_counter % self.every:
            return
        try:
            n = min(self.ENVS, driver.num_envs)
            snap = self._rl_monitor.snapshot(driver.snapshot_rows(range(n)),
                                             t=driver.common_step_counter * driver.step_dt,
                                             overlays=self._overlays)
            att = driver._ep_attempts[:n].cpu().tolist()
            suc = driver._ep_successes[:n].cpu().tolist()
            snap["eval_sr"] = [[int(suc[i]), int(att[i])] for i in range(n)]
            snap["paused"] = self.paused
            self._server.publish(snap)
        except Exception as e:
            print(f"[telemetry] disabled after snapshot error: {e!r}", flush=True)
            driver._telem = NO_DASHBOARD


class _NoDashboard:
    """What a headless run holds — every hook a no-op, so the step loop needs no telemetry branch."""

    paused = False

    def drain(self, driver) -> None: ...
    def note_pause(self, driver, paused: bool) -> None: ...
    def publish(self, driver, obs) -> None: ...


NO_DASHBOARD = _NoDashboard()
